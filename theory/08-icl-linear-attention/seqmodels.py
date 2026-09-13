"""Four token mixers at matched depth/width for in-context regression: softmax attention, linear attention,
DeltaNet, Gated DeltaNet.  Minimal, explicit parameterizations; fast paths use flash-linear-attention kernels,
and every recurrence has a pure-torch reference (see test_seqmodels.py for the equivalence check).

Prompt format (one token per (x,y) pair, so the model only has to learn the regression algorithm):
    [ (x_1, 0, 1), (x_1, y_1, 0), (x_2, 0, 1), (x_2, y_2, 0), ... ]      token = (x, y, is_query)
Prediction for y_t is read out at the query token (x_t, 0, 1), which sees only pairs < t (causal).
"""
import os
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")  # Triton tl.dot defaults to TF32; use IEEE fp32
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------------------------ recurrences
def linear_attn_ref(q, k, v):
    """S_t = S_{t-1} + v_t k_t^T,  o_t = S_t q_t / t   (causal-mean linear attention; [B,T,H,*])."""
    B, T, H, Dk = q.shape
    S = q.new_zeros(B, H, v.shape[-1], Dk)
    out = []
    for t in range(T):
        S = S + torch.einsum("bhv,bhk->bhvk", v[:, t], k[:, t])
        out.append(torch.einsum("bhvk,bhk->bhv", S, q[:, t]) / (t + 1))
    return torch.stack(out, 1)


def linear_attn_par(q, k, v):
    """Same as linear_attn_ref, computed in parallel with a causal mask (O(T^2), fine for T<=200)."""
    T = q.shape[1]
    A = torch.einsum("bthk,bshk->bhts", q, k)
    A = A * torch.tril(torch.ones(T, T, device=q.device, dtype=q.dtype))
    A = A / torch.arange(1, T + 1, device=q.device, dtype=q.dtype)[:, None]
    return torch.einsum("bhts,bshv->bthv", A, v)


def delta_ref(q, k, v, beta, log_alpha=None):
    """(Gated) delta rule, pure torch:  S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,  o_t = S_t q_t.
    With alpha = 1 this is DeltaNet: one SGD step on 1/2 ||S k_t - v_t||^2 with learning rate beta_t."""
    B, T, H, Dk = q.shape
    S = q.new_zeros(B, H, v.shape[-1], Dk)
    out = []
    for t in range(T):
        if log_alpha is not None:
            S = S * log_alpha[:, t].exp()[..., None, None]
        kt, vt, bt = k[:, t], v[:, t], beta[:, t][..., None]
        pred = torch.einsum("bhvk,bhk->bhv", S, kt)
        S = S + torch.einsum("bhv,bhk->bhvk", bt * (vt - pred), kt)
        out.append(torch.einsum("bhvk,bhk->bhv", S, q[:, t]))
    return torch.stack(out, 1)


def delta_fast(q, k, v, beta, log_alpha=None):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    dt = q.dtype
    if log_alpha is None:
        log_alpha = torch.zeros_like(beta)
    o, _ = chunk_gated_delta_rule(q.float(), k.float(), v.float(), log_alpha.float(), beta.float(), scale=1.0)
    return o.to(dt)


# ------------------------------------------------------------------------------------------------ mixers
class Mixer(nn.Module):
    def __init__(self, kind, width, heads, fast=True):
        super().__init__()
        self.kind, self.h, self.hd, self.fast = kind, heads, width // heads, fast
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        if kind in ("delta", "gdelta"):
            self.b = nn.Linear(width, heads)            # beta_t = sigmoid(b(h_t))
        if kind == "gdelta":                             # alpha_t = exp(-exp(A_log) softplus(a(h_t) + dt_bias)), as in fla
            self.a = nn.Linear(width, heads, bias=False)
            self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(1, 16)))
            dt = torch.exp(torch.rand(heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        B, T, _ = x.shape
        sh = lambda t: t.view(B, T, self.h, self.hd)
        q, k, v = sh(self.q(x)), sh(self.k(x)), sh(self.v(x))
        if self.kind == "softmax":
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        elif self.kind == "linear":
            o = linear_attn_par(q, k, v) / math.sqrt(self.hd)
        else:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            beta = torch.sigmoid(self.b(x))
            la = None
            if self.kind == "gdelta":
                la = -self.A_log.exp() * F.softplus(self.a(x) + self.dt_bias)
            o = (delta_fast if self.fast else delta_ref)(q, k, v, beta, la)
        return self.o(o.reshape(B, T, -1))


class ICLModel(nn.Module):
    def __init__(self, kind, d, width=128, heads=4, layers=2, mlp=True):
        super().__init__()
        self.kind, self.d = kind, d
        self.inp = nn.Linear(d + 2, width)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            blk = nn.ModuleDict({"ln1": nn.LayerNorm(width), "mix": Mixer(kind, width, heads)})
            if mlp:
                blk["ln2"] = nn.LayerNorm(width)
                blk["mlp"] = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
            self.blocks.append(blk)
        self.lnf = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)

    def set_fast(self, fast):
        for b in self.blocks:
            b["mix"].fast = fast

    def forward(self, X, y, xq=None):
        """X [B,n,d], y [B,n].  Returns predictions for every y_t from pairs < t, shape [B,n].
        If xq [B,d] is given, also appends one final query and returns its prediction as [B]."""
        B, n, d = X.shape
        z = X.new_zeros(B, n, 1)
        qtok = torch.cat([X, z, z + 1], -1)
        ctok = torch.cat([X, y[..., None], z], -1)
        tok = torch.stack([qtok, ctok], 2).reshape(B, 2 * n, d + 2)
        if xq is not None:
            tok = torch.cat([tok, torch.cat([xq, xq.new_zeros(B, 1), xq.new_ones(B, 1)], -1)[:, None]], 1)
        h = self.inp(tok)
        for b in self.blocks:
            h = h + b["mix"](b["ln1"](h))
            if "mlp" in b:
                h = h + b["mlp"](b["ln2"](h))
        p = self.out(self.lnf(h))[..., 0]
        if xq is not None:
            return p[:, -1]
        return p[:, 0::2]
