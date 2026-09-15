"""Reference implementations of linear-attention recurrences, plus thin wrappers around the fla kernels.

Everything uses shapes [B, T, H, D] (batch, time, heads, head dim), the same convention as fla.

The one recurrence to keep in mind: a matrix-valued memory S_t (d_v x d_k) that is read with a query q_t and
written with a key/value pair (k_t, v_t).

    linear attention         S_t = a_t S_{t-1} + v_t k_t^T                           (a_t = 1: vanilla; a_t < 1: decay)
    (gated) delta rule       S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T    (a_t = 1: DeltaNet)
    output                   o_t = S_t q_t

Reading the delta rule as online learning: S_t is one SGD step (learning rate b_t) on the loss
1/2 ||S k_t - v_t||^2, taken after shrinking the old memory by a_t. Plain linear attention skips the
"- S k_t" error-correction term, so it only ever adds.

`*_ref` functions are step-by-step loops (slow, obviously correct; use float64 as ground truth).
`delta_par` is an exact parallel form, copied from theory/08-icl-linear-attention/seqmodels.py (Fan Pu's project).
`fla_*` wrap the Triton kernels actually used for training (float32 / bf16, CUDA only).
Gates are passed in log space (log_a = log a_t <= 0), matching fla's `g` argument.
"""
import os

# Triton tl.dot defaults to TF32 on Ampere+ GPUs, and fla only overrides that on older cards. On the GB10 this makes
# the chunk delta-rule kernel ~30x less accurate (rel. err ~2e-3 vs ~5e-5). Must be set before fla compiles kernels.
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import torch


def linear_ref(q, k, v, log_a=None, S0=None):
    """(Decayed) linear attention, step by step. Returns (outputs [B,T,H,Dv], final state [B,H,Dv,Dk])."""
    B, T, H, Dk = q.shape
    S = q.new_zeros(B, H, v.shape[-1], Dk) if S0 is None else S0.clone()
    out = []
    for t in range(T):
        if log_a is not None:
            S = S * log_a[:, t].exp()[..., None, None]
        S = S + torch.einsum("bhv,bhk->bhvk", v[:, t], k[:, t])
        out.append(torch.einsum("bhvk,bhk->bhv", S, q[:, t]))
    return torch.stack(out, 1), S


def delta_ref(q, k, v, beta, log_a=None, S0=None):
    """(Gated) delta rule, step by step. beta: [B,T,H]. Returns (outputs, final state)."""
    B, T, H, Dk = q.shape
    S = q.new_zeros(B, H, v.shape[-1], Dk) if S0 is None else S0.clone()
    out = []
    for t in range(T):
        if log_a is not None:
            S = S * log_a[:, t].exp()[..., None, None]
        kt, bt = k[:, t], beta[:, t][..., None]
        pred = torch.einsum("bhvk,bhk->bhv", S, kt)                     # what the memory currently returns for k_t
        S = S + torch.einsum("bhv,bhk->bhvk", bt * (v[:, t] - pred), kt)  # move it toward v_t
        out.append(torch.einsum("bhvk,bhk->bhv", S, q[:, t]))
    return torch.stack(out, 1), S


def delta_par(q, k, v, beta, log_a=None):
    """Exact parallel (gated) delta rule via one triangular solve; O(T^2) memory, fine for T <= ~1000.

    Unrolling gives S_t = sum_{i<=t} b_i g_{t,i} u_i k_i^T with g_{t,i} = prod_{m=i+1..t} a_m and "pseudo-values"
    u_t = v_t - sum_{i<t} b_i g_{t,i} (k_t.k_i) u_i. In matrix form (I + M) U = V and O = C U with
    M[t,i] = b_i g_{t,i} k_t.k_i (i<t) and C[t,i] = b_i g_{t,i} q_t.k_i (i<=t).
    (From theory/08-icl-linear-attention/seqmodels.py.)
    """
    B, T, H, _ = q.shape
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B,H,T,*]
    G = beta.transpose(1, 2)[:, :, None, :].expand(B, H, T, T)
    if log_a is not None:
        cum = log_a.transpose(1, 2).cumsum(-1)
        G = G * torch.exp(torch.clamp(cum[..., :, None] - cum[..., None, :], max=0.0))
    tril = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    strict = tril & ~torch.eye(T, dtype=torch.bool, device=q.device)
    M = torch.where(strict, G * (kh @ kh.transpose(-1, -2)), 0.0)
    I = torch.eye(T, device=q.device, dtype=q.dtype)
    U = torch.linalg.solve_triangular(I + M, vh, upper=False, unitriangular=True)
    C = torch.where(tril, G * (qh @ kh.transpose(-1, -2)), 0.0)
    return (C @ U).transpose(1, 2)


# ---------------------------------------------------------------------------------------------- fla kernels
def fla_delta(q, k, v, beta, log_a=None, mode="chunk"):
    """fla gated delta rule (log_a=None -> DeltaNet). mode: 'chunk' (training kernel) or 'recurrent'."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    if log_a is None:
        log_a = torch.zeros_like(beta)
    fn = chunk_gated_delta_rule if mode == "chunk" else fused_recurrent_gated_delta_rule
    o, S = fn(q, k, v, g=log_a, beta=beta, scale=1.0, output_final_state=True)
    return o, S.transpose(-1, -2)  # fla stores the state as [B,H,Dk,Dv]; we use [B,H,Dv,Dk]


def fla_linear(q, k, v, log_a=None):
    """fla (decayed) linear attention via simple_gla (scalar per-head, per-token decay)."""
    from fla.ops.simple_gla import chunk_simple_gla

    if log_a is None:
        log_a = q.new_zeros(q.shape[:3])
    o, S = chunk_simple_gla(q, k, v, g=log_a, scale=1.0, output_final_state=True)
    return o, S.transpose(-1, -2)  # fla stores the state as [B,H,Dk,Dv]; we use [B,H,Dv,Dk]
