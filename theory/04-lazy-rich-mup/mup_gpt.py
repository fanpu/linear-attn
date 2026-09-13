"""A small pre-LN GPT with switchable standard (SP) or maximal-update (muP) parameterization, plus depth-muP.

Width rules follow Tensor Programs V (Yang et al. 2021), Table 3 in its "multiplier" form (Table 8), with base
width d0 = 128 so that SP and muP are the *same model* at d = 128.  r = d / d0 is the width multiplier.

                        SP                          muP
  token/pos embedding   N(0, 1),  Adam LR eta       N(0, 1),  Adam LR eta
  hidden matrices       N(0, 1/fan_in), LR eta      N(0, 1/fan_in), LR eta / r
  LayerNorm gain/bias   LR eta                      LR eta
  readout (unembed)     N(0, 1/d), LR eta           N(0, 1/d), LR eta, logits multiplied by 1/r
  attention logits      q.k / sqrt(d_head)          q.k * sqrt(d_head0) / d_head        (1/d attention)

Depth (Tensor Programs VI, "Depth-muP", block depth caveat noted in the post): every residual branch is
multiplied by sqrt(L0 / L) and the hidden-matrix Adam LR by sqrt(L0 / L), with base depth L0 = 4. With
depth_mup=False, branches have multiplier 1 and LRs do not depend on depth.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    d: int = 128
    L: int = 4
    n_head: int = 4
    vocab: int = 8192
    ctx: int = 256
    param: str = "mup"  # "sp" | "mup"
    d0: int = 128
    L0: int = 4
    depth_mup: bool = False

    @property
    def r(self):
        return self.d / self.d0 if self.param == "mup" else 1.0

    @property
    def branch(self):
        return math.sqrt(self.L0 / self.L) if self.depth_mup else 1.0


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        d, h = c.d, c.n_head
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
        dh, dh0 = d // h, c.d0 // h
        self.attn_scale = 1 / math.sqrt(dh) if c.param == "sp" else math.sqrt(dh0) / dh

    def attn(self, x):
        B, T, d = x.shape
        h = self.c.n_head
        q, k, v = (lin(x).view(B, T, h, d // h).transpose(1, 2) for lin in (self.q, self.k, self.v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
        return self.o(y.transpose(1, 2).reshape(B, T, d))

    def forward(self, x):
        a = self.attn(self.ln1(x)) * self.c.branch
        x = x + a
        m = self.fc2(F.gelu(self.fc1(self.ln2(x)))) * self.c.branch
        return x + m, a, m


class GPT(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.L))
        self.lnf = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.reset(seed=0)

    @torch.no_grad()
    def reset(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        for name, p in self.named_parameters():
            if p.ndim == 1:
                p.copy_(torch.ones_like(p) if name.endswith("weight") else torch.zeros_like(p))
            elif name.startswith(("tok", "pos")):
                p.copy_(torch.randn(p.shape, generator=g))
            else:  # hidden matrices and readout: variance 1/fan_in
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))

    def hidden_matrices(self):
        return [(n, p) for n, p in self.named_parameters() if p.ndim == 2 and n.startswith("blocks")]

    def param_groups(self, lr):
        c = self.c
        hid = {id(p) for _, p in self.hidden_matrices()}
        hidden_lr = lr / c.r * (c.branch if c.depth_mup else 1.0)
        return [
            {"params": [p for p in self.parameters() if id(p) in hid], "lr": hidden_lr, "base": hidden_lr},
            {"params": [p for p in self.parameters() if id(p) not in hid], "lr": lr, "base": lr},
        ]

    def forward(self, idx, taps=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        if taps is not None:
            taps["embed"] = x
        for i, b in enumerate(self.blocks):
            x, a, m = b(x)
            if taps is not None:
                taps[f"attn{i}"], taps[f"mlp{i}"] = a, m
        logits = self.head(self.lnf(x)) / self.c.r
        if taps is not None:
            taps["logits"] = logits
        return logits


def lr_at(step, total, warmup_frac=0.05, final_frac=0.1):
    w = max(1, int(total * warmup_frac))
    if step < w:
        return (step + 1) / w
    prog = (step - w) / max(1, total - w)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * prog))
