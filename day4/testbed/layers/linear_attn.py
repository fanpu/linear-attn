"""Linear-attention mixer with a switchable update rule. [AI]

after fla/layers/delta_net.py (the path with `use_short_conv=False`,
`use_gate=False`, `qk_activation='silu'`, `qk_norm='l2'`): q/k/v projections,
SiLU on all three, L2-normalized q and k, a per-head beta = sigmoid(b_proj x)
for the delta rule, a per-head RMSNorm on the output, and an output
projection. The only difference between `rule="additive"` and `rule="delta"`
is the op that is called, so a comparison between the two changes exactly
one thing: whether the write subtracts what the state already predicts.

`impl="chunk"` calls fla's kernels (GPU); `impl="naive"` calls the reference
recurrences in `testbed/ops/*/naive.py`, which run anywhere once written.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from testbed.ops.delta_rule import chunk as delta_chunk
from testbed.ops.delta_rule import naive as delta_naive
from testbed.ops.gated_delta_rule import chunk as gated_chunk
from testbed.ops.linear_attn import chunk as linear_chunk
from testbed.ops.linear_attn import naive as linear_naive

RULES = ("additive", "delta", "gated_delta")


class LinearAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 2, rule: str = "delta", impl: str = "chunk", norm_eps: float = 1e-5):
        super().__init__()
        assert rule in RULES, rule
        assert impl in ("chunk", "naive"), impl
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rule = rule
        self.impl = impl
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        if rule in ("delta", "gated_delta"):
            self.b_proj = nn.Linear(d_model, num_heads, bias=False)
        if rule == "gated_delta":
            # after fla/layers/gated_deltanet.py: g = -softplus(a_proj x) * exp(A_log)
            self.a_proj = nn.Linear(d_model, num_heads, bias=False)
            self.A_log = nn.Parameter(torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32)))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.last_beta_mean = None    # mean write gate over the last forward, delta rules only
        self.last_decay_mean = None   # mean decay alpha = exp(g) over the last forward, gated rule only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = F.silu(self.q_proj(x))
        k = F.silu(self.k_proj(x))
        v = F.silu(self.v_proj(x))
        q, k, v = (rearrange(t, "b t (h d) -> b t h d", d=self.head_dim) for t in (q, k, v))
        # F.normalize is an autocast-to-fp32 op (it goes through linalg_vector_norm),
        # so under bf16 autocast q and k come back fp32 while v stays bf16 and fla's
        # kernels reject the mix. Normalize in fp32, hand the kernels one dtype.
        q = F.normalize(q, p=2, dim=-1).to(v.dtype)
        k = F.normalize(k, p=2, dim=-1).to(v.dtype)

        if self.rule == "additive":
            fn = linear_chunk.chunk_linear_attn if self.impl == "chunk" else linear_naive.naive_linear_attn
            o, _ = fn(q, k, v)
        elif self.rule == "delta":
            beta = torch.sigmoid(self.b_proj(x))  # [B, T, H]
            self.last_beta_mean = beta.detach().float().mean()  # a tensor; train_mqar.py reads it at log time (no sync per step)
            fn = delta_chunk.chunk_delta_rule if self.impl == "chunk" else delta_naive.naive_delta_rule
            o, _ = fn(q, k, v, beta)
        else:
            assert self.impl == "chunk", "the gated rule has no naive path in this repository"
            beta = torch.sigmoid(self.b_proj(x))
            self.last_beta_mean = beta.detach().float().mean()
            g = -F.softplus(self.a_proj(x).float()) * torch.exp(self.A_log)  # [B, T, H], <= 0
            self.last_decay_mean = g.detach().exp().mean()
            o, _ = gated_chunk.chunk_gated_delta_rule(q, k, v, g, beta)

        o = self.o_norm(o.float()).to(x.dtype)  # RMSNorm in fp32, as fla's o_norm does
        return self.o_proj(rearrange(o, "b t h d -> b t (h d)"))

    def state_elements(self, seq_len: int) -> int:
        """Numbers held per layer at generation time: one d_k x d_v matrix per head."""
        return self.num_heads * self.head_dim * self.head_dim
