"""Multi-head softmax attention mixer. [AI]

after zoology/mixers/attention.py (MHA): one fused qkv projection with bias,
dropout on the attention weights, an output projection. The softmax itself
runs through PyTorch's scaled_dot_product_attention (SDPA) instead of
Zoology's explicit einsum, which is what the sprint's transformer baseline
uses; the arithmetic is the same.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 2, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = rearrange(self.Wqkv(x), "b t (three h d) -> b h t three d", three=3, d=self.head_dim)
        q, k, v = qkv.unbind(dim=3)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(rearrange(o, "b h t d -> b t (h d)"))

    def state_elements(self, seq_len: int) -> int:
        """Numbers held per layer at generation time: the KV cache, 2 * T * d_model."""
        return 2 * seq_len * self.d_model
