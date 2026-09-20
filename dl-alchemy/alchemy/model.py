"""Baseline transformer. `Block` and `Transformer` follow the CS 312 example snippet
(source/cs312-site-2026-09-19.md); quiz diffs are written against this file."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    depth: int = 8
    width: int = 384
    head_dim: int = 64
    mlp_ratio: int = 4
    context: int = 512


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, width, head_dim):
        super().__init__()
        assert width % head_dim == 0
        self.heads = width // head_dim
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        b, t, w = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, w))


class MLP(nn.Module):
    def __init__(self, width, mlp_ratio):
        super().__init__()
        self.up = nn.Linear(width, mlp_ratio * width, bias=False)
        self.proj = nn.Linear(mlp_ratio * width, width, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, width, head_dim, mlp_ratio):
        super().__init__()
        self.attn_norm = RMSNorm(width)
        self.attn = Attention(width, head_dim)
        self.mlp_norm = RMSNorm(width)
        self.mlp = MLP(width, mlp_ratio)

    def update(self, x):
        attn = self.attn(self.attn_norm(x))
        mlp = self.mlp(self.mlp_norm(x + attn))
        return attn + mlp

    def forward(self, x):
        return x + self.update(x)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position_embed = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.ModuleList(
            [Block(cfg.width, cfg.head_dim, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.final_norm = RMSNorm(cfg.width)
        self.lm_head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.apply(self._init)
        # GPT-2 style: residual-branch output projections scaled down with depth
        for block in self.blocks:
            for proj in (block.attn.proj, block.mlp.proj):
                nn.init.normal_(proj.weight, std=0.02 / math.sqrt(2 * cfg.depth))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embed(tokens)
        x = x + self.position_embed(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def param_groups(self):
        """Named groups for logging norms."""
        g = {"embed": [self.token_embed.weight], "pos": [self.position_embed.weight],
             "head": [self.lm_head.weight], "attn": [], "mlp": [], "norm": [self.final_norm.weight]}
        for b in self.blocks:
            g["attn"] += [b.attn.qkv.weight, b.attn.proj.weight]
            g["mlp"] += [b.mlp.up.weight, b.mlp.proj.weight]
            g["norm"] += [b.attn_norm.weight, b.mlp_norm.weight]
        return g

    def accounting(self):
        """Parameter counts and training FLOPs per token (fwd+bwd = 3x fwd; matmuls only)."""
        c = self.cfg
        body = sum(p.numel() for b in self.blocks for p in b.parameters())
        embed = self.token_embed.weight.numel() + self.position_embed.weight.numel()
        head = self.lm_head.weight.numel()
        attn_flops = 6 * c.depth * c.context * c.width  # QK^T and AV, causal not discounted
        return {
            "params_body": body, "params_embed": embed, "params_head": head,
            "flops_per_token_body": 6 * body, "flops_per_token_head": 6 * head,
            "flops_per_token_attn": attn_flops,
            "flops_per_token_total": 6 * (body + head) + attn_flops,
        }
