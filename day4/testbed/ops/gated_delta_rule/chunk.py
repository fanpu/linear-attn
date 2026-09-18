"""Fast gated delta rule (the Day 1 kernel), for the optional ablation. [AI]

after fla/ops/gated_delta_rule/chunk.py. Same contract as
`ops/delta_rule/chunk.py` plus the log-decay `g` of shape `[B, T, H]`
(g <= 0; exp(g_t) multiplies the state before the write).
"""
from __future__ import annotations

import torch


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla  # noqa: WPS433

    o, S = _fla(q, k, v, g, beta, scale=scale, output_final_state=True)
    return o, S
