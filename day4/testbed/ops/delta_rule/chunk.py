"""Fast delta rule: a thin wrapper around fla's chunked kernel. [AI]

after fla/ops/delta_rule/chunk.py. The kernel's `use_qk_l2norm_in_kernel`
is left off so that the L2 normalization happens in the layer, identically
for the additive and the delta arms.
"""
from __future__ import annotations

import torch


def chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same contract as `naive.naive_delta_rule`; GPU only."""
    from fla.ops.delta_rule import chunk_delta_rule as _fla_chunk_delta_rule  # noqa: WPS433

    o, S = _fla_chunk_delta_rule(q, k, v, beta, scale=scale, output_final_state=True)
    return o, S
