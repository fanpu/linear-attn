"""Fast additive linear attention: a thin wrapper around fla's chunked kernel. [AI]

after fla/ops/linear_attn/chunk.py. The wrapper fixes the two settings the
day uses (no output normalization, no initial state) so that the layer and
the tests call one function with one signature.
"""
from __future__ import annotations

import torch


def chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same contract as `naive.naive_linear_attn`; GPU only.

    fla's `chunk_linear_attn(..., normalize=False)` computes
    S_t = S_{t-1} + k_t v_t^T and o_t = S_t^T (scale * q_t), which is the
    reference recurrence; `normalize=True` (fla's default) would divide by a
    running sum of keys, which is Katharopoulos et al.'s normalized variant
    and not what today's layer uses.
    """
    from fla.ops.linear_attn import chunk_linear_attn as _fla_chunk_linear_attn  # noqa: WPS433

    o, S = _fla_chunk_linear_attn(q, k, v, scale=scale, output_final_state=True, normalize=False)
    return o, S
