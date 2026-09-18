"""Reference recurrence for additive linear attention. [you]

after fla/ops/linear_attn/naive.py: the reference lives beside the fast op
(`chunk.py`) and is the oracle the fast op is tested against.
"""
from __future__ import annotations

import torch


def naive_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive (unnormalized) linear attention, one token at a time.

    For every batch element and head, starting from S_0 = 0 (a d_k x d_v
    matrix of zeros), for t = 1..T:

        S_t = S_{t-1} + k_t v_t^T
        o_t = S_t^T (scale * q_t)

    Args:
        q, k: `[B, T, H, K]`.
        v:    `[B, T, H, V]`.
        scale: multiplier on the queries; `None` means `K ** -0.5`, which is
            what the fast op uses by default.

    Returns:
        o: `[B, T, H, V]` in the dtype of `v`.
        S: `[B, H, K, V]` in fp32, the final state.

    The state and the arithmetic are fp32 regardless of the input dtype.
    """
    raise NotImplementedError
