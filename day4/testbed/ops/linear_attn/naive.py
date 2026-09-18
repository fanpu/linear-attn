"""Reference recurrence for additive linear attention. [you]

after fla/ops/linear_attn/naive.py: the reference lives beside the fast op
(`chunk.py`) and is the oracle the fast op is tested against.
"""

from __future__ import annotations

import torch
from einops import einsum


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
    B, T, H, K = q.shape
    V = v.shape[-1]

    S = torch.zeros((B, H, K, V), dtype=torch.float32, device=v.device)
    o = torch.empty((B, T, H, V), dtype=v.dtype, device=v.device)

    if scale is None:
        scale = K**-0.5

    for t in range(T):
        q_t = scale * q[:, t].float()  # (B, H, K)
        k_t = k[:, t].float()  # (B, H, K)
        v_t = v[:, t].float()  # (B, H, V)

        # (B, H, K, V)
        S += einsum(k_t, v_t, "b h k, b h v -> b h k v")

        o[:, t] = einsum(S, q_t, "b h k v, b h k -> b h v")

    return (o, S)
