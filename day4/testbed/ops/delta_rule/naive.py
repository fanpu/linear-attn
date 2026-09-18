"""Reference recurrence for the delta rule (DeltaNet, no decay gate). [you]

after fla/ops/delta_rule/naive.py. Note the layout: fla's own naive file uses
`[B, H, T, D]`; this one uses `[B, T, H, D]`, the layout of the chunked op
it is tested against, so that no transpose sits between the two.
"""

from __future__ import annotations

import torch
from einops import einsum


def naive_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The delta rule, one token at a time.

    For every batch element and head, starting from S_0 = 0, for t = 1..T:

        S_t = S_{t-1} + beta_t (v_t - S_{t-1}^T k_t) k_t^T
        o_t = S_t^T (scale * q_t)

    The bracket is the error between the value to store and what the state
    already predicts for the key; beta_t in [0, 1] is how much of it is
    written. beta_t = 1 stores v_t exactly for the key k_t (when |k_t| = 1);
    beta_t = 0 writes nothing. With the bracket replaced by v_t this is
    `naive_linear_attn`.

    Args:
        q, k:  `[B, T, H, K]`. The layer L2-normalizes k before calling this;
               the function itself does not normalize anything.
        v:     `[B, T, H, V]`.
        beta:  `[B, T, H]`.
        scale: multiplier on the queries; `None` means `K ** -0.5`.

    Returns:
        o: `[B, T, H, V]` in the dtype of `v`.
        S: `[B, H, K, V]` in fp32, the final state.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K**-0.5

    S = torch.zeros((B, H, K, V), dtype=torch.float32, device=v.device)
    o = torch.empty((B, T, H, V), dtype=v.dtype, device=v.device)

    for t in range(T):
        q_t = q[:, t].float()
        k_t = k[:, t].float()
        v_t = v[:, t].float()

        S_update = einsum(
            (v_t - einsum(S, k_t, "b h k v, b h k -> b h v")),
            k_t,
            "b h v, b h k -> b h k v",
        )
        S += einsum(beta[:, t].float(), S_update, "b h, b h k v -> b h k v")
        o[:, t] = einsum(S, scale * q_t, "b h k v, b h k -> b h v")

    return (o, S)
