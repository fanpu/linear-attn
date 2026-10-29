"""Reference solution: symmetries/saxe.py (Problem saxe).

Whitened inputs (Σxx = I), Σyx = Σ_α s_α u_α v_αᵀ (SVD, s_α > 0). From the aligned balanced
init W1 = R √D Vᵀ, W2 = U √D Rᵀ every mode is the anchor with target s_α:
    W(t) = W2 W1 = Σ_α u_α(t) u_α v_αᵀ,   u_α(t) = s_α / (1 + (s_α/u0_α - 1) e^{-2 s_α t}).
From a small random init the early (linear) phase grows x + y like e^{s t}, with
x = W1 v_α, y = W2ᵀ u_α, so the mode starts its sigmoid from u0_eff = |x + y|² / 4.
"""
from __future__ import annotations

import numpy as np


def _modes(Syx, tol=1e-10):
    U, S, Vt = np.linalg.svd(np.asarray(Syx, dtype=float), full_matrices=False)
    keep = S > tol * max(S.max(), 1e-300)
    return U[:, keep], S[keep], Vt[keep]


def saxe_prediction(Syx, u0, t):
    U, S, Vt = _modes(Syx)
    u0 = np.broadcast_to(np.asarray(u0, dtype=float), S.shape)
    u = S / (1.0 + (S / u0 - 1.0) * np.exp(-2.0 * S * float(t)))
    return (U * u) @ Vt


def aligned_init(Syx, u0, hidden):
    U, S, Vt = _modes(Syx)
    r = len(S)
    if hidden < r:
        raise ValueError(f"hidden={hidden} < rank {r}")
    root = np.sqrt(np.broadcast_to(np.asarray(u0, dtype=float), S.shape))
    R = np.eye(hidden)[:, :r]                       # any hidden×r matrix with orthonormal columns
    W1 = R @ np.diag(root) @ Vt                     # (hidden, d_in)
    W2 = U @ np.diag(root) @ R.T                    # (d_out, hidden)
    return W1, W2


def effective_u0(W1, W2, Syx):
    U, S, Vt = _modes(Syx)
    W1, W2 = np.asarray(W1, dtype=float), np.asarray(W2, dtype=float)
    x = W1 @ Vt.T                                   # column α is W1 v_α
    y = W2.T @ U                                    # column α is W2ᵀ u_α
    return np.sum((x + y) ** 2, axis=0) / 4.0
