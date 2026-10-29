"""Reference solution: prereqs/modes.py (Problem modes).

W(t) = W* + (W0 - W*) U exp(-Λ t) Uᵀ,   W* = Σ_yx Σ_xx⁻¹,   Σ_xx = U Λ Uᵀ.
"""
from __future__ import annotations

import numpy as np


def mode_dynamics(W0, Sxx, Syx, t):
    W0, Sxx, Syx = (np.asarray(m, dtype=float) for m in (W0, Sxx, Syx))
    lam, U = np.linalg.eigh(Sxx)                     # Σ_xx = U diag(λ) Uᵀ
    W_star = np.linalg.solve(Sxx, Syx.T).T           # Σ_yx Σ_xx⁻¹ (Σ_xx symmetric)
    decay = U @ np.diag(np.exp(-lam * t)) @ U.T      # exp(-Σ_xx t)
    return W_star + (W0 - W_star) @ decay
