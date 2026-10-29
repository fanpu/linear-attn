"""Reference solution: wdecay/finite_step.py (Problem finite-step).

GD θ' = θ - η∇L_γ has the same fixed points as the flow; one is linearly stable iff every Hessian
eigenvalue λ satisfies |1 - ηλ| < 1, i.e. 0 < λ < 2/η. Hence
    γ < ½ (nonzero minimum, eigenvalues 4γ and 2 - 4γ):  η < 2 / max(4γ, 2 - 4γ)
    γ > ½ (origin, eigenvalues 2γ ± 1):                  η < 2 / (2γ + 1)
For γ < ¼ the first eigenvalue to cross 2/η is 2 - 4γ (direction (1, 1)); for ¼ < γ < ½ it is 4γ
(direction (1, -1)).
"""
from __future__ import annotations

import numpy as np


def max_stable_step(gamma):
    g = np.asarray(gamma, dtype=float)
    out = np.where(g < 0.5, 2.0 / np.maximum(4.0 * g, 2.0 - 4.0 * g), 2.0 / (2.0 * g + 1.0))
    return out[()] if out.ndim == 0 else out


def predicted_phase(eta, gamma):
    eta, gamma = float(eta), float(gamma)
    if eta >= max_stable_step(gamma):
        return "unstable"
    return "nonzero" if gamma < 0.5 else "zero"
