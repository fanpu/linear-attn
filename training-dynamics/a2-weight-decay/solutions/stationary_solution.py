"""Reference solution: wdecay/stationary.py (Problem stationary).

L_γ(a, b) = ½(ab - 1)² + γ(a² + b²), γ > 0. Weight decay forces a² = b² at every stationary point
(Problem balance), which leaves the origin and, for γ < ½, the two balanced points ±(u, u) with
u² = 1 - 2γ. Hessian [[b² + 2γ, 2ab - 1], [2ab - 1, a² + 2γ]]: at the origin [[2γ, -1], [-1, 2γ]]
(eigenvalues 2γ ± 1), at ±(u, u) [[1, 1 - 4γ], [1 - 4γ, 1]] (eigenvalues 4γ and 2 - 4γ).
"""
from __future__ import annotations

import numpy as np


def hessian_wd(theta, gamma):
    theta = np.asarray(theta, dtype=float)
    a, b = theta[..., 0], theta[..., 1]
    off = 2.0 * a * b - 1.0
    return np.stack([np.stack([b * b + 2 * gamma, off], -1),
                     np.stack([off, a * a + 2 * gamma], -1)], -2)


def _kind(H, tol=1e-9):
    lam = np.linalg.eigvalsh(H)
    eps = tol * max(1.0, float(np.abs(H).max()))
    if np.any(np.abs(lam) <= eps):
        return "degenerate"
    if np.all(lam > 0):
        return "minimum"
    if np.all(lam < 0):
        return "maximum"
    return "saddle"


def stationary_points(gamma):
    gamma = float(gamma)
    out = [(np.zeros(2), _kind(hessian_wd(np.zeros(2), gamma)))]
    if gamma < 0.5:
        u = np.sqrt(1.0 - 2.0 * gamma)
        for sgn in (1.0, -1.0):
            p = sgn * np.array([u, u])
            out.append((p, _kind(hessian_wd(p, gamma))))
    return out


def learned_product(gamma):
    g = np.asarray(gamma, dtype=float)
    return np.where(g < 0.5, 1.0 - 2.0 * g, 0.0)
