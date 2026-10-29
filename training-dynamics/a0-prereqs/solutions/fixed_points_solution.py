"""Reference solution: prereqs/fixed_points.py (Problem phase-portrait)."""
from __future__ import annotations

import numpy as np


def classify_fixed_point(J, tol=1e-9):
    J = np.asarray(J, dtype=float)
    assert J.shape == (2, 2), "classify_fixed_point expects a 2x2 Jacobian"
    eps = tol * max(1.0, float(np.abs(J).max()))
    lam = np.linalg.eigvals(J)
    re, im = lam.real, lam.imag
    if np.any(np.abs(lam) <= eps):
        return "degenerate"                       # a zero eigenvalue: linearization says nothing
    if np.all(np.abs(im) > eps):                  # complex-conjugate pair
        if np.all(np.abs(re) <= eps):
            return "center"
        return "stable spiral" if re[0] < 0 else "unstable spiral"
    if np.all(re < 0):
        return "stable node"
    if np.all(re > 0):
        return "unstable node"
    return "saddle"
