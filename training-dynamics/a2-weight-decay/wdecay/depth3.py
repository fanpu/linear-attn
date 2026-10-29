"""Depth 3: the scalar network abc with weight decay.  [you]
Handout: Problem (depth3). Loss (5): L_γ(a, b, c) = ½(abc - s)² + γ(a² + b² + c²), target s > 0.
"""
from __future__ import annotations

import numpy as np


def balanced_roots(gamma, s=1.0) -> np.ndarray:
    """Return the values u > 0 such that (u, u, u) is a stationary point of loss (5), in ascending
    order: an array of length 2 for γ < γ*(s), and of length 0 for γ > γ*(s).

    Problem (depth3)(b): the stationarity condition on the balanced ray is a polynomial in u;
    np.roots is fine.
    """
    raise NotImplementedError("Problem (depth3): implement balanced_roots")


def critical_gamma(s=1.0) -> float:
    """Return γ*(s), the largest γ for which loss (5) has a nonzero stationary point, in closed form
    (Problem (depth3)(b) deliverable)."""
    raise NotImplementedError("Problem (depth3): implement critical_gamma")


def coexistence_gamma(s=1.0) -> float:
    """Return γ_c(s), the γ at which the nonzero minimum and the origin have the same loss, in closed
    form (Problem (depth3)(d) deliverable)."""
    raise NotImplementedError("Problem (depth3): implement coexistence_gamma")


def balanced_stationary(gamma, s=1.0) -> list:
    """Return [(u, kind), ...] for the balanced roots u (ascending), kind "minimum" or "saddle",
    from your Hessian eigenvalues at (u, u, u) in Problem (depth3)(c)."""
    raise NotImplementedError("Problem (depth3): implement balanced_stationary")
