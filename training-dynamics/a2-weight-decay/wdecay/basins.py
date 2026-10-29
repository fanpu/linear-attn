"""Basins of the zero solution at depth 3.  [you]
Handout: Problem (basins). Loss (5): L_γ(a, b, c) = ½(abc - s)² + γ(a² + b² + c²).
"""
from __future__ import annotations

import numpy as np


def balanced_endpoint(u0, gamma, s=1.0):
    """Return u∞ = lim_{t→∞} u(t) for gradient flow on loss (5) started on the balanced ray
    a0 = b0 = c0 = u0 (any real u0). Problem (basins)(a): use the phase line of the 1-D flow, no ODE
    solver (your depth3.balanced_roots may help). Broadcasts over an array u0.
    """
    raise NotImplementedError("Problem (basins): implement balanced_endpoint")


def zero_basin_radius(gamma, s=1.0) -> float:
    """Return the largest u such that every balanced start with 0 < u0 < u flows to the origin, or
    np.inf if every balanced start does (γ > γ*). Problem (basins)(a)."""
    raise NotImplementedError("Problem (basins): implement zero_basin_radius")
