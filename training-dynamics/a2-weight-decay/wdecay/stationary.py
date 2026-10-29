"""Stationary points of the two-layer anchor with weight decay.  [you]
Handout: Problem (stationary). Loss (1): L_γ(a, b) = ½(ab - 1)² + γ(a² + b²), γ > 0.
"""
from __future__ import annotations

import numpy as np

KINDS = ("minimum", "saddle", "maximum", "degenerate")


def hessian_wd(theta, gamma):
    """Return the Hessian ∇²L_γ(θ) of loss (1), shape (..., 2, 2), for points θ of shape (..., 2).

    Write it in closed form (no autograd); it is the anchor's Hessian plus the weight-decay term.
    """
    raise NotImplementedError("Problem (stationary): implement hessian_wd")


def stationary_points(gamma):
    """Return every stationary point of L_γ for a given γ > 0, with its type.

    Use your derivation in Problem (stationary)(a): the points in closed form (no root finding),
    each classified from the eigenvalues of hessian_wd at that point.

    Returns:
        a list of (point, kind) pairs, point a float array of shape (2,), kind one of KINDS:
        "minimum" (all eigenvalues > 0), "saddle" (both signs), "maximum" (all < 0), or
        "degenerate" (an eigenvalue within 1e-9 of zero). Any order.
    """
    raise NotImplementedError("Problem (stationary): implement stationary_points")


def learned_product(gamma):
    """Return lim_{t→∞} a(t)b(t) under gradient flow on L_γ from a small balanced start a0 = b0 > 0.

    Problem (stationary)(a) deliverable. Broadcasts over a numpy array of γ > 0 (γ ≠ ½).
    """
    raise NotImplementedError("Problem (stationary): implement learned_product")
