"""Classify a fixed point of a 2-D ODE from its Jacobian.  [you]  Handout: Problem (phase-portrait)."""
from __future__ import annotations

import numpy as np

LABELS = ("stable node", "unstable node", "saddle", "stable spiral", "unstable spiral",
          "center", "degenerate")


def classify_fixed_point(J, tol: float = 1e-9) -> str:
    """Classify the fixed point θ* of dθ/dt = F(θ) from the Jacobian J = ∂F/∂θ at θ*,
    using the eigenvalue criterion of Table 1 in the handout.

    Args:
        J:   real 2×2 array-like.
        tol: an eigenvalue whose real part has |Re λ| <= tol * max(1, max|J_ij|) counts as
             having zero real part; likewise for imaginary parts.

    Returns one of LABELS:
        "stable node"      both eigenvalues real and negative
        "unstable node"    both real and positive
        "saddle"           real, one positive and one negative
        "stable spiral"    complex pair with negative real part
        "unstable spiral"  complex pair with positive real part
        "center"           purely imaginary nonzero pair
        "degenerate"       at least one eigenvalue is zero (linearization is inconclusive)
    """
    raise NotImplementedError("Problem (phase-portrait): implement classify_fixed_point")
