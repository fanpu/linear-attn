"""Closed-form gradient flow of linear regression.  [you]  Handout: Problem (modes)."""
from __future__ import annotations

import numpy as np


def mode_dynamics(W0, Sxx, Syx, t: float) -> np.ndarray:
    """Return W(t), the solution at time t of the gradient flow (8)

        dW/dt = Σ_yx - W Σ_xx,   W(0) = W0,

    using the closed form you derive in Problem (modes), not a numerical integrator.

    Args:
        W0:  (d_out, d_in) initial weights.
        Sxx: (d_in, d_in) input covariance Σ_xx, symmetric positive definite.
        Syx: (d_out, d_in) input-output correlation Σ_yx = E[y xᵀ].
        t:   time t >= 0 (a float).

    Returns:
        (d_out, d_in) float64 array W(t).
    """
    raise NotImplementedError("Problem (modes): implement mode_dynamics")
