"""Critical slowing down near γ = ½.  [you]
Handout: Problem (slowing). Loss (1): L_γ(a, b) = ½(ab - 1)² + γ(a² + b²).
"""
from __future__ import annotations

import numpy as np


def relaxation_time(gamma):
    """Return τ(γ) = 1 / λ_min, where λ_min is the smallest Hessian eigenvalue of L_γ at the fixed
    point that gradient flow from a generic start converges to (the nonzero minimum for γ < ½, the
    origin for γ > ½). Return np.inf at γ = ½. Problem (slowing)(b). Broadcasts over γ > 0.
    """
    raise NotImplementedError("Problem (slowing): implement relaxation_time")


def time_to_within(gamma, u0, eps):
    """Return the first time t at which gradient flow on L_γ from the balanced start a0 = b0 = u0 > 0
    is within Euclidean distance eps of its limit point (u*, u*), i.e. √2 |u(t) - u*| ≤ eps.
    Return 0.0 if the start is already within eps.

    Use your closed form from Problem (slowing)(d) (the logistic solution of A1 with the effective
    target of (a)); no ODE solver. Scalars in, float out; any γ > 0 including γ = ½.
    """
    raise NotImplementedError("Problem (slowing): implement time_to_within")
