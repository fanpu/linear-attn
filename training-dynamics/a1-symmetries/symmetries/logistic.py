"""Balanced learning of one mode is logistic.  [you]  Handout: Problem (logistic)."""
from __future__ import annotations

import numpy as np


def balanced_product(u0, s, t):
    """Return u(t) = a(t) b(t) for gradient flow on ½(s - ab)² from a balanced start a0 = b0 > 0.

    Type in your solution of the ODE of Problem (logistic)(a); no numerical integration. It must
    broadcast over numpy arrays (e.g. a vector of times) and must not overflow or return NaN
    for large t (rewrite the formula if e^{2st} would overflow).

    Args:
        u0: initial product u(0) = a0 b0, with 0 < u0.
        s:  target s > 0.
        t:  time(s) t >= 0.

    Returns:
        u(t), broadcast shape of the inputs.
    """
    raise NotImplementedError("Problem (logistic): implement balanced_product")


def time_to_learn(u0, s, frac: float = 0.5):
    """Return the time at which u(t) first equals frac * s, for 0 < u0 < frac * s < s.

    Your deliverable from Problem (logistic)(b) (the time t_f defined in equation (3)). Broadcasts.
    """
    raise NotImplementedError("Problem (logistic): implement time_to_learn")
