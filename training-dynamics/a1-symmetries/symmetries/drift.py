"""How finite steps, momentum and weight decay break the conservation of c.  [you]
Handout: Problem (broken). Conventions: L = ½(ab - 1)², r = ab - 1, c = a² - b², p = ab,
weight decay L_γ = L + γ(a² + b²), heavy-ball momentum θ_{k+1} = θ_k - η ∇L(θ_k) + β(θ_k - θ_{k-1}).
"""
from __future__ import annotations

import numpy as np


def imbalance_after_step(a, b, eta: float, gamma: float = 0.0):
    """Return c_{k+1}, the imbalance after ONE GD step on L_γ (update (7)) from (a, b).

    Type in your exact formula from Problem (broken)(a) (it is a function of c, r, η, γ); do not
    take the step. Broadcasts over arrays a, b.
    """
    raise NotImplementedError("Problem (broken): implement imbalance_after_step")


def drift_integral(p0, c):
    """Return I(p0, c) = ∫_0^∞ r(t)² dt along gradient flow from a start with product p0 and
    imbalance c ≠ 0, in your closed form of Problem (broken)(c). Broadcasts."""
    raise NotImplementedError("Problem (broken): implement drift_integral")


def predicted_log_drift(a0, b0, eta: float, beta: float = 0.0) -> float:
    """Return the first-order prediction of ln(c∞ / c0) for GD (beta = 0) or heavy-ball momentum
    with coefficient beta, step eta, started at rest from (a0, b0) with c0 ≠ 0.

    Problem (broken)(c) for beta = 0 and (d) for beta > 0.
    """
    raise NotImplementedError("Problem (broken): implement predicted_log_drift")
