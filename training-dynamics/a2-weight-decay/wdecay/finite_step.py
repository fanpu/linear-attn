"""Gradient descent with a finite step on the two-layer anchor with weight decay.  [you]
Handout: Problem (finite-step). GD (6): θ_{k+1} = θ_k - η ∇L_γ(θ_k) on loss (1).
"""
from __future__ import annotations

import numpy as np


def max_stable_step(gamma):
    """Return η_max(γ): GD (6) is linearly stable at the fixed point gradient flow converges to (the
    nonzero minimum for γ < ½, the origin for γ > ½) iff η < η_max(γ). Problem (finite-step)(b).
    Broadcasts over γ > 0.
    """
    raise NotImplementedError("Problem (finite-step): implement max_stable_step")


def predicted_phase(eta, gamma) -> str:
    """Linear-stability prediction for GD (6) from a generic start near the fixed points: "nonzero"
    (converges to the nonzero minimum), "zero" (converges to the origin) or "unstable" (no stable
    fixed point: oscillation or divergence). Scalars in. Problem (finite-step)(b)."""
    raise NotImplementedError("Problem (finite-step): implement predicted_phase")
