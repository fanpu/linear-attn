"""Gradient descent that records its trajectory.  [you]  Handout: Problem (gd-stability)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def gd(grad_fn: Callable[[np.ndarray], np.ndarray], theta0, eta: float, steps: int) -> np.ndarray:
    """Run full-batch gradient descent, the update of equation (2) in the handout:

        θ_{k+1} = θ_k - η ∇L(θ_k),   k = 0, ..., steps-1.

    Args:
        grad_fn: maps a parameter vector θ of shape (d,) to ∇L(θ), shape (d,).
        theta0:  initial parameters, array-like of shape (d,). Do not modify it in place.
        eta:     step size η > 0.
        steps:   number of updates K.

    Returns:
        float64 array of shape (steps + 1, d); row k is θ_k, so row 0 equals theta0.
        Do not stop early or clip: a diverging run returns its (large) iterates.
    """
    raise NotImplementedError("Problem (gd-stability): implement gd in prereqs/gd.py")
