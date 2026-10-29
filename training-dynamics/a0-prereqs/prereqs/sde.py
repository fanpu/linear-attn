"""Euler-Maruyama simulation of the Ornstein-Uhlenbeck process.  [you]  Handout: Problem (ou)."""
from __future__ import annotations

import numpy as np


def ou_simulate(x0: float, theta: float, sigma: float, dt: float, steps: int, n_paths: int,
                rng: np.random.Generator) -> np.ndarray:
    """Simulate n_paths independent paths of  dX = -θ X dt + σ dW  by Euler-Maruyama,
    equation (10):

        X_{k+1} = X_k - θ X_k Δt + σ √Δt ξ_k,   ξ_k ~ N(0, 1) i.i.d.,   X_0 = x0.

    Draw all randomness from `rng` (a numpy Generator), e.g. rng.standard_normal.

    Returns:
        float64 array of shape (steps + 1, n_paths); row k holds X at time kΔt for every path.
    """
    raise NotImplementedError("Problem (ou): implement ou_simulate")
