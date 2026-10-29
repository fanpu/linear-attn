"""The rescaling symmetry of the anchor and its conserved quantity.  [you]  Handout: Problem (noether).

θ = (a, b) is the LAST axis of every input, so each function must also work on an array of
points of shape (..., 2) (use theta[..., 0], theta[..., 1] and np.stack(..., axis=-1)).
"""
from __future__ import annotations

import numpy as np


def rescale(theta, alpha):
    """Apply the rescaling symmetry ψ_α of equation (2) to θ = (a, b).

    Args:
        theta: array (..., 2).
        alpha: nonzero scalar α.

    Returns:
        array (..., 2), ψ_α(θ). ψ_1 must be the identity and ψ_α(ψ_β(θ)) = ψ_{αβ}(θ).
    """
    raise NotImplementedError("Problem (noether): implement rescale")


def generator(theta):
    """Return the generator X(θ) = ∂ψ_α(θ)/∂α at α = 1 (equation (1)), shape (..., 2)."""
    raise NotImplementedError("Problem (noether): implement generator")


def conserved(theta):
    """Return the conserved quantity Q(θ) of Problem (noether)(b), shape (...).

    Your deliverable: a function of (a, b) that is constant along gradient flow for EVERY loss of
    the form L = ℓ(ab), and is not constant on the plane.
    """
    raise NotImplementedError("Problem (noether): implement conserved")
