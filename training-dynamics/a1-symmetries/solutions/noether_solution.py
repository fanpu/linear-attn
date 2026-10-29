"""Reference solution: symmetries/noether.py (Problem noether).

Rescaling symmetry ψ_α(a, b) = (α a, b/α); generator X(θ) = ∂_α ψ_α(θ)|_{α=1} = (a, -b);
conserved quantity c = a² - b² (any function of c is also conserved).
"""
from __future__ import annotations

import numpy as np


def rescale(theta, alpha):
    theta = np.asarray(theta, dtype=float)
    a, b = theta[..., 0], theta[..., 1]
    return np.stack([alpha * a, b / alpha], axis=-1)


def generator(theta):
    theta = np.asarray(theta, dtype=float)
    a, b = theta[..., 0], theta[..., 1]
    return np.stack([a, -b], axis=-1)


def conserved(theta):
    theta = np.asarray(theta, dtype=float)
    a, b = theta[..., 0], theta[..., 1]
    return a * a - b * b
