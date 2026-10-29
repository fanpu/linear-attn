"""Gradient-descent loops that record the whole trajectory.  [AI]

Everything works on flat numpy parameter vectors θ of shape (d,). For torch models,
use tdyn.hessian.flat_loss to get a numpy-compatible loss/grad of the flat vector.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def gd_trajectory(grad_fn: Callable[[np.ndarray], np.ndarray], theta0, eta: float,
                  steps: int, *, stop_if_above: float | None = 1e12) -> np.ndarray:
    """Full-batch GD θ_{k+1} = θ_k - η ∇L(θ_k). Returns (steps+1, d), row k = θ_k.

    If |θ| exceeds `stop_if_above` (divergence) the remaining rows are filled with inf,
    so a diverging run still returns an array of the promised shape.
    """
    theta = np.array(theta0, dtype=float)
    out = np.full((steps + 1,) + theta.shape, np.inf)
    out[0] = theta
    for k in range(steps):
        theta = theta - eta * np.asarray(grad_fn(theta), dtype=float)
        if stop_if_above is not None and not np.all(np.abs(theta) < stop_if_above):
            break
        out[k + 1] = theta
    return out


def sgd_trajectory(grad_fn: Callable[[np.ndarray, Any], np.ndarray], theta0, eta: float,
                   steps: int, batch_fn: Callable[[np.random.Generator], Any],
                   rng: np.random.Generator, *,
                   noise_fn: Callable[[Any, np.random.Generator], Any] | None = None,
                   record_every: int = 1) -> np.ndarray:
    """Minibatch SGD θ_{k+1} = θ_k - η ∇L_B(θ_k).

    grad_fn(θ, batch)    gradient of the loss on one batch
    batch_fn(rng)        draws a batch (e.g. indices, or (x, y) arrays)
    noise_fn(batch, rng) optional hook applied to each batch before the gradient;
                         label-noise SGD (A4) adds fresh N(0, σ²) noise to the targets here
    Returns the iterates every `record_every` steps, including θ_0.
    """
    theta = np.array(theta0, dtype=float)
    out = [theta.copy()]
    for k in range(steps):
        batch = batch_fn(rng)
        if noise_fn is not None:
            batch = noise_fn(batch, rng)
        theta = theta - eta * np.asarray(grad_fn(theta, batch), dtype=float)
        if (k + 1) % record_every == 0:
            out.append(theta.copy())
    return np.array(out)
