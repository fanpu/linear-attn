"""Hessian-vector products, sharpness and trace.  [you]  Handout: Problem (hessian).

Conventions (the same as tdyn.hessian, which later assignments use): the parameters are ONE
flat float64 torch tensor `params` of shape (d,), and loss_fn(params) returns a scalar tensor.
"""
from __future__ import annotations

from typing import Callable

import torch


def hvp(loss_fn: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor,
        v: torch.Tensor) -> torch.Tensor:
    """Return H v, where H = ∇²L(params), by double backprop, equation (4):

        H v = ∇_θ ( ∇_θ L(θ) · v ).

    Never form H. Returns a detached tensor of shape (d,). Do not modify `params`.
    """
    raise NotImplementedError("Problem (hessian): implement hvp")


def power_iteration(hvp_fn: Callable[[torch.Tensor], torch.Tensor], dim: int, iters: int,
                    seed: int = 0) -> tuple[float, torch.Tensor]:
    """Estimate the top eigenpair of the symmetric matrix behind hvp_fn, equation (5):

        v_0 random unit vector (torch.Generator seeded with `seed`, float64),
        v_{k+1} = H v_k / |H v_k|,   λ ≈ v_kᵀ H v_k.

    Returns (lambda_max: float, v: unit tensor of shape (dim,)). Assume the eigenvalue of
    largest magnitude is positive (true for every Hessian this assignment gives you).
    """
    raise NotImplementedError("Problem (hessian): implement power_iteration")


def hutchinson_trace(hvp_fn: Callable[[torch.Tensor], torch.Tensor], dim: int, samples: int,
                     seed: int = 0) -> float:
    """Estimate tr H with Hutchinson's estimator, equation (6):

        tr H ≈ (1/S) Σ_s z_sᵀ H z_s,   z_s with i.i.d. entries ±1 (probability ½ each).

    Draw the z_s from a torch.Generator seeded with `seed`. Returns a float.
    """
    raise NotImplementedError("Problem (hessian): implement hutchinson_trace")
