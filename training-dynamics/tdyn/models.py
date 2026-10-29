"""The anchor model and a tiny MLP.  [AI]

Anchor (James Ang's post, papers/ang2026-training-dynamics-post.md, §2):
    f_{a,b}(x) = a b x,   L(a,b) = ½(ab - 1)²   (after absorbing (1/n)Σx_i² = 1)
    ∇L   = (b r, a r)                    with residual r = ab - 1
    ∇²L  = [[b², 2ab-1], [2ab-1, a²]]    on ab = 1 this is [[b², 1], [1, a²]]
θ = (a, b) is the last axis of every numpy input, so all functions vectorize over
leading axes (a grid of points works).
"""
from __future__ import annotations

import numpy as np


def anchor_loss(theta) -> np.ndarray:
    """L(a,b) = ½(ab-1)²."""
    theta = np.asarray(theta, dtype=float)
    a, b = theta[..., 0], theta[..., 1]
    return 0.5 * (a * b - 1.0) ** 2


def anchor_grad(theta) -> np.ndarray:
    """∇L = (b(ab-1), a(ab-1))."""
    theta = np.asarray(theta, dtype=float)
    a, b = theta[..., 0], theta[..., 1]
    r = a * b - 1.0
    return np.stack([b * r, a * r], axis=-1)


def anchor_flow(theta) -> np.ndarray:
    """Gradient-flow vector field -∇L, the right-hand side of (ȧ, ḃ)."""
    return -anchor_grad(theta)


def anchor_hessian(theta) -> np.ndarray:
    """∇²L = [[b², 2ab-1], [2ab-1, a²]], shape (..., 2, 2)."""
    theta = np.asarray(theta, dtype=float)
    a, b = theta[..., 0], theta[..., 1]
    off = 2.0 * a * b - 1.0
    return np.stack([np.stack([b * b, off], -1), np.stack([off, a * a], -1)], -2)


def anchor_flow_jacobian(theta) -> np.ndarray:
    """Jacobian of the gradient-flow vector field, J = -∇²L."""
    return -anchor_hessian(theta)


def anchor_loss_torch(theta):
    """Torch version of L for autograd (theta: tensor of shape (2,))."""
    a, b = theta[0], theta[1]
    return 0.5 * (a * b - 1.0) ** 2


# --------------------------------------------------------------------------
# A tiny MLP and helpers to treat its parameters as one flat vector.
# --------------------------------------------------------------------------
def mlp(d_in: int, width: int, d_out: int, depth: int = 2, act: str = "tanh", seed: int = 0):
    """Fully connected net with `depth` linear layers and `act` between them (torch.nn.Sequential).

    depth=2 is one hidden layer: Linear(d_in, width) -> act -> Linear(width, d_out).
    Default PyTorch init, seeded.
    """
    import torch
    from torch import nn

    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "identity": nn.Identity}
    torch.manual_seed(seed)
    dims = [d_in] + [width] * (depth - 1) + [d_out]
    layers = []
    for i in range(depth):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < depth - 1:
            layers.append(acts[act]())
    return nn.Sequential(*layers).double()


def regression_batch(n: int, d_in: int, d_out: int = 1, seed: int = 0):
    """A fixed random regression batch (x ~ N(0, I), y = sin of a random projection), float64 torch."""
    import torch

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_in, generator=g, dtype=torch.float64)
    w = torch.randn(d_in, d_out, generator=g, dtype=torch.float64) / d_in ** 0.5
    return x, torch.sin(2.0 * x @ w)
