"""Hessian tools by automatic differentiation.  [AI, staff versions]

These are the reference implementations of what you write in A0 Problem (hessian);
later assignments import them from here. Conventions: the parameters are ONE flat
float64 tensor `params` of shape (d,), and `loss_fn(params)` returns a scalar tensor.
`flat_loss` turns a torch.nn.Module plus a batch into such a loss_fn.

    hvp(loss_fn, params, v)        H v by double backprop (Pearlmutter's trick): ∇(∇L·v)
    power_iteration(hvp_fn, d, k)  top eigenpair of H from k Hessian-vector products
    hutchinson_trace(hvp_fn, d, s) tr H ≈ mean of zᵀHz over s Rademacher vectors z
    exact_hessian(loss_fn, params) the full d×d matrix (only for tiny models)
"""
from __future__ import annotations

from typing import Callable

import torch

LossFn = Callable[[torch.Tensor], torch.Tensor]
HvpFn = Callable[[torch.Tensor], torch.Tensor]


def hvp(loss_fn: LossFn, params: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """H(params) @ v without forming H: differentiate g(θ)·v, where g = ∇L(θ) keeps its graph."""
    p = params.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(loss_fn(p), p, create_graph=True)
    (hv,) = torch.autograd.grad(g @ v.to(g.dtype), p)
    return hv.detach()


def make_hvp_fn(loss_fn: LossFn, params: torch.Tensor) -> HvpFn:
    """Close over (loss_fn, params): returns v -> H v."""
    return lambda v: hvp(loss_fn, params, v)


def power_iteration(hvp_fn: HvpFn, dim: int, iters: int = 100, *, seed: int = 0,
                    tol: float | None = None, dtype=torch.float64) -> tuple[float, torch.Tensor]:
    """Top eigenpair of a symmetric H by power iteration: v <- Hv / |Hv|.

    Converges to the eigenvalue of largest MAGNITUDE; this is the sharpness λ_max(H)
    whenever that eigenvalue is positive (true near a minimum). The error shrinks like
    |λ_2/λ_1|^k. The returned eigenvalue is the Rayleigh quotient vᵀHv of the final unit v.
    Stops early if the Rayleigh quotient changes by less than tol (relative).
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g, dtype=dtype)
    v = v / v.norm()
    lam = float("nan")
    for _ in range(iters):
        hv = hvp_fn(v)
        new_lam = float(v @ hv)
        nrm = hv.norm()
        if nrm == 0:
            return 0.0, v
        v = hv / nrm
        if tol is not None and abs(new_lam - lam) <= tol * abs(new_lam):
            lam = new_lam
            break
        lam = new_lam
    return float(v @ hvp_fn(v)), v


def hutchinson_trace(hvp_fn: HvpFn, dim: int, samples: int = 100, *, seed: int = 0,
                     dtype=torch.float64) -> float:
    """Hutchinson's estimator: E[zᵀHz] = tr H for z with i.i.d. ±1 entries (Rademacher)."""
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(samples):
        z = torch.randint(0, 2, (dim,), generator=g).to(dtype) * 2 - 1
        total += float(z @ hvp_fn(z))
    return total / samples


def exact_hessian(loss_fn: LossFn, params: torch.Tensor) -> torch.Tensor:
    """Full Hessian via torch.autograd.functional.hessian. Only for d up to a few thousand."""
    return torch.autograd.functional.hessian(loss_fn, params.detach())


def flat_loss(model: torch.nn.Module, loss_of_output: Callable[[torch.Tensor], torch.Tensor],
              x: torch.Tensor) -> tuple[LossFn, torch.Tensor]:
    """Turn `model` into a function of one flat parameter vector.

    Returns (loss_fn, theta0) with loss_fn(theta) = loss_of_output(model_theta(x)) and theta0
    the model's current parameters concatenated in `model.named_parameters()` order.
    """
    names, shapes, sizes = [], [], []
    for n, p in model.named_parameters():
        names.append(n)
        shapes.append(p.shape)
        sizes.append(p.numel())
    theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def loss_fn(theta: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(theta, sizes)
        params = {n: c.view(s) for n, c, s in zip(names, chunks, shapes)}
        out = torch.func.functional_call(model, params, (x,))
        return loss_of_output(out)

    return loss_fn, theta0
