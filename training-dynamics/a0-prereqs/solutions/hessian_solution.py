"""Reference solution: prereqs/hessian.py (Problem hessian)."""
from __future__ import annotations

import torch


def hvp(loss_fn, params, v):
    p = params.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(loss_fn(p), p, create_graph=True)   # keep the graph of ∇L
    (hv,) = torch.autograd.grad(g @ v.to(g.dtype), p)              # ∇(∇L · v) = H v
    return hv.detach()


def power_iteration(hvp_fn, dim, iters, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g, dtype=torch.float64)
    v = v / v.norm()
    for _ in range(iters):
        hv = hvp_fn(v)
        n = hv.norm()
        if n == 0:                    # v is in the null space: the top eigenvalue we can see is 0
            return 0.0, v
        v = hv / n
    return float(v @ hvp_fn(v)), v    # Rayleigh quotient of the final unit vector


def hutchinson_trace(hvp_fn, dim, samples, seed=0):
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(samples):
        z = torch.randint(0, 2, (dim,), generator=g).to(torch.float64) * 2 - 1
        total += float(z @ hvp_fn(z))
    return total / samples
