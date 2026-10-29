"""Tests for hvp, power_iteration, hutchinson_trace.  [AI-owned; do not edit]

Problem (hessian), equations (4), (5), (6). References: the explicit anchor
Hessian [[b², 2ab-1], [2ab-1, a²]] and torch.autograd.functional.hessian on a 193-parameter MLP.
"""
import numpy as np
import pytest
import torch
from adapters import run_hutchinson_trace, run_hvp, run_power_iteration

from tdyn.models import anchor_hessian, anchor_loss_torch, mlp, regression_batch
from tdyn.hessian import flat_loss

POINTS = [(1.5, 0.5), (2.0, 0.5), (0.3, -1.2), (0.0, 0.0), (-1.0, -1.0)]


def _mlp_problem():
    net = mlp(4, 32, 1, seed=0)                      # 4*32 + 32 + 32 + 1 = 193 parameters
    x, y = regression_batch(64, 4, 1, seed=0)
    loss_fn, theta = flat_loss(net, lambda out: 0.5 * ((out - y) ** 2).mean(), x)
    H = torch.autograd.functional.hessian(loss_fn, theta)
    return loss_fn, theta, H


@pytest.mark.parametrize("ab", POINTS)
def test_hvp_anchor(ab):
    theta = torch.tensor(ab, dtype=torch.float64)
    H = torch.tensor(anchor_hessian(ab))
    for v in (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.3, -2.0])):
        hv = run_hvp(anchor_loss_torch, theta, v.double())
        torch.testing.assert_close(hv, H @ v.double(), rtol=1e-12, atol=1e-12)


def test_hvp_mlp():
    loss_fn, theta, H = _mlp_problem()
    theta_before = theta.clone()
    v = torch.randn(theta.numel(), generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    hv = run_hvp(loss_fn, theta, v)
    assert not hv.requires_grad
    torch.testing.assert_close(hv, H @ v, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(theta, theta_before, rtol=0, atol=0)


@pytest.mark.parametrize("a", [0.5, 1.0, 1.7, 3.0])
def test_power_iteration_anchor_manifold(a):
    # on ab = 1 the Hessian is [[b², 1], [1, a²]] with eigenvalues 0 and a² + b²
    H = torch.tensor(anchor_hessian([a, 1.0 / a]))
    lam, v = run_power_iteration(lambda u: H @ u, 2, 50)
    assert abs(lam - (a * a + 1.0 / (a * a))) < 1e-9 * (a * a + 1.0 / (a * a))
    assert abs(float(torch.as_tensor(v).norm()) - 1.0) < 1e-9


def test_power_iteration_mlp():
    _, _, H = _mlp_problem()
    lam, v = run_power_iteration(lambda u: H @ u, H.shape[0], 300)
    top = float(torch.linalg.eigvalsh(H)[-1])        # 3.319..., next largest |λ| is 2.457
    assert abs(lam - top) < 1e-6 * top
    v = torch.as_tensor(v)
    torch.testing.assert_close(H @ v, top * v, rtol=0, atol=1e-5)


def test_hutchinson_trace_diagonal_is_exact():
    # with ±1 probes, zᵀDz = tr D exactly for diagonal D: one sample is enough
    d = torch.arange(1.0, 11.0, dtype=torch.float64)
    est = run_hutchinson_trace(lambda u: d * u, 10, 1)
    assert abs(est - 55.0) < 1e-12


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
def test_hutchinson_trace_anchor_manifold(a):
    # tr = a² + b²; zᵀHz = tr + 2 z1 z2 H12 with H12 = 1, so the per-sample std is 2
    H = torch.tensor(anchor_hessian([a, 1.0 / a]))
    S = 4000
    est = run_hutchinson_trace(lambda u: H @ u, 2, S)
    assert abs(est - (a * a + 1.0 / (a * a))) < 5 * 2.0 / np.sqrt(S)


def test_hutchinson_trace_mlp():
    _, _, H = _mlp_problem()
    S = 1000
    est = run_hutchinson_trace(lambda u: H @ u, H.shape[0], S)
    off = float((H ** 2).sum() - (H.diagonal() ** 2).sum())
    std = np.sqrt(2.0 * off / S)                     # Var(zᵀHz) = 2 Σ_{i≠j} H_ij²
    assert abs(est - float(H.trace())) < 5 * std
