"""Tests for the two-layer linear network.  [AI-owned; do not edit]

Problem (matrix-law): gradients of (4) (against torch autograd), conservation of
D = W1 W1ᵀ - W2ᵀ W2 (5) under RK4 gradient flow, and the exact one-step change under GD.
"""
import numpy as np
import pytest
import torch
from _a1_helpers import two_layer_problem
from adapters import run_gd_imbalance_change, run_imbalance, run_two_layer_grads

from tdyn.integrate import rk4


def _torch_grads(W1, W2, Sxx, Syx):
    w1 = torch.tensor(W1, requires_grad=True)
    w2 = torch.tensor(W2, requires_grad=True)
    W = w2 @ w1
    L = 0.5 * torch.trace(W @ torch.tensor(Sxx) @ W.T) - torch.trace(W @ torch.tensor(Syx).T)
    L.backward()
    return w1.grad.numpy(), w2.grad.numpy()


def test_grads_match_autograd():
    for seed in (0, 1):
        W1, W2, Sxx, Syx = two_layer_problem(seed, d_in=5, m=3, d_out=2)
        g1, g2 = run_two_layer_grads(W1, W2, Sxx, Syx)
        r1, r2 = _torch_grads(W1, W2, Sxx, Syx)
        np.testing.assert_allclose(g1, r1, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(g2, r2, rtol=1e-10, atol=1e-12)


def test_grads_scalar_case_is_anchor():
    a, b = 1.3, 0.4
    g1, g2 = run_two_layer_grads(np.array([[a]]), np.array([[b]]), np.eye(1), np.eye(1))
    r = a * b - 1
    np.testing.assert_allclose([np.asarray(g1).item(), np.asarray(g2).item()], [b * r, a * r], rtol=1e-14)


def test_imbalance_value():
    W1, W2, _, _ = two_layer_problem(2)
    D = np.asarray(run_imbalance(W1, W2))
    assert D.shape == (W1.shape[0], W1.shape[0])
    np.testing.assert_allclose(D, W1 @ W1.T - W2.T @ W2, rtol=1e-13, atol=1e-14)


def test_imbalance_conserved_under_flow():
    W1, W2, Sxx, Syx = two_layer_problem(3)
    n1 = W1.size

    def f(th):
        g1, g2 = run_two_layer_grads(th[:n1].reshape(W1.shape), th[n1:].reshape(W2.shape), Sxx, Syx)
        return -np.concatenate([np.ravel(g1), np.ravel(g2)])

    traj = rk4(f, np.concatenate([W1.ravel(), W2.ravel()]), 0.005, 1000)    # t in [0, 5]
    D0 = np.asarray(run_imbalance(W1, W2))
    for th in traj[::100]:
        D = np.asarray(run_imbalance(th[:n1].reshape(W1.shape), th[n1:].reshape(W2.shape)))
        assert np.linalg.norm(D - D0) < 1e-6          # RK4 error; a wrong gradient gives O(1)


@pytest.mark.parametrize("seed,eta", [(0, 0.05), (4, 0.2)])
def test_gd_imbalance_change_exact(seed, eta):
    W1, W2, Sxx, Syx = two_layer_problem(seed)
    for _ in range(5):                      # check along a short GD run, not only at the start
        g1, g2 = run_two_layer_grads(W1, W2, Sxx, Syx)
        pred = np.asarray(run_gd_imbalance_change(W1, W2, Sxx, Syx, eta))
        N1, N2 = W1 - eta * np.asarray(g1), W2 - eta * np.asarray(g2)
        actual = (N1 @ N1.T - N2.T @ N2) - (W1 @ W1.T - W2.T @ W2)
        np.testing.assert_allclose(pred, actual, rtol=1e-9, atol=1e-13)
        W1, W2 = N1, N2


def test_gd_imbalance_change_scalar():
    a, b, eta = 1.5, 0.5, 0.3
    d = np.asarray(run_gd_imbalance_change(np.array([[a]]), np.array([[b]]), np.eye(1), np.eye(1), eta)).item()
    r, c = a * b - 1, a * a - b * b
    assert abs(d - (-(eta * r) ** 2 * c)) < 1e-14
