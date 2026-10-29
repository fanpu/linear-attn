"""Tests for the Saxe mode decomposition.  [AI-owned; do not edit]

Problem (saxe): the aligned balanced initialization, the mode-by-mode prediction (6)
against an accurate solution of the full matrix gradient flow (Σxx = I), and the effective
initial strength of a small random initialization.
"""
import numpy as np
import pytest
from _a1_helpers import saxe_problem, two_layer_flow_field
from adapters import run_aligned_init, run_effective_u0, run_saxe_prediction

from tdyn.integrate import solve_flow

SYX, U, S, V = saxe_problem()
D_OUT, D_IN = SYX.shape


def test_aligned_init_product_and_balance():
    W1, W2 = (np.asarray(w) for w in run_aligned_init(SYX, 0.01, 4))
    assert W1.shape == (4, D_IN) and W2.shape == (D_OUT, 4)
    np.testing.assert_allclose(W2 @ W1, 0.01 * U @ V.T, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(W1 @ W1.T, W2.T @ W2, rtol=1e-10, atol=1e-13)


def test_aligned_init_per_mode_strengths():
    u0 = np.array([0.02, 0.005, 0.001])
    W1, W2 = run_aligned_init(SYX, u0, 3)
    np.testing.assert_allclose(np.diag(U.T @ (np.asarray(W2) @ np.asarray(W1)) @ V), u0, rtol=1e-10)


def test_prediction_at_zero_and_infinity():
    np.testing.assert_allclose(run_saxe_prediction(SYX, 1e-3, 0.0), 1e-3 * U @ V.T, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(run_saxe_prediction(SYX, 1e-3, 500.0), SYX, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("u0", [1e-4, np.array([0.01, 1e-3, 1e-5])])
def test_prediction_matches_matrix_flow(u0):
    hidden = 4
    W1, W2 = (np.asarray(w) for w in run_aligned_init(SYX, u0, hidden))
    f = two_layer_flow_field(SYX, hidden, D_IN, D_OUT)
    tr = solve_flow(f, np.concatenate([W1.ravel(), W2.ravel()]), 20.0, n_points=11)
    for t, th in zip(tr.t, tr.y):
        W = th[hidden * D_IN:].reshape(D_OUT, hidden) @ th[:hidden * D_IN].reshape(hidden, D_IN)
        np.testing.assert_allclose(run_saxe_prediction(SYX, u0, t), W, rtol=1e-6, atol=1e-8)


def test_effective_u0_of_aligned_init():
    u0 = np.array([0.02, 0.005, 0.001])
    W1, W2 = run_aligned_init(SYX, u0, 3)
    np.testing.assert_allclose(run_effective_u0(W1, W2, SYX), u0, rtol=1e-10)


def test_effective_u0_predicts_top_mode_from_random_init():
    rng = np.random.default_rng(7)
    hidden = 5
    W1 = 1e-3 * rng.standard_normal((hidden, D_IN))
    W2 = 1e-3 * rng.standard_normal((D_OUT, hidden))
    u_eff = np.asarray(run_effective_u0(W1, W2, SYX), dtype=float)
    assert u_eff.shape == (3,)
    f = two_layer_flow_field(SYX, hidden, D_IN, D_OUT)
    tr = solve_flow(f, np.concatenate([W1.ravel(), W2.ravel()]), 6.0, n_points=6001)
    strength = [U[:, 0] @ (th[hidden * D_IN:].reshape(D_OUT, hidden) @ th[:hidden * D_IN].reshape(hidden, D_IN)) @ V[:, 0]
                for th in tr.y]
    t_half = np.interp(S[0] / 2, strength, tr.t)
    pred = np.log((S[0] - u_eff[0]) / u_eff[0]) / (2 * S[0])
    assert abs(t_half - pred) < 0.01 * pred
