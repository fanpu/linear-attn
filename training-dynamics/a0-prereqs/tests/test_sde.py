"""Tests for ou_simulate.  [AI-owned; do not edit]

Problem (ou): Euler-Maruyama (10) for dX = -θX dt + σ dW. Statistical tests use 5-sigma
tolerances plus the O(Δt) bias of Euler-Maruyama, so they fail only for a wrong scheme.
"""
import numpy as np
from adapters import run_ou_simulate


def test_ou_simulate_shape_and_start():
    X = np.asarray(run_ou_simulate(1.5, 1.0, 0.5, 0.01, 30, 7, np.random.default_rng(0)))
    assert X.shape == (31, 7)
    np.testing.assert_array_equal(X[0], np.full(7, 1.5))


def test_ou_simulate_reproducible():
    a = run_ou_simulate(0.0, 1.0, 1.0, 0.1, 20, 50, np.random.default_rng(42))
    b = run_ou_simulate(0.0, 1.0, 1.0, 0.1, 20, 50, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_ou_simulate_no_noise_is_euler():
    X = np.asarray(run_ou_simulate(2.0, 3.0, 0.0, 0.1, 10, 3, np.random.default_rng(0)))
    np.testing.assert_allclose(X[:, 0], 2.0 * 0.7 ** np.arange(11), rtol=1e-12)


def test_ou_simulate_mean():
    theta, sigma, dt, n = 2.0, 1.0, 0.01, 20000
    X = np.asarray(run_ou_simulate(2.0, theta, sigma, dt, 50, n, np.random.default_rng(1)))
    t = 50 * dt
    mean = 2.0 * np.exp(-theta * t)
    var = sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * t))
    assert abs(X[-1].mean() - mean) < 5 * np.sqrt(var / n) + 0.01   # 0.01 covers the O(Δt) bias


def test_ou_simulate_stationary_variance():
    theta, sigma, dt, n = 1.0, 0.5, 0.01, 10000
    X = np.asarray(run_ou_simulate(0.0, theta, sigma, dt, 800, n, np.random.default_rng(2)))
    v_inf = sigma**2 / (2 * theta)                                    # 0.125
    assert abs(X[-1].var() / v_inf - 1.0) < 5 * np.sqrt(2.0 / n) + 0.01
