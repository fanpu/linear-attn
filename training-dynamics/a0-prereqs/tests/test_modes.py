"""Tests for mode_dynamics.  [AI-owned; do not edit]

Problem (modes): the closed-form solution of dW/dt = Σ_yx - W Σ_xx (8), checked against
an RK4 integration of the same ODE.
"""
import numpy as np
import pytest
from adapters import run_mode_dynamics

from tdyn.integrate import rk4


def _problem(seed=0, d_in=5, d_out=3):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
    lam = np.logspace(-1, 1, d_in)                    # eigenvalues 0.1 ... 10
    Sxx = U @ np.diag(lam) @ U.T
    Syx = rng.standard_normal((d_out, d_in)) @ Sxx
    W0 = 0.1 * rng.standard_normal((d_out, d_in))
    return W0, Sxx, Syx


def _rk4(W0, Sxx, Syx, t, dt=1e-3):
    shape = W0.shape
    f = lambda w: (Syx - w.reshape(shape) @ Sxx).ravel()
    n = int(round(t / dt))
    return rk4(f, W0.ravel(), t / n, n)[-1].reshape(shape)


def test_mode_dynamics_at_zero():
    W0, Sxx, Syx = _problem()
    np.testing.assert_allclose(run_mode_dynamics(W0, Sxx, Syx, 0.0), W0, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("t", [0.3, 2.0, 15.0])
def test_mode_dynamics_matches_rk4(t):
    W0, Sxx, Syx = _problem()
    W = np.asarray(run_mode_dynamics(W0, Sxx, Syx, t))
    assert W.shape == W0.shape
    np.testing.assert_allclose(W, _rk4(W0, Sxx, Syx, t), rtol=1e-7, atol=1e-9)


def test_mode_dynamics_long_time_limit():
    W0, Sxx, Syx = _problem(seed=3)
    W_star = Syx @ np.linalg.inv(Sxx)                 # the least-squares solution
    np.testing.assert_allclose(run_mode_dynamics(W0, Sxx, Syx, 500.0), W_star, rtol=1e-8, atol=1e-10)


def test_mode_dynamics_scalar():
    # one input, one output: dw/dt = s - λ w
    lam, s, w0, t = 2.0, 3.0, -1.0, 0.7
    w = float(np.asarray(run_mode_dynamics(np.array([[w0]]), np.array([[lam]]), np.array([[s]]), t)).item())
    assert abs(w - (s / lam + (w0 - s / lam) * np.exp(-lam * t))) < 1e-12
