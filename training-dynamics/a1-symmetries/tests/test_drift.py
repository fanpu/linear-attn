"""Tests for the broken conservation law.  [AI-owned; do not edit]

Problem (broken): the exact one-step change of c under the GD update with weight decay (7), the
closed form of I(p0, c) = ∫ r² dt (against numerical quadrature), and the first-order drift
prediction for GD and heavy-ball momentum (against simulation at small η).
"""
import numpy as np
import pytest
from _a1_helpers import gd_run
from adapters import run_drift_integral, run_imbalance_after_step, run_predicted_log_drift
from scipy.integrate import quad

RNG = np.random.default_rng(0)
PTS = RNG.uniform(-2, 2, size=(50, 2))


@pytest.mark.parametrize("eta,gamma", [(0.1, 0.0), (0.37, 0.0), (0.1, 0.05)])
def test_imbalance_after_step_exact(eta, gamma):
    a, b = PTS[:, 0], PTS[:, 1]
    r = a * b - 1
    a1, b1 = a - eta * (b * r + 2 * gamma * a), b - eta * (a * r + 2 * gamma * b)
    got = np.asarray(run_imbalance_after_step(a, b, eta, gamma), dtype=float)
    np.testing.assert_allclose(got, a1 ** 2 - b1 ** 2, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("p0,c", [(0.03, 0.08), (2.5, -0.3), (-0.2, 0.6)])
def test_drift_integral_matches_quadrature(p0, c):
    ref = quad(lambda p: (1 - p) / np.sqrt(c * c + 4 * p * p), p0, 1.0, epsabs=1e-13, epsrel=1e-12)[0]
    assert abs(float(run_drift_integral(p0, c)) - ref) < 1e-10


def test_drift_integral_is_integral_of_r_squared():
    # ∫ r(t)² dt along the actual flow from (0.6, 0.2), by the trapezoid rule on a fine RK4 run.
    from tdyn.integrate import rk4
    from tdyn.models import anchor_flow
    tr = rk4(anchor_flow, [0.6, 0.2], 1e-3, 30000)
    r = tr[:, 0] * tr[:, 1] - 1
    ref = np.sum(0.5 * (r[1:] ** 2 + r[:-1] ** 2)) * 1e-3
    assert abs(float(run_drift_integral(0.12, 0.32)) - ref) < 1e-6


@pytest.mark.parametrize("beta,tol", [(0.0, 0.01), (0.5, 0.03)])
def test_predicted_log_drift_matches_simulation(beta, tol):
    a0, b0, eta = 0.3, 0.1, 0.003
    tr = gd_run(a0, b0, eta, int(50 / eta / (1 - beta)), beta=beta)
    c0, c_end = a0 ** 2 - b0 ** 2, tr[-1, 0] ** 2 - tr[-1, 1] ** 2
    measured = np.log(c_end / c0)
    pred = float(run_predicted_log_drift(a0, b0, eta, beta))
    assert pred < 0
    assert abs(measured / pred - 1) < tol


def test_predicted_log_drift_unbalanced_start():
    a0, b0, eta = -0.4, 1.5, 0.002        # c0 < 0, p0 < 0
    tr = gd_run(a0, b0, eta, int(40 / eta))
    measured = np.log((tr[-1, 0] ** 2 - tr[-1, 1] ** 2) / (a0 ** 2 - b0 ** 2))
    assert abs(measured / float(run_predicted_log_drift(a0, b0, eta)) - 1) < 0.02
