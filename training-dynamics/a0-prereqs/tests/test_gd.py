"""Tests for gd.  [AI-owned; do not edit]

Problem (gd-stability): the GD update (2) on L(θ) = ½ θᵀHθ, whose iterates are exactly
θ_k = (I - ηH)^k θ_0. H has eigenvalues 1 and 10 in a basis rotated by 30 degrees.
"""
import numpy as np
import pytest
from adapters import run_gd

c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
R = np.array([[c, -s], [s, c]])
H = R @ np.diag([1.0, 10.0]) @ R.T
THETA0 = np.array([1.0, -0.5])


def grad(theta):
    return H @ theta


def closed_form(eta, steps):
    M = np.eye(2) - eta * H
    out = [THETA0]
    for _ in range(steps):
        out.append(M @ out[-1])
    return np.array(out)


def test_gd_shape_and_start():
    tr = np.asarray(run_gd(grad, THETA0, 0.05, 7))
    assert tr.shape == (8, 2)
    np.testing.assert_allclose(tr[0], THETA0, rtol=0, atol=0)


def test_gd_does_not_modify_theta0():
    th = THETA0.copy()
    run_gd(grad, th, 0.05, 5)
    np.testing.assert_array_equal(th, THETA0)


def test_gd_worked_example():
    # Example (gd-by-hand): λ = 1, η = 1.5, θ_0 = 1  ->  θ_k = (-0.5)^k
    tr = np.asarray(run_gd(lambda th: 1.0 * th, np.array([1.0]), 1.5, 4))
    np.testing.assert_allclose(tr[:, 0], [1.0, -0.5, 0.25, -0.125, 0.0625], rtol=0, atol=1e-15)


@pytest.mark.parametrize("eta", [0.05, 0.15])   # ηλ_max = 0.5 (monotone), 1.5 (oscillating)
def test_gd_matches_closed_form(eta):
    tr = np.asarray(run_gd(grad, THETA0, eta, 60))
    np.testing.assert_allclose(tr, closed_form(eta, 60), rtol=1e-10, atol=1e-13)


def test_gd_diverges_above_threshold():
    # η = 0.21 > 2/λ_max = 0.2: the stiff mode grows by |1 - 2.1| = 1.1 per step.
    tr = np.asarray(run_gd(grad, THETA0, 0.21, 200))
    np.testing.assert_allclose(tr, closed_form(0.21, 200), rtol=1e-8)
    assert np.linalg.norm(tr[-1]) > 1e6
