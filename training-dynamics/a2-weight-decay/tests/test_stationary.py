"""Tests for the stationary points of L_γ.  [AI-owned; do not edit]

Problem (stationary): the Hessian of loss (1) against central differences of its gradient, the
stationary points (zero gradient, nothing missed, types from the numerical Hessian), and the learned
product against an accurate gradient-flow solution.
"""
import numpy as np
import pytest
from _a2_helpers import flow2_end, grad2, gd2, num_hessian
from adapters import run_hessian_wd, run_learned_product, run_stationary_points

RNG = np.random.default_rng(0)
PTS = RNG.uniform(-2, 2, size=(20, 2))


@pytest.mark.parametrize("gamma", [0.05, 1.3])
def test_hessian_matches_finite_differences(gamma):
    for th in PTS:
        H = np.asarray(run_hessian_wd(th, gamma), dtype=float)
        assert H.shape == (2, 2)
        np.testing.assert_allclose(H, num_hessian(lambda x: grad2(x, gamma), th), atol=1e-7)


def test_hessian_broadcasts_and_origin():
    H = np.asarray(run_hessian_wd(PTS.reshape(4, 5, 2), 0.2), dtype=float)
    assert H.shape == (4, 5, 2, 2)
    np.testing.assert_allclose(run_hessian_wd(np.zeros(2), 0.3), [[0.6, -1.0], [-1.0, 0.6]], atol=1e-14)


def _check_points(gamma):
    pts = run_stationary_points(gamma)
    for p, kind in pts:
        p = np.asarray(p, dtype=float)
        assert p.shape == (2,)
        assert np.linalg.norm(grad2(p, gamma)) < 1e-10, f"gradient not zero at {p}"
        lam = np.linalg.eigvalsh(num_hessian(lambda x: grad2(x, gamma), p))
        want = ("degenerate" if np.any(np.abs(lam) < 1e-6) else "minimum" if np.all(lam > 0)
                else "maximum" if np.all(lam < 0) else "saddle")
        assert kind == want, f"{p}: got {kind}, eigenvalues {lam}"
    return pts


@pytest.mark.parametrize("gamma,n", [(0.1, 3), (0.49, 3), (0.7, 1)])
def test_stationary_points_count_and_types(gamma, n):
    assert len(_check_points(gamma)) == n


def test_stationary_points_none_missed():
    # GD and gradient flow from 60 random starts must end at one of the returned points.
    for gamma in (0.15, 0.8):
        pts = np.array([np.asarray(p, dtype=float) for p, _ in run_stationary_points(gamma)])
        for th0 in RNG.uniform(-2.5, 2.5, size=(30, 2)):
            for end in (flow2_end(th0, gamma, T=300.0), gd2(th0, gamma, 0.05, 6000)):
                assert np.min(np.linalg.norm(pts - end, axis=1)) < 1e-6


def test_stationary_points_at_critical_value():
    pts = run_stationary_points(0.5)
    assert len(pts) == 1 and pts[0][1] == "degenerate"
    np.testing.assert_allclose(pts[0][0], [0.0, 0.0], atol=1e-14)


@pytest.mark.parametrize("gamma", [0.02, 0.4, 0.6])
def test_learned_product_matches_flow(gamma):
    end = flow2_end([0.05, 0.05], gamma, T=600.0)
    assert abs(float(run_learned_product(gamma)) - end[0] * end[1]) < 1e-8


def test_learned_product_broadcasts():
    g = np.array([0.1, 0.45, 0.55, 0.9])
    got = np.asarray(run_learned_product(g), dtype=float)
    assert got.shape == (4,)
    assert got[2] == 0 and got[3] == 0 and got[0] > got[1] > 0
