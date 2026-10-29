"""Tests for GD with a finite step.  [AI-owned; do not edit]

Problem (finite-step): the stability threshold against the numerical Hessian at the attracting fixed
point, and against GD runs started near that fixed point just inside and just outside it.
"""
import numpy as np
import pytest
from _a2_helpers import gd2, grad2, num_hessian
from adapters import run_max_stable_step, run_predicted_phase


def _fixed_point(gamma):
    u = np.sqrt(1 - 2 * gamma) if gamma < 0.5 else 0.0
    return np.array([u, u])


@pytest.mark.parametrize("gamma", [0.02, 0.3, 0.45, 0.7])
def test_max_stable_step_from_hessian(gamma):
    lam = np.linalg.eigvalsh(num_hessian(lambda x: grad2(x, gamma), _fixed_point(gamma)))
    assert abs(float(run_max_stable_step(gamma)) - 2 / lam.max()) < 1e-6


@pytest.mark.parametrize("gamma", [0.1, 0.35, 0.7])
def test_max_stable_step_against_gd(gamma):
    em = float(run_max_stable_step(gamma))
    th0 = _fixed_point(gamma) + np.array([1e-3, -2e-3])
    inside = gd2(th0, gamma, 0.97 * em, 4000)
    outside = gd2(th0, gamma, 1.03 * em, 4000)
    assert np.linalg.norm(inside - _fixed_point(gamma)) < 1e-8
    assert not np.linalg.norm(outside - _fixed_point(gamma)) < 1e-4


def test_max_stable_step_broadcasts():
    got = np.asarray(run_max_stable_step(np.array([0.1, 0.25, 0.4, 0.6])), dtype=float)
    assert got.shape == (4,) and np.argmax(got) == 1


def test_predicted_phase():
    cases = [(0.5, 0.1, "nonzero"), (1.5, 0.1, "unstable"), (1.9, 0.25, "nonzero"),
             (1.3, 0.45, "unstable"), (0.5, 0.8, "zero"), (0.9, 0.8, "unstable")]
    for eta, gamma, want in cases:
        assert run_predicted_phase(eta, gamma) == want, (eta, gamma)
