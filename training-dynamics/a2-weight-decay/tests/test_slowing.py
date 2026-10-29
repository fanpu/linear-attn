"""Tests for critical slowing down.  [AI-owned; do not edit]

Problem (slowing): the relaxation time against the late-time decay rate of an accurate gradient-flow
solution (distance to the fixed point falling from 1e-4 to 1e-8), and the closed-form time to come
within eps against event location on the same solution.
"""
import numpy as np
import pytest
from _a2_helpers import first_time_within
from adapters import run_relaxation_time, run_time_to_within


def _fixed_point(gamma):
    u = np.sqrt(1 - 2 * gamma) if gamma < 0.5 else 0.0
    return np.array([u, u])


@pytest.mark.parametrize("gamma", [0.4, 0.49, 0.51, 0.6])
def test_relaxation_time_is_late_decay_time(gamma):
    tau = float(run_relaxation_time(gamma))
    th0 = [0.5, 0.3]
    T = 60 * tau + 100
    t1 = first_time_within(th0, _fixed_point(gamma), 1e-4, gamma, T)
    t2 = first_time_within(th0, _fixed_point(gamma), 1e-8, gamma, T)
    assert abs((t2 - t1) / np.log(1e4) / tau - 1) < 0.01


def test_relaxation_time_formula_shape():
    g = np.array([0.45, 0.499, 0.501, 0.55])
    tau = np.asarray(run_relaxation_time(g), dtype=float)
    assert tau.shape == (4,)
    assert tau[1] > 100 and tau[2] > 100 and tau[1] < tau[2]
    assert np.isinf(float(run_relaxation_time(0.5)))


@pytest.mark.parametrize("gamma,u0,eps", [(0.2, 0.1, 1e-3), (0.48, 0.05, 1e-6), (0.8, 0.3, 1e-7),
                                          (0.5, 0.6, 1e-2)])
def test_time_to_within_matches_flow(gamma, u0, eps):
    got = float(run_time_to_within(gamma, u0, eps))
    want = first_time_within([u0, u0], _fixed_point(gamma), eps, gamma, 1e5)
    assert abs(got - want) < 1e-6 * max(1.0, want)


def test_time_to_within_already_close():
    u = np.sqrt(1 - 2 * 0.3)
    assert float(run_time_to_within(0.3, u + 1e-6, 1e-3)) == 0.0
