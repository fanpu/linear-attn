"""Checks against known answers. Run from this directory: ../../.venv/bin/python -m pytest -q test_sim.py"""
import math

import numpy as np
import pytest
import torch

from sim import simulate
from theory import gdn_excess_risk, gdn_recursion, optimal_gdn


@pytest.mark.parametrize("alpha,beta,q,sigma2,d", [
    (1.0, 1.0, 0.0, 0.1, 16),     # stationary, noisy
    (0.99, 0.7, 0.01, 0.0, 16),   # mismatched decay, noiseless
    (0.97, 1.3, 0.02, 0.5, 32),   # everything on
])
def test_closed_form_recursion_matches_simulation(alpha, beta, q, sigma2, d):
    T = 400
    out = simulate(d, q, sigma2, T, batch=6000, alphas=[alpha], betas=[beta], kalman=False, seed=1)
    sim = out["gdn"][0].numpy()
    th = gdn_recursion(alpha, beta, q, sigma2, d, T)
    # compare on blocks of 50 steps so Monte-Carlo noise averages out; 3% relative
    s, t = sim.reshape(8, 50).mean(1), th.reshape(8, 50).mean(1)
    assert np.all(np.abs(s - t) / t < 0.03), (s, t)


def test_fixed_point_is_limit_of_recursion():
    th = gdn_recursion(0.98, 0.9, 0.01, 0.2, 32, 20000)
    assert math.isclose(th[-1], gdn_excess_risk(0.98, 0.9, 0.01, 0.2, 32), rel_tol=1e-9)


def test_noiseless_stationary_kalman_is_exact_after_d_steps():
    d = 12
    out = simulate(d, 0.0, 0.0, d + 3, batch=64, alphas=[1.0], betas=[1.0], seed=2)
    assert out["kf"][d:].max() < 1e-12 and out["kf"][0] > 0.5


def test_noiseless_stationary_delta_rule_is_kaczmarz():
    """alpha = beta = 1, q = sigma = 0: each step removes the error along one random direction, m_t = (1 - 1/d)^t."""
    d, T = 20, 200
    out = simulate(d, 0.0, 0.0, T, batch=8000, alphas=[1.0], betas=[1.0], kalman=False, seed=3)
    th = (1 - 1 / d) ** np.arange(T)
    assert np.allclose(out["gdn"][0].numpy(), th, rtol=0.04)


def test_kalman_beats_best_delta_rule():
    d, q, s2 = 16, 0.02, 0.1
    a_, b_, _ = optimal_gdn(q, s2, d)
    out = simulate(d, q, s2, 1500, batch=2000, alphas=[a_], betas=[b_], seed=4)
    assert out["kf"][500:].mean() < out["gdn"][0, 500:].mean()


def test_optimum_is_a_minimum():
    a_, b_, r = optimal_gdn(0.01, 0.1, 64)
    for da, db in [(-1e-3, 0), (0, 0.02), (0, -0.02)]:
        if a_ + da <= 1:
            assert gdn_excess_risk(a_ + da, b_ + db, 0.01, 0.1, 64) >= r - 1e-12
