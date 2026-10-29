"""Tests for the balanced (logistic) solution.  [AI-owned; do not edit]

Problem (logistic): u = ab on a = b obeys the logistic ODE; balanced_product is checked against
an accurate solution of the full 2-D gradient flow on ½(s - ab)² started at a = b = √u0.
"""
import numpy as np
import pytest
from _a1_helpers import anchor_flow_s
from adapters import run_balanced_product, run_time_to_learn

from tdyn.integrate import solve_flow


def _flow_product(u0, s, T, n=201):
    tr = solve_flow(anchor_flow_s(s), [np.sqrt(u0), np.sqrt(u0)], T, n_points=n)
    return tr.t, tr.y[:, 0] * tr.y[:, 1]


def test_balanced_product_at_zero():
    assert abs(float(run_balanced_product(0.01, 2.0, 0.0)) - 0.01) < 1e-15


@pytest.mark.parametrize("s,u0", [(0.5, 1e-3), (3.0, 1e-6)])
def test_balanced_product_matches_flow(s, u0):
    t, u = _flow_product(u0, s, 12.0 / s)
    got = np.asarray(run_balanced_product(u0, s, t), dtype=float)
    assert got.shape == t.shape
    np.testing.assert_allclose(got, u, rtol=1e-8, atol=1e-12)


def test_balanced_product_from_above():
    # u0 > s: the product comes down to s (still the same ODE).
    t, u = _flow_product(3.0, 1.0, 5.0)
    np.testing.assert_allclose(run_balanced_product(3.0, 1.0, t), u, rtol=1e-8)


def test_balanced_product_no_overflow():
    out = np.asarray(run_balanced_product(1e-12, 2.0, np.array([0.0, 1e3, 1e5])), dtype=float)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[1:], 2.0, rtol=1e-12)


def test_time_to_learn_matches_flow():
    for frac, s, u0 in [(0.5, 1.0, 1e-4), (0.5, 2.5, 1e-2), (0.9, 1.0, 1e-4)]:
        T = float(run_time_to_learn(u0, s, frac))
        t, u = _flow_product(u0, s, 1.5 * T, n=30001)
        crossing = np.interp(frac * s, u, t)          # u is increasing
        assert abs(T - crossing) < 1e-3 * T


def test_time_to_learn_small_init_scaling():
    # For u0 << s the half-time is ln(s/u0)/(2s) up to O(u0/s).
    for s in (0.5, 1.0, 4.0):
        for u0 in (1e-6, 1e-9):
            T = float(run_time_to_learn(u0, s, 0.5))
            assert abs(2 * s * T - np.log(s / u0)) < 2 * u0 / s + 1e-12
