"""Tests for flow_endpoint.  [AI-owned; do not edit]

Problem (endpoint): the limit of gradient flow on ½(s - ab)², checked against an accurate
ODE solution (tdyn.integrate.solve_flow, DOP853 at rtol 1e-10) run to t = 60.
"""
import numpy as np
import pytest
from _a1_helpers import flow_end
from adapters import run_flow_endpoint

STARTS = [(1.5, 0.5), (0.2, 1.7), (-1.2, -0.3), (2.0, -0.5)]


@pytest.mark.parametrize("a0,b0", STARTS)
def test_endpoint_matches_flow(a0, b0):
    got = np.asarray(run_flow_endpoint(a0, b0, 1.0), dtype=float)
    assert got.shape == (2,)
    np.testing.assert_allclose(got, flow_end(a0, b0, 1.0), rtol=1e-7, atol=1e-8)


def test_endpoint_other_targets_and_small_init():
    for s in (0.25, 3.0):
        for a0, b0 in [(1.5, 0.5), (-0.3, -1.4), (0.01, 0.02)]:
            np.testing.assert_allclose(run_flow_endpoint(a0, b0, s), flow_end(a0, b0, s), rtol=1e-7, atol=1e-8)


def test_endpoint_balanced_starts():
    np.testing.assert_allclose(run_flow_endpoint(0.3, 0.3, 2.0), [np.sqrt(2), np.sqrt(2)], rtol=1e-12)
    np.testing.assert_allclose(run_flow_endpoint(-0.3, -0.3, 1.0), [-1.0, -1.0], rtol=1e-12)


def test_endpoint_saddle_line():
    np.testing.assert_allclose(run_flow_endpoint(0.8, -0.8, 1.0), [0.0, 0.0], atol=1e-12)


def test_endpoint_invariants():
    rng = np.random.default_rng(5)
    for a0, b0 in rng.uniform(-3, 3, size=(100, 2)):
        if abs(a0 + b0) < 1e-3:
            continue
        a, b = np.asarray(run_flow_endpoint(a0, b0, 1.5), dtype=float)
        assert abs(a * b - 1.5) < 1e-10
        assert abs((a * a - b * b) - (a0 * a0 - b0 * b0)) < 1e-10
        assert np.sign(a) == np.sign(a0 + b0)
