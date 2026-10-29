"""Tests for the basins of the zero solution at depth 3.  [AI-owned; do not edit]

Problem (basins): the predicted endpoint of the balanced flow against an accurate gradient-flow
solution of loss (5) in 3-D, and the zero-basin radius against the endpoint changing on the ray.
"""
import numpy as np
import pytest
from _a2_helpers import flow3_end
from adapters import run_balanced_endpoint, run_zero_basin_radius


@pytest.mark.parametrize("gamma,s", [(0.05, 1.0), (0.1, 0.6)])
def test_balanced_endpoint_matches_flow(gamma, s):
    u0s = np.array([-0.8, 0.02, 0.1, 0.25, 0.4, 0.7, 1.0, 1.8])
    got = np.asarray(run_balanced_endpoint(u0s, gamma, s), dtype=float)
    assert got.shape == u0s.shape
    for u0, g in zip(u0s, got):
        end = flow3_end([u0, u0, u0], gamma, s, T=800.0)
        assert abs(g - end[0]) < 1e-6, f"u0 = {u0}: predicted {g}, flow {end[0]}"


def test_balanced_endpoint_above_critical_is_zero():
    got = np.asarray(run_balanced_endpoint(np.array([0.3, 1.0, 3.0]), 0.3, 1.0), dtype=float)
    np.testing.assert_allclose(got, 0.0, atol=1e-14)


@pytest.mark.parametrize("gamma", [0.02, 0.2])
def test_zero_basin_radius_is_the_boundary(gamma):
    r = float(run_zero_basin_radius(gamma))
    lo = flow3_end([r * 0.995] * 3, gamma, T=3000.0)
    hi = flow3_end([r * 1.005] * 3, gamma, T=3000.0)
    assert abs(lo[0]) < 1e-6 and hi[0] > 0.5


def test_zero_basin_radius_small_gamma_and_none():
    assert abs(float(run_zero_basin_radius(1e-3)) - 2e-3) < 1e-8
    assert np.isinf(float(run_zero_basin_radius(0.25)))
