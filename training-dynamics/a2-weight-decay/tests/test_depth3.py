"""Tests for the depth-3 stationary points.  [AI-owned; do not edit]

Problem (depth3): balanced roots are stationary points of loss (5) and are all of them on the ray;
γ*(s) is where they disappear (checked by bisection on the root count of the quartic); γ_c(s) is
where the nonzero minimum's loss equals the origin's; types from the numerical Hessian.
"""
import numpy as np
import pytest
from _a2_helpers import grad3, num_hessian
from adapters import run_balanced_roots, run_balanced_stationary, run_coexistence_gamma, run_critical_gamma


def _n_pos_roots(gamma, s):
    r = np.roots([1.0, 0, 0, -s, 2 * gamma])
    return int(np.sum((np.abs(r.imag) < 1e-12) & (r.real > 0)))


@pytest.mark.parametrize("gamma,s", [(0.01, 1.0), (0.2, 1.0), (0.5, 2.5)])
def test_balanced_roots_are_stationary(gamma, s):
    r = np.asarray(run_balanced_roots(gamma, s), dtype=float)
    assert r.shape == (2,) and r[0] < r[1] and r[0] > 0
    for u in r:
        assert np.linalg.norm(grad3(np.array([u, u, u]), gamma, s)) < 1e-10


@pytest.mark.parametrize("gamma,s", [(0.24, 1.0), (0.05, 0.1)])
def test_balanced_roots_empty_above_critical(gamma, s):
    assert len(np.asarray(run_balanced_roots(gamma, s))) == 0


@pytest.mark.parametrize("s", [1.0, 0.3])
def test_critical_gamma_is_where_roots_vanish(s):
    lo, hi = 0.0, 2.0 * s ** 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _n_pos_roots(mid, s) == 2 else (lo, mid)
    assert abs(float(run_critical_gamma(s)) - lo) < 1e-6 * s


def test_critical_gamma_value():
    assert abs(float(run_critical_gamma(1.0)) - 0.23623519685528) < 1e-12


def test_coexistence_gamma_equal_losses():
    for s in (1.0, 0.5):
        g = float(run_coexistence_gamma(s))
        assert 0 < g < float(run_critical_gamma(s))
        u = max(np.roots([1.0, 0, 0, -s, 2 * g]).real)
        loss = 0.5 * (u ** 3 - s) ** 2 + 3 * g * u ** 2
        assert abs(loss - 0.5 * s ** 2) < 1e-10


@pytest.mark.parametrize("gamma,s", [(0.2, 1.0), (0.05, 0.5)])
def test_balanced_stationary_types(gamma, s):
    out = run_balanced_stationary(gamma, s)
    assert len(out) == 2
    for u, kind in out:
        lam = np.linalg.eigvalsh(num_hessian(lambda x: grad3(x, gamma, s), np.array([u, u, u])))
        assert kind == ("minimum" if np.all(lam > 0) else "saddle")
    assert [k for _, k in out] == ["saddle", "minimum"]
