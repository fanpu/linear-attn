"""Adapters connecting the tests to your implementations.  [you: edit only if needed]

Each run_<slug> is called by tests/test_<file>.py. They are already wired to the stubs
in wdecay/ with the signatures the handout states, so if you keep those signatures you
never touch this file. If your function differs (another name, argument order, return
type), make the change here; never edit the test files.

impl(ROOT, "wdecay.stationary") returns your wdecay/stationary.py, or
solutions/stationary_solution.py when the tests run with --solutions (TDYN_SOLUTIONS=1).
"""
from __future__ import annotations

from pathlib import Path

from tdyn.swap import impl

ROOT = Path(__file__).resolve().parents[1]


# Problem (stationary)
def run_hessian_wd(theta, gamma):
    """Return ∇²L_γ(θ), shape (..., 2, 2)."""
    return impl(ROOT, "wdecay.stationary").hessian_wd(theta, gamma)


def run_stationary_points(gamma):
    """Return a list of (point of shape (2,), kind) pairs."""
    return impl(ROOT, "wdecay.stationary").stationary_points(gamma)


def run_learned_product(gamma):
    """Return lim ab under the flow from a small balanced start."""
    return impl(ROOT, "wdecay.stationary").learned_product(gamma)


# Problem (slowing)
def run_relaxation_time(gamma):
    """Return τ(γ) = 1/λ_min at the attracting fixed point."""
    return impl(ROOT, "wdecay.slowing").relaxation_time(gamma)


def run_time_to_within(gamma, u0, eps):
    """Return the first time the balanced flow from u0 is within eps of its limit."""
    return impl(ROOT, "wdecay.slowing").time_to_within(gamma, u0, eps)


# Problem (depth3)
def run_balanced_roots(gamma, s=1.0):
    """Return the positive balanced stationary values u, ascending."""
    return impl(ROOT, "wdecay.depth3").balanced_roots(gamma, s)


def run_critical_gamma(s=1.0):
    """Return γ*(s)."""
    return impl(ROOT, "wdecay.depth3").critical_gamma(s)


def run_coexistence_gamma(s=1.0):
    """Return γ_c(s)."""
    return impl(ROOT, "wdecay.depth3").coexistence_gamma(s)


def run_balanced_stationary(gamma, s=1.0):
    """Return [(u, kind), ...] for the balanced roots."""
    return impl(ROOT, "wdecay.depth3").balanced_stationary(gamma, s)


# Problem (basins)
def run_balanced_endpoint(u0, gamma, s=1.0):
    """Return u∞ for the flow started at (u0, u0, u0)."""
    return impl(ROOT, "wdecay.basins").balanced_endpoint(u0, gamma, s)


def run_zero_basin_radius(gamma, s=1.0):
    """Return the radius of the zero basin on the balanced ray."""
    return impl(ROOT, "wdecay.basins").zero_basin_radius(gamma, s)


# Problem (finite-step)
def run_max_stable_step(gamma):
    """Return η_max(γ)."""
    return impl(ROOT, "wdecay.finite_step").max_stable_step(gamma)


def run_predicted_phase(eta, gamma):
    """Return "nonzero", "zero" or "unstable"."""
    return impl(ROOT, "wdecay.finite_step").predicted_phase(eta, gamma)
