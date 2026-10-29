"""Adapters connecting the tests to your implementations.  [you: edit only if needed]

Each run_<slug> is called by tests/test_<file>.py. They are already wired to the stubs
in prereqs/ with the signatures the handout states, so if you keep those signatures you
never touch this file. If your function differs (another name, argument order, return
type), make the change here; never edit the test files.

impl(ROOT, "prereqs.gd") returns your prereqs/gd.py, or solutions/gd_solution.py when the
tests run with --solutions (TDYN_SOLUTIONS=1).
"""
from __future__ import annotations

from pathlib import Path

from tdyn.swap import impl

ROOT = Path(__file__).resolve().parents[1]


# Problem (gd-stability)
def run_gd(grad_fn, theta0, eta: float, steps: int):
    """Return the (steps+1, d) array of GD iterates θ_0, ..., θ_K."""
    return impl(ROOT, "prereqs.gd").gd(grad_fn, theta0, eta, steps)


# Problem (phase-portrait)
def run_classify_fixed_point(J) -> str:
    """Return the label of the fixed point whose Jacobian is J (2×2)."""
    return impl(ROOT, "prereqs.fixed_points").classify_fixed_point(J)


# Problem (hessian)
def run_hvp(loss_fn, params, v):
    """Return H v for H = ∇²loss_fn(params) (flat float64 tensors)."""
    return impl(ROOT, "prereqs.hessian").hvp(loss_fn, params, v)


def run_power_iteration(hvp_fn, dim: int, iters: int, seed: int = 0):
    """Return (lambda_max, v) after `iters` power-iteration steps."""
    return impl(ROOT, "prereqs.hessian").power_iteration(hvp_fn, dim, iters, seed=seed)


def run_hutchinson_trace(hvp_fn, dim: int, samples: int, seed: int = 0) -> float:
    """Return the Hutchinson estimate of tr H from `samples` Rademacher probes."""
    return impl(ROOT, "prereqs.hessian").hutchinson_trace(hvp_fn, dim, samples, seed=seed)


# Problem (modes)
def run_mode_dynamics(W0, Sxx, Syx, t: float):
    """Return W(t) from your closed form."""
    return impl(ROOT, "prereqs.modes").mode_dynamics(W0, Sxx, Syx, t)


# Problem (ou)
def run_ou_simulate(x0, theta, sigma, dt, steps, n_paths, rng):
    """Return the (steps+1, n_paths) Euler-Maruyama paths."""
    return impl(ROOT, "prereqs.sde").ou_simulate(x0, theta, sigma, dt, steps, n_paths, rng)
