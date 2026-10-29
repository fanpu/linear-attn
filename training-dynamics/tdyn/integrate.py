"""Fixed-step ODE integrators and an adaptive reference solver.  [AI]

All functions integrate an autonomous system  dy/dt = f(y)  (gradient flow is
autonomous: dθ/dt = -∇L(θ)). `y` is a 1-D numpy array; f returns the same shape.
Every function returns the whole trajectory, one row per time point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

VectorField = Callable[[np.ndarray], np.ndarray]


def euler(f: VectorField, y0, dt: float, steps: int) -> np.ndarray:
    """Forward Euler, y_{k+1} = y_k + dt f(y_k). Returns (steps+1, d).

    Gradient descent with step η is exactly this with f = -∇L and dt = η (A0, Problem flow-limit).
    """
    y = np.array(y0, dtype=float)
    out = np.empty((steps + 1,) + y.shape)
    out[0] = y
    for k in range(steps):
        y = y + dt * f(y)
        out[k + 1] = y
    return out


def rk4(f: VectorField, y0, dt: float, steps: int) -> np.ndarray:
    """Classical fourth-order Runge-Kutta (global error O(dt^4)). Returns (steps+1, d)."""
    y = np.array(y0, dtype=float)
    out = np.empty((steps + 1,) + y.shape)
    out[0] = y
    for k in range(steps):
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = y
    return out


def rk4_at(f: VectorField, y0, times, substeps: int = 20) -> np.ndarray:
    """RK4 evaluated at increasing `times` (times[0] is the start), `substeps` per interval.

    Useful to compare a discrete method against the flow at t_k = kη. Returns (len(times), d).
    """
    times = np.asarray(times, dtype=float)
    y = np.array(y0, dtype=float)
    out = np.empty((len(times),) + y.shape)
    out[0] = y
    for i in range(1, len(times)):
        h = (times[i] - times[i - 1]) / substeps
        y = rk4(f, y, h, substeps)[-1]
        out[i] = y
    return out


@dataclass
class Trajectory:
    """Dense solution of dy/dt = f(y): t (n,), y (n, d), and sol(t) -> y(t) for any t in range."""
    t: np.ndarray
    y: np.ndarray
    sol: Callable[[np.ndarray], np.ndarray]


def solve_flow(f: VectorField, y0, t_end: float, n_points: int = 1001, *,
               rtol: float = 1e-10, atol: float = 1e-12, method: str = "DOP853",
               events=None) -> Trajectory:
    """Adaptive high-accuracy reference solution of dy/dt = f(y) on [0, t_end].

    A thin wrapper around scipy.integrate.solve_ivp with dense output; `y` has shape
    (n_points, d) on a uniform grid. Use it as "the true flow" when checking a derivation.
    """
    t_eval = np.linspace(0.0, t_end, n_points)
    res = solve_ivp(lambda t, y: f(y), (0.0, t_end), np.asarray(y0, dtype=float),
                    method=method, t_eval=t_eval, dense_output=True,
                    rtol=rtol, atol=atol, events=events)
    if not res.success:
        raise RuntimeError(res.message)
    return Trajectory(t=res.t, y=res.y.T, sol=lambda t: res.sol(t).T)
