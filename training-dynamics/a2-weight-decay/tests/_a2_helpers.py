"""Shared helpers for the A2 tests: reference simulations built only from tdyn.  [AI-owned; do not edit]"""
from __future__ import annotations

import numpy as np

from tdyn.integrate import solve_flow


def grad2(th, gamma):
    """∇L_γ for L_γ = ½(ab - 1)² + γ(a² + b²) on arrays (..., 2)."""
    a, b = th[..., 0], th[..., 1]
    r = a * b - 1.0
    return np.stack([b * r + 2 * gamma * a, a * r + 2 * gamma * b], axis=-1)


def grad3(th, gamma, s=1.0):
    """∇L_γ for L_γ = ½(abc - s)² + γ(a² + b² + c²) on arrays (..., 3)."""
    a, b, c = th[..., 0], th[..., 1], th[..., 2]
    r = a * b * c - s
    return np.stack([b * c * r + 2 * gamma * a, a * c * r + 2 * gamma * b,
                     a * b * r + 2 * gamma * c], axis=-1)


def flow2_end(theta0, gamma, T=400.0):
    return solve_flow(lambda th: -grad2(th, gamma), theta0, T, n_points=2).y[-1]


def flow3_end(theta0, gamma, s=1.0, T=400.0):
    return solve_flow(lambda th: -grad3(th, gamma, s), theta0, T, n_points=2).y[-1]


def first_time_within(theta0, target, eps, gamma, T):
    """First time the 2-layer flow from theta0 is within eps of target (event location)."""
    from scipy.integrate import solve_ivp
    target = np.asarray(target, float)

    def ev(t, y):
        return np.linalg.norm(y - target) - eps
    ev.terminal = True
    ev.direction = -1
    res = solve_ivp(lambda t, y: -grad2(y, gamma), (0.0, T), np.asarray(theta0, float),
                    method="DOP853", rtol=1e-11, atol=1e-13, events=ev)
    return float(res.t_events[0][0])


def num_hessian(grad, theta, h=1e-5):
    """Central-difference Hessian of a gradient function at one point."""
    theta = np.asarray(theta, float)
    d = theta.size
    H = np.empty((d, d))
    for i in range(d):
        e = np.zeros(d); e[i] = h
        H[:, i] = (grad(theta + e) - grad(theta - e)) / (2 * h)
    return 0.5 * (H + H.T)


def gd2(theta0, gamma, eta, steps):
    """GD on the two-layer loss; returns the last iterate (inf if it blew up)."""
    th = np.array(theta0, float)
    with np.errstate(all="ignore"):
        for _ in range(steps):
            th = th - eta * grad2(th, gamma)
            if not np.all(np.abs(th) < 1e6):
                return np.full(2, np.inf)
    return th
