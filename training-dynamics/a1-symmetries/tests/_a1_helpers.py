"""Shared helpers for the A1 tests: reference simulations built only from tdyn.  [AI-owned; do not edit]"""
from __future__ import annotations

import numpy as np

from tdyn.integrate import solve_flow


def anchor_flow_s(s):
    """Gradient-flow field of L = ½(s - ab)² on arrays (..., 2)."""
    def f(th):
        a, b = th[..., 0], th[..., 1]
        r = a * b - s
        return np.stack([-b * r, -a * r], axis=-1)
    return f


def flow_end(a0, b0, s=1.0, T=60.0):
    return solve_flow(anchor_flow_s(s), [a0, b0], T, n_points=2).y[-1]


def gd_run(a0, b0, eta, steps, beta=0.0, gamma=0.0):
    """Heavy-ball GD on L_γ = ½(ab-1)² + γ(a²+b²), started at rest. Returns (steps+1, 2)."""
    th = np.array([a0, b0], dtype=float)
    prev = th.copy()
    out = np.empty((steps + 1, 2))
    out[0] = th
    for k in range(steps):
        a, b = th
        r = a * b - 1.0
        g = np.array([b * r + 2 * gamma * a, a * r + 2 * gamma * b])
        th, prev = th - eta * g + beta * (th - prev), th
        out[k + 1] = th
    return out


def two_layer_problem(seed=0, d_in=4, m=3, d_out=2, scale=0.5):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
    Sxx = Q @ np.diag(np.linspace(0.5, 2.0, d_in)) @ Q.T
    Syx = rng.standard_normal((d_out, d_in))
    W1 = scale * rng.standard_normal((m, d_in))
    W2 = scale * rng.standard_normal((d_out, m))
    return W1, W2, Sxx, Syx


def saxe_problem(seed=0, d_in=5, d_out=4, svals=(3.0, 1.5, 0.5)):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d_out, d_out)))
    V, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
    k = len(svals)
    return U[:, :k] @ np.diag(svals) @ V[:, :k].T, U[:, :k], np.array(svals), V[:, :k]


def two_layer_flow_field(Syx, hidden, d_in, d_out):
    """Gradient flow of ½E|W2W1x - y|² with Σxx = I on the flat vector (W1, W2)."""
    n1 = hidden * d_in

    def f(th):
        W1 = th[:n1].reshape(hidden, d_in)
        W2 = th[n1:].reshape(d_out, hidden)
        G = Syx - W2 @ W1
        return np.concatenate([(W2.T @ G).ravel(), (G @ W1.T).ravel()])
    return f
