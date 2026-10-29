"""Reference solution: prereqs/sde.py (Problem ou)."""
from __future__ import annotations

import numpy as np


def ou_simulate(x0, theta, sigma, dt, steps, n_paths, rng):
    out = np.empty((steps + 1, n_paths))
    x = np.full(n_paths, float(x0))
    out[0] = x
    sq = sigma * np.sqrt(dt)
    for k in range(steps):
        x = x - theta * x * dt + sq * rng.standard_normal(n_paths)
        out[k + 1] = x
    return out
