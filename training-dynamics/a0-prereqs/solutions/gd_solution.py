"""Reference solution: prereqs/gd.py (Problem gd-stability)."""
from __future__ import annotations

import numpy as np


def gd(grad_fn, theta0, eta, steps):
    theta = np.array(theta0, dtype=float)          # a copy: theta0 is never modified
    out = np.empty((steps + 1,) + theta.shape)
    out[0] = theta
    with np.errstate(over="ignore", invalid="ignore"):   # diverging runs may overflow
        for k in range(steps):
            theta = theta - eta * np.asarray(grad_fn(theta), dtype=float)
            out[k + 1] = theta
    return out
