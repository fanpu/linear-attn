"""Reference solution: symmetries/logistic.py (Problem logistic).

Balanced flow (a = b > 0) on L = ½(s - ab)²: u = ab obeys u' = 2u(s - u), so
    u(t) = s / (1 + (s/u0 - 1) e^{-2st})                      (written to avoid overflow)
    t_f  = ln[ f (s - u0) / ((1 - f) u0) ] / (2s)            time for u to reach f·s
         ≈ ln(s/u0) / (2s)   for f = ½ and u0 << s.
"""
from __future__ import annotations

import numpy as np


def balanced_product(u0, s, t):
    u0, s, t = (np.asarray(x, dtype=float) for x in (u0, s, t))
    return s / (1.0 + (s / u0 - 1.0) * np.exp(-2.0 * s * t))


def time_to_learn(u0, s, frac=0.5):
    u0, s = np.asarray(u0, dtype=float), np.asarray(s, dtype=float)
    return np.log(frac * (s - u0) / ((1.0 - frac) * u0)) / (2.0 * s)
