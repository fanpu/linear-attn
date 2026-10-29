"""Reference solution: wdecay/basins.py (Problem basins).

On the balanced ray a = b = c = u the flow is u̇ = -u(u⁴ - s u + 2γ). The bracket is positive at
u = 0 and for every u < 0, negative exactly between the two positive roots u_s < u_+ (which exist
for γ < γ*). So u(t) → 0 from u0 ≤ 0 and from 0 < u0 < u_s, stays at u_s from u_s, and → u_+ from
u0 > u_s. u_s is the radius of the zero basin on this ray; u_s ≈ 2γ/s for small γ.
"""
from __future__ import annotations

import numpy as np


def balanced_endpoint(u0, gamma, s=1.0):
    u0 = np.asarray(u0, dtype=float)
    roots = _roots(gamma, s)
    if len(roots) < 2:
        out = np.zeros_like(u0)
    else:
        us, up = roots[0], roots[-1]
        out = np.where(u0 > us, up, 0.0)
        out = np.where(u0 == us, us, out)
    return out[()] if out.ndim == 0 else out


def zero_basin_radius(gamma, s=1.0):
    roots = _roots(gamma, s)
    return float(roots[0]) if len(roots) >= 2 else float("inf")


def _roots(gamma, s):
    gamma, s = float(gamma), float(s)
    r = np.roots([1.0, 0.0, 0.0, -s, 2.0 * gamma])
    r = np.sort(r[(np.abs(r.imag) < 1e-9) & (r.real > 0)].real)
    for _ in range(3):
        r = r - (r ** 4 - s * r + 2 * gamma) / (4 * r ** 3 - s)
    return np.sort(r)
