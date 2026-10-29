"""Reference solution: wdecay/depth3.py (Problem depth3).

L_γ(a, b, c) = ½(abc - s)² + γ(a² + b² + c²). Every nonzero stationary point has |a| = |b| = |c| = u
with abc = u³ > 0 and u⁴ - s u + 2γ = 0, i.e. γ = u(s - u³)/2. This has two positive roots for
γ < γ* and none for γ > γ*, where (maximizing over u, u*³ = s/4)
    γ*(s) = (3/8) 4^(-1/3) s^(4/3)   (s = 1: 3·2^(1/3)/16 ≈ 0.23624),  p jumps from s/4 to 0.
Hessian at a balanced root: eigenvalue u(4u³ - s) along (1,1,1), 4γ (twice) across, so the larger
root is a minimum and the smaller a saddle. The minimum ties with the origin, L = s²/2, at
p = s/2, i.e. γ_c(s) = (s/4)(s/2)^(1/3)  (s = 1: 2^(-1/3)/4 ≈ 0.19843).
"""
from __future__ import annotations

import numpy as np


def balanced_roots(gamma, s=1.0):
    gamma, s = float(gamma), float(s)
    r = np.roots([1.0, 0.0, 0.0, -s, 2.0 * gamma])
    r = np.sort(r[(np.abs(r.imag) < 1e-9) & (r.real > 0)].real)
    for _ in range(3):                                  # polish with Newton
        r = r - (r ** 4 - s * r + 2 * gamma) / (4 * r ** 3 - s)
    return np.sort(r)


def critical_gamma(s=1.0):
    s = float(s)
    return 0.375 * 4.0 ** (-1.0 / 3.0) * s ** (4.0 / 3.0)


def coexistence_gamma(s=1.0):
    s = float(s)
    return 0.25 * s * (0.5 * s) ** (1.0 / 3.0)


def balanced_stationary(gamma, s=1.0):
    return [(float(u), "minimum" if 4 * u ** 3 > s else "saddle") for u in balanced_roots(gamma, s)]
