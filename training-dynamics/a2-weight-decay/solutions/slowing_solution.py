"""Reference solution: wdecay/slowing.py (Problem slowing).

On the balanced line a = b = u the product p = u² obeys A1's logistic equation with the effective
target s = 1 - 2γ:  ṗ = 2p(s - p). The slowest Hessian eigenvalue at the stable fixed point is
    γ < ½:  λ_min = min(4γ, 2 - 4γ)   (near ½ it is 2 - 4γ = 4(½ - γ))
    γ > ½:  λ_min = 2γ - 1 = 2(γ - ½)  (the origin)
so τ = 1/λ_min diverges like 1/(4(½ - γ)) from below and 1/(2(γ - ½)) from above.
"""
from __future__ import annotations

import numpy as np


def relaxation_time(gamma):
    g = np.asarray(gamma, dtype=float)
    lam = np.where(g < 0.5, np.minimum(4.0 * g, 2.0 - 4.0 * g), 2.0 * g - 1.0)
    with np.errstate(divide="ignore"):
        tau = np.where(lam > 0, 1.0 / np.where(lam > 0, lam, 1.0), np.inf)
    return tau[()] if tau.ndim == 0 else tau


def time_to_within(gamma, u0, eps):
    gamma, u0, eps = float(gamma), float(u0), float(eps)
    s = 1.0 - 2.0 * gamma
    ustar = np.sqrt(s) if s > 0 else 0.0
    if np.sqrt(2.0) * abs(u0 - ustar) <= eps:
        return 0.0
    ue = ustar + np.sign(u0 - ustar) * eps / np.sqrt(2.0)
    p0, pe = u0 * u0, ue * ue
    if s == 0.0:
        return 0.5 * (1.0 / pe - 1.0 / p0)
    return float(np.log(pe * (s - p0) / (p0 * (s - pe))) / (2.0 * s))
