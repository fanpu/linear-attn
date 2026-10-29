"""Reference solution: symmetries/drift.py (Problem broken).

One GD step with weight decay (L_γ = L + γ(a² + b²)) maps c = a² - b² to
    c' = c [ (1 - 2ηγ)² - η² r² ],   r = ab - 1        (exact; γ = 0 gives c(1 - η² r²)).
Modified flow θ' = -∇L - (η/2) H ∇L gives c' = -η r² c, so to first order in η
    ln(c∞/c0) ≈ -η I(p0, c0),   I = ∫_0^∞ r² dt = ∫_{p0}^{1} (1 - p) / √(c² + 4p²) dp
      = ½[asinh(2/|c|) - asinh(2p0/|c|)] - ¼[√(c² + 4) - √(c² + 4p0²)].
Heavy-ball momentum β multiplies the drift at fixed η by (1 + β)/(1 - β)².
"""
from __future__ import annotations

import numpy as np


def imbalance_after_step(a, b, eta, gamma=0.0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    c, r = a * a - b * b, a * b - 1.0
    return c * ((1.0 - 2.0 * eta * gamma) ** 2 - eta ** 2 * r ** 2)


def drift_integral(p0, c):
    p0, c = np.asarray(p0, dtype=float), np.abs(np.asarray(c, dtype=float))
    return (0.5 * (np.arcsinh(2.0 / c) - np.arcsinh(2.0 * p0 / c))
            - 0.25 * (np.sqrt(c * c + 4.0) - np.sqrt(c * c + 4.0 * p0 * p0)))


def predicted_log_drift(a0, b0, eta, beta=0.0):
    a0, b0 = float(a0), float(b0)
    return -eta * (1.0 + beta) / (1.0 - beta) ** 2 * drift_integral(a0 * b0, a0 * a0 - b0 * b0)
