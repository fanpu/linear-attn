"""Reference solution: symmetries/endpoint.py (Problem endpoint).

Gradient flow on L = ½(s - ab)² conserves c = a² - b² and ends on ab = s, so
    a∞² = (c + √(c² + 4s²)) / 2,   b∞² = (-c + √(c² + 4s²)) / 2,
with the common sign of (a∞, b∞) equal to sign(a0 + b0); the line a0 = -b0 flows to the origin.
"""
from __future__ import annotations

import numpy as np


def flow_endpoint(a0, b0, s=1.0):
    a0, b0, s = float(a0), float(b0), float(s)
    if a0 + b0 == 0.0:                       # c = 0 on the line a = -b: the flow goes to the saddle
        return np.zeros(2)
    c = a0 * a0 - b0 * b0
    root = np.sqrt(c * c + 4.0 * s * s)
    sign = np.sign(a0 + b0)
    return sign * np.array([np.sqrt((c + root) / 2.0), np.sqrt((root - c) / 2.0)])
