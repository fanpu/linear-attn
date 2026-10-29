"""Where gradient flow on the anchor ends.  [you]  Handout: Problem (endpoint)."""
from __future__ import annotations

import numpy as np


def flow_endpoint(a0, b0, s: float = 1.0) -> np.ndarray:
    """Return (a∞, b∞) = lim_{t→∞} of gradient flow on L(a, b) = ½(s - ab)² from (a0, b0).

    Use your closed form from Problem (endpoint) (conservation + the zero-loss curve), not an ODE
    solver. Cover every start: both signs of c = a0² - b0², c = 0, and the line a0 = -b0
    (which flows to the origin).

    Args:
        a0, b0: floats, the initialization.
        s:      target s > 0.

    Returns:
        float64 array of shape (2,).
    """
    raise NotImplementedError("Problem (endpoint): implement flow_endpoint")
