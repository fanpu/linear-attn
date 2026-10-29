"""Two-layer linear network: gradients and the matrix conservation law.  [you]
Handout: Problem (matrix-law).

Shapes: W1 (m, d_in), W2 (d_out, m), Sxx = E[x xᵀ] (d_in, d_in), Syx = E[y xᵀ] (d_out, d_in).
Loss (equation (4)): L(W1, W2) = ½ E|W2 W1 x - y|².
"""
from __future__ import annotations

import numpy as np


def two_layer_grads(W1, W2, Sxx, Syx):
    """Return (∇_{W1} L, ∇_{W2} L), the gradients you derive in Problem (matrix-law)(a).

    Write them with matrix products only (no autograd, no finite differences).

    Returns:
        tuple of float64 arrays with the shapes of W1 and W2.
    """
    raise NotImplementedError("Problem (matrix-law): implement two_layer_grads")


def imbalance(W1, W2):
    """Return the (m, m) matrix D = W1 W1ᵀ - W2ᵀ W2 of equation (5)."""
    raise NotImplementedError("Problem (matrix-law): implement imbalance")


def gd_imbalance_change(W1, W2, Sxx, Syx, eta: float):
    """Return D_{k+1} - D_k for ONE gradient-descent step of size eta from (W1, W2).

    Type in your exact formula from Problem (matrix-law)(c) in terms of the gradients;
    do not take the step and subtract.

    Returns:
        (m, m) float64 array.
    """
    raise NotImplementedError("Problem (matrix-law): implement gd_imbalance_change")
