"""Saxe et al. (2014) mode decomposition of a two-layer linear network.  [you]  Handout: Problem (saxe).

Whitened inputs (Σxx = I) throughout. Shapes: Syx (d_out, d_in), W1 (hidden, d_in),
W2 (d_out, hidden). "Modes" are the singular triples (s_α, u_α, v_α) of Syx with s_α > 0
(treat s_α <= 1e-10 * max s as zero), in the order np.linalg.svd returns (largest first).
"""
from __future__ import annotations

import numpy as np


def aligned_init(Syx, u0, hidden: int):
    """Return (W1, W2): the aligned, balanced initialization of Problem (saxe)(a).

    Every mode α with s_α > 0 starts with W2 W1 = Σ_α u0_α u_α v_αᵀ and the two layers balanced
    (W1 W1ᵀ = W2ᵀ W2); all other directions are zero.

    Args:
        Syx:    (d_out, d_in).
        u0:     scalar, or array with one entry per mode, initial mode strength(s) > 0.
        hidden: hidden width m >= number of modes.
    """
    raise NotImplementedError("Problem (saxe): implement aligned_init")


def saxe_prediction(Syx, u0, t: float):
    """Return the predicted product W(t) = W2(t) W1(t), shape (d_out, d_in), equation (6).

    The prediction for gradient flow on ½ E|W2 W1 x - y|² with Σxx = I started from
    aligned_init(Syx, u0, hidden) (any hidden). u0 is a scalar or one entry per mode.
    Use your solution of Problem (logistic) mode by mode; no numerical integration.
    """
    raise NotImplementedError("Problem (saxe): implement saxe_prediction")


def effective_u0(W1, W2, Syx):
    """Return, for each mode, the effective initial strength of Problem (saxe)(d).

    For a SMALL, not aligned initialization (W1, W2), the value of u0_α for which the logistic
    curve of mode α best describes its rise. Array of shape (number of modes,).
    """
    raise NotImplementedError("Problem (saxe): implement effective_u0")
