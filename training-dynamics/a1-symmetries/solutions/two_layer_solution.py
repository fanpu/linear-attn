"""Reference solution: symmetries/two_layer.py (Problem matrix-law).

L(W1, W2) = ½ E|W2 W1 x - y|² = ½ tr(W Σxx Wᵀ) - tr(W Σyxᵀ) + const,  W = W2 W1.
With G = ∇_W L = W Σxx - Σyx:   ∇_{W1} L = W2ᵀ G,   ∇_{W2} L = G W1ᵀ.
D = W1 W1ᵀ - W2ᵀ W2 is conserved by gradient flow; one GD step changes it by exactly
    D_{k+1} - D_k = η² (∇_{W1} ∇_{W1}ᵀ - ∇_{W2}ᵀ ∇_{W2}).
"""
from __future__ import annotations

import numpy as np


def two_layer_grads(W1, W2, Sxx, Syx):
    W1, W2, Sxx, Syx = (np.asarray(m, dtype=float) for m in (W1, W2, Sxx, Syx))
    G = W2 @ W1 @ Sxx - Syx
    return W2.T @ G, G @ W1.T


def imbalance(W1, W2):
    W1, W2 = np.asarray(W1, dtype=float), np.asarray(W2, dtype=float)
    return W1 @ W1.T - W2.T @ W2


def gd_imbalance_change(W1, W2, Sxx, Syx, eta):
    g1, g2 = two_layer_grads(W1, W2, Sxx, Syx)
    return eta ** 2 * (g1 @ g1.T - g2.T @ g2)
