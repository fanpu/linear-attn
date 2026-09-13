"""Shared rendering helpers (no GPU)."""
import numpy as np


def cdf_img(x, x_ref=None, buffer=0.25):
    """Sohl-Dickstein's restretch: histogram-equalise negatives (converged) into
    [-1,-buffer] and positives (diverged) into [buffer,1], keep the sign, then negate.
    Output: +ve = converged, -ve = diverged (his convention)."""
    if x_ref is None:
        x_ref = x
    u = np.sort(x_ref.ravel())
    nn = int((u < 0).sum()); npos = u.size - nn
    v = np.concatenate([np.linspace(-1, -buffer, nn), np.linspace(buffer, 1, npos)])
    return -np.interp(x, u, v)


def edges(M):
    """His extract_edges: 2x2 neighbourhood contains both signs."""
    Y = np.stack((M[1:, 1:], M[:-1, 1:], M[1:, :-1], M[:-1, :-1]), -1)
    return np.sign(Y.max(-1) * Y.min(-1)) < 0


def speed01(M):
    """Per-phase rank of the measure in [0,1]: 1 = fastest convergence/divergence
    (small |measure|), 0 = slowest (near the boundary). Returned with the
    converged mask."""
    conv = M < 0
    s = np.zeros(M.shape)
    for mask in (conv, ~conv):
        a = np.abs(M[mask])
        if a.size:
            r = np.argsort(np.argsort(a))
            s[mask] = 1.0 - r / max(1, a.size - 1)
    return s, conv
