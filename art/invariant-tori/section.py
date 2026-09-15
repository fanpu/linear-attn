"""Poincare-section seeding and return map at fixed energy.

section_seed is COPIED from art/game-chaos/compute_poincare.py (section_seed), unchanged except that
it uses chart.rps_matrix and returns strategies; seeds match the game-chaos Poincare plates.
Section plane coordinates (as in game-chaos render_poincare.py): (x_R, y_P) = (x[0], y[1]).
"""
import numpy as np
from scipy.optimize import brentq

import replicator_c as rc
from chart import rps_matrix


def section_seed(p, q, H0, eps, root=None):
    """Point on section g=0 with x0=p, y1=q, energy H0 and g increasing, or None.
    Each (p, q) can carry two such points (the energy curve meets g = 0 twice): root=None returns the
    first valid one as game-chaos did; root=0 / 1 asks for the small-a / large-a root specifically
    (a = x_P)."""
    def parts(a):
        return np.array([p, a, 1 - p - a]), np.array([a - p + q, q, 1 - a + p - 2 * q])

    lo = max(0.0, p - q); hi = min(1 - p, 1 + p - 2 * q)
    if hi - lo < 1e-9:
        return None
    H = lambda a: rc.energy(*parts(a)) - H0
    grid = np.linspace(lo, hi, 402)[1:-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        vals = np.array([H(a) for a in grid])
    if not np.any(np.isfinite(vals)):
        return None
    i = np.nanargmin(vals)
    if vals[i] > 0:
        return None
    A = rps_matrix(eps)
    brackets = ((lo + 1e-12, grid[i]), (grid[i], hi - 1e-12))
    if root is not None:
        brackets = brackets[root:root + 1]
    for a_lo, a_hi in brackets:
        try:
            a = brentq(H, a_lo, a_hi, xtol=1e-14)
        except ValueError:
            continue
        x, y = parts(a)
        if np.any(x <= 0) or np.any(y <= 0):
            continue
        xd = x * (A @ y - x @ A @ y); B = rps_matrix(-eps); yd = y * (B @ x - y @ B @ x)
        if xd[1] - xd[0] + yd[1] - yd[0] > 0:
            return x, y
    return None


def first_returns(x, y, eps, T=200.0, nret=1):
    """Section crossings (probabilities, 6-vectors) of orbits started at (x, y)."""
    # advance 0.5 time units first so the start point itself (g = 0) is not counted as a crossing
    L = rc.integrate(rc.probs_to_logits(np.atleast_2d(x), np.atleast_2d(y)), eps, -eps, h=0.01, T=0.5,
                     every=50, traj_every=50, ntraj=1)['traj'][:, 0]
    r = rc.integrate(L, eps, -eps, h=0.01, T=T, every=100, maxsec=nret + 1)
    return r['sec'][:, :nret], r['nsec']
