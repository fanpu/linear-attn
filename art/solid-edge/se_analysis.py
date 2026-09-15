"""CPU analysis helpers: 3D boundary voxels and box counting (spec §11.1).

2D edge sets and the least-squares fit are the source's own
(art/trainability-fractal/common_render.edges, boxcount.fit_dimension), imported read-only.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/fzeng/ml/research/art/trainability-fractal')
from boxcount import box_counts, fit_dimension, dimension_of_measure  # noqa: E402,F401
from common_render import edges  # noqa: E402,F401


def boundary3d(L):
    """Boolean (Z,Y,X): voxel has at least one 6-neighbour with a different label."""
    L = np.asarray(L)
    B = np.zeros(L.shape, bool)
    for ax in range(3):
        d = np.diff(L, axis=ax) != 0
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[ax] = slice(0, -1)
        sl_hi[ax] = slice(1, None)
        B[tuple(sl_lo)] |= d
        B[tuple(sl_hi)] |= d
    return B


def edges3d(L):
    """3D analogue of his extract_edges: a 2x2x2 cell (indexed by its lowest corner,
    padded back to L's shape) is an edge cell if it contains both labels. One layer thick for
    a plane, so box counting gives exactly 2 there; the two-sided 6-neighbour set
    (boundary3d) doubles the b=1 count and biases a b=1..16 fit upward by ~0.2."""
    L = np.asarray(L).astype(np.int8)
    mx = L.copy(); mn = L.copy()
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                if dz == dy == dx == 0:
                    continue
                sub = np.roll(np.roll(np.roll(L, -dz, 0), -dy, 1), -dx, 2)
                np.maximum(mx, sub, out=mx); np.minimum(mn, sub, out=mn)
    E = mx != mn
    E[-1, :, :] = False; E[:, -1, :] = False; E[:, :, -1] = False   # no wrap-around cells
    return E          # same shape as L (last layers empty) so b-tilings cover the full grid


def box_counts3d(E, sizes=None):
    """Occupied cubes of side b tiling the largest b-divisible corner cube."""
    L = min(E.shape)
    if sizes is None:
        sizes = [2 ** j for j in range(0, int(np.log2(L)) + 1) if 2 ** j <= L // 4]
    counts = []
    for b in sizes:
        m = (L // b) * b
        e = E[:m, :m, :m]
        counts.append(int(e.reshape(m // b, b, m // b, b, m // b, b).any(axis=(1, 3, 5)).sum()))
    return np.array(sizes), np.array(counts)


def dimension3d(L, bmin=1, bmax=16):
    E = edges3d(L)
    s, c = box_counts3d(E)
    return fit_dimension(s, c, bmin, bmax), s, c, E


def local_slopes(s, c):
    s = np.asarray(s, float); c = np.asarray(c, float)
    return -np.diff(np.log10(c)) / np.diff(np.log10(s))
