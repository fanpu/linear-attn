"""Fractal-verification helpers (fractals doc §11): boundary extraction, box counting, fits."""
import numpy as np


def boundary(labels):
    """Pixel is on the boundary if any 4-neighbour carries a different label."""
    L = np.asarray(labels)
    b = np.zeros(L.shape, bool)
    d = L[1:, :] != L[:-1, :]; b[1:, :] |= d; b[:-1, :] |= d
    d = L[:, 1:] != L[:, :-1]; b[:, 1:] |= d; b[:, :-1] |= d
    return b


def box_counts(mask, min_box=1, max_box=None):
    """N(eps) for box sides eps = 2^k pixels (only full boxes; mask cropped to a multiple)."""
    H, W = mask.shape
    max_box = max_box or min(H, W) // 4
    sizes, counts = [], []
    s = min_box
    while s <= max_box:
        h, w = (H // s) * s, (W // s) * s
        m = mask[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3))
        sizes.append(s); counts.append(int(m.sum()))
        s *= 2
    return np.array(sizes), np.array(counts)


def fit_dimension(sizes, counts, lo=None, hi=None):
    """Slope of log N vs log(1/eps) over sizes in [lo, hi]; returns (D, stderr, n points)."""
    sizes = np.asarray(sizes, float); counts = np.asarray(counts, float)
    sel = (counts > 0)
    if lo is not None: sel &= sizes >= lo
    if hi is not None: sel &= sizes <= hi
    x = np.log(1 / sizes[sel]); y = np.log(counts[sel])
    if sel.sum() < 3:
        return np.nan, np.nan, int(sel.sum())
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    s2 = ((y - yhat) ** 2).sum() / max(len(x) - 2, 1)
    se = np.sqrt(s2 / ((x - x.mean()) ** 2).sum())
    return float(coef[0]), float(se), int(sel.sum())


def local_slopes(sizes, counts):
    x = np.log(1 / np.asarray(sizes, float)); y = np.log(np.maximum(counts, 1))
    return (x[:-1] + x[1:]) / 2, np.diff(y) / np.diff(x)


def refinement_flip_fraction(coarse, fine):
    """Fraction of coarse pixels whose label disagrees with any of the co-located fine pixels.
    coarse (R,R) sampled on linspace grid, fine (2R-1,2R-1) sharing every other sample is the ideal;
    here we use nearest co-located samples: fine[::2, ::2] vs coarse when fine = 2R-1 points."""
    f = fine[::2, ::2]
    n = min(f.shape[0], coarse.shape[0])
    return float((f[:n, :n] != coarse[:n, :n]).mean())
