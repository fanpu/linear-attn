"""Box counting on the converge/diverge boundary (his sign-change edge set)."""
import numpy as np

from common_render import edges


def box_counts(E, sizes=None):
    """E: boolean edge map (H,W). Returns sizes (px), counts N(b) of occupied boxes of
    side b tiling the largest b-divisible top-left square region."""
    H, W = E.shape
    L = min(H, W)
    if sizes is None:
        sizes = [2 ** j for j in range(0, int(np.log2(L)) + 1) if 2 ** j <= L // 2]
    counts = []
    for b in sizes:
        m = (L // b) * b
        e = E[:m, :m]
        counts.append(int(e.reshape(m // b, b, m // b, b).any(axis=(1, 3)).sum()))
    return np.array(sizes), np.array(counts)


def fit_dimension(sizes, counts, bmin, bmax):
    sel = (sizes >= bmin) & (sizes <= bmax) & (counts > 0)
    x = np.log10(1.0 / sizes[sel]); y = np.log10(counts[sel])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, icpt = coef
    yhat = A @ coef
    ss = np.sum((y - yhat) ** 2); st = np.sum((y - y.mean()) ** 2)
    # standard error of slope
    n = len(x)
    se = np.sqrt(ss / max(1, n - 2) / np.sum((x - x.mean()) ** 2)) if n > 2 else np.nan
    return dict(D=float(slope), se=float(se), r2=float(1 - ss / st) if st > 0 else np.nan,
                bmin=int(sizes[sel].min()), bmax=int(sizes[sel].max()),
                decades=float(np.log10(sizes[sel].max() / sizes[sel].min())), npts=int(n))


def dimension_of_measure(M, bmin=2, bmax=None):
    E = edges(M)
    R = M.shape[0]
    if bmax is None:
        bmax = R // 8
    s, c = box_counts(E)
    return fit_dimension(s, c, bmin, bmax), s, c
