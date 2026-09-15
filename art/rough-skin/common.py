"""Shared math for 'Rough Skin' (art/ml-art-3d.md §7): level sets of random Heaviside networks on S^3.

Model: the finite-width net of art/depth-roughness/nets.py (Di Lillo, Marinucci, Salvi & Vigogna 2025,
arXiv:2504.06250, Gamma_b = 0), lifted from S^2 to S^3 (x in R^4):
    h_1 = W0 x,  h_{l+1} = sqrt(2/n) W_l sigma(h_l),  T_L = sqrt(2/n) v . sigma(h_L),  all entries N(0,1).
Gamma_W = 2 for both Heaviside (E[H(Z)^2] = 1/2) and ReLU (E[ReLU(Z)^2] = 1/2), so one weight draw
serves both activations (the ReLU twin is the null) and every depth (depth L = first L layers, shared v).

Chart: exponential map at a base point p of S^3 of the cube [-a, a]^3 in T_p S^3 (a = 0.25 rad).
"""
import numpy as np

HALF = 0.25                                   # half side of the tangent cube, rad (side 0.5 rad)
BASE = np.array([0.3, -0.5, 0.8, 0.1])        # base point on S^3 (declared, arbitrary), normalised below
BASE = BASE / np.linalg.norm(BASE)
ACTS = ["heaviside", "relu"]
LMAX = 4


# ---------------------------------------------------------------------------------------------
# chart
# ---------------------------------------------------------------------------------------------

def tangent_frame(p=BASE):
    """Orthonormal 4x3 matrix E whose columns span T_p S^3."""
    M = np.concatenate([p[:, None], np.eye(4)], 1)
    Q, _ = np.linalg.qr(M)
    E = Q[:, 1:4]
    E = E - p[:, None] * (p @ E)[None]
    E, _ = np.linalg.qr(E)
    return E


def grid_coords(R, half=HALF):
    """1D node coordinates of an R-node axis: t_i = (i - (R-1)/2) * h, h = 2*half/R.
    R = 256 -> h = 1.953e-3 rad. The R/2-node grid with the same h*2 spacing is NOT the even
    subsample; use grid_coords(256)[::2] for the 128^3 grid so the two grids share nodes."""
    h = 2 * half / R
    return (np.arange(R) - (R - 1) / 2) * h


def expmap(t, p=BASE, E=None):
    """t [..., 3] tangent coordinates (rad) -> unit vectors [..., 4] on S^3."""
    if E is None:
        E = tangent_frame(p)
    r = np.linalg.norm(t, axis=-1, keepdims=True)
    sinc = np.where(r > 0, np.sin(r) / np.where(r > 0, r, 1), 1.0)
    return np.cos(r) * p + sinc * (t @ E.T)


def stretch_singular_values(t):
    """Analytic singular values of d(exp)/dt at t: 1 (radial) and sin r / r (twice, transverse)."""
    r = np.linalg.norm(t, axis=-1)
    s = np.where(r > 0, np.sin(r) / np.where(r > 0, r, 1), 1.0)
    return np.ones_like(r), s


# ---------------------------------------------------------------------------------------------
# networks (torch)
# ---------------------------------------------------------------------------------------------

def make_weights(n, seed, L=LMAX):
    """Float64 CPU weights; float32 copies are exact casts of these, so the float64 slab uses the
    same draw. Order: W0 [n,4], v [n], W1..W_{L-1} [n,n]."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    W0 = torch.randn(n, 4, generator=g, dtype=torch.float64)
    v = torch.randn(n, generator=g, dtype=torch.float64)
    Ws = [torch.randn(n, n, generator=g, dtype=torch.float64) for _ in range(L - 1)]
    return W0, v, Ws


def forward_all(X, W0, v, Ws, act):
    """X [P,4] -> [L, P]: outputs T_1..T_L of the nested nets, and a list of the per-layer
    pre-activation sign codes if requested (not stored: too big). dtype follows the weights."""
    import torch
    n = W0.shape[0]
    s = (2.0 / n) ** 0.5
    sig = (lambda h: (h > 0).to(h.dtype)) if act == "heaviside" else torch.relu
    h = X @ W0.T
    outs = []
    for l in range(len(Ws) + 1):
        a = sig(h)
        outs.append(s * (a @ v))
        if l < len(Ws):
            h = s * (a @ Ws[l].T)
    return torch.stack(outs)


def forward_codes(X, W0, v, Ws, act):
    """Like forward_all, also returns per-layer boolean codes (h > 0) [L, P, n] for flip counting."""
    import torch
    n = W0.shape[0]
    s = (2.0 / n) ** 0.5
    sig = (lambda h: (h > 0).to(h.dtype)) if act == "heaviside" else torch.relu
    h = X @ W0.T
    outs, codes = [], []
    for l in range(len(Ws) + 1):
        codes.append(h > 0)
        a = sig(h)
        outs.append(s * (a @ v))
        if l < len(Ws):
            h = s * (a @ Ws[l].T)
    return torch.stack(outs), codes


# ---------------------------------------------------------------------------------------------
# levels, box counting, fits
# ---------------------------------------------------------------------------------------------

def median_level(f):
    """Midpoint between the two central order statistics (avoids ties with a cell value)."""
    x = np.sort(f.ravel())
    m = x.size // 2
    return 0.5 * (float(x[m - 1]) + float(x[m]))


def boundary_mask(f, level):
    """Voxels (pixels) with at least one 6- (4-) neighbour on the other side of `level`.
    Sides: f > level vs f <= level. No wrap-around."""
    s = f > level
    B = np.zeros(s.shape, bool)
    for ax in range(s.ndim):
        d = np.diff(s, axis=ax)
        lo = [slice(None)] * s.ndim; lo[ax] = slice(0, -1)
        hi = [slice(None)] * s.ndim; hi[ax] = slice(1, None)
        B[tuple(lo)] |= d
        B[tuple(hi)] |= d
    return B


def boxcount_mask(B, sizes):
    """Number of aligned boxes of side b (voxels) containing a boundary voxel. Partial edge boxes dropped."""
    out = []
    for b in sizes:
        m = [(k // b) * b for k in B.shape]
        C = B[tuple(slice(0, k) for k in m)]
        shp = []
        for k in m:
            shp += [k // b, b]
        C = C.reshape(shp)
        C = C.any(axis=tuple(range(1, 2 * B.ndim, 2)))
        out.append(int(C.sum()))
    return np.array(out)


def fit_slope(sizes, counts, smin, smax):
    sizes = np.asarray(sizes, float); counts = np.asarray(counts, float)
    sel = (sizes >= smin) & (sizes <= smax) & (counts > 0)
    x = np.log(1.0 / sizes[sel]); y = np.log(counts[sel])
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    se = np.sqrt((r ** 2).sum() / max(len(x) - 2, 1) / ((x - x.mean()) ** 2).sum()) if len(x) > 2 else np.nan
    return float(coef[0]), float(se)


SIZES3 = [1, 2, 4, 8, 16, 32, 64]
FIT3 = (2, 64)           # 3D fit range b = 2..64 voxels (plan)
SIZES2 = [1, 2, 4, 8, 16, 32, 64]
FIT2 = (2, 32)           # slice fit range (slices are 176^2 at the 256^3 spacing; 64 px = 2.75 boxes)


def dim3(f, level=None):
    lvl = median_level(f) if level is None else level
    c = boxcount_mask(boundary_mask(f, lvl), SIZES3)
    return fit_slope(SIZES3, c, *FIT3)[0], c


def dim2(f, level=None, fit=FIT2):
    lvl = median_level(f) if level is None else level
    c = boxcount_mask(boundary_mask(f, lvl), SIZES2)
    return fit_slope(SIZES2, c, *fit)[0], c


# ---------------------------------------------------------------------------------------------
# oblique slices
# ---------------------------------------------------------------------------------------------

def slice_planes(nslice=12, seed=2026, side_frac=0.68, offset_frac=0.1):
    """Random oblique planes through the cube, in tangent coordinates normalised to the half side
    (cube = [-1, 1]^3). Returns list of (centre c [3], in-plane orthonormal u, w [3]).
    Normals uniform on S^2 but rejected within 15 deg of any axis (axis-aligned slices can be atypical,
    spec §11.2); centre offset uniform in +-offset_frac along the normal. The square of side
    side_frac*2 fits inside the cube for any orientation (half-diagonal <= sqrt(1 - offset^2))."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < nslice:
        nrm = rng.normal(size=3); nrm /= np.linalg.norm(nrm)
        if np.max(np.abs(nrm)) > np.cos(np.radians(15)):
            continue
        a = rng.normal(size=3); u = a - (a @ nrm) * nrm; u /= np.linalg.norm(u)
        w = np.cross(nrm, u)
        c = rng.uniform(-offset_frac, offset_frac) * nrm
        out.append((c, u, w))
    return out


def slice_points(plane, npx, step):
    """Plane sample points (normalised cube coords) on an npx x npx grid of spacing `step`."""
    c, u, w = plane
    s = (np.arange(npx) - (npx - 1) / 2) * step
    A, B = np.meshgrid(s, s, indexing="ij")
    return c + A[..., None] * u + B[..., None] * w


def sample_trilinear(vol, pts, stride=1, R0=256):
    """vol = V[::stride, ::stride, ::stride] of an R0^3 volume on nodes (i - (R0-1)/2) * (2/R0)
    (normalised coords); pts [..., 3] normalised. Linear interpolation (declared)."""
    from scipy.ndimage import map_coordinates
    idx = (pts / (2.0 / R0) + (R0 - 1) / 2) / stride
    return map_coordinates(vol, np.moveaxis(idx, -1, 0), order=1, mode="nearest")


SLICE_NPX = {1: 176, 2: 88}      # stride -> slice pixels at the volume's own spacing (side 0.68 of the cube)
FIT2_BY_STRIDE = {1: (2, 32), 2: (2, 16)}


def theory(act, L):
    return 3.0 - 2.0 ** -L if act == "heaviside" else 2.0


def measure_field(f, T):
    """3D box D (fit 2-64 on 256^3, 2-32 on 128^3) and 12 oblique-slice 1 + D, median level. Used identically
    for calibration fields and nets."""
    stride = 256 // T
    fit3 = (2, 64) if T == 256 else (2, 32)
    lvl = median_level(f)
    c = boxcount_mask(boundary_mask(f, lvl), SIZES3)
    D3 = fit_slope(SIZES3, c, *fit3)[0]
    d2 = []
    for pl in slice_planes():
        P = slice_points(pl, SLICE_NPX[stride], stride * 2.0 / 256)
        d2.append(dim2(sample_trilinear(f, P, stride=stride), level=lvl, fit=FIT2_BY_STRIDE[stride])[0])
    return dict(D3=D3, counts=c.tolist(), slice_1pD=(1 + np.array(d2)).tolist())
