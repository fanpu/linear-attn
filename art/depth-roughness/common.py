"""Shared math for 'Depth as the Dial'.

Model (Di Lillo, Marinucci, Salvi & Vigogna 2025, arXiv:2504.06250, Sec. 2.2) with Gamma_b = 0:
    x in S^2 (unit vectors in R^3)
    T_0(x) = W0 x,              W0_ij ~ N(0, 1)
    T_s(x) = W_s sigma(T_{s-1}(x)), W_s,ij ~ N(0, Gamma_W / n)   (paper writes n^{-1/2}; the
                                                                 variance must be 1/n for eq. (2.3))
    Gamma_W = 1 / E[sigma(Z)^2]
Infinite-width limit: isotropic GP with covariance kappa_L(<x,y>) = kappa o ... o kappa (L times),
    kappa(u) = Gamma_W E[sigma(Z1) sigma(u Z1 + sqrt(1-u^2) Z2)],  kappa(1) = 1.

We carry d_L(theta) = 1 - kappa_L(cos theta) because all the interesting structure is in the
tiny deviation from 1 at small angle, which cancels catastrophically if computed as 1 - kappa.
"""
import numpy as np
from scipy.special import roots_hermitenorm, erf, roots_legendre

ACTS = ["heaviside", "relu", "gelu", "tanh", "sin"]
ACT_LABEL = {"heaviside": "Heaviside", "relu": "ReLU", "gelu": "GELU", "tanh": "tanh",
             "sin": "sin", "gauss_sparse": "Gaussian a=1+√2", "rbf": "RBF (null)"}

# ----------------------------------------------------------------------------------------------
# kernel maps d -> 1 - kappa(1 - d)
# ----------------------------------------------------------------------------------------------

def _half_angle(d):
    """phi = arccos(1 - d), computed stably for small d."""
    return 2.0 * np.arcsin(np.sqrt(np.clip(d, 0.0, 2.0) / 2.0))


def dmap_heaviside(d):
    # kappa(u) = 1 - arccos(u)/pi  (arc-cosine kernel of order 0, normalised by Gamma_W = 2)
    return _half_angle(d) / np.pi


def dmap_relu(d):
    # kappa(u) = (sqrt(1-u^2) + (pi - arccos u) u) / pi  (arc-cosine order 1, Gamma_W = 2)
    phi = _half_angle(d)
    small = phi < 1e-2
    p = np.where(small, phi, 0.0)
    ser = p**3 / 3 - p**5 / 30 + p**7 / 840 - p**9 / 45360
    s_minus = np.where(small, ser, np.sin(phi) - phi * np.cos(phi))
    return (2 * np.pi * np.sin(phi / 2) ** 2 - s_minus) / np.pi


_SMOOTH = {
    "tanh": np.tanh,
    "sin": np.sin,
    "gelu": lambda z: z * 0.5 * (1 + erf(z / np.sqrt(2))),
    "gauss_sparse": lambda z: np.exp(-(1 + np.sqrt(2)) / 2 * z**2),
    "gauss_low": lambda z: np.exp(-0.5 * z**2),
}
_HERM_CACHE = {}


def hermite_coeffs(name, K=300, nq=1000):
    """a_k >= 0 with kappa(u) = sum_k a_k u^k, sum a_k = 1 (Mehler expansion)."""
    if name in _HERM_CACHE:
        return _HERM_CACHE[name]
    x, w = roots_hermitenorm(nq)
    w = w / w.sum()
    f = _SMOOTH[name](x)
    H = np.zeros((K + 1, nq))
    H[0] = 1
    H[1] = x
    for k in range(1, K):
        H[k + 1] = (x * H[k] - np.sqrt(k) * H[k - 1]) / np.sqrt(k + 1)
    a = (H @ (w * f)) ** 2
    a = a / a.sum()
    _HERM_CACHE[name] = a
    return a


def dmap_smooth(name):
    a = hermite_coeffs(name)
    k = np.arange(len(a))

    def f(d):
        d = np.asarray(d, dtype=np.float64)
        out = np.empty_like(d)
        lo = d <= 0.5
        dl = d[lo]
        # 1 - (1-d)^k = -expm1(k log1p(-d)), exact for tiny d
        out[lo] = (-np.expm1(np.outer(np.log1p(-dl), k)) * a).sum(1) if dl.size else dl
        u = 1 - d[~lo]
        if u.size:
            kap = np.polynomial.polynomial.polyval(u, a)
            out[~lo] = 1 - kap
        return out
    return f


def dmap(name):
    if name == "heaviside":
        return dmap_heaviside
    if name == "relu":
        return dmap_relu
    return dmap_smooth(name)


def d_of_theta(name, L, theta):
    """1 - kappa_L(cos theta) for L >= 0 (L = 0 is the linear field)."""
    d = 2 * np.sin(np.asarray(theta, dtype=np.float64) / 2) ** 2
    f = dmap(name)
    for _ in range(L):
        d = f(d)
    return d


def kappa_prime_1(name):
    """kappa'(1): 1 for ReLU (sparse), >1 high-disorder, inf for Heaviside."""
    if name == "heaviside":
        return np.inf
    if name == "relu":
        return 1.0
    a = hermite_coeffs(name)
    return float((np.arange(len(a)) * a).sum())


def theory(name, L):
    """Paper's predictions on S^2 (d = 2)."""
    if name == "heaviside":
        beta = 0.5  # kappa(1-t) = 1 - (sqrt2/pi) t^{1/2} + ...
        return dict(cls="fractal", cri=beta**L, dimH=2 - beta**L, exp_len=np.inf)
    kp = kappa_prime_1(name)
    return dict(cls="Kac-Rice", cri=1.5 if name == "relu" else 2.0, dimH=1.0,
                exp_len=2 * np.pi * kp ** (L / 2))  # E H^1(T_L^{-1}(0)) = omega_1 kappa'(1)^{L/2}


# ----------------------------------------------------------------------------------------------
# angular power spectra via graded composite Gauss-Legendre in theta
# ----------------------------------------------------------------------------------------------

def theta_nodes(lmax, per_panel=12, ngeo=55):
    g, gw = roots_legendre(per_panel)
    h = 2 * np.pi / max(lmax, 64)
    npan = int(np.ceil(np.pi / h))
    edges = np.linspace(0, np.pi, npan + 1)
    a, b = edges[1], edges[-2]
    # geometric panels in [0, a] and [b, pi]
    geo = a * 2.0 ** -np.arange(ngeo + 1)[::-1]            # a*2^-55 ... a
    left = np.concatenate([[0.0], geo])
    right = np.pi - left[::-1]
    allp = np.concatenate([left, edges[2:-2], right])
    allp = np.unique(allp)
    lo, hi = allp[:-1], allp[1:]
    t = (lo[:, None] + hi[:, None]) / 2 + (hi - lo)[:, None] / 2 * g[None, :]
    w = (hi - lo)[:, None] / 2 * gw[None, :]
    return t.ravel(), w.ravel()


def spectra(dfuncs, lmax, device="cpu"):
    """dfuncs: list of callables theta -> d(theta). Returns C[n_kernels, lmax+1] with
    kappa(cos t) = sum_l (2l+1)/(4 pi) C_l P_l(cos t)."""
    import torch
    t, w = theta_nodes(lmax)
    x = np.cos(t)
    sw = np.sin(t) * w
    D = np.stack([f(t) for f in dfuncs])                      # [K, N]
    dev = torch.device(device)
    Dw = torch.tensor(D * sw[None], dtype=torch.float64, device=dev)
    X = torch.tensor(x, dtype=torch.float64, device=dev)
    C = torch.zeros(len(dfuncs), lmax + 1, dtype=torch.float64, device=dev)
    C[:, 0] = 4 * np.pi - 2 * np.pi * Dw.sum(1)
    Pm, P = torch.ones_like(X), X.clone()
    for l in range(1, lmax + 1):
        C[:, l] = -2 * np.pi * (Dw @ P)
        Pm, P = P, ((2 * l + 1) * X * P - l * Pm) / (l + 1)
    return C.cpu().numpy()


# ----------------------------------------------------------------------------------------------
# alm with common random numbers (the same z_lm for every kernel and every lmax)
# ----------------------------------------------------------------------------------------------

def white_alm(lmax, seed):
    """Unit-variance complex white noise in healpy ordering, reproducible per (seed, l) so that
    raising lmax only appends new modes."""
    idx_m0 = lambda m: m * (2 * lmax + 1 - m) // 2
    z = np.zeros((lmax + 1) * (lmax + 2) // 2, dtype=np.complex128)
    mstart = np.array([idx_m0(m) for m in range(lmax + 1)])
    for l in range(lmax + 1):
        rng = np.random.default_rng([seed, l])
        g = rng.standard_normal((2, l + 1))
        m = np.arange(l + 1)
        v = (g[0] + 1j * g[1]) / np.sqrt(2)
        v[0] = g[0, 0]
        z[mstart[m] + l] = v
    return z


def alm_from_white(z, C, lmax_z):
    """Scale white alm (built with lmax_z) by sqrt(C_l); output lmax = len(C)-1 <= lmax_z."""
    lmax = len(C) - 1
    out = np.zeros((lmax + 1) * (lmax + 2) // 2, dtype=np.complex128)
    s = np.sqrt(np.clip(C, 0, None))
    for m in range(lmax + 1):
        a0 = m * (2 * lmax_z + 1 - m) // 2
        b0 = m * (2 * lmax + 1 - m) // 2
        n = lmax + 1 - m
        out[b0 + m:b0 + m + n] = z[a0 + m:a0 + m + n] * s[m:]
    return out


def synth(alm, lmax, theta, phi, nthreads=8):
    import ducc0
    loc = np.stack([theta.ravel(), phi.ravel() % (2 * np.pi)], 1).astype(np.float64)
    out = ducc0.sht.synthesis_general(alm=alm[None], spin=0, lmax=lmax, loc=loc, epsilon=1e-10,
                                      nthreads=nthreads)
    return out[0].reshape(theta.shape)


# ----------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------

def rot_frame(center):
    """Orthonormal (e1, e2, c) with c = center direction."""
    c = np.asarray(center, float)
    c = c / np.linalg.norm(c)
    tmp = np.array([0, 0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(tmp, c)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    return e1, e2, c


def lambert_patch(center, half_width, n):
    """Square n x n patch in Lambert azimuthal equal-area coordinates around `center`.
    half_width is the planar half side in units of the unit sphere (small patch: ~radians).
    Returns unit vectors [n, n, 3]."""
    e1, e2, c = rot_frame(center)
    s = np.linspace(-half_width, half_width, n)
    X, Y = np.meshgrid(s, -s)          # row 0 = top
    r = np.hypot(X, Y)
    rho = np.clip(r / 2, 0, 1)
    ang = 2 * np.arcsin(rho)           # colatitude from center
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(r > 0, X / r, 0)
        uy = np.where(r > 0, Y / r, 0)
    v = (np.cos(ang)[..., None] * c + np.sin(ang)[..., None] * (ux[..., None] * e1 + uy[..., None] * e2))
    return v


def gnomonic_patch(center, half_width, n, rot=0.0):
    e1, e2, c = rot_frame(center)
    cr, sr = np.cos(rot), np.sin(rot)
    e1, e2 = cr * e1 + sr * e2, -sr * e1 + cr * e2
    s = (np.arange(n) + 0.5) / n * 2 * half_width - half_width
    X, Y = np.meshgrid(s, -s)
    v = c + X[..., None] * e1 + Y[..., None] * e2
    return v / np.linalg.norm(v, axis=-1, keepdims=True), X, Y


def vec_to_thetaphi(v):
    th = np.arccos(np.clip(v[..., 2], -1, 1))
    ph = np.arctan2(v[..., 1], v[..., 0]) % (2 * np.pi)
    return th, ph


# ----------------------------------------------------------------------------------------------
# box counting on a sampled field
# ----------------------------------------------------------------------------------------------

def boxcount(f, level, sizes=None):
    """Count boxes (side s pixels, aligned on the sample lattice, box includes its (s+1)^2
    corner samples) where the field straddles `level`. Returns sizes, counts."""
    import torch
    F = torch.as_tensor(f, dtype=torch.float64)
    n = min(F.shape) - 1
    if sizes is None:
        sizes = [2**k for k in range(int(np.log2(n)) + 1) if 2**k <= n]
    # cell-level min / max over 2x2 corners
    mx = torch.maximum(torch.maximum(F[:-1, :-1], F[1:, :-1]), torch.maximum(F[:-1, 1:], F[1:, 1:]))
    mn = torch.minimum(torch.minimum(F[:-1, :-1], F[1:, :-1]), torch.minimum(F[:-1, 1:], F[1:, 1:]))
    counts = []
    for s in sizes:
        m = (mx.shape[0] // s) * s
        k = (mx.shape[1] // s) * s
        a = torch.nn.functional.max_pool2d(mx[None, None, :m, :k], s)[0, 0]
        b = -torch.nn.functional.max_pool2d(-mn[None, None, :m, :k], s)[0, 0]
        counts.append(int(((a > level) & (b < level)).sum()))
    return np.array(sizes), np.array(counts)


def fit_dim(sizes, counts, smin, smax):
    sel = (sizes >= smin) & (sizes <= smax) & (counts > 0)
    x = np.log(1.0 / sizes[sel])
    y = np.log(counts[sel])
    A = np.stack([x, np.ones_like(x)], 1)
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss = ((y - yhat) ** 2).sum()
    se = np.sqrt(ss / max(len(x) - 2, 1) / ((x - x.mean()) ** 2).sum()) if len(x) > 2 else np.nan
    return float(coef[0]), float(se), int(sel.sum())


def rbf_d(s):
    """Null model: squared-exponential-on-the-sphere kernel exp(-(1-u)/s^2) (C^infinity)."""
    return lambda t: -np.expm1(-(2 * np.sin(np.asarray(t, float) / 2) ** 2) / s**2)


def kernel_d(name, L):
    """theta -> 1 - kappa(cos theta) for a named kernel; name 'rbf' ignores L."""
    if name == "rbf":
        return rbf_d(RBF_S)
    return lambda t: d_of_theta(name, L, t)


RBF_S = 0.02


def flat_spectrum(dfunc, ks, K0=60.0, nper=24):
    """Planar (flat-sky) spectral density S(k) = -2 pi int_0^inf D(theta) J0(k theta) theta dtheta,
    regularised by a Gaussian window exp(-(k theta / K0)^2) (relative spectral blur ~ 1/K0).
    Valid for large k (theta-support ~ K0/k << 1). Matches C_l at l = k."""
    from scipy.special import j0
    g, gw = roots_legendre(nper)
    smax = 6 * K0
    # panels in s = k theta: geometric near 0, then width pi/2 (J0 half-period ~ pi)
    geo = (np.pi / 2) * 2.0 ** -np.arange(60)[::-1]
    uni = np.arange(np.pi / 2, smax + 1e-9, np.pi / 2)
    edges = np.unique(np.concatenate([[0.0], geo, uni]))
    lo, hi = edges[:-1], edges[1:]
    s = ((lo + hi)[:, None] / 2 + (hi - lo)[:, None] / 2 * g).ravel()
    w = ((hi - lo)[:, None] / 2 * gw).ravel()
    base = w * j0(s) * s * np.exp(-(s / K0) ** 2)
    out = []
    for k in ks:
        out.append(-2 * np.pi / k**2 * np.dot(base, dfunc(s / k)))
    return np.array(out)
