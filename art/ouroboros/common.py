"""Shared pieces for Ouroboros: targets, common-random-number pools, batched models, metrics.

Everything here is batched over independent chains on the GPU in float64.
A "chain" is one (target, model, regime, lambda, n, seed) retraining loop.
"""
import hashlib
import math

import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
EXTENT = (-1.45, 1.45)  # square window all targets live in (data units)

RING_K = 8
RING_R = 1.0
RING_SIGMA = 0.08
RING_CENTERS = np.stack([RING_R * np.cos(2 * np.pi * np.arange(RING_K) / RING_K),
                         RING_R * np.sin(2 * np.pi * np.arange(RING_K) / RING_K)], 1)


# ----------------------------------------------------------------------------- targets
def sample_ring(n, rng):
    j = rng.integers(0, RING_K, n)
    return RING_CENTERS[j] + RING_SIGMA * rng.standard_normal((n, 2))


def sample_spiral(n, rng):
    # Archimedean spiral r ∝ theta, sampled uniformly in arc length (approx: theta ∝ sqrt(u)),
    # plus isotropic Gaussian noise.
    u = rng.random(n)
    th = 0.6 * np.pi + (4.2 * np.pi - 0.6 * np.pi) * np.sqrt(u)
    r = th / (4.2 * np.pi) * 1.25
    x = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    x = x - np.array([0.13, -0.15])
    return x + 0.035 * rng.standard_normal((n, 2))


FERN_MAPS = np.array([  # Barnsley (1988) fern: a, b, c, d, e, f, p
    [0.00, 0.00, 0.00, 0.16, 0.00, 0.00, 0.01],
    [0.85, 0.04, -0.04, 0.85, 0.00, 1.60, 0.85],
    [0.20, -0.26, 0.23, 0.22, 0.00, 1.60, 0.07],
    [-0.15, 0.28, 0.26, 0.24, 0.00, 0.44, 0.07],
])


def sample_fern(n, rng, burn=64):
    """Chaos game, vectorised over n independent walkers (each walker burns in, then 1 sample)."""
    p = FERN_MAPS[:, 6] / FERN_MAPS[:, 6].sum()
    x = np.zeros((n, 2))
    for _ in range(burn):
        k = rng.choice(4, size=n, p=p)
        a, b, c, d, e, f = [FERN_MAPS[k, i] for i in range(6)]
        x = np.stack([a * x[:, 0] + b * x[:, 1] + e, c * x[:, 0] + d * x[:, 1] + f], 1)
    # fern spans x in [-2.18, 2.66], y in [0, 9.99]; map into the window, upright
    x[:, 0] = (x[:, 0] - 0.24) * 0.27
    x[:, 1] = (x[:, 1] - 5.0) * 0.27
    return x


TARGETS = {"ring": sample_ring, "spiral": sample_spiral, "fern": sample_fern}


def seed_int(*keys):
    h = hashlib.sha256("/".join(map(str, keys)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 62) - 1)


def real_pool(target, seed, n_max):
    """Fixed real dataset for a seed; nested: the dataset of size n is the first n points."""
    rng = np.random.default_rng(seed_int("real", target, seed))
    return torch.tensor(TARGETS[target](n_max, rng), dtype=DT, device=DEV)


def reference_sample(target, m=200_000):
    rng = np.random.default_rng(seed_int("reference", target))
    return TARGETS[target](m, rng)


def crn_pools(seed, gen, n_max, tag="train"):
    """Common random numbers for (seed, generation): uniform [n_max] and normal [n_max, 2].

    Every grid cell with this seed uses the *prefix* of these pools, so neighbouring cells
    in (lambda, n) share their sampling noise.
    """
    g = torch.Generator(device=DEV).manual_seed(seed_int(tag, seed, gen))
    u = torch.rand(n_max, generator=g, device=DEV, dtype=DT)
    z = torch.randn(n_max, 2, generator=g, device=DEV, dtype=DT)
    return u, z


# ----------------------------------------------------------------------------- batched GMM
REG_COVAR = 1e-6  # absolute covariance floor (data units^2): std floor 1e-3


def gmm_logpdf_terms(X, pi, mu, cov):
    """X [B,N,2]; returns log pi_k + log N(x|mu_k,cov_k), shape [B,N,K]."""
    a = cov[..., 0, 0][:, None, :]
    b = cov[..., 0, 1][:, None, :]
    c = cov[..., 1, 1][:, None, :]
    det = (a * c - b * b).clamp_min(1e-300)
    dx = X[:, :, None, 0] - mu[:, None, :, 0]
    dy = X[:, :, None, 1] - mu[:, None, :, 1]
    maha = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
    return torch.log(pi.clamp_min(1e-300))[:, None, :] - math.log(2 * math.pi) - 0.5 * torch.log(det) - 0.5 * maha


def gmm_em(X, w, pi, mu, cov, iters):
    """Weighted batched EM, warm-started. X [B,N,2], w [B,N] (0 = padding)."""
    for _ in range(iters):
        lp = gmm_logpdf_terms(X, pi, mu, cov)
        r = torch.softmax(lp, dim=2) * w[:, :, None]
        Nk = r.sum(1)  # [B,K]
        alive = Nk > 1e-3
        pi = Nk / Nk.sum(1, keepdim=True)
        Nk_safe = Nk.clamp_min(1e-12)
        mu_new = torch.einsum("bnk,bnd->bkd", r, X) / Nk_safe[..., None]
        mu = torch.where(alive[..., None], mu_new, mu)
        d = X[:, :, None, :] - mu[:, None, :, :]
        cov_new = torch.einsum("bnk,bnkd,bnke->bkde", r, d, d) / Nk_safe[..., None, None]
        cov_new = cov_new + REG_COVAR * torch.eye(2, dtype=X.dtype, device=X.device)
        cov = torch.where(alive[..., None, None], cov_new, cov)
    return pi, mu, cov


def gmm_init(X, K):
    """Deterministic farthest-point init on the first chain dimension-wise. X [B,N,2]."""
    B, N, _ = X.shape
    idx = torch.zeros(B, K, dtype=torch.long, device=X.device)
    dmin = ((X - X[:, :1]) ** 2).sum(-1)
    for k in range(1, K):
        idx[:, k] = dmin.argmax(1)
        c = X[torch.arange(B), idx[:, k]]
        dmin = torch.minimum(dmin, ((X - c[:, None]) ** 2).sum(-1))
    mu = X[torch.arange(B)[:, None], idx]
    var = X.var(1).mean(1) / K  # [B]
    cov = var[:, None, None, None] * torch.eye(2, dtype=X.dtype, device=X.device).expand(B, K, 2, 2).clone()
    pi = torch.full((B, K), 1.0 / K, dtype=X.dtype, device=X.device)
    return pi, mu, cov


def gmm_sample(pi, mu, cov, u, z):
    """Inverse-CDF component choice with uniforms u [B,M], normals z [B,M,2]."""
    cdf = torch.cumsum(pi, 1)
    cdf = cdf / cdf[:, -1:]
    k = torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp_max(pi.shape[1] - 1)
    L = torch.linalg.cholesky(cov)  # [B,K,2,2]
    Lk = torch.gather(L, 1, k[..., None, None].expand(-1, -1, 2, 2))
    muk = torch.gather(mu, 1, k[..., None].expand(-1, -1, 2))
    return muk + torch.einsum("bmde,bme->bmd", Lk, z)


# ----------------------------------------------------------------------------- batched KDE
def kde_bandwidth_loocv(X, cnt, q_u, grid=None):
    """Isotropic Gaussian KDE bandwidth maximising leave-one-out log-likelihood.

    X [B,N,2] with valid prefix of length cnt[b]. The LOO likelihood is averaged over
    Q query points drawn (with CRN uniforms q_u [B,Q]) from the valid prefix; when cnt <= Q
    this is effectively the full LOO average (queries resampled with replacement).
    Grid search over 48 log-spaced h in [1e-3, 1] (declared).
    """
    if grid is None:
        grid = torch.logspace(-3, 0, 48, dtype=X.dtype, device=X.device)
    B, N, _ = X.shape
    qi = (q_u * cnt[:, None]).long().clamp_max(N - 1)  # [B,Q]
    Qx = torch.gather(X, 1, qi[..., None].expand(-1, -1, 2))
    D2 = torch.cdist(Qx, X) ** 2  # [B,Q,N]
    valid = torch.arange(N, device=X.device)[None, None, :] < cnt[:, None, None]
    notself = torch.arange(N, device=X.device)[None, None, :] != qi[..., None]
    logmask = torch.where(valid & notself, 0.0, -float("inf")).to(X.dtype)
    lcnt = torch.log((cnt - 1).clamp_min(1).to(X.dtype))[:, None]
    best = torch.full((B,), -float("inf"), dtype=X.dtype, device=X.device)
    hbest = torch.ones(B, dtype=X.dtype, device=X.device)
    for h in grid:
        lk = torch.logsumexp(-0.5 * D2 / h**2 + logmask, 2) - lcnt - math.log(2 * math.pi) - 2 * torch.log(h)
        ll = lk.mean(1)
        better = ll > best
        best = torch.where(better, ll, best)
        hbest = torch.where(better, h.expand(B), hbest)
    return hbest


def kde_sample(X, cnt, h, u, z):
    """Resample a training point (index = floor(u * count)) and add N(0, h^2 I). X [B,N,2] (valid prefix)."""
    idx = (u * cnt[:, None]).long().clamp_max(X.shape[1] - 1)
    base = torch.gather(X, 1, idx[..., None].expand(-1, -1, 2))
    return base + h[:, None, None] * z


# ----------------------------------------------------------------------------- metrics
_DIRS = None


def sliced_w2(Y, R, n_dirs=64):
    """Sliced 2-Wasserstein between batch Y [B,M,2] and reference R [M,2] (equal M)."""
    global _DIRS
    if _DIRS is None or _DIRS.shape[0] != n_dirs:
        th = torch.arange(n_dirs, dtype=DT, device=DEV) * math.pi / n_dirs
        _DIRS = torch.stack([th.cos(), th.sin()], 1)
    py = torch.sort(Y @ _DIRS.T, dim=1).values
    pr = torch.sort(R @ _DIRS.T, dim=0).values
    return ((py - pr[None]) ** 2).mean((1, 2)).sqrt()


def ring_mode_mass(Y, radius=3 * RING_SIGMA):
    c = torch.tensor(RING_CENTERS, dtype=Y.dtype, device=Y.device)
    d = torch.cdist(Y, c.expand(Y.shape[0], -1, -1))  # [B,M,8]
    return (d < radius).double().mean(1)  # [B,8]


def hist_overlap(Y, p_ref, bins=24):
    """Histogram overlap sum_c min(p_gen, p_ref) on a bins x bins grid over EXTENT."""
    lo, hi = EXTENT
    ij = ((Y - lo) / (hi - lo) * bins).floor().long().clamp(0, bins - 1)
    flat = ij[..., 0] * bins + ij[..., 1]
    B, M = flat.shape
    counts = torch.zeros(B, bins * bins, dtype=DT, device=Y.device)
    counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=DT))
    p = counts / M
    return torch.minimum(p, p_ref[None]).sum(1)


def ref_hist(R, bins=24):
    lo, hi = EXTENT
    H, _, _ = np.histogram2d(R[:, 0], R[:, 1], bins=bins, range=[[lo, hi], [lo, hi]])
    return torch.tensor(H.ravel() / H.sum(), dtype=DT, device=DEV)
