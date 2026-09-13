"""Core math for in-context linear regression: data, closed forms, and classical baselines.

Conventions (following Zhang, Frei & Bartlett 2024, "ZFB"):
  x_i, x_q ~ N(0, Lam)   (d-dim)      w ~ N(0, I_d)      y_i = <w, x_i> + sigma * eps_i
  A prompt with M context pairs is embedded as E = [[x_1 .. x_M x_q], [y_1 .. y_M 0]] in R^{(d+1) x (M+1)}.

Everything is batched torch; use float64 when comparing against closed forms.
"""
from __future__ import annotations

import math
import torch

# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------

def make_cov(kind: str, d: int, rho: float = 0.8, dtype=torch.float64) -> torch.Tensor:
    """'iso' -> I;  'ar1' -> Lam_ij = rho^|i-j| (trace d, smooth correlations, tridiagonal inverse)."""
    if kind == "iso":
        return torch.eye(d, dtype=dtype)
    if kind == "ar1":
        i = torch.arange(d, dtype=dtype)
        return rho ** (i[:, None] - i[None, :]).abs()
    raise ValueError(kind)


def sample_prompts(B, M, d, Lam=None, sigma=0.0, w=None, device="cpu", dtype=torch.float64, gen=None):
    """Returns X [B,M,d], y [B,M], xq [B,d], yq [B], w [B,d]. yq is noiseless-plus-noise like the context."""
    kw = dict(device=device, dtype=dtype, generator=gen)
    Z = torch.randn(B, M + 1, d, **kw)
    if Lam is not None:
        L = torch.linalg.cholesky(Lam.to(device=device, dtype=dtype))
        Z = Z @ L.T
    if w is None:
        w = torch.randn(B, d, **kw)
    X, xq = Z[:, :M], Z[:, M]
    y = torch.einsum("bmd,bd->bm", X, w)
    yq = torch.einsum("bd,bd->b", xq, w)
    if sigma > 0:
        y = y + sigma * torch.randn(B, M, **kw)
        yq = yq + sigma * torch.randn(B, **kw)
    return X, y, xq, yq, w


def embed(X, y, xq):
    """ZFB embedding E in R^{B x (d+1) x (M+1)}."""
    B, M, d = X.shape
    top = torch.cat([X, xq[:, None]], 1)                          # [B, M+1, d]
    bot = torch.cat([y, torch.zeros(B, 1, dtype=y.dtype, device=y.device)], 1)
    return torch.cat([top, bot[..., None]], 2).transpose(1, 2)    # [B, d+1, M+1]


# ----------------------------------------------------------------------------------------------
# one-layer linear self-attention (ZFB eq. 3.3) and its closed-form global minimum (Thm 4.1)
# ----------------------------------------------------------------------------------------------

def lsa_predict(E, W_pv, W_kq, rho=None):
    """y_hat = [E + W_pv E E^T W_kq E / rho]_{d+1, M+1}  (bottom-right entry)."""
    M = E.shape[-1] - 1
    rho = M if rho is None else rho
    G = E @ E.transpose(1, 2) / rho                               # [B, d+1, d+1]
    out = W_pv @ G @ (W_kq @ E[..., -1:])                          # [B, d+1, 1]
    return out[:, -1, 0]                                           # residual y-entry of query is 0


def zfb_gamma(Lam: torch.Tensor, N: int) -> torch.Tensor:
    d = Lam.shape[0]
    return (1 + 1 / N) * Lam + torch.trace(Lam) / N * torch.eye(d, dtype=Lam.dtype)


def zfb_global_min(Lam: torch.Tensor, N: int):
    """W*_PV, W*_KQ from ZFB Theorem 4.1 (balanced scaling tr(Gamma^-2)^{+-1/4})."""
    d = Lam.shape[0]
    Gi = torch.linalg.inv(zfb_gamma(Lam, N))
    c = torch.trace(Gi @ Gi) ** 0.25
    W_pv = torch.zeros(d + 1, d + 1, dtype=Lam.dtype)
    W_pv[d, d] = c
    W_kq = torch.zeros(d + 1, d + 1, dtype=Lam.dtype)
    W_kq[:d, :d] = Gi / c
    return W_pv, W_kq


def effective_preconditioner(W_pv, W_kq, d):
    """For the prediction path used by ZFB: y_hat ~ x_q^T B^T (1/M) sum y_i x_i with B = w22 * W11^KQ
    (+ the rank-one path w21^PV (w21^KQ)^T which is zero at the ZFB optimum)."""
    return W_pv[d, d] * W_kq[:d, :d] + torch.outer(W_pv[d, :d], W_kq[d, :d])


def risk_precond_gd1(A, Lam, M, sigma=0.0, Lam_q=None, w_cov=None, include_target_noise=False):
    """Exact E[(x_q^T A (1/M) sum y_i x_i - x_q^T w)^2] for x_i ~ N(0,Lam), x_q ~ N(0,Lam_q), w ~ N(0, w_cov).

    Uses E[S B S] = Lam B Lam + (Lam B^T Lam + tr(B Lam) Lam)/M for S = (1/M) sum x_i x_i^T, B symmetric.
    """
    d = Lam.shape[0]
    I = torch.eye(d, dtype=Lam.dtype)
    Lq = Lam if Lam_q is None else Lam_q
    C = I if w_cov is None else w_cov
    Bm = A.T @ Lq @ A                                            # symmetric
    ESBS = Lam @ Bm @ Lam * (1 + 1 / M) + torch.trace(Bm @ Lam) * Lam / M
    r = torch.trace(Lq @ C) - 2 * torch.trace(Lq @ A @ Lam @ C) + torch.trace(ESBS @ C)
    r = r + sigma**2 * torch.trace(Bm @ Lam) / M
    if include_target_noise:
        r = r + sigma**2
    return r


# ----------------------------------------------------------------------------------------------
# classical in-context algorithms (all batched)
# ----------------------------------------------------------------------------------------------

def ridge_w(X, y, lam):
    """argmin ||Xw - y||^2 + lam ||w||^2.  X [B,M,d] -> [B,d]."""
    d = X.shape[-1]
    I = torch.eye(d, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(X.transpose(1, 2) @ X + lam * I, (X.transpose(1, 2) @ y[..., None]))[..., 0]


def rls_prefix_w(X, y, lam):
    """Recursive least squares. Returns W [B, M+1, d]: W[:, t] = ridge solution on the first t pairs.
    Sherman-Morrison on P_t = (lam I + sum_{s<=t} x_s x_s^T)^{-1}; exactly equals ridge_w on each prefix."""
    B, M, d = X.shape
    P = torch.eye(d, dtype=X.dtype, device=X.device).expand(B, d, d) / lam
    w = torch.zeros(B, d, dtype=X.dtype, device=X.device)
    out = [w]
    for t in range(M):
        x = X[:, t]
        Px = torch.einsum("bij,bj->bi", P, x)
        g = Px / (1 + (x * Px).sum(-1, keepdim=True))            # gain
        w = w + g * (y[:, t] - (w * x).sum(-1))[:, None]
        P = P - g[:, :, None] * Px[:, None, :]
        out.append(w)
    return torch.stack(out, 1)


def gd_w(X, y, lrs, A=None, w0=None):
    """k steps of (preconditioned) full-batch GD on R(w) = (1/2M) ||Xw - y||^2 from w0 = 0.
    lrs: list of scalars (or [d,d] matrices when A is None and entries are tensors)."""
    B, M, d = X.shape
    w = torch.zeros(B, d, dtype=X.dtype, device=X.device) if w0 is None else w0
    S = X.transpose(1, 2) @ X / M
    b = torch.einsum("bmd,bm->bd", X, y) / M
    for eta in lrs:
        g = torch.einsum("bij,bj->bi", S, w) - b
        if isinstance(eta, torch.Tensor) and eta.dim() == 2:
            w = w - g @ eta.T
        else:
            w = w - eta * (g if A is None else g @ A.T)
    return w


def lms_prefix_w(X, y, beta, normalize=True):
    """Online SGD / delta rule (single pass): w <- w + beta (y_t - w.x_t) x_t / (|x_t|^2 if normalize).
    This is exactly a one-head DeltaNet state with k_t = x_t/|x_t|, v_t = y_t/|x_t| read out with q = x_q."""
    B, M, d = X.shape
    w = torch.zeros(B, d, dtype=X.dtype, device=X.device)
    out = [w]
    for t in range(M):
        x = X[:, t]
        s = (x * x).sum(-1, keepdim=True) if normalize else 1.0
        w = w + beta * (y[:, t] - (w * x).sum(-1))[:, None] * x / s
        out.append(w)
    return torch.stack(out, 1)


def dmmse_prefix_w(X, y, tasks, sigma, max_elems=6e7):
    """Posterior mean of w under a uniform prior on `tasks` [T,d] (Raventos et al. eq. 2).
    Returns W [B, M+1, d] for prefixes 0..M. Chunked over batch and tasks to bound memory."""
    B, M, d = X.shape
    T = tasks.shape[0]
    bs = max(1, int(max_elems // (T * (M + 1))))
    outs = []
    for b0 in range(0, B, bs):
        Xb, yb = X[b0:b0 + bs], y[b0:b0 + bs]
        nb = Xb.shape[0]
        logl = torch.zeros(nb, T, M + 1, dtype=X.dtype, device=X.device)
        chunk = max(1, int(max_elems // max(1, nb * M * d)))
        for s in range(0, T, chunk):
            tk = tasks[s:s + chunk]
            r = yb[:, None, :] - torch.einsum("bmd,td->btm", Xb, tk)
            logl[:, s:s + chunk, 1:] = -(r**2).cumsum(-1) / (2 * sigma**2)
        p = torch.softmax(logl, dim=1)                             # [nb, T, M+1]
        outs.append(torch.einsum("btm,td->bmd", p, tasks))
        del logl, p
    return torch.cat(outs, 0)


# ----------------------------------------------------------------------------------------------
# proportional-limit (Marchenko-Pastur) formulas for isotropic data, used by the JS widget
# ----------------------------------------------------------------------------------------------

def mp_quadrature(gamma: float, n: int = 2000):
    """Nodes/weights for the MP law of S = X^T X / M with aspect gamma = d/M (continuous part only)
    plus the mass of the atom at 0 when gamma > 1."""
    a, b = (1 - math.sqrt(gamma)) ** 2, (1 + math.sqrt(gamma)) ** 2
    t = torch.linspace(0, math.pi, n + 1, dtype=torch.float64)
    t = 0.5 * (t[1:] + t[:-1])
    s = a + (b - a) * (1 - torch.cos(t)) / 2                       # Chebyshev-like spacing
    ds = (b - a) * torch.sin(t) / 2 * (math.pi / n)
    dens = torch.sqrt(torch.clamp((b - s) * (s - a), min=0)) / (2 * math.pi * gamma * s)
    wts = dens * ds
    atom = max(0.0, 1 - 1 / gamma)
    return s, wts, atom
