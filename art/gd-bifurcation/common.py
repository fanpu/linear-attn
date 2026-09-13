"""Shared dynamics for the Cascade project.

Every map here is an exact gradient-descent step (or the logistic map, used as
a reference/null), written so that a whole sweep over step sizes eta is one
vectorised batch.  Everything is float64.

Objectives
----------
prod-k   : f(x) = 1/2 (x_1 x_2 ... x_k - 1)^2,          x in R^k   (full GD, all k coords)
bal-k    : the same objective restricted to the balanced line x_i = u
           u <- u - eta (u^k - 1) u^(k-1)                 (exact invariant sub-dynamics)
dln      : f(W) = 1/2 || W_L ... W_1 - M ||_F^2, W_l in R^{d x d}
logistic : x <- r x (1 - x)
"""
import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
ROOT = "/home/fzeng/ml/research/art/gd-bifurcation"
CACHE = ROOT + "/cache"
GALLERY = ROOT + "/gallery"

BIG = 1e6  # divergence threshold on |coordinate|


# ----------------------------------------------------------------- scalar products
def others_prod(x):
    """prod_{j != i} x_j for each i, without division.  x: (..., k)."""
    k = x.shape[-1]
    ones = torch.ones_like(x[..., :1])
    left = torch.cumprod(torch.cat([ones, x[..., :-1]], -1), -1)
    right = torch.flip(torch.cumprod(torch.cat([ones, torch.flip(x[..., 1:], [-1])], -1), -1), [-1])
    return left * right


def prod_step(x, eta):
    """One GD step on 1/2(prod x - 1)^2.  x: (N,k), eta: (N,1) or scalar."""
    g = others_prod(x)                       # dP/dx_i
    P = x[..., :1] * g[..., :1]              # full product
    return x - eta * (P - 1.0) * g


def prod_loss(x):
    return 0.5 * (torch.prod(x, -1) - 1.0) ** 2


def prod_hessian(x):
    """Hessian of 1/2(P-1)^2: g g^T + (P-1) d2P.  x: (N,k) -> (N,k,k)."""
    N, k = x.shape
    g = others_prod(x)
    P = x[:, 0] * g[:, 0]
    H = g[:, :, None] * g[:, None, :]
    d2 = torch.zeros(N, k, k, dtype=x.dtype, device=x.device)
    for i in range(k):
        for j in range(k):
            if i != j:
                m = torch.ones(k, dtype=torch.bool, device=x.device)
                m[i] = False
                m[j] = False
                d2[:, i, j] = torch.prod(x[:, m], -1) if m.any() else 1.0
    return H + (P - 1.0)[:, None, None] * d2


def prod_tangent(x, v, eta):
    """(I - eta H) v, the linearised GD map, without forming H.  x,v: (N,k)."""
    k = x.shape[-1]
    g = others_prod(x)
    P = x[:, :1] * g[:, :1]
    gv = (g * v).sum(-1, keepdim=True)
    # (d2P v)_i = sum_{j != i} prod_{l != i,j} x_l v_j ;  use derivative of others_prod along v
    # d/ds others_prod(x + s v) at s=0
    d2v = torch.zeros_like(x)
    for i in range(k):
        acc = torch.zeros_like(x[:, 0])
        for j in range(k):
            if j == i:
                continue
            m = [l for l in range(k) if l != i and l != j]
            term = v[:, j]
            for l in m:
                term = term * x[:, l]
            acc = acc + term
        d2v[:, i] = acc
    return v - eta * (g * gv + (P - 1.0) * d2v)


def bal_step(u, eta, k):
    return u - eta * (u ** k - 1.0) * u ** (k - 1)


def bal_deriv(u, eta, k):
    return 1.0 - eta * ((2 * k - 1) * u ** (2 * k - 2) - (k - 1) * u ** (k - 2))


def bal_transverse(u, eta, k):
    """Per-step multiplier of any imbalance x_i^2 - x_j^2 at a balanced point.

    Exact identity for GD on the product objective:
        x_i'^2 - x_j'^2 = (x_i^2 - x_j^2) * (1 - eta^2 r^2 P^2 / (x_i^2 x_j^2)),  r = P - 1.
    On the balanced line P = u^k, x_i x_j = u^2.
    """
    return 1.0 - eta ** 2 * (u ** k - 1.0) ** 2 * u ** (2 * k - 4)


# ----------------------------------------------------------------- logistic
def logistic_step(x, r):
    return r * x * (1.0 - x)


def logistic_deriv(x, r):
    return r * (1.0 - 2.0 * x)


# ----------------------------------------------------------------- deep linear network
def dln_target(d=5, svals=(10.0, 6.0, 3.0), seed=0):
    """Ghosh et al. (2025) App. B.1: 5x5 rank-3 target with singular values 10, 6, 3."""
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=DT))
    V, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=DT))
    s = torch.zeros(d, dtype=DT)
    s[: len(svals)] = torch.tensor(svals, dtype=DT)
    return (U * s) @ V.T


def dln_forward(W):
    """W: (N,L,d,d) -> end-to-end W_L ... W_1 (N,d,d)."""
    out = W[:, 0]
    for l in range(1, W.shape[1]):
        out = W[:, l] @ out
    return out


def dln_grad(W, M):
    N, L, d, _ = W.shape
    # prefix[l] = W_{l-1}...W_1 (identity for l=0), suffix[l] = W_L...W_{l+1}
    eye = torch.eye(d, dtype=W.dtype, device=W.device).expand(N, d, d)
    pre = [eye]
    for l in range(L - 1):
        pre.append(W[:, l] @ pre[-1])
    suf = [eye] * L
    acc = eye
    for l in range(L - 1, -1, -1):
        suf[l] = acc
        acc = acc @ W[:, l]
    R = acc - M                                  # acc is now W_L...W_1
    G = torch.stack([suf[l].transpose(-1, -2) @ R @ pre[l].transpose(-1, -2) for l in range(L)], 1)
    return G, R


def dln_step(W, M, eta):
    G, R = dln_grad(W, M)
    return W - eta[:, None, None, None] * G, R


def dln_tangent(W, V, M, eta):
    """(I - eta H) V via forward-mode AD through the analytic gradient."""
    def gfun(Wx):
        return dln_grad(Wx, M)[0]
    _, jv = torch.func.jvp(gfun, (W,), (V,))
    return V - eta[:, None, None, None] * jv
