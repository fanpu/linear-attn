"""Pure-numpy fits for Number Knot M1 (CPU). No model code here.

Helix (Kantamneni & Tegmark 2025, arXiv:2502.00873 §4): project the residual stream to its top-100
principal components, then regress PCA(h_a) = C B(a) with B(a) = [1, a, cos 2πa/T, sin 2πa/T].
R² is the variance-weighted fraction of the 100-dim PCA cloud explained, 1 - ||Y - XC||² / ||Y||².
ΔR²_T = R²_T - R²_lin isolates the circle of period T beyond the linear term.

Calendar circles (supervised, not SAE clustering as in Engels et al. 2024): class means over
templates, the top-2 principal plane of the class means ("mean-difference plane"), then a 2-D
regression P = c + A [cos 2πk/K, sin 2πk/K].
"""
from __future__ import annotations

import numpy as np

PERIODS = (2, 5, 10, 100)


def pca(X: np.ndarray, k: int = 100):
    """X (N, D) -> scores (N, k'), components (k', D), mean (D,), explained-variance ratio (k',)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    k = min(k, X.shape[0] - 1, X.shape[1])
    # Gram-matrix SVD is cheaper when N < D
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    tot = (S ** 2).sum()
    return U[:, :k] * S[:k], Vt[:k], mu, (S[:k] ** 2) / tot


def basis(a: np.ndarray, T: int | None) -> np.ndarray:
    """[1, a, cos, sin] (T=None -> [1, a]). Columns that vanish identically (sin πa for T=2) are dropped."""
    a = np.asarray(a, dtype=np.float64)
    cols = [np.ones_like(a), a / max(a.max(), 1.0)]   # scaling a leaves R² unchanged
    if T is not None:
        w = 2 * np.pi * a / T
        cols += [np.cos(w), np.sin(w)]
    B = np.stack(cols, 1)
    keep = np.linalg.norm(B, axis=0) > 1e-8 * np.sqrt(len(a))
    return B[:, keep]


def r2_fit(Y: np.ndarray, B: np.ndarray) -> float:
    """Variance-weighted R² of least squares Y ≈ B C (Y is centred by the constant column)."""
    Yc = Y - Y.mean(0)
    Q, _ = np.linalg.qr(B)
    return float((np.einsum("nq,nd->qd", Q, Yc) ** 2).sum() / (Yc ** 2).sum())


def r2_perm(Y: np.ndarray, B: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """R² for each row-permutation of the labels: equivalent to refitting on shuffled a."""
    Yc = Y - Y.mean(0)
    Q, _ = np.linalg.qr(B)
    tot = (Yc ** 2).sum()
    out = np.empty(len(perms))
    for i, p in enumerate(perms):
        # label a[j] now belongs to row p[j]
        out[i] = (np.einsum("nq,nd->qd", Q, Yc[p]) ** 2).sum() / tot
    return out


def helix_table(Y: np.ndarray, a: np.ndarray, perms: np.ndarray, periods=PERIODS):
    """Returns dict: r2[T or 'lin'] (float) and null[T or 'lin'] (n_perm,) for shuffled labels."""
    r2, null = {}, {}
    for T in (None,) + tuple(periods):
        B = basis(a, T)
        key = "lin" if T is None else T
        r2[key] = r2_fit(Y, B)
        null[key] = r2_perm(Y, B, perms)
    return r2, null


def helix_coords(Y: np.ndarray, a: np.ndarray, T: int):
    """Least-squares decode of each measured point into the fitted [a, cos, sin] directions.
    Returns coords (N, 3), C (k, 3) fitted directions in PCA space, intercept (k,)."""
    B = basis(a, T)
    assert B.shape[1] == 4, "helix_coords needs a non-degenerate circle (T != 2)"
    coef, *_ = np.linalg.lstsq(B, Y, rcond=None)          # (4, k)
    c0, C = coef[0], coef[1:].T                          # C: (k, 3)
    coords = (Y - c0) @ np.linalg.pinv(C).T              # (N, 3) in units of the basis functions
    return coords, C, c0


def fourier_power(Y: np.ndarray, a: np.ndarray, detrend: bool):
    """K&T-style spectrum over a: centre, optionally remove the linear fit in a, rfft along a
    (a must be 0..N-1 in order), sum power over dims, normalise to fraction of non-DC power."""
    order = np.argsort(a)
    Yo = Y[order] - Y.mean(0)
    if detrend:
        B = basis(a[order], None)
        coef, *_ = np.linalg.lstsq(B, Yo, rcond=None)
        Yo = Yo - B @ coef
    P = (np.abs(np.fft.rfft(Yo, axis=0)) ** 2).sum(1)
    P[0] = 0.0
    return P / P.sum()


# ---- calendar circles -----------------------------------------------------------------------------

def mean_plane(H: np.ndarray, lab: np.ndarray, K: int):
    """H (n, D), integer class labels -> grand mean (D,), plane (2, D), class means (K, D), var ratio."""
    mu = H.mean(0)
    M = np.stack([H[lab == k].mean(0) for k in range(K)])
    Mc = M - M.mean(0)
    _, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    return M.mean(0), Vt[:2], M, float((S[:2] ** 2).sum() / (S ** 2).sum())


def circle_design(pos: np.ndarray, K: int) -> np.ndarray:
    th = 2 * np.pi * pos / K
    return np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)


def circle_r2(P_fit, pos_fit, P_eval, pos_eval, K):
    """Fit P = c + A[cos, sin] on (P_fit, pos_fit), R² of that fit on (P_eval, pos_eval)."""
    coef, *_ = np.linalg.lstsq(circle_design(pos_fit, K), P_fit, rcond=None)
    res = P_eval - circle_design(pos_eval, K) @ coef
    return float(1 - (res ** 2).sum() / ((P_eval - P_eval.mean(0)) ** 2).sum())


def calendar_scores(H: np.ndarray, lab: np.ndarray, tmpl: np.ndarray, K: int, order=None):
    """H (n, D) one layer; lab class ids 0..K-1 used to build the plane; order maps class -> cyclic
    position (identity = true order). Returns in-sample R², held-out R² (plane and circle fitted on
    even templates, scored on odd, and vice versa, averaged), cyclic-order flag, plane var ratio."""
    order = np.arange(K) if order is None else np.asarray(order)
    pos = order[lab]
    c, U, M, vr = mean_plane(H, lab, K)
    P = (H - c) @ U.T
    r2_in = circle_r2(P, pos, P, pos, K)
    ho = []
    for fit_par in (0, 1):
        f, e = (tmpl % 2 == fit_par), (tmpl % 2 != fit_par)
        cf, Uf, _, _ = mean_plane(H[f], lab[f], K)
        Pf, Pe = (H[f] - cf) @ Uf.T, (H[e] - cf) @ Uf.T
        ho.append(circle_r2(Pf, pos[f], Pe, pos[e], K))
    Pm = (M - c) @ U.T
    ang = np.arctan2(Pm[:, 1], Pm[:, 0])
    seq = order[np.argsort(ang)]                          # cyclic positions in angular order
    steps = np.diff(np.r_[seq, seq[0]]) % K
    cyclic = bool(np.all(steps == 1) or np.all(steps == K - 1))
    return dict(r2_in=r2_in, r2_ho=float(np.mean(ho)), cyclic=cyclic, plane_var=vr, proj_means=Pm, P=P)


def basis_mod(a: np.ndarray, m: int) -> np.ndarray:
    """[1, a, one-hot(a mod m) minus one level]: every function of a mod m plus the linear term.
    The T = m, m/2, m/3, ... circles all live inside this span."""
    a = np.asarray(a)
    oh = (a[:, None] % m == np.arange(1, m)[None]).astype(np.float64)
    return np.concatenate([basis(a, None), oh], 1)


def basis_poly3(a: np.ndarray) -> np.ndarray:
    """[1, a, a², a³]: same column count as the circle basis; a smooth non-periodic comparator."""
    x = np.asarray(a, dtype=np.float64) / max(np.max(a), 1)
    x = 2 * x - 1
    return np.stack([np.ones_like(x), x, x ** 2, x ** 3], 1)
