"""Intrinsic-dimension estimators, implemented from the papers (no skdim).

- TwoNN  (Facco, d'Errico, Rodriguez & Laio, Sci. Rep. 2017)
- MLE    (Levina & Bickel, NIPS 2005), with the MacKay-Ghahramani pooling used by
          Pope et al. 2021 ("average the inverse, then invert")
- correlation-integral curve N(r) (Grassberger-Procaccia), used only for pictures

All neighbour searches are exact, chunked brute force on the GPU (float64 distances).
"""
import numpy as np
import torch


def knn_dists(X, k, device="cuda", chunk=2048):
    """Distances to the k nearest neighbours (self excluded). X: (n, D) array/tensor.
    Exact, float64. Returns (n, k) numpy array sorted ascending."""
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=device)
    n = X.shape[0]
    sq = (X * X).sum(1)
    out = torch.empty(n, k, dtype=torch.float64, device=device)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * X[s:e] @ X.T
        d2.clamp_(min=0.0)
        d2[torch.arange(e - s), torch.arange(s, e)] = float("inf")  # exclude self
        vals, _ = torch.topk(d2, k, dim=1, largest=False)
        out[s:e] = vals.sqrt()
    return out.cpu().numpy()


def twonn(r, discard=0.1):
    """TwoNN from an (n, >=2) neighbour-distance array.
    Returns dict with d_fit (Facco's linear fit through the origin of -log(1-F) vs log mu,
    after discarding the top `discard` fraction of mu), d_mle (closed-form ML: (n-1)/sum log mu),
    and the arrays needed to draw the plot."""
    r1, r2 = r[:, 0], r[:, 1]
    ok = (r1 > 0) & np.isfinite(r2)
    mu = np.sort(r2[ok] / r1[ok])
    n = mu.size
    F = np.arange(1, n + 1) / n
    keep = int(np.floor(n * (1 - discard)))
    x = np.log(mu[:keep])
    y = -np.log(1 - F[:keep])
    d_fit = float((x * y).sum() / (x * x).sum())
    d_mle = float((n - 1) / np.log(mu).sum())
    return dict(d_fit=d_fit, d_mle=d_mle, x=x, y=y, mu=mu, n=n, n_zero=int((~ok).sum()))


def twonn_bootstrap(r, B=200, seed=0, discard=0.1):
    rng = np.random.default_rng(seed)
    n = r.shape[0]
    ds = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        ds[b] = twonn(r[idx], discard)["d_fit"]
    return np.percentile(ds, [2.5, 97.5])


def mle_levina_bickel(r, k):
    """Levina-Bickel MLE with k neighbours, pooled as in MacKay & Ghahramani / Pope et al. 2021:
    m_hat = [ 1/(n(k-1)) sum_i sum_{j<k} log(T_k(x_i)/T_j(x_i)) ]^{-1}.
    Also returns LB05's original pooling (mean of per-point inverse, with the unbiased k-2)."""
    T = r[:, :k]
    ok = (T[:, 0] > 0)
    T = T[ok]
    L = np.log(T[:, -1:] / T[:, :-1]).sum(1)  # sum_{j<k} log(T_k/T_j), shape (n,)
    m_mg = float(1.0 / (L.mean() / (k - 1)))
    per_point = (k - 2) / L
    m_lb = float(per_point.mean())
    return dict(d=m_mg, d_lb=m_lb, per_point=per_point)


def mle_bootstrap(r, k, B=200, seed=0):
    rng = np.random.default_rng(seed)
    n = r.shape[0]
    ds = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        ds[b] = mle_levina_bickel(r[idx], k)["d"]
    return np.percentile(ds, [2.5, 97.5])


def neighbour_count_curve(X, radii, n_centres=500, device="cuda", seed=0):
    """Mean number of points within radius r of a random data point (self excluded):
    N(r) ~ r^d over the scaling range. X: (n, D)."""
    rng = np.random.default_rng(seed)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=device)
    idx = rng.choice(Xt.shape[0], size=min(n_centres, Xt.shape[0]), replace=False)
    C = Xt[idx]
    rr = torch.as_tensor(np.asarray(radii), dtype=torch.float64, device=device)
    counts = torch.zeros(len(radii), dtype=torch.float64, device=device)
    for s in range(0, len(idx), 64):
        d = torch.cdist(C[s:s + 64], Xt)  # (c, n)
        counts += (d[:, :, None] <= rr[None, None, :]).sum(1).sum(0).double() - d.shape[0]  # remove self per centre
    return (counts / len(idx)).cpu().numpy()


def estimate_all(X, ks=(5, 10, 20), device="cuda", bootstrap=True):
    kmax = max(max(ks), 2)
    r = knn_dists(X, kmax, device=device)
    tw = twonn(r)
    out = dict(twonn=tw["d_fit"], twonn_mle=tw["d_mle"], n_zero=tw["n_zero"])
    if bootstrap:
        out["twonn_ci"] = twonn_bootstrap(r).tolist()
    for k in ks:
        m = mle_levina_bickel(r, k)
        out[f"mle{k}"] = m["d"]
        if bootstrap:
            out[f"mle{k}_ci"] = mle_bootstrap(r, k).tolist()
    return out, r, tw
