"""Build-on #2: steepest descent geometries for a linear multiclass classifier W in R^{k x d}, cross-entropy loss.

Optimizers (all float64, full batch, constant step unless noted), vectorized over S random separable datasets:
  gd        W -= eta * G                           (Frobenius geometry)
  ngd       W -= eta * G/||G||_F
  sign      W -= eta * sign(G)                     (entrywise max-norm geometry)
  spec      W -= eta * U V^T,  G = U S V^T         (spectral-norm steepest descent, exact SVD)
  spec_dec  same with eta_t = eta0/sqrt(t)          (the schedule analysed by Fan, Schmidt & Thrampoulidis 2025)
  muon_svd  M = b M + G ;  W -= eta * polar(G + b M)   (Nesterov momentum b = 0.95, exact orthogonalization)
  muon_ns   same, orthogonalized by 5 Newton-Schulz steps with Keller Jordan's quintic (3.4445, -4.7750, 2.0315)
  adam      Adam, eps = 0 (log-scale gradients), beta = (0.9, 0.999)

The per-sample gradient weights are carried in log domain (G = e^C * Ghat) so scale-free methods never underflow.
Reference max-margin solutions: min ||W|| s.t. (W_y - W_c) x_i >= 1 for Frobenius, max-entry, spectral, nuclear (cvxpy).
    python compute_spectral.py [T]
"""
import sys, time
import numpy as np
import cvxpy as cp
from core import log_checkpoints

T = int(float(sys.argv[1])) if len(sys.argv) > 1 else 300000
k, d, n, Sn = 4, 24, 40, 6
OPTS = ["gd", "ngd", "sign", "spec", "spec_dec", "muon_svd", "muon_ns", "adam"]


def make(seed):
    rng = np.random.default_rng(500 + seed)
    X = rng.standard_normal((n, d)) / np.sqrt(d)
    y = rng.integers(0, k, n)
    return X, y


def maxmargin(X, y, norm):
    W = cp.Variable((k, d))
    cons = []
    for i in range(n):
        for c in range(k):
            if c != y[i]:
                cons.append((W[y[i]] - W[c]) @ X[i] >= 1)
    obj = {"fro": cp.norm(W, "fro"), "max": cp.max(cp.abs(W)), "spec": cp.sigma_max(W), "nuc": cp.normNuc(W)}[norm]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver="CLARABEL")
    return np.asarray(W.value), prob.value


def norms(W):
    s = np.linalg.svd(W, compute_uv=False)
    return {"fro": np.sqrt((W ** 2).sum((-2, -1))), "max": np.abs(W).max((-2, -1)), "spec": s[..., 0], "nuc": s.sum(-1)}


def ns5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    Xm = G / (np.linalg.norm(G, axis=(-2, -1), keepdims=True) + 1e-300)  # k <= d: iterate on the k x k Gram side
    for _ in range(steps):
        A = Xm @ np.swapaxes(Xm, -1, -2)
        B = b * A + c * A @ A
        Xm = a * Xm + B @ Xm
    return Xm


def polar(G):
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    return U @ Vt


if __name__ == "__main__":
    data = [make(s) for s in range(Sn)]
    X = np.stack([D[0] for D in data])  # S n d
    Y = np.stack([D[1] for D in data])  # S n
    onehot = np.eye(k)[Y]               # S n k
    refs = {nm: np.stack([maxmargin(Xs, ys, nm)[0] for Xs, ys in data]) for nm in ["fro", "max", "spec", "nuc"]}
    print("references done", {nm: np.round(1 / norms(R)[nm], 4) for nm, R in refs.items()}, flush=True)

    def grad_hat(W):
        logits = np.einsum("skd,snd->snk", W, X)
        lse = np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        logp = logits - lse                                   # S n k
        # weight of (i, c != y_i) is p_ic ; carry in log domain
        logw = np.where(onehot > 0, -np.inf, logp)
        C = logw.max(axis=(1, 2), keepdims=True)
        Pw = np.exp(logw - C)                                  # S n k, zero on the true class
        coef = Pw - onehot * Pw.sum(-1, keepdims=True)         # d/dlogits of the loss (scaled)
        G = np.einsum("snk,snd->skd", coef, X)
        return G, C[:, 0, 0]

    Wst = {o: np.zeros((Sn, k, d)) for o in OPTS}
    mom = {o: np.zeros((Sn, k, d)) for o in OPTS}
    momC = {o: np.zeros(Sn) for o in OPTS}
    adamM, adamV, adamC = np.zeros((Sn, k, d)), np.zeros((Sn, k, d)), np.zeros(Sn)
    ts = log_checkpoints(T, 30)
    rec = {o: np.empty((len(ts), Sn, k, d)) for o in OPTS}
    eta, beta = 0.01, 0.95
    kk = 0
    t0 = time.time()
    for t in range(1, T + 1):
        for o in OPTS:
            W = Wst[o]
            G, C = grad_hat(W)
            if o == "gd":
                W -= 0.5 * np.exp(C)[:, None, None] * G
            elif o == "ngd":
                W -= eta * G / np.linalg.norm(G, axis=(1, 2), keepdims=True)
            elif o == "sign":
                W -= eta * np.sign(G)
            elif o == "spec":
                W -= eta * polar(G)
            elif o == "spec_dec":
                W -= 0.1 / np.sqrt(t) * polar(G)
            elif o in ("muon_svd", "muon_ns"):
                # raw-gradient momentum, carried in the same log scale as G (buffer rescaled when the scale moves)
                fac = np.exp(momC[o] - C)[:, None, None]
                mom[o] = beta * mom[o] * fac + G
                momC[o] = C
                upd = G + beta * mom[o]
                W -= eta * (polar(upd) if o == "muon_svd" else ns5(upd))
            elif o == "adam":
                fac = np.exp(adamC - C)[:, None, None]
                adamM = 0.9 * adamM * fac + 0.1 * G
                adamV = 0.999 * adamV * fac ** 2 + 0.001 * G ** 2
                adamC = C
                W -= eta * (adamM / (1 - 0.9 ** t)) / (np.sqrt(adamV / (1 - 0.999 ** t)) + 1e-300)
        if t == ts[kk]:
            for o in OPTS:
                rec[o][kk] = Wst[o]
            kk += 1
            if kk % 30 == 0:
                print(t, f"{time.time() - t0:.0f}s", flush=True)
    np.savez("cache/spectral.npz", steps=ts, X=X, Y=Y, **{f"W_{o}": rec[o] for o in OPTS}, **{f"ref_{nm}": R for nm, R in refs.items()})
    print("done", time.time() - t0)
