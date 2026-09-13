"""1-D random Fourier features, min-norm least squares, n = 20 noisy points.  -> cache/hero.npz

Features phi_j(x) = sqrt(2/P) cos(w_j x + b_j), w_j ~ N(0, SCALE^2), b_j ~ U[0, 2pi). Features are nested: the model
with P features uses the first P of one fixed sequence, so the sweep over P is a single continuous "growing" model.
"""
import os, pathlib
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)

N_PTS, NOISE, SCALE, JITTER, DATA_SEED = 20, 0.10, 10.0, 0.45, 2
PMAX = 5000


def target(t):
    return 0.8 * np.sin(2.0 * np.pi * t) + 0.3 * t


def make_data():
    rng = np.random.default_rng(DATA_SEED)
    n = N_PTS
    x = np.sort(np.linspace(-1, 1, n) + rng.uniform(-JITTER, JITTER, n) * (2 / (n - 1)))
    y = target(x) + NOISE * rng.standard_normal(n)
    return x, y


def features(seed):
    r = np.random.default_rng(seed)
    W = r.standard_normal(PMAX) * SCALE
    B = r.uniform(0, 2 * np.pi, PMAX)
    return W, B


def phi(t, W, B, P):
    return np.sqrt(2.0 / P) * np.cos(np.outer(t, W[:P]) + B[:P])


def fit(x, y, W, B, P, lam=0.0):
    F = phi(x, W, B, P)
    if lam == 0:
        return np.linalg.lstsq(F, y, rcond=None)[0]
    if P > len(x):
        return F.T @ np.linalg.solve(F @ F.T + lam * np.eye(len(x)), y)
    return np.linalg.solve(F.T @ F + lam * np.eye(P), F.T @ y)


def p_list():
    return np.unique(np.concatenate([np.arange(1, 61), np.round(np.geomspace(60, PMAX, 70)).astype(int)]))


if __name__ == "__main__":
    x, y = make_data()
    xs = np.linspace(-1, 1, 700)
    fs = target(xs)
    Ps = p_list()
    # risk statistics over many feature draws (same data)
    S = 200
    R = np.zeros((len(Ps), S))
    SM = np.zeros((len(Ps), S))
    for s in range(S):
        Ws, Bs = features(1000 + s)
        for i, P in enumerate(Ps):
            R[i, s] = np.mean((phi(xs, Ws, Bs, P) @ fit(x, y, Ws, Bs, P) - fs) ** 2)
            SM[i, s] = np.linalg.svd(phi(x, Ws, Bs, P), compute_uv=False)[-1]
    # the animated draw: the feature seed whose log-risk curve is closest to the median one ("a typical draw")
    rep = int(np.argmin(np.mean(np.abs(np.log(R) - np.log(np.median(R, 1, keepdims=True))), axis=0)))
    W, B = features(1000 + rep)
    fits = np.array([phi(xs, W, B, P) @ fit(x, y, W, B, P) for P in Ps])
    norms = np.array([np.linalg.norm(fit(x, y, W, B, P)) for P in Ps])
    risk_shown = np.mean((fits - fs) ** 2, axis=1)
    smin_shown = np.array([np.linalg.svd(phi(x, W, B, P), compute_uv=False)[-1] for P in Ps])
    # ridge comparison curves: a few lambdas for the same draws (fewer draws)
    lams = np.array([1e-3, 1e-2, 1e-1])
    RL = np.zeros((len(lams), len(Ps), 50))
    for s in range(50):
        Ws, Bs = features(1000 + s)
        for i, P in enumerate(Ps):
            for k, lam in enumerate(lams):
                RL[k, i, s] = np.mean((phi(xs, Ws, Bs, P) @ fit(x, y, Ws, Bs, P, lam) - fs) ** 2)
    np.savez(CACHE / "hero.npz", x=x, y=y, xs=xs, fs=fs, Ps=Ps, fits=fits.astype(np.float32), norms=norms,
             rep=rep, risk_shown=risk_shown, smin_shown=smin_shown, SM=SM, R=R, lams=lams, RL=RL)
    med = np.median(R, 1)
    for P in [5, 10, 12, 15, 18, 19, 20, 21, 22, 25, 40, 100, 1000, 5000]:
        i = np.searchsorted(Ps, P)
        print(f"P={P:5d} median risk {med[i]:.4g}  shown {risk_shown[i]:.4g}  ||a|| {norms[i]:.3g}")
