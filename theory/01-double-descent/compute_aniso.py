"""Where does the peak go under anisotropy + misspecification?  CPU.  -> cache/aniso.npz

Ambient covariates x in R^D with Sigma = diag(e_j), e_j ∝ j^-alpha (trace normalised to D), isotropic target beta ~ N(0, I/D),
noise sigma^2. The model only sees the first p covariates (largest variance first), so everything beyond p acts as noise
(Hastie et al. 2022, Sec. 5, "misspecified model"). Theory: tail + deterministic equivalent on the first p coordinates
with effective noise sigma^2 + tail.
"""
import os, pathlib, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool

import dd_core as C

HERE = pathlib.Path(__file__).resolve().parent
D, N, SIG2 = 2000, 200, 0.05
ALPHAS = [0.0, 0.5, 1.0, 1.5]
LAMS = [0.0, 1e-3, 1e-2]
PS = np.unique(np.concatenate([np.round(np.geomspace(5, D, 48)).astype(int), N + np.array([-20, -10, -5, -2, 2, 5, 10, 20, 40])]))
PS = PS[PS != N]


def spectrum(alpha):
    e = np.arange(1, D + 1, dtype=float) ** (-alpha)
    return e * D / e.sum()


def theory(alpha, lam):
    e = spectrum(alpha)
    b2 = np.full(D, 1.0 / D)
    out = []
    for p in PS:
        tail = np.sum(e[p:] * b2[p:])
        r = C.general_ridge_risk(e[:p], b2[:p], p / N, lam, sigma2=SIG2 + tail)
        out.append(tail + r[0])
    return np.array(out)


def job(args):
    alpha, seed = args
    rng = np.random.default_rng([int(alpha * 10), seed])
    e = spectrum(alpha); se = np.sqrt(e)
    beta = rng.standard_normal(D) / np.sqrt(D)
    X = rng.standard_normal((N, D)) * se
    y = X @ beta + np.sqrt(SIG2) * rng.standard_normal(N)
    res = np.zeros((len(PS), len(LAMS)))
    for i, p in enumerate(PS):
        Xp = X[:, :p]
        U, S, Vt = np.linalg.svd(Xp / np.sqrt(N), full_matrices=False)
        uy = U.T @ y / np.sqrt(N)
        tail = np.sum(e[p:] * beta[p:] ** 2)
        for k, lam in enumerate(LAMS):
            if lam == 0:
                keep = S > S.max() * 1e-12
                coef = np.where(keep, uy / np.where(keep, S, 1), 0)
            else:
                coef = S * uy / (S ** 2 + lam)
            err = Vt.T @ coef - beta[:p]
            res[i, k] = tail + err @ (e[:p] * err)
    return res


if __name__ == "__main__":
    t0 = time.time()
    seeds = 30
    th = np.array([[theory(a, l) for l in LAMS] for a in ALPHAS])        # [alpha, lam, p]
    with Pool(4) as pool:
        sims = pool.map(job, [(a, s) for a in ALPHAS for s in range(seeds)])
    sim = np.array(sims).reshape(len(ALPHAS), seeds, len(PS), len(LAMS))
    np.savez(HERE / "cache" / "aniso.npz", PS=PS, alphas=ALPHAS, lams=LAMS, th=th, sim=sim, N=N, D=D, sig2=SIG2)
    print(f"done {time.time() - t0:.0f}s")
    for ia, a in enumerate(ALPHAS):
        for il, l in enumerate(LAMS):
            m = sim[ia, :, :, il].mean(0)
            j = int(np.argmax(np.where(np.isfinite(th[ia, il]), th[ia, il], -1)))
            js = int(np.argmax(m))
            far = np.abs(PS / N - 1) > 0.2
            rel = np.median(np.abs(m[far] - th[ia, il][far]) / th[ia, il][far])
            print(f"alpha={a} lam={l}: theory peak p={PS[j]} ({th[ia, il][j]:.3g}), sim peak p={PS[js]} ({m[js]:.3g}); "
                  f"min theory {th[ia, il].min():.3g} at p={PS[np.argmin(th[ia, il])]}; median rel err {rel:.3f}")
