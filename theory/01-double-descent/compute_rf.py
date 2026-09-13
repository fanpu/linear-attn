"""Random-features regression (ReLU, sphere data, linear target + noise) vs Mei & Montanari asymptotics. CPU.

    python compute_rf.py        -> cache/rf.npz
"""
import os, pathlib, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
from scipy import optimize

import dd_core as C

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)

PSI2 = 3.0
TAU2 = 0.25             # F1^2 = 1  -> SNR rho = 4
LAMS = np.array([0.0, 1e-3, 1e-2, 1e-1])


def opt_lam(psi1):
    f = lambda ll: C.mm_risk(psi1, PSI2, np.exp(ll), 1.0, TAU2)[0]
    r = optimize.minimize_scalar(f, bounds=(np.log(1e-7), np.log(10.0)), method="bounded", options=dict(xatol=1e-4))
    return float(np.exp(r.x))


def n_grid(d, m=30):
    n = int(PSI2 * d)
    Ns = set(np.round(np.geomspace(0.1, 30, m) * d).astype(int))
    for f in [0.8, 0.9, 0.95, 1.05, 1.1, 1.25]:
        Ns.add(int(round(f * n)))
    Ns.discard(n)
    return np.array(sorted(Ns))


def job(args):
    d, N, lams, seed = args
    return C.simulate_rf(d, int(PSI2 * d), N, lams, F1sq=1.0, tau2=TAU2, rng=np.random.SeedSequence([d, N, seed]))


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    # theory curves on a fine grid
    psi1 = np.geomspace(0.1, 30, 400)
    out["th_psi1"] = psi1
    out["th_lams"] = LAMS
    th = np.full((len(LAMS), len(psi1), 3), np.nan)
    for i, lam in enumerate(LAMS):
        for j, p1 in enumerate(psi1):
            if lam == 0 and abs(p1 - PSI2) < 1e-9:
                continue
            th[i, j] = C.mm_risk(p1, PSI2, lam, 1.0, TAU2)
    out["th"] = th
    lopt = np.array([opt_lam(p1) for p1 in psi1])
    out["th_lopt"] = lopt
    out["th_opt"] = np.array([C.mm_risk(p1, PSI2, l, 1.0, TAU2) for p1, l in zip(psi1, lopt)])
    print(f"theory done {time.time() - t0:.0f}s")

    reps = 20
    for d in [50, 100, 200]:
        Ns = n_grid(d)
        tasks = [(d, int(N), np.concatenate([LAMS, [opt_lam(N / d)]]), s) for N in Ns for s in range(reps)]
        with Pool(4) as pool:
            res = pool.map(job, tasks, chunksize=2)
        out[f"sim_d{d}_N"] = Ns
        out[f"sim_d{d}"] = np.array(res).reshape(len(Ns), reps, -1)
        print(f"d={d} done {time.time() - t0:.0f}s")
    out["psi2"] = PSI2; out["tau2"] = TAU2
    np.savez(CACHE / "rf.npz", **out)
