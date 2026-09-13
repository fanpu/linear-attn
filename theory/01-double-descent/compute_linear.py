"""Isotropic linear-model sweeps (CPU, float64). Writes cache/linear_*.npz.

    python compute_linear.py main     # n=400, gamma 0.1..10, 50 seeds, many ridge values
    python compute_linear.py sizes    # n in {50, 200, 800}: error bars shrink onto the curve
"""
import sys, os, pathlib, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool

import dd_core as C

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)

FIXED_LAMS = np.array([0.0, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0])
SIG2 = np.array([1.0, 0.5, 0.25, 0.1, 0.04])          # r^2 = 1  ->  SNR = 1, 2, 4, 10, 25


def lams_for(gamma):
    return np.concatenate([FIXED_LAMS, C.optimal_lambda(gamma, 1.0, SIG2)])


def p_grid(n, m, lo=0.1, hi=10.0):
    ps = set(np.round(np.geomspace(lo, hi, m) * n).astype(int))
    for d in [1, 2, 3, 5, 8, 12, 20, 30]:          # densify around the threshold (p = n excluded: E risk = inf)
        if n - d > 0:
            ps.add(n - d)
        ps.add(n + d)
    ps.discard(n)
    return np.array(sorted(p for p in ps if p >= 1))


def job(args):
    n, p, seed = args
    out, lmin = C.simulate_linear_parts(n, p, lams_for, rng=np.random.SeedSequence([n, p, seed]))
    return out, lmin


def run(n, m, seeds, tag):
    ps = p_grid(n, m)
    t0 = time.time()
    tasks = [(n, int(p), s) for p in ps for s in range(seeds)]
    with Pool(4) as pool:
        res = pool.map(job, tasks, chunksize=4)
    parts = np.array([r[0] for r in res]).reshape(len(ps), seeds, -1, 4)
    lmin = np.array([r[1] for r in res]).reshape(len(ps), seeds)
    np.savez(CACHE / f"linear_{tag}.npz", n=n, p=ps, gamma=ps / n, parts=parts, lmin=lmin,
             fixed_lams=FIXED_LAMS, sig2=SIG2)
    print(f"{tag}: n={n} {len(ps)} p values x {seeds} seeds in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "main"
    if what == "main":
        run(400, 64, 50, "n400")
    elif what == "sizes":
        for n in [50, 200, 800]:
            run(n, 28, 50, f"n{n}")
