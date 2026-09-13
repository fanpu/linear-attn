"""Measurements for verify.py (CPU). Three jobs, each cached separately; rerun skips finished jobs.

usage: OMP_NUM_THREADS=2 OUROBOROS_DT=float32 python verify_compute.py [native|seeds|floor|all]

native: ring/GMM replace phase map at NATIVE resolution in a window around the collapse boundary:
        every integer n in [32, 96] and every distinct real count n_r = floor(lambda n) with lambda in
        [0.03, 0.35] (lambda = n_r / n exactly), seeds 0 and 1, G = 80, same settings as the phase map.
        Rows with equal n_r are the same chain (CRN), so this is the finest map that exists.
seeds:  film settings (ring GMM n = 128, G = 200) over seeds 0-4: replace (lambda 0), anchored (lambda 0.25),
        accumulate (lambda 0).
floor:  sliced-W2 / modes / var_ratio of TRUE samples (2048) vs the evaluator reference, 64 draws.
"""
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
from chains import Evaluator, N_EVAL, run_batch  # noqa: E402
from common import DEV, DT, sample_ring  # noqa: E402

job = sys.argv[1] if len(sys.argv) > 1 else "all"
t0 = time.time()
log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)  # noqa: E731

if job in ("floor", "all") and not os.path.exists("cache/verify_floor.npz"):
    ev = Evaluator("ring")
    R = sample_ring(200_000, np.random.default_rng(999))  # independent of the evaluator reference stream
    rng = np.random.default_rng(123)
    Y = torch.tensor(np.stack([R[rng.choice(len(R), N_EVAL, replace=False)] for _ in range(64)]), dtype=DT, device=DEV)
    m = ev(Y)
    np.savez("cache/verify_floor.npz", **{k: v.cpu().numpy() for k, v in m.items()})
    log(f"floor sw2 {m['sw2'].mean():.4f} +- {m['sw2'].std():.4f}")

if job in ("seeds", "all") and not os.path.exists("cache/verify_seeds.npz"):
    out = {}
    seeds = [0, 1, 2, 3, 4]
    for name, regime, lam in [("replace", "replace", 0.0), ("anchored", "replace", 0.25), ("accumulate", "accumulate", 0.0)]:
        r = run_batch("ring", "gmm", regime, [lam] * 5, 128, seeds, 200, K=8)
        for k in ("sw2", "modes", "var_ratio", "overlap"):
            out[f"{name}/{k}"] = r[k]
        log(f"seeds {name} sw2(G) {r['sw2'][:, -1].round(3)} modes(G) {r['modes'][:, -1]}")
    np.savez("cache/verify_seeds.npz", **out)

if job in ("native", "all") and not os.path.exists("cache/verify_native.npz"):
    rows = []
    order = range(96, 31, -1) if "--rev" in sys.argv else range(32, 97)  # --rev: 2nd worker from the top
    for n in order:
        part = f"cache/parts/verify_native_n{n}.npz"
        if os.path.exists(part):
            rows.append(dict(np.load(part))); continue
        nr = np.arange(int(np.floor(0.03 * n)), int(np.ceil(0.35 * n)) + 1)
        lam = list((nr / n).astype(float)) * 2
        sd = [0] * len(nr) + [1] * len(nr)
        r = run_batch("ring", "gmm", "replace", lam, n, sd, 80, K=8, em_iters=10)
        d = dict(n=np.full(len(lam), n), nr=np.concatenate([nr, nr]), seed=np.array(sd),
                 sw2=r["sw2"].astype(np.float32))
        np.savez(part, **d); rows.append(d)
        log(f"native n={n} rows={len(nr)}")
    rows.sort(key=lambda r: int(r["n"][0]))
    np.savez_compressed("cache/verify_native.npz", **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
    log("wrote verify_native")
