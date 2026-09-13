"""Long single-seed chains with display samples every generation (film, spiral, fern, ridgeline).

usage: python compute_film.py <target> <model> <n> <G> [--K 8] [--nd 4000] [--seed 0] [--every 1]
Runs three regimes side by side: replace lambda=0, replace lambda=LAM_ANCHOR, accumulate lambda=0.
-> cache/film_<target>_<model>_n<n>.npz
"""
import argparse, time, os
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
from chains import run_batch

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("model"); ap.add_argument("n", type=int); ap.add_argument("G", type=int)
ap.add_argument("--K", type=int, default=8); ap.add_argument("--nd", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--every", type=int, default=1)
ap.add_argument("--anchor", type=float, default=0.25); ap.add_argument("--tag", default="")
ap.add_argument("--regimes", default="replace,anchored,accumulate")
a = ap.parse_args()
t0 = time.time()
out = {}
ap2 = [("replace", "replace", 0.0), ("anchored", "replace", a.anchor), ("accumulate", "accumulate", 0.0)]
for name, regime, lam in [r for r in ap2 if r[0] in a.regimes.split(",")]:
    r = run_batch(a.target, a.model, regime, [lam], a.n, [a.seed], a.G, K=a.K, keep_samples_every=a.every,
                  n_display=a.nd, log=lambda s: print(f"[{time.time()-t0:6.0f}s] {name} {s}", flush=True))
    for k, v in r.items():
        v = v[0]
        out[f"{name}/{k}"] = v.astype(np.float16) if k == "display" else v
out["meta"] = np.array([a.target, a.model, a.n, a.G, a.K, a.seed, a.every, a.anchor], dtype=object)
fn = f"cache/film_{a.target}_{a.model}_n{a.n}{a.tag}.npz"
np.savez_compressed(fn, **{k: v for k, v in out.items() if k != "meta"}, meta=str(list(out["meta"])))
print("wrote", fn, f"{time.time()-t0:.0f}s")
