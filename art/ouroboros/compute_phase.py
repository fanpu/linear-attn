"""Phase map over (lambda = real fraction, n = dataset size), per generation metric histories.

usage: python compute_phase.py <target> <model> <regime> --L 96 --N 64 --nmin 16 --nmax 1024 --G 60
       [--lam0 0 --lam1 1] [--seeds 0] [--bs 96] [--K 8] [--tag ""]
-> cache/phase_<target>_<model>_<regime><tag>.npz with arrays [S, L, Nn, G+1] (float32)
Resumable: finished columns are stored in cache/parts/ and skipped on rerun.
"""
import argparse, os, time
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
from chains import run_batch

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("model"); ap.add_argument("regime")
ap.add_argument("--L", type=int, default=96); ap.add_argument("--N", type=int, default=64)
ap.add_argument("--nmin", type=int, default=16); ap.add_argument("--nmax", type=int, default=1024)
ap.add_argument("--lam0", type=float, default=0.0); ap.add_argument("--lam1", type=float, default=1.0)
ap.add_argument("--G", type=int, default=60); ap.add_argument("--seeds", default="0")
ap.add_argument("--bs", type=int, default=96); ap.add_argument("--K", type=int, default=8)
ap.add_argument("--tag", default="")
a = ap.parse_args()

lams = np.linspace(a.lam0, a.lam1, a.L)
# n is an integer: rounded log-spaced columns are kept with repeats (repeated n = identical column, computed once),
# so grids with 2k+1 and k+1 columns nest exactly.
ns = np.round(np.geomspace(a.nmin, a.nmax, a.N)).astype(int)
seeds = [int(s) for s in a.seeds.split(",")]
name = f"phase_{a.target}_{a.model}_{a.regime}{a.tag}"
os.makedirs("cache/parts", exist_ok=True)
keys = ["sw2", "var_ratio", "overlap"] + (["modes"] if a.target == "ring" else []) + (["h"] if a.model == "kde" else [])
t0 = time.time()
cols = []
for j, n in enumerate(np.unique(ns)[::-1]):  # biggest first: fail fast on memory
    part = f"cache/parts/{name.replace(a.tag, '')}_L{a.L}_{a.lam0:g}-{a.lam1:g}_G{a.G}_s{a.seeds}_n{n}.npz"
    if os.path.exists(part):
        cols.append((n, dict(np.load(part)))); continue
    grid = [(s, l) for s in seeds for l in lams]
    acc = {k: [] for k in keys}
    for i in range(0, len(grid), a.bs):
        chunk = grid[i:i + a.bs]
        r = run_batch(a.target, a.model, a.regime, [c[1] for c in chunk], int(n), [c[0] for c in chunk], a.G, K=a.K)
        for k in keys:
            acc[k].append(r[k].astype(np.float32))
    col = {k: np.concatenate(v, 0).reshape(len(seeds), len(lams), a.G + 1) for k, v in acc.items()}
    np.savez(part, **col)
    cols.append((n, col))
    print(f"[{time.time()-t0:7.0f}s] {name} n={n} ({j+1}/{len(ns)}) sw2[lam=0,G]={col['sw2'][0,0,-1]:.3f}", flush=True)
bycol = dict(cols)
out = {k: np.stack([bycol[n][k] for n in ns], 2) for k in keys}  # [S, L, Nn, G+1]
np.savez_compressed(f"cache/{name}.npz", lams=lams, ns=ns, seeds=np.array(seeds), **out)
print("wrote", name, f"{time.time()-t0:.0f}s")
