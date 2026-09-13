"""Multi-decade box counting of GP level sets via nested windows (zoom_engine).
For each kernel/seed: windows of side F_j = B_0/2 * 4^-j centred on a point of the level set
(level u = full-resolution field value at the zoom centre), 2048^2 samples, box counting
s = 1..1024 px. Band content of window j stops at wavelength F_j/256 (8 px), so fits use s >= 8.
Outputs cache/multiscale_<kernel>_s<seed>.npz (+ window fields for plates when --store)."""
import sys, time, json, numpy as np, torch
torch.cuda.set_per_process_memory_fraction(0.10)
from zoom_engine import *

kernel = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")]
store = "--store" in sys.argv
NBANDS = 0 if kernel == "rbf" else 10
CENTER = np.array([0.2, 0.6, 0.77])
N = 2048
for seed in seeds:
    t0 = time.time()
    M = MultiScaleField(kernel, seed, CENTER, nbands=NBANDS, verbose=False)
    B0 = 128 * 2 * np.pi / LS
    js = list(range(-2, 10))
    # level through the centre: evaluate the finest-resolution field at the origin
    f_c = M.eval(0.0, 0.0, B0 / 2 * 4.0 ** -9, 2, fade=False)
    u = float(f_c.mean())
    rows, fields = [], {}
    for j in js:
        F = B0 / 2 * 4.0 ** -j
        f = M.eval(0.0, 0.0, F, N, fade=False)
        s, c = boxcount(f, u)
        rows.append(dict(j=j, F=F, sizes=s, counts=c, frac=float((f > u).mean()),
                         std=float(f.std())))
        if store:
            fields[f"f{j}"] = f.astype(np.float32)
        print(kernel, seed, j, f"F={F:.3e}", "counts", c[:8], flush=True)
    np.savez(f"cache/multiscale_{kernel}_s{seed}.npz", u=u, F=np.array([r["F"] for r in rows]),
             j=np.array(js), sizes=rows[0]["sizes"], counts=np.stack([r["counts"] for r in rows]),
             frac=np.array([r["frac"] for r in rows]), **fields)
    print("done", kernel, seed, time.time() - t0, flush=True)
    del M
