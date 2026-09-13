"""1D transects across the Zhu et al. basin boundary (eta = 0.2), for a many-decade dimension estimate.

At high zoom the converge/diverge boundary of the degree-4 model is a stack of nearly parallel stripes:
locally (Cantor set) x (smooth curve).  A vertical line x = x0 crosses the stripes transversally, so the
set of label changes along it is the Cantor factor.  We sample 2^22 points on nested vertical segments
(lengths 0.4, 0.4/100, 0.4/10^4, centred on the same point) and box-count the label-change set in 1D.
If the product picture holds, D_2D = 1 + D_1D.  Output: cache/transect_<level>.npz
"""
import sys
import time
import numpy as np
import torch
from compute_basins import run

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"


def main():
    x0, yc = 3.296875, 0.309375
    n = 1 << 22
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.10)
    for lvl, L in enumerate([0.4, 0.004, 0.00004]):
        t0 = time.time()
        y0 = yc - L / 2 if lvl > 0 else 0.1
        ys = y0 + L * (np.arange(n) + 0.5) / n
        o = run("zhu4", 0.2, np.array([x0]), ys, 10000, 1e-12, dev=dev)
        np.savez_compressed(f"{CACHE}/transect_{lvl}.npz", ys=ys, x0=x0, L=L, status=o["status"][:, 0], tev=o["tev"][:, 0],
                            nu=o["nu"][:, 0])
        print(f"level {lvl} L={L} done {time.time() - t0:.0f}s div frac {(o['status'] == 2).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
