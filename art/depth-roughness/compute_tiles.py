"""Exact (band-limited) GP samples of the depth-L limiting fields on a Lambert equal-area patch of S^2,
for every activation and depth, with common random numbers (same white a_lm for all kernels).
Also: box counting at 2048^2 (lmax 4096) and the resolution check at 4096^2 (lmax 8192).
Outputs: cache/tiles_2048.npz, cache/boxcount_tiles.json"""
import numpy as np, json, time, os
from common import *

os.environ.setdefault("OMP_NUM_THREADS", "8")
SEED = 11
CENTER = np.array([0.3, -0.5, 0.8])
HW = 0.7               # Lambert half-width (patch spans ~ 82 deg edge to edge)
LS = list(range(1, 11))
sp = np.load("cache/spectra.npz")
names = list(sp["names"]); C = sp["C"]
l = np.arange(C.shape[1])

def spec(name):
    return C[names.index(name)]

def run(n, lmax, kernels, store):
    z = white_alm(lmax, SEED)
    v = lambert_patch(CENTER, HW, n)
    th, ph = vec_to_thetaphi(v)
    out, bc = {}, {}
    for key in kernels:
        c = spec(key)[: lmax + 1]
        t0 = time.time()
        f = synth(alm_from_white(z, c, lmax), lmax, th, ph, nthreads=16)
        cap = float(((2 * l[: lmax + 1] + 1) * c).sum() / (4 * np.pi))
        const = float(c[0] / (4 * np.pi))
        med = float(np.median(f))
        s, cnt = boxcount(f, med)
        s0, cnt0 = boxcount(f, 0.0)
        bc[key] = dict(sizes=s.tolist(), counts=cnt.tolist(), counts_zero=cnt0.tolist(), median=med,
                       var_captured=cap, var_const=const, std=float(f.std()))
        if store:
            out[key] = f.astype(np.float32)
        print(n, key, f"{time.time()-t0:.1f}s cap={cap:.3f} const={const:.3f}", flush=True)
    return out, bc

kern = [f"{a}_L{L}" for a in ACTS for L in LS] + ["rbf"]
fields, bc2048 = run(2048, 4096, kern, True)
rng = np.random.default_rng(SEED)
np.savez("cache/tiles_2048.npz", **fields, tail_noise=rng.standard_normal((2048, 2048)).astype(np.float32),
         center=CENTER, hw=HW, seed=SEED)
_, bc4096 = run(4096, 8192, [f"heaviside_L{L}" for L in range(1, 7)] + ["relu_L3", "tanh_L6", "rbf"], False)
json.dump(dict(n2048=bc2048, n4096=bc4096, center=CENTER.tolist(), hw=HW, seed=SEED),
          open("cache/boxcount_tiles.json", "w"))
