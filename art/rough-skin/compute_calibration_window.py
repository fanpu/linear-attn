"""CPU: estimator calibration, window protocol (primary; supersedes the periodic protocol of
compute_calibration.py, which is kept as a comparison).

Why: the net's cube is a 0.5 rad window of a field on all of S^3, so its level set carries large-scale trends
from wavelengths longer than the cube. A periodic box of exactly the grid size has no such trends; on the first
width-4096 field the local slopes matched H = 1/2 fields at b <= 8 but fell below them at b >= 16, where boxes
saturate. Here each power-law field lives on a periodic 1024^3 box, and the calibration volumes are
non-periodic central windows of it, subsampled to the target grid T with k octaves of sub-voxel roughness:

    config      crop   subsample   T    k   window / box
    T256_k0      256      1       256   0      1/4
    T256_k1      512      2       256   1      1/2
    T128_k0      128      1       128   0      1/8
    T128_k1      256      2       128   1      1/4
    T128_k2      512      4       128   2      1/2

Same estimators as compute_calibration.py (median level, 3D fit 2-64 / 2-32, 12 oblique slices).
Output cache/calibration3d_window.json (resumable per field).
"""
import json, os, sys, time
import numpy as np
import scipy.fft as sfft
from common import *


OUT = "cache/calibration3d_window.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
HS = [0.9, 0.75, 0.5, 0.375, 0.25, 0.125, 0.0625, 0.03125, "smooth"]
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 4
M = 1024
CONFIGS = [("T256_k0", 256, 1, 256), ("T256_k1", 512, 2, 256), ("T128_k0", 128, 1, 128),
           ("T128_k1", 256, 2, 128), ("T128_k2", 512, 4, 128)]


def make_field(H, seed):
    rng = np.random.default_rng(5000 + seed)
    w = rng.standard_normal((M, M, M), dtype=np.float32)
    F = sfft.rfftn(w, workers=4, overwrite_x=True)
    del w
    k1 = np.fft.fftfreq(M).astype(np.float32) ** 2
    k3 = np.fft.rfftfreq(M).astype(np.float32) ** 2
    for i in range(M):                      # slab-wise to keep memory ~ one extra 1024^2 x 513 array
        kk = np.sqrt(k1[i] + k1[:, None] + k3[None, :])
        if H == "smooth":
            ell = 48.0 * 4                  # 48 voxels of the 256-voxel grid, in 1024-box voxels
            amp = np.exp(-(2 * np.pi * kk * ell) ** 2 / 4)
        else:
            amp = np.where(kk > 0, np.maximum(kk, 1e-12) ** (-(1.5 + H)), 0.0)
        F[i] *= amp.astype(np.float32)
    F[0, 0, 0] = 0
    return sfft.irfftn(F, s=(M, M, M), workers=4, overwrite_x=True).real.astype(np.float32)


for H in HS:
    for seed in range(NSEED):
        if all(f"{c}_H{H}_s{seed}" in res for c, *_ in CONFIGS):
            continue
        t0 = time.time()
        f = make_field(H, seed)
        for name, crop, sub, T in CONFIGS:
            o = (M - crop) // 2
            g = np.ascontiguousarray(f[o:o + crop:sub, o:o + crop:sub, o:o + crop:sub])
            m = measure_field(g, T)
            T_, k_ = name.split("_")
            m.update(T=T, k=int(k_[1:]), crop=crop, sub=sub, H=H, D_true=(2.0 if H == "smooth" else 3.0 - H), seed=seed)
            res[f"{name}_H{H}_s{seed}"] = m
            print(f"{name} H={H} s{seed} true {m['D_true']:.4f} D3 {m['D3']:.3f} slice {np.mean(m['slice_1pD']):.3f}", flush=True)
        del f
        json.dump(res, open(OUT + ".tmp", "w"), indent=1); os.replace(OUT + ".tmp", OUT)
        print(f"  field H={H} s{seed} {time.time()-t0:.0f}s", flush=True)
