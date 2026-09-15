"""CPU: calibrate the 3D boundary-voxel box-counting estimator (and the oblique-slice estimator) on
synthetic isotropic Gaussian fields of known level-set dimension.

A Gaussian field on R^3 with spectral density S(k) ~ k^-(3 + 2H), 0 < H < 1, has variogram ~ r^(2H) and
level sets of Hausdorff dimension 3 - H (Adler 1981; Falconer). Heaviside depth L has H = 2^-L (our
derivation, see NOTES.md), i.e. 3 - 2^-L.

Protocol (3D analogue of art/depth-roughness/compute_calibration.py):
  A periodic M^3 box with the power law up to its own Nyquist is subsampled to the target grid T (256 or 128),
  keeping k = log2(M/T) octaves of sub-voxel roughness (aliased into the samples, as for the nets):
     T = 256: k = 0 (M = 256), 1 (M = 512);   T = 128: k = 0, 1, 2 (M = 128, 256, 512).
  Matching rule (finite-width cutoff ~4/n rad, art/depth-roughness §4.4): width 4096 on 256^3 (h = 2.0e-3,
  4/n = 1.0e-3) -> k = 1; width 4096 on 128^3 (h = 3.9e-3) -> k = 2; width 1024 on 128^3 (4/n = 3.9e-3 = h) -> k = 0.
  The spread over k is reported as a systematic.
  level = median; 3D fit b = 2-64 (T = 256) and 2-32 (T = 128); 12 oblique slices by linear interpolation of the
  volume at its own spacing, 2D fit 2-32 / 2-16.
  smooth null: Gaussian spectrum exp(-(2 pi k l)^2 / 4) with l = 48/256 of the box side (D = 2).
Output cache/calibration3d.json (resumable per field).
"""
import json, os, sys, time
import numpy as np
import scipy.fft as sfft
from common import *

OUT = "cache/calibration3d.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
HS = [0.9, 0.75, 0.5, 0.375, 0.25, 0.125, 0.0625, 0.03125, "smooth"]
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CONFIGS = [(256, 256), (512, 256), (128, 128), (256, 128), (512, 128)]   # (M, T)
PLANES = slice_planes()


def make_field(M, H, seed):
    rng = np.random.default_rng(1000 + seed)
    w = rng.standard_normal((M, M, M), dtype=np.float32)
    F = sfft.rfftn(w, workers=4)
    del w
    k1 = np.fft.fftfreq(M).astype(np.float32)
    k3 = np.fft.rfftfreq(M).astype(np.float32)
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + k3[None, None, :] ** 2)
    if H == "smooth":
        ell = 48.0 * (M / 256)                     # 48/256 of the box side
        amp = np.exp(-(2 * np.pi * kk * ell) ** 2 / 4)
    else:
        amp = np.where(kk > 0, np.maximum(kk, 1e-12) ** (-(1.5 + H)), 0.0)
    amp[0, 0, 0] = 0.0
    F *= amp.astype(np.float32)
    del amp, kk
    return sfft.irfftn(F, s=(M, M, M), workers=4).astype(np.float32)


def measure(f, T):
    stride = 256 // T
    fit3 = (2, 64) if T == 256 else (2, 32)
    lvl = median_level(f)
    c = boxcount_mask(boundary_mask(f, lvl), SIZES3)
    D3 = fit_slope(SIZES3, c, *fit3)[0]
    d2 = []
    for pl in PLANES:
        P = slice_points(pl, SLICE_NPX[stride], stride * 2.0 / 256)
        s = sample_trilinear(f, P, stride=stride)
        d2.append(dim2(s, level=lvl, fit=FIT2_BY_STRIDE[stride])[0])
    return dict(D3=D3, counts=c.tolist(), slice_1pD=(1 + np.array(d2)).tolist())


for M, T in CONFIGS:
    for H in HS:
        for seed in range(NSEED):
            key = f"M{M}_T{T}_H{H}_s{seed}"
            if key in res:
                continue
            t0 = time.time()
            st = M // T
            f = make_field(M, H, seed)[::st, ::st, ::st].copy()
            m = measure(f, T)
            m.update(M=M, T=T, k=int(np.log2(st)), H=H, D_true=(2.0 if H == "smooth" else 3.0 - H), seed=seed,
                     seconds=time.time() - t0)
            res[key] = m
            json.dump(res, open(OUT + ".tmp", "w"), indent=1); os.replace(OUT + ".tmp", OUT)
            print(key, f"true {m['D_true']:.4f}  D3 {m['D3']:.3f}  slice {np.mean(m['slice_1pD']):.3f}  {m['seconds']:.0f}s", flush=True)
