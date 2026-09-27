"""Checks on the boiling-film measurement (cache/boil): are the frames independent draws of the same
sampling distribution, and does their average converge to the smooth mean surface?

    python boil_stats.py [--n 96]   -> cache/boil_stats.json
"""
import argparse, json, os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from dl_common import CACHE, grid_axis

p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=96); a = p.parse_args()
bd = os.path.join(CACHE, "boil")
n = min(a.n, len([f for f in os.listdir(bd) if f.endswith("_acc.npy")]))
L = np.stack([np.load(os.path.join(bd, f"B_r256s_f{k:03d}_loss.npy")).astype(np.float64) for k in range(n)])
Z = np.log10(L)                                              # (n, 256, 256)
ax = grid_axis(1024)[::4][:256]
EB = RectBivariateSpline(grid_axis(128), grid_axis(128), np.log10(np.load(os.path.join(CACHE, "grid", "A16_n1024_r128_loss.npy"))))(ax, ax)
B1 = np.log10(np.load(os.path.join(CACHE, "grid", "B_s1_r1024_loss.npy")).astype(np.float64))[::4, ::4][:256, :256]
C = np.log10(np.load(os.path.join(CACHE, "grid", "C_r1024_loss.npy")).astype(np.float64))[::4, ::4][:256, :256]


def hp(x, s=1.0):
    return x - gaussian_filter(x, s)


def corr(x, y):
    x = x.ravel() - x.mean(); y = y.ravel() - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


hpc = [corr(hp(Z[k]), hp(Z[k + 1])) for k in range(n - 1)]
sd_pt = Z.std(0, ddof=1)
out = dict(frames=n, grid="256^2 subgrid of the 1024^2 plane (every 4th point)",
           frame_to_frame_highpass_corr_mean=float(np.mean(hpc)), frame_to_frame_highpass_corr_max_abs=float(np.max(np.abs(hpc))),
           frame0_vs_B1sub_highpass_corr=corr(hp(Z[0]), hp(B1)),
           per_point_sd_across_frames_log10_median=float(np.median(sd_pt)),
           per_point_sd_across_frames_log10_mean=float(sd_pt.mean()),
           raw_corr_frame0_frame1=corr(Z[0], Z[1]))
for m in (1, 4, 16, n):
    Mz = np.log10(L[:m].mean(0))                                  # log of the Monte-Carlo mean loss
    out[f"mean_of_{m}_vs_EB_rmse_log10"] = float(np.sqrt(((Mz - EB) ** 2).mean()))
    out[f"mean_of_{m}_vs_EB_corr"] = corr(Mz, EB)
    out[f"mean_of_{m}_rms_second_diff_log10"] = float(np.sqrt((np.diff(Mz, 2, axis=1) ** 2).mean()))
out["C_sub_rms_second_diff_log10"] = float(np.sqrt((np.diff(C, 2, axis=1) ** 2).mean()))
json.dump(out, open(os.path.join(CACHE, "boil_stats.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
