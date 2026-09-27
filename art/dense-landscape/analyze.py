"""Noise-vs-landscape statistics (CPU).  All in log10(loss), the rendered height.

Writes cache/noise_stats.json and gallery/study_noise.png.
"""
import json, os
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from dl_common import CACHE, GALLERY, grid_axis

G = os.path.join(CACHE, "grid")
ld = lambda n: np.load(os.path.join(G, n + ".npy")).astype(np.float64)
lg = lambda a: np.log10(a)

A256 = lg(ld("A_n1024_r256_loss"))
Arows = lg(ld("A_n1024_r1024_rows508-516_loss"))
B1 = lg(ld("B_s1_r1024_loss")); B2 = lg(ld("B_s2_r1024_loss"))
C = lg(ld("C_r1024_loss"))
D256 = lg(ld("D_s1_r256_loss"))
A16 = lg(ld("A16_n1024_r128_loss"))
ax1024, ax256, ax128 = grid_axis(1024), grid_axis(256), grid_axis(128)


def up(F, ax_src, ax_dst=ax1024, rows=None):
    s = RectBivariateSpline(ax_src, ax_src, F, kx=3, ky=3)
    ry = ax_dst if rows is None else ax_dst[rows]
    return s(ry, ax_dst)                        # (rows, cols): first arg = y (beta, rows)


def hp(X, sigma=4.0):
    return X - ndimage.gaussian_filter(X, sigma, mode="nearest")


def acf(X, maxlag=12):
    X = X - X.mean()
    v = (X * X).mean()
    return [float((X[:, :-k] * X[:, k:]).mean() / v) if k else 1.0 for k in range(maxlag + 1)], \
           [float((X[:-k] * X[k:]).mean() / v) if k else 1.0 for k in range(maxlag + 1)]


def corr_len(r):
    """lag (px, linear interp) where the ACF first drops below 1/e."""
    r = np.array(r)
    k = np.argmax(r < 1 / np.e)
    return float(k - 1 + (r[k - 1] - 1 / np.e) / (r[k - 1] - r[k]))


S = {}
# 1. smooth truth: is A smooth at 1024 spacing? interpolation 256->1024 vs exact rows
Aup_rows = up(A256, ax256, rows=np.arange(508, 516))
e = np.abs(Aup_rows - Arows)
S["A_interp_256_to_1024"] = dict(median_abs_log10=float(np.median(e)), max_abs_log10=float(e.max()),
                                 range_log10=float(A256.max() - A256.min()))
d2A = np.diff(Arows, 2, axis=1)
S["A_exact_rows_rms_second_diff_log10"] = float(np.sqrt((d2A ** 2).mean()))
S["A_centre_loss"] = float(10 ** A256[127:129, 127:129].mean())
S["A_min_loss"] = float(10 ** A256.min()); S["A_max_loss"] = float(10 ** A256.max())

# 2. two independent Ideami-style measurements of the same plane
dB = B1 - B2
var_tot = float(B1.var()); var_noise = float(dB.var() / 2)
S["B_total_var_log10"] = var_tot
S["B_noise_var_log10_from_B1_minus_B2"] = var_noise
S["B_noise_std_log10"] = float(np.sqrt(var_noise))
S["B_noise_fraction_of_total_var"] = var_noise / var_tot
S["B1_B2_corr_raw"] = float(np.corrcoef(B1.ravel(), B2.ravel())[0, 1])
for sg in (1.0, 2.0, 4.0, 8.0):
    h1, h2 = hp(B1, sg), hp(B2, sg)
    c = float(np.corrcoef(h1.ravel(), h2.ravel())[0, 1])
    S[f"B1_B2_corr_highpass_sigma{sg:g}px"] = c
    S[f"B_highpass_sigma{sg:g}_noise_fraction"] = float(1 - max(c, 0))
S["B_rms_second_diff_log10"] = float(np.sqrt((np.diff(B1, 2, axis=1) ** 2).mean()))
S["B_over_A_second_diff_ratio"] = S["B_rms_second_diff_log10"] / S["A_exact_rows_rms_second_diff_log10"]

# 3. residual against the measured mean surface E[B] (A16, 128^2, bicubic to 1024^2)
EB = up(A16, ax128)
r = B1 - EB
S["B_minus_EB_mean_log10"] = float(r.mean())   # Jensen offset: E[log] < log E
r0 = r - r.mean()
S["B_residual_var_fraction"] = float(r0.var() / B1.var())
ax_, ay_ = acf(r0); S["B_residual_acf_x"] = ax_; S["B_residual_acf_y"] = ay_
S["B_residual_corr_len_px"] = corr_len(ax_)
S["B_residual_corr_len_alpha"] = S["B_residual_corr_len_px"] * (2 / 1023)
bx, by = acf(dB); S["B1_minus_B2_acf_x_lag1"] = bx[1]; S["B1_minus_B2_acf_y_lag1"] = by[1]
S["EB_range_log10"] = float(A16.max() - A16.min())
S["signal_to_noise_std_ratio"] = float(EB.std() / np.sqrt(var_noise))
# noise vs level
bins = np.quantile(EB, np.linspace(0, 1, 9))
S["noise_std_by_EB_level"] = [dict(EB_log10_lo=float(bins[i]), EB_log10_hi=float(bins[i + 1]),
                                   std=float(dB[(EB >= bins[i]) & (EB <= bins[i + 1])].std() / np.sqrt(2)))
                              for i in range(8)]

# 4. fixed minibatch (C) and BN mode split
S["C_rms_second_diff_log10"] = float(np.sqrt((np.diff(C, 2, axis=1) ** 2).mean()))
S["C_minus_EB_std_log10"] = float((C - EB).std())
A1024 = up(A256, ax256)
S["EB_minus_A_mean_log10"] = float((EB - A1024).mean()); S["EB_minus_A_std_log10"] = float((EB - A1024).std())
# 5. sampling noise without BN-train (D, fresh 16, BN eval) vs A at 256^2 on the same grid
S["D_noise_std_log10"] = float((D256 - A256).std())
B1_256 = B1[np.round(np.linspace(0, 1023, 256)).astype(int)][:, np.round(np.linspace(0, 1023, 256)).astype(int)]
S["B_noise_std_log10_check"] = S["B_noise_std_log10"]
print(json.dumps(S, indent=1))
json.dump(S, open(os.path.join(CACHE, "noise_stats.json"), "w"), indent=1)

# ------------------------------------------------------------- study figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axs = plt.subplots(2, 4, figsize=(16, 8.4), facecolor="#0b0b0c")
vm = (min(A1024.min(), B1.min()), max(A1024.max(), B1.max()))
panels = [(A1024, "A  fixed 1024 imgs, BN eval\n(256² measured, bicubic to 1024²)", vm, "magma"),
          (EB, "E[B]  64 fixed minibatches of 16,\nBN train (128² measured, bicubic)", vm, "magma"),
          (B1, "B  fresh 16 per point, BN train\n(seed 1, 1024² measured)", vm, "magma"),
          (B2, "B′  same plane, seed 2", vm, "magma"),
          (B1 - EB - r.mean(), "B − E[B]  (residual)", (-3 * np.sqrt(var_noise), 3 * np.sqrt(var_noise)), "RdBu_r"),
          (hp(B1, 2) , "high-pass B (σ=2 px)", (-3 * np.sqrt(var_noise), 3 * np.sqrt(var_noise)), "RdBu_r"),
          (hp(B2, 2), "high-pass B′ (σ=2 px)", (-3 * np.sqrt(var_noise), 3 * np.sqrt(var_noise)), "RdBu_r")]
for a_, (X, t, v, cm) in zip(axs.flat, panels):
    a_.imshow(X[384:640, 384:640] if "high" in t or "resid" in t else X, origin="lower", vmin=v[0], vmax=v[1], cmap=cm)
    a_.set_title(t + ("  [centre 256²]" if ("high" in t or "resid" in t) else ""), color="w", fontsize=8.5)
    a_.axis("off")
a_ = axs.flat[7]; a_.set_facecolor("#0b0b0c")
a_.plot(ax_, "o-", color="#f0a040", label="ACF of B − E[B] (α lag)")
a_.plot(bx, "s--", color="#60b0ff", label="ACF of B − B′")
a_.axhline(1 / np.e, color="#888", lw=0.6); a_.set_xlabel("lag (grid cells, 0.00196 α)", color="w")
a_.tick_params(colors="w"); a_.legend(fontsize=8, facecolor="#222", labelcolor="w")
a_.set_title(f"noise corr. length {S['B_residual_corr_len_px']:.2f} px;\ncorr(hp B, hp B′) σ=2: {S['B1_B2_corr_highpass_sigma2px']:.3f}", color="w", fontsize=8.5)
for s_ in a_.spines.values(): s_.set_color("#666")
fig.suptitle("Is the texture landscape or sampling noise?  log10 loss on one filter-normalised plane, α,β∈[−1,1]",
             color="w", fontsize=11)
plt.tight_layout(); os.makedirs(GALLERY, exist_ok=True)
plt.savefig(os.path.join(GALLERY, "study_noise.png"), dpi=110, facecolor=fig.get_facecolor())
