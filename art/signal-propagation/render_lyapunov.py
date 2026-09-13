"""Extra idea 1 renders: the finite-time Lyapunov exponent lambda(sigma_w, sigma_b) of one fixed random
erf network (N = 100, depth 1000) - the continuous signed field under the binary frontier.

  python render_lyapunov.py
"""
import os
import numpy as np
from render_common import *

Z = np.load(os.path.join(CACHE, "lyap_full_N100_s0_D1000_f32_r1024.npz"))
lam = Z["lam"]
W0 = np.load(os.path.join(CACHE, "widthmap_mf_D1000_r512.npz"))
ext = [0, 4, 0, 4]
N = int(Z["N"])


def plate(rgb, name, title, cap, bg, fg):
    W, H = 2600, 3000
    fig = fig_px(W, H, bg=bg)
    ax = fig.add_axes([180 / W, 1 - (260 + 2240) / H, 2240 / W, 2240 / H])
    ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
    for s in ax.spines.values():
        s.set_color(fg)
    ax.tick_params(colors=fg, labelsize=12)
    ax.set_xlabel(r"$\sigma_w$", color=fg, fontsize=16); ax.set_ylabel(r"$\sigma_b$", color=fg, fontsize=16)
    fig.text(180 / W, 1 - 110 / H, title, color=fg, fontsize=28, va="center")
    fig.text(180 / W, 1 - 2640 / H, cap, color=fg, fontsize=12.5, va="top", linespacing=1.6)
    savefig(fig, name)


cap0 = (f"Finite-time maximal Lyapunov exponent of one random erf MLP (width N = {N}, layers 201-1000,\ntangent vector renormalised every layer), "
        "1024 x 1024 pixels, float32. lambda < 0: nearby inputs merge (order); lambda > 0: they separate (chaos). "
        f"\nMeasured fraction with lambda > 0: {np.mean(lam > 0):.3f}; range {lam.min():.2f} to {lam.max():.2f} per layer.\n")
plate(split_rgb(lam > 0, np.abs(lam), "sd_spectral", near_boundary="small"), "lyapunov_spectral.png",
      "The exponent under the frontier", cap0 + "Declared Spectral split at lambda = 0, rank-normalised per side (dark seam = lambda = 0).", "#111014", "#e9e4da")
plate(split_rgb(lam > 0, np.abs(lam), "cyanotype_vandyke", near_boundary="small"), "lyapunov_cyanotype_vandyke.png",
      "The exponent under the frontier (cyanotype / Van Dyke)", cap0 + "Declared split palette 'cyanotype_vandyke', rank-normalised per side.", PAPER, INK)
vmax = np.percentile(np.abs(lam), 99)
rgb = plt.get_cmap("cmc.vik")(np.clip(lam / vmax / 2 + 0.5, 0, 1))[..., :3]
plate(rgb, "lyapunov_vik_linear.png", "The exponent under the frontier (linear diverging)",
      cap0 + f"Linear diverging map (Crameri vik) centred at 0, clipped at +/-{vmax:.2f} (99th percentile of |lambda|).", "#0b0c10", "#e9e4da")
print("ok", np.mean(lam > 0))
