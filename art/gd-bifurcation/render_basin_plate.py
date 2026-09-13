"""Verification plate for the Zhu et al. basin boundary (fractals doc Sec. 11), scientific-plate idiom.

Top: zoom sequence wide -> z1 -> z2 (measured labels; boxes mark the next zoom).
Bottom: box counting with nulls, resolution check (same sub-square at 1024..8192), label-flip scaling,
1-D transects (barcodes of converge/diverge along x = x0 on nested segments) with their box counting.
"""
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from render_lib import CACHE, GAL
from analyze_boxcount import boundary

INK, RED, BLUE, PAPER = "#1b1b22", "#b3261e", "#1f5f99", "#f3eee2"


def label_img(status, conv_rgb=(0.95, 0.93, 0.88), div_rgb=(0.11, 0.11, 0.14)):
    img = np.zeros(status.shape + (3,))
    img[status == 2] = div_rgb
    img[status != 2] = conv_rgb
    return img


def main():
    B = json.load(open(f"{CACHE}/boxcount.json"))
    fig = plt.figure(figsize=(18, 11.5), dpi=200, facecolor=PAPER)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.32, wspace=0.28, left=0.05, right=0.98, top=0.9, bottom=0.07)
    zooms = [("wideT", 4096, (0, 3.8, 0, 3.8)), ("z1", 8192, (3.0, 3.4, 0.1, 0.5)), ("z2", 8192, (3.29375, 3.3, 0.30625, 0.3125))]
    for i, (tag, res, ext) in enumerate(zooms):
        ax = fig.add_subplot(gs[0, i])
        d = np.load(f"{CACHE}/basin_zhu4_{tag}_{res}.npz")
        st = d["status"]
        step = max(1, st.shape[0] // 2048)
        ax.imshow(label_img(st[::step, ::step]), origin="lower", extent=ext, interpolation="nearest")
        if i < 2:
            nx = zooms[i + 1][2]
            ax.add_patch(Rectangle((nx[0], nx[2]), nx[1] - nx[0], nx[3] - nx[2], fill=False, ec=RED, lw=1.0))
        ax.set_title(f"x₀ ∈ [{ext[0]}, {ext[1]}], y₀ ∈ [{ext[2]}, {ext[3]}]\nzoom ×{3.8 / (ext[1] - ext[0]):.0f}, {res}² runs",
                     family="serif", fontsize=9, loc="left", color=INK)
        ax.tick_params(labelsize=7, colors=INK)
    # transect barcodes
    ax = fig.add_subplot(gs[0, 3])
    ax.set_facecolor(PAPER)
    tr = sorted(glob.glob(f"{CACHE}/transect_*.npz"))
    for j, f in enumerate(tr):
        t = np.load(f)
        s = (t["status"] == 2).astype(float)
        n = len(s)
        # show 1200 bins, grey level = diverged fraction in each bin (declared)
        bins = s[: n // 1200 * 1200].reshape(1200, -1).mean(1)
        ax.imshow(bins[None, :], extent=[0, 1, j + 0.1, j + 0.9], cmap="Greys", vmin=0, vmax=1, aspect="auto")
        ax.text(1.01, j + 0.5, f"L = {float(t['L']):.0e}", fontsize=7, family="serif", va="center", color=INK)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, max(1, len(tr)))
    ax.set_yticks([])
    ax.set_xlabel("position along the vertical segment (fraction of L)", family="serif", fontsize=8)
    ax.set_title(f"1-D transects at x₀ = 3.296875, 2²² runs each,\nnested ×100 (dark = diverges)", family="serif", fontsize=9, loc="left")
    # box counting
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(PAPER)
    for key, lab, col in [("zhu4_z1", "Zhu degree-4, z1", INK), ("zhu4_z2", "Zhu degree-4, z2", "#555"),
                          ("prod2_null", "null: ½(xy−1)², proven smooth", BLUE), ("lmreg_pos", "positive control (L&M reg.)", RED)]:
        if key not in B:
            continue
        bc = B[key]["boxcount"]
        e = np.array(bc["eps"])
        N = np.array(bc["counts"], float)
        ax.loglog(1 / e, N, "o-", ms=3, lw=0.8, color=col, label=f"{lab}: D = {bc['D']:.3f} ± {bc['se']:.3f}")
    ax.axvspan(1 / 0.125, 1 / 2.44e-4, color="#999", alpha=0.08, lw=0)
    ax.set_xlabel("1/ε (box side as a fraction of the region)", family="serif", fontsize=8)
    ax.set_ylabel("occupied boxes N(ε)", family="serif", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("box counting at 8192² (shaded: fit range, 2.7 decades)", family="serif", fontsize=9, loc="left")
    # resolution check crops
    ax = fig.add_subplot(gs[1, 1])
    crops = []
    for r in [1024, 2048, 4096, 8192]:
        d = np.load(f"{CACHE}/basin_zhu4_z2_{r}.npz")["status"]
        c = d[: r // 8, : r // 8]
        c = np.kron(c, np.ones((8192 // r, 8192 // r), dtype=c.dtype))
        crops.append(label_img(c))
    top = np.hstack([crops[0], np.ones((1024, 20, 3)) * 0.95, crops[1]])
    bot = np.hstack([crops[2], np.ones((1024, 20, 3)) * 0.95, crops[3]])
    ax.imshow(np.vstack([top, np.ones((20, top.shape[1], 3)) * 0.95, bot]), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for (x, y, t) in [(10, 60, "1024²"), (1054, 60, "2048²"), (10, 1104, "4096²"), (1054, 1104, "8192²")]:
        ax.text(x, y, t, color=RED, fontsize=8, family="monospace")
    ax.set_title("resolution check: the same 1/8 × 1/8 corner of z2,\nrecomputed at 4 resolutions (not resampled)", family="serif", fontsize=9, loc="left")
    # flips and resolution scaling
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(PAPER)
    for key, lab, col in [("zhu4_z1", "Zhu z1", INK), ("zhu4_z2", "Zhu z2", "#555"), ("prod2_null", "null", BLUE), ("lmreg_pos", "pos. control", RED)]:
        if key not in B:
            continue
        fl = B[key]["label_flip"]
        R = [int(k.split("->")[0]) for k in fl]
        v = list(fl.values())
        ax.loglog(R, v, "o-", ms=3, lw=0.8, color=col, label=f"{lab}: boundary px ∝ R^{B[key]['D_resolution_scaling']:.2f}")
    ax.set_xlabel("resolution R (then 2R)", family="serif", fontsize=8)
    ax.set_ylabel("fraction of labels that flip under 2× refinement", family="serif", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("uncertainty exponent: flips ∝ R^−α, D = 2 − α\nZhu: α ≈ 0.27 → D ≈ 1.73; null: α ≈ 1 → D = 1",
                 family="serif", fontsize=9, loc="left")
    # transect box counting
    ax = fig.add_subplot(gs[1, 3])
    ax.set_facecolor(PAPER)
    txt = []
    for j, key in enumerate(sorted(k for k in B if k.startswith("transect_"))):
        t = B[key]
        ax.loglog(1 / np.array(t["eps"]), np.array(t["counts"], float), "o-", ms=2.5, lw=0.8,
                  color=[INK, "#555", "#999"][j % 3], label=f"L = {t['L']:.0e}: D₁ = {t['D1']:.3f}")
        txt.append(t["D1"])
    ax.set_xlabel("1/ε (absolute, in y₀)", family="serif", fontsize=8)
    ax.set_ylabel("segments containing a label change", family="serif", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    ttl = "1-D box counting of the transects"
    if txt:
        ttl += f"\nproduct picture: D₂ ≈ 1 + D₁ = {1 + np.mean(txt):.2f}"
    ax.set_title(ttl, family="serif", fontsize=9, loc="left")
    fig.suptitle("Is the converge/diverge boundary of GD on ½(1 − x²y²)² (Zhu et al., η = 0.2) fractal?  Fractals doc §11, measured",
                 family="serif", fontsize=13, color=INK, x=0.05, ha="left")
    fig.savefig(f"{GAL}/plate_basin_verification.png", facecolor=PAPER)
    print("wrote plate_basin_verification.png")


if __name__ == "__main__":
    main()
