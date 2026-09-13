"""Box-counting survey plate (fractals doc section 11 as a picture): for chain B (sync label), chain A
(paper threshold L_avg > 1e-5) and the infinite-width null model, the frontier edge cells of the native
256^2 float64 maps at five magnifications, with the occupied 16-px boxes tinted, and the N(s) curves.

  python render_boxcount_plate.py
"""
import json, os
import numpy as np
from PIL import Image
from render_common import *
from sp_core import edge_cells, box_counts

ZB = np.load(os.path.join(CACHE, "zoom_B_N100_D1000_s0_f64_r256.npz"))
ZA = np.load(os.path.join(CACHE, "zoom_A_N100_D1000_s0_f64_r256.npz"))
RB = json.load(open(os.path.join(CACHE, "fractal_report_B_N100_sync.json")))
RA = json.load(open(os.path.join(CACHE, "fractal_report_A_N100_tau.json")))
NB = np.load(os.path.join(CACHE, "null_labels_sync_r256.npz"))
LEVELS = [0, 2, 4, 6, 8]
S_BOX = 16
TINT = np.array([0.86, 0.62, 0.45])  # declared ochre tint for occupied boxes
PAPER_RGB = np.array([int(PAPER[i:i + 2], 16) for i in (1, 3, 5)]) / 255
INK_RGB = np.array([int(INK[i:i + 2], 16) for i in (1, 3, 5)]) / 255

rows = [
    ("finite width, sync label", "pair never merged (L < 1e-10) within 1000 layers", [ZB["t_hit"][k] > 1000 for k in LEVELS], RB["levels"], "#b8473a"),
    ("finite width, paper threshold", "L_avg > 1e-5 (arXiv:2508.03222)", [ZA["L_avg"][k] > 1e-5 for k in LEVELS], RA["levels"], "#2f6aa3"),
    ("null model: infinite width", "mean-field L, same pipeline, own boundary-centred zooms", [NB["B"][k] for k in LEVELS], RB["null"], "#4a4a4a"),
]


def panel(B, up=2):
    E = edge_cells(B)
    R = E.shape[0]
    Ep = np.pad(E, ((0, (-R) % S_BOX), (0, (-R) % S_BOX)))
    occ = Ep.reshape(Ep.shape[0] // S_BOX, S_BOX, -1, S_BOX).any((1, 3))
    occ_px = np.kron(occ, np.ones((S_BOX, S_BOX)))[:R, :R] > 0
    rgb = np.where(occ_px[..., None], PAPER_RGB * 0.45 + TINT * 0.55, PAPER_RGB)
    rgb = np.where(E[..., None], INK_RGB, rgb)
    rgb = np.kron(rgb, np.ones((up, up, 1)))
    g = S_BOX * up
    rgb[::g, :, :] = rgb[::g, :, :] * 0.82 + 0.18 * INK_RGB
    rgb[:, ::g, :] = rgb[:, ::g, :] * 0.82 + 0.18 * INK_RGB
    return rgb[::-1]  # origin lower


PW, GAP, LM, TOP, ROWH, CAP = 510, 34, 150, 250, 510 + 150, 170
PLOTW = 820
W = LM + 5 * PW + 4 * GAP + 90 + PLOTW + 90
H = TOP + 3 * ROWH + CAP
fig = fig_px(W, H, dpi=140, bg=PAPER)
fig.text(LM / W, 1 - 80 / H, "Box-counting survey of the order/chaos frontier  (one random erf network, width 100, depth 1000)",
         fontsize=26, color=INK, va="center")
fig.text(LM / W, 1 - 140 / H, "Ink: edge cells (2x2 neighbourhoods containing both outcomes) of the native 256 x 256 float64 map at each zoom.  "
         "Ochre: occupied 16-px boxes (one scale of the count).  Fit: log N vs log(1/s), s = 2 ... 32 px.",
         fontsize=13, color=INK, va="center", alpha=0.85)
cm = plt.get_cmap("cmc.batlow")
for r, (name, sub, Bs, rep, col) in enumerate(rows):
    y0 = TOP + r * ROWH
    fig.text(LM / W, 1 - (y0 - 30) / H, name, fontsize=17, color=col, va="center", weight="bold")
    fig.text((LM + 640) / W, 1 - (y0 - 30) / H, sub, fontsize=12, color=INK, va="center", alpha=0.8)
    for i, (k, B) in enumerate(zip(LEVELS, Bs)):
        x0 = LM + i * (PW + GAP)
        ax = img_axes(fig, x0, y0, PW, PW, W, H)
        ax.imshow(panel(B), interpolation="nearest")
        ax.set_xlim(-0.5, 511.5); ax.set_ylim(511.5, -0.5)
        d = rep[k]
        ac = d.get("adj_corr", np.nan)
        fig.text((x0 + PW / 2) / W, 1 - (y0 + PW + 40) / H, f"x{4 ** k:,}      slope {d['slope']:.2f}",
                 fontsize=15, color=INK, ha="center", va="center")
        fig.text((x0 + PW / 2) / W, 1 - (y0 + PW + 80) / H, f"neighbour corr {ac:.2f}   edge cells {int(edge_cells(B).sum()):,}",
                 fontsize=10.5, color=INK, ha="center", va="center", alpha=0.75, family="DejaVu Sans Mono")
    # N(s) curves for every level of the chain
    ax = fig.add_axes([(LM + 5 * PW + 4 * GAP + 150) / W, 1 - (y0 + PW) / H, (PLOTW - 60) / W, PW / H])
    ax.set_facecolor(PAPER)
    nlev = len(rep)
    for d in rep:
        s = np.array(d["sizes"], float); n = np.array(d["counts"], float)
        n0 = n[0]
        ax.plot(1 / s, n / n0, "o-", color=cm(d["level"] / max(nlev - 1, 1)), lw=1.3, ms=4,
                label=f"x{4 ** d['level']:,}: {d['slope']:.2f}")
    xx = np.array([1 / 32, 1 / 2])
    for sl, ls in ((1, ":"), (2, "--")):
        ax.plot(xx, (xx / 0.5) ** sl, ls, color=INK, lw=0.8)
        ax.text(xx[0] * 0.9, (xx[0] / 0.5) ** sl, f"D = {sl}", fontsize=9, ha="right", va="center", color=INK)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlim(1 / 64, 0.62)
    ax.set_xlabel("1 / box size (px)", fontsize=10); ax.set_ylabel("N(s) / N(2 px)", fontsize=10)
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(INK); sp.set_linewidth(0.6)
    ax.legend(fontsize=9, frameon=False, loc="upper left", title="zoom: local slope", title_fontsize=10, ncol=2)
fig.text(LM / W, 1 - (TOP + 3 * ROWH + 10) / H,
         "Reading the plate.  Row 1: the slope climbs from 1.08 to a peak of 1.87 at x256, then falls to 1.48 at x262,144 as laminated stripes become resolved "
         "(neighbour correlation 0.47 -> 0.87): no single dimension.\n"
         "Row 2: with the paper's threshold the edge cells fill the window past x256 (slope 1.98-1.99, neighbour correlation 0.07-0.13): area-filling noise at "
         "this resolution, not a curve with a stable non-integer dimension.\n"
         "Row 3: the infinite-width limit through the same pipeline stays a smooth curve (slope 0.98-1.03 at every zoom).  "
         "Zoom windows are centred where ordered/chaotic mixing is strongest, which biases counts upward.  Panels shown at 2x nearest-neighbour.",
         fontsize=12, color=INK, va="top", linespacing=1.7)
savefig(fig, "verification_boxcount_plate.png", dpi=140)
print("ok", W, H)
