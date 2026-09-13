"""Training curves (train / test / restricted / excluded loss) with measured phase bands, in styles
matched to the rest of the gallery.  Output: gallery/curves_*.png"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from styles import STYLES, fig_canvas, save_png
from phases import phases

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
A = dict(np.load("cache/analysis.npz"))
s = args.seed
ph = phases(A, s)
steps = np.arange(A["tr_loss"].shape[0])
tag = f"seed{int(A['init_seeds'][s])}"


def curves(style):
    st = STYLES[style]
    fig = fig_canvas(3000, 1700, style)
    ax = fig.add_axes([0.08, 0.13, 0.88, 0.74]); ax.set_facecolor(st["bg"])
    ink, muted = st["ink"], st["muted"]
    tr = np.maximum(A["tr_loss"][:, s], 1e-12)
    # declared: train loss thinned to every 10th step for drawing (no smoothing)
    ax.plot(steps[::10], tr[::10], color=st["train"], lw=1.4, label="train loss (3,830 pairs)")
    ls_test = "-" if style != "plotter" else (0, (5, 2))
    ax.plot(A["ev_steps"], A["te_loss"][:, s], color=st["test"], lw=1.4, ls=ls_test, label="test loss (8,939 held-out pairs)")
    fs = A["full_steps"]
    ls_r = (0, (1, 1.6)); ls_e = (0, (6, 2, 1, 2))
    rc = st["ink"] if style != "nocturne" else "#c9a9ff"
    ec = st["muted"] if style != "nocturne" else "#7fd6a8"
    ax.plot(fs, A["restricted"][:, s], color=rc, lw=1.3, ls=ls_r, label="restricted loss (only key frequencies kept)")
    ax.plot(fs, A["excluded"][:, s], color=ec, lw=1.3, ls=ls_e, label="excluded loss (key frequencies ablated, train set)")
    ax.set_yscale("log")
    ax.set_xlim(0, steps[-1])
    lo = min(tr[tr > 0].min(), np.nanmin(A["restricted"][:, s])) / 3
    ax.set_ylim(max(lo, 1e-9), 80)
    bounds = [0, ph["t_circ"], ph["t_clean"], ph["t_done"] if ph["t_done"] > 0 else steps[-1], steps[-1]]
    names = ["memorization", "circuit formation", "cleanup", "stable"]
    for i in range(4):
        if style == "nocturne":
            ax.axvspan(bounds[i], bounds[i + 1], color="white", alpha=[0.0, 0.035, 0.08, 0.0][i], lw=0)
        elif style == "plate":
            ax.axvspan(bounds[i], bounds[i + 1], color=st["grid"], alpha=[0.0, 0.35, 0.7, 0.0][i], lw=0)
        else:
            ax.axvspan(bounds[i], bounds[i + 1], facecolor="none", edgecolor=st["grid"], hatch=["", "", "////", ""][i], lw=0)
        if i < 3:
            ax.axvline(bounds[i + 1], color=muted, lw=0.6, ls=":")
        mid = (bounds[i] + bounds[i + 1]) / 2
        ax.text(mid, 1.02, names[i], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=12, color=muted,
                style="italic" if style == "plate" else "normal")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(muted); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=muted, labelsize=12)
    ax.set_xlabel("training step (full-batch AdamW)", color=muted, fontsize=13)
    ax.set_ylabel("cross-entropy loss (log scale)", color=muted, fontsize=13)
    ax.grid(True, axis="y", color=st["grid"], lw=0.5, which="major")
    leg = ax.legend(loc="lower left", frameon=False, fontsize=12, labelcolor=ink)
    title = {"nocturne": "Grokking  —  (a + b) mod 113", "plotter": "GROKKING / (a+b) mod 113 / one-layer transformer",
             "riso": "grokking", "plate": "Plate II.  Loss during training, with the three phases of Nanda et al."}[style]
    fig.text(0.08, 0.955, title, fontsize=20, color=ink, ha="left", va="center",
             style="italic" if style == "plate" else "normal")
    fig.text(0.96, 0.955, f"{tag} · 30% train split · wd 1.0", fontsize=12, color=muted, ha="right", va="center")
    return fig


for style in ["nocturne", "plotter", "plate"]:
    save_png(curves(style), f"gallery/curves_{tag}_{style}.png")
print(ph)
