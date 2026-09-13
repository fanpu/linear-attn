"""Star polygons {113/k}: small multiples per key frequency and overlay of all key stars (main seed),
in four styles.  Input: cache/analysis.npz.  Output: gallery/stars_*.png"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from styles import STYLES, draw_star, normalize, fig_canvas, fig_to_array, riso_composite, coverage_from_fig, save_png, P

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0, help="index into the 12 runs")
ap.add_argument("--step", type=int, default=-1, help="index into emb checkpoints (default final)")
args = ap.parse_args()
A = dict(np.load("cache/analysis.npz"))
s = args.seed
K = int(A["nkeys"][s]); keys = A["keys"][s][:K]
rings = [normalize(A["ring"][args.step, s, j].astype(np.float64)) for j in range(K)]
order = np.argsort(keys)
keys = keys[order]; rings = [rings[i] for i in order]
Rf = A["R"][args.step, s][order]
step = int(A["emb_steps"][args.step])
tag = f"seed{int(A['init_seeds'][s])}"


def star_name(k):
    kk = min(k, P - k)
    return "{113/%d}" % kk


# ------------------------------------------------------------------ small multiples
def small_multiples(style):
    st = STYLES[style]
    W, Hh = 3000, 3000 // K + 520
    if style == "riso":
        # two separate ink layers: stars in blue, labels + circle guides in pink, misregistered
        layers = []
        for layer in range(2):
            fig = plt.figure(figsize=(W / 200, Hh / 200), dpi=200, facecolor="white")
            for j, k in enumerate(keys):
                ax = fig.add_axes([j / K + 0.01, 0.2, 1 / K - 0.02, 0.72])
                if layer == 0:
                    draw_star(ax, rings[j], style, color="black", lw=0.7)
                else:
                    t = np.linspace(0, 2 * np.pi, 400)
                    ax.plot(1.42 * np.cos(t), 1.42 * np.sin(t), color="black", lw=3.0)
                    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.75); ax.set_aspect("equal"); ax.axis("off")
                    fig.text(j / K + 0.5 / K, 0.09, f"k = {k}", ha="center", fontsize=17, color="black", weight="bold")
            layers.append(coverage_from_fig(fig)); plt.close(fig)
        img = riso_composite(layers, [st["ink"], st["ink2"]], st["bg"], offsets=[(0, 0), (5, -7)], grain=0.12)
        return img
    fig = fig_canvas(W, Hh, style)
    for j, k in enumerate(keys):
        ax = fig.add_axes([j / K + 0.01, 0.2, 1 / K - 0.02, 0.72])
        ax.set_facecolor(st["bg"])
        draw_star(ax, rings[j], style, lw=st["lw"] * 1.4)
        lab = f"k = {k}   {star_name(k)}"
        if style == "plate":
            lab = f"Fig. {j + 1}.  k = {k}\n{star_name(k)}   R = {Rf[j]:.3f}"
        fig.text(j / K + 0.5 / K, 0.1, lab, ha="center", va="center", fontsize=13, color=st["ink"] if style != "nocturne" else st["muted"])
    if style == "plate":
        fig.text(0.02, 0.965, f"PLATE I.  Embedding circles of a one-layer transformer, (a+b) mod 113, {tag}, step {step}",
                 fontsize=13, color=st["ink"], va="top", style="italic")
        ax = fig.add_axes([0.005, 0.005, 0.99, 0.99]); ax.set_facecolor("none")
        for sp in ax.spines.values():
            sp.set_color(st["ink"]); sp.set_linewidth(1.2)
        ax.set_xticks([]); ax.set_yticks([])
    return fig


# ------------------------------------------------------------------ overlay
def overlay(style):
    st = STYLES[style]
    W = 2400
    if style == "riso":
        layers = []
        for layer in range(2):
            fig = plt.figure(figsize=(W / 200, W / 200), dpi=200, facecolor="white")
            ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
            for j in range(K):
                if j % 2 == layer:
                    draw_star(ax, rings[j], style, color="black", lw=0.75, alpha=0.85, lim=1.6)
            ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.axis("off")
            layers.append(coverage_from_fig(fig)); plt.close(fig)
        return riso_composite(layers, [st["ink"], st["ink2"]], st["bg"], offsets=[(0, 0), (6, -8)], grain=0.1)
    fig = fig_canvas(W, W, style)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92]); ax.set_facecolor(st["bg"])
    for j in range(K):
        if style == "nocturne":
            draw_star(ax, rings[j], style, lw=0.5, alpha=0.55, lim=1.6)
        else:
            draw_star(ax, rings[j], style, lw=0.4, alpha=0.8, lim=1.6)
    if style == "plate":
        fig.text(0.04, 0.975, f"All {K} key-frequency stars superposed  (k = {', '.join(map(str, keys))})", fontsize=12,
                 color=st["ink"], style="italic", va="top")
    return fig


for style in STYLES:
    save_png(small_multiples(style), f"gallery/stars_smallmultiples_{tag}_{style}.png")
    save_png(overlay(style), f"gallery/stars_overlay_{tag}_{style}.png")
    print("wrote", style, flush=True)


# ------------------------------------------------------------------ labelled ring: the scrambled order
def labelled(style, j=None):
    st = STYLES[style]
    j = int(np.argmax(Rf)) if j is None else j                  # the cleanest ring
    k = int(keys[j]); z = rings[j]
    inv = pow(k, -1, P)
    fig = fig_canvas(2400, 2400, style)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9]); ax.set_facecolor(st["bg"])
    draw_star(ax, z, style, lw=0.5, alpha=0.5 if style == "nocturne" else 0.35, lim=1.45)
    cols = st["cyclic"](np.arange(P) / P) if style == "nocturne" else [st["ink"]] * P
    ax.scatter(z[:, 0], z[:, 1], s=9, c=cols, zorder=3, lw=0)
    for a_ in range(P):
        r = np.hypot(*z[a_]); u = z[a_] / r
        ax.text(*(z[a_] + 0.07 * u), str(a_), fontsize=7, ha="center", va="center", color=cols[a_] if style == "nocturne" else st["ink"],
                rotation=np.degrees(np.arctan2(u[1], u[0])) - 90 if False else 0)
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
    fig.text(0.05, 0.965, f"k = {k}: token a sits at angle 2π·{k}·a/113", fontsize=18, color=st["ink"],
             style="italic" if style == "plate" else "normal")
    fig.text(0.05, 0.94, f"neighbours around the ring differ by {inv} = {k}⁻¹ mod 113; the lines join a → a+1, drawing {star_name(k)}",
             fontsize=12, color=st["muted"])
    return fig


def lissajous(style):
    """x = learned coordinate on ring k_i, y = learned coordinate on ring k_j (real data), joined a -> a+1."""
    st = STYLES[style]
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    cols = min(len(pairs), 5); rows = int(np.ceil(len(pairs) / cols))
    W = 3000; Hh = int(W / cols * rows) + 200
    fig = fig_canvas(W, Hh, style)
    for n, (i, j) in enumerate(pairs):
        r, c = divmod(n, cols)
        ax = fig.add_axes([c / cols + 0.01, 1 - 120 / Hh - (r + 1) * (1 - 200 / Hh) / rows + 0.02, 1 / cols - 0.02, (1 - 200 / Hh) / rows - 0.04])
        ax.set_facecolor(st["bg"])
        zz = np.stack([rings[i][:, 0], rings[j][:, 1]], 1)
        draw_star(ax, zz, style, lw=0.5, lim=1.6)
        ax.text(0, -1.6, f"cos-axis of k={keys[i]}  ×  sin-axis of k={keys[j]}", ha="center", fontsize=8, color=st["muted"])
    fig.text(0.02, 1 - 60 / Hh, "Lissajous figures from pairs of learned rings", fontsize=16, color=st["ink"])
    return fig


for style in ["nocturne", "plotter", "plate"]:
    save_png(labelled(style), f"gallery/stars_labelled_{tag}_{style}.png")
    save_png(lissajous(style), f"gallery/lissajous_{tag}_{style}.png")
