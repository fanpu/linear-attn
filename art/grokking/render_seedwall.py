"""Seed wall (12 seeds, all key stars superposed per seed) and the specimen sheet (every seed's key-frequency
stars, arranged by k).  Final checkpoints.  Output: gallery/seedwall_*.png, gallery/specimen_*.png"""
import numpy as np
import matplotlib.pyplot as plt
from styles import STYLES, draw_star, normalize, fig_canvas, riso_composite, coverage_from_fig, save_png, P

A = dict(np.load("cache/analysis.npz"))
S = len(A["init_seeds"])


def seed_label(s):
    lab = f"seed {int(A['init_seeds'][s])}"
    if A["data_seeds"][s] != 598:
        lab += f" · split {int(A['data_seeds'][s])}"
    return lab


def keys_of(s):
    K = int(A["nkeys"][s]); ks = A["keys"][s][:K]
    o = np.argsort(ks)
    return ks[o], [normalize(A["ring"][-1, s, j].astype(np.float64)) for j in o]


# ------------------------------------------------------------------ seed wall
def seedwall(style, cols=4):
    st = STYLES[style]
    rows = int(np.ceil(S / cols))
    W, Hh = 3200, int(3200 / cols * rows * 1.1) + 200
    cw, ch = 1 / cols, (1 - 200 / Hh) / rows

    def cell_ax(fig, i):
        r, c = divmod(i, cols)
        return fig.add_axes([c * cw + 0.012, 1 - 150 / Hh - (r + 1) * ch + 0.035 * ch * 1.0 + 0.02, cw - 0.024, ch * 0.86])

    if style == "riso":
        layers = []
        for layer in range(2):
            fig = plt.figure(figsize=(W / 200, Hh / 200), dpi=200, facecolor="white")
            for i in range(S):
                ks, rs = keys_of(i)
                ax = cell_ax(fig, i)
                for j, z in enumerate(rs):
                    if j % 2 == layer:
                        draw_star(ax, z, style, color="black", lw=0.55, alpha=0.9, lim=1.6)
                ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.axis("off")
                if layer == 1:
                    r, c = divmod(i, cols)
                    fig.text(c * cw + cw / 2, 1 - 150 / Hh - (r + 1) * ch + 0.012, f"{seed_label(i)}   k = {', '.join(map(str, ks))}",
                             ha="center", fontsize=10, color="black")
            layers.append(coverage_from_fig(fig)); plt.close(fig)
        return riso_composite(layers, [st["ink"], st["ink2"]], st["bg"], offsets=[(0, 0), (5, -6)], grain=0.1)
    fig = fig_canvas(W, Hh, style)
    for i in range(S):
        ks, rs = keys_of(i)
        ax = cell_ax(fig, i); ax.set_facecolor(st["bg"])
        for z in rs:
            if style == "nocturne":   # declared: one warm ink, low alpha, so overlaps accumulate like exposure
                draw_star(ax, z, style, lw=0.5, alpha=0.45, color="#ffd9a8", lim=1.35)
            else:
                draw_star(ax, z, style, lw=0.4, alpha=0.8, lim=1.35)
        r, c = divmod(i, cols)
        fig.text(c * cw + cw / 2, 1 - 150 / Hh - (r + 1) * ch + 0.012, f"{seed_label(i)}    k = {', '.join(map(str, ks))}",
                 ha="center", fontsize=10, color=st["muted"], style="italic" if style == "plate" else "normal")
    title = {"nocturne": "Twelve seeds, twelve sets of key frequencies",
             "plotter": "12 SEEDS / KEY FREQUENCIES / {113/k} SUPERPOSED",
             "plate": "Plate III.  The same algorithm, twelve times, with arbitrary frequencies"}[style]
    fig.text(0.02, 1 - 60 / Hh, title, fontsize=18, color=st["ink"], va="center", style="italic" if style == "plate" else "normal")
    return fig


# ------------------------------------------------------------------ specimen sheet (arranged by k)
def specimen(style):
    st = STYLES[style]
    items = []
    for s in range(S):
        ks, rs = keys_of(s)
        for k, z in zip(ks, rs):
            items.append((min(int(k), P - int(k)), s, z))
    items.sort(key=lambda x: (x[0], x[1]))
    n = len(items)
    cols = int(np.ceil(np.sqrt(n * 1.0)))
    rows = int(np.ceil(n / cols))
    W = 3600; cell = W / cols; Hh = int(cell * rows * 1.12) + 260
    cw = 1 / cols; ch = (Hh - 260) / Hh / rows
    fig = fig_canvas(W, Hh, style)
    for i, (k, s, z) in enumerate(items):
        r, c = divmod(i, cols)
        y0 = 1 - 200 / Hh - (r + 1) * ch
        ax = fig.add_axes([c * cw + 0.006, y0 + ch * 0.12, cw - 0.012, ch * 0.86]); ax.set_facecolor(st["bg"])
        draw_star(ax, z, style, lw=0.35 if style != "nocturne" else 0.45, lim=1.55)
        fig.text(c * cw + cw / 2, y0 + ch * 0.05, f"{{113/{k}}}  s{int(A['init_seeds'][s])}", ha="center", fontsize=7.5,
                 color=st["muted"], family="DejaVu Sans Mono")
    title = {"nocturne": "Specimens: every key-frequency star from twelve seeds, ordered by k",
             "plotter": "SPECIMEN SHEET — {113/k}, ALL SEEDS, ORDERED BY k",
             "plate": "Plate IV.  Specimens of learned star polygons, arranged by frequency"}[style]
    fig.text(0.015, 1 - 90 / Hh, title, fontsize=18, color=st["ink"], va="center", style="italic" if style == "plate" else "normal")
    fig.text(0.985, 1 - 90 / Hh, f"{n} stars · k and 113−k give the same star", fontsize=11, color=st["muted"], va="center", ha="right")
    return fig


if __name__ == "__main__":
    for style in ["nocturne", "plotter", "plate"]:
        save_png(seedwall(style), f"gallery/seedwall_{style}.png")
        if True:
            save_png(specimen(style), f"gallery/specimen_{style}.png")
        print("wrote", style, flush=True)
