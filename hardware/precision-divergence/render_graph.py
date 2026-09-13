"""Draw the exact functional graph of x -> 4x(1-x) on float16 / bfloat16 (all values in [0,1]).

    python render_graph.py graph [rivers plotter riso] [float16 bfloat16 FP8_E4M3 FP8_E5M2]
    python render_graph.py line  [float16 bfloat16]      # basin-coloured number line

Exact: nodes (every representable value in [0,1]), edges x -> round(4x(1-x)), cycles, depth, basin,
traffic (seed measure flowing through each edge).  Layout: radial trees on cycle rings, angular span
proportional to leaf count, children ordered by value (a declared, deterministic layout choice).
Aesthetic: colours, width mapping (traffic^0.5), paper/riso treatment.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import colorcet  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

HERE = Path(__file__).parent
CACHE, GALLERY = HERE / "cache", HERE / "gallery"
SERIF, MONO = "P052", "Nimbus Mono PS"
STACK = "exact integer rounding (ties-to-even), verified = torch/numpy native · 2026-09"
BG = {"rivers": "#050608", "plotter": "#f3eee2", "riso": "#f4efe6"}
INK = {"rivers": "#e8e4d8", "plotter": "#1b1a17", "riso": "#1f5fbf"}
BASIN_COLS = ["#1f5fbf", "#ff4f8b", "#e0a030", "#3f8f5a", "#7a4fb0", "#20a0a0"]  # declared categorical palette


def load(fmt):
    return dict(np.load(CACHE / f"layout_{fmt}.npz"))


def comp_geometry(L, k):
    nodes, th, r = L[f"c{k}_nodes"], L[f"c{k}_theta"], L[f"c{k}_r"]
    pos = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    return nodes, pos, L[f"c{k}_edges"], L[f"c{k}_cyc_edges"]


def draw_component(ax, L, k, style, layer=None):
    nodes, pos, edges, cyc_edges = comp_geometry(L, k)
    tr = L["traffic"][nodes]
    child = edges[:, 0]
    segs = np.stack([pos[edges[:, 0]], pos[edges[:, 1]]], 1)
    t = tr[child]
    tn = t / L["traffic"].max()
    rmax = np.abs(pos).max() if len(pos) > 1 else 1.0
    scale = 1.0 / max(rmax, 1e-9)
    if style == "rivers":
        lt = np.log10(np.maximum(t, 1e-30))
        lo, hi = np.log10(np.percentile(L["cell"], 5)), 0.0
        c = np.clip((lt - lo) / (hi - lo), 0, 1)
        order = np.argsort(c)
        cols = plt.get_cmap("cet_fire")(0.12 + 0.88 * c[order])
        cols[:, 3] = 0.35 + 0.65 * c[order]
        lw = 0.15 + 5.5 * tn[order] ** 0.5
        ax.add_collection(LineCollection(segs[order] * scale, colors=cols, linewidths=lw, capstyle="round"))
        if len(cyc_edges):
            cs = np.stack([pos[cyc_edges[:, 0]], pos[cyc_edges[:, 1]]], 1) * scale
            ax.add_collection(LineCollection(cs, colors="#bfe3ff", linewidths=2.2))
        ax.plot(*(pos[[i for i in range(len(nodes)) if L["depth"][nodes[i]] == 0]] * scale).T, "o", ms=3.5, color="#bfe3ff")
    else:
        ink = "black" if layer else INK[style]
        red = "black" if layer else ("#b8322a" if style == "plotter" else "#ff4f8b")
        if layer in (None, "A"):
            lw = 0.12 + 2.2 * tn ** 0.5
            ax.add_collection(LineCollection(segs * scale, colors=ink if style == "plotter" else (ink if layer else "#ff4f8b"),
                                             linewidths=lw, capstyle="round", alpha=0.9))
        if layer in (None, "B"):
            if len(cyc_edges):
                cs = np.stack([pos[cyc_edges[:, 0]], pos[cyc_edges[:, 1]]], 1) * scale
                ax.add_collection(LineCollection(cs, colors=red if style == "plotter" else (ink if layer else "#1f5fbf"),
                                                 linewidths=2.0))
            cyc = pos[L["depth"][nodes] == 0] * scale
            ax.plot(cyc[:, 0], cyc[:, 1], "o", ms=3.2, color=red if style == "plotter" else (ink if layer else "#1f5fbf"))
    lim = 1.03
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def component_caption(L, k, vals):
    cyc = L[f"c{k}_cycle"]
    nodes = L[f"c{k}_nodes"]
    return (f"cycle length {len(cyc)}   ·   {len(nodes)} values   ·   max depth {int(L['depth'][nodes].max())}\n"
            f"catches {100 * L['basin_measure'][k]:.1f}% of random real seeds   ·   "
            + (f"fixed point {vals[cyc[0]]:g}" if len(cyc) == 1 else f"smallest cycle value {vals[cyc].min():.6g}"))


def graph_plate(fmt, style, layer=None):
    L = load(fmt)
    vals = L["vals"]
    ncyc = int(L["cycle_id"].max()) + 1
    order = list(np.argsort(-np.array([len(L[f"c{k}_nodes"]) for k in range(ncyc)])))
    big = [k for k in order if len(L[f"c{k}_nodes"]) >= 40][:2]
    small = [k for k in order if k not in big]
    bg = "white" if layer else BG[style]
    ink = "black" if layer else INK[style]
    W = 12 * len(big) + (4 if small else 0)
    fig = plt.figure(figsize=(W, 14), dpi=300, facecolor=bg)
    x = 0.0
    for k in big:
        ax = fig.add_axes([x / W + 0.005, 0.08, 12 / W - 0.01, 12 / 14])
        ax.set_facecolor(bg)
        draw_component(ax, L, k, style, layer)
        if layer in (None, "B"):
            fig.text(x / W + 0.02, 0.045, component_caption(L, k, vals), family=MONO, fontsize=12, color=ink, va="center")
        x += 12
    y = 0.62
    for k in small:
        h = 3.2
        ax = fig.add_axes([x / W + 0.01, y, 3.4 / W, h / 14])
        ax.set_facecolor(bg)
        draw_component(ax, L, k, style, layer)
        if layer in (None, "B"):
            c = L[f"c{k}_cycle"]
            fig.text(x / W + 0.01, y - 0.012, f"cycle {len(c)} at {vals[c].min():g}\n{len(L[f'c{k}_nodes'])} values · "
                     f"{100 * L['basin_measure'][k]:.2f}% of seeds", family=MONO, fontsize=10, color=ink, va="top")
        y -= 0.3
    if layer in (None, "B"):
        name = {"float16": "float16", "bfloat16": "bfloat16", "FP8_E4M3": "FP8 E4M3", "FP8_E5M2": "FP8 E5M2"}[fmt]
        fig.text(0.012, 0.965, f"Every orbit ends in a cycle: {name}", family=SERIF, fontsize=36, color=ink, va="center")
        how = ("edge width and brightness = fraction of random real seeds whose rounded orbit flows through it (fire, log)"
               if style == "rivers" else "edge width = fraction of random real seeds flowing through it")
        fig.text(0.012, 0.935, f"all {len(vals)} representable x in [0,1]; edge x -> round(4x(1-x)); trees hang outward from the "
                 f"cycle (radius = steps to the cycle). {how}", family=MONO, fontsize=11, color=ink, va="center")
        fig.text(0.99, 0.012, STACK, family=MONO, fontsize=9, color=ink, ha="right")
    return fig


def fig_cov(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255
    plt.close(fig)
    return 1 - a.mean(2)


def graph(fmt, style):
    if style == "riso":
        A = fig_cov(graph_plate(fmt, style, "A"))
        B = fig_cov(graph_plate(fmt, style, "B"))
        B = np.roll(np.roll(B, 6, 0), -5, 1)
        paper = np.array(matplotlib.colors.to_rgb(BG["riso"]))
        out = np.ones(A.shape + (3,)) * paper
        for cov, col in ((A * 0.9, "#ff4f8b"), (B * 0.95, "#1f5fbf")):
            c = np.array(matplotlib.colors.to_rgb(col))
            out *= 1 - cov[..., None] * (1 - c)
        from PIL import Image
        p = GALLERY / f"graph_{fmt}_riso.png"
        Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(p, optimize=True)
    else:
        fig = graph_plate(fmt, style)
        p = GALLERY / f"graph_{fmt}_{style}.png"
        fig.savefig(p, facecolor=fig.get_facecolor())
        plt.close(fig)
    print("wrote", p)


def number_line(fmt, style="plotter"):
    """Every representable x in [0,1]: a vertical line of height = steps to its cycle, coloured by basin.
    Top: linear x. Bottom: log2 x (the tiny values that dominate the count)."""
    L = load(fmt)
    vals, depth, basin = L["vals"], L["depth"], L["basin"]
    bg, ink = BG[style if style != "rivers" else "rivers"], INK[style if style != "rivers" else "rivers"]
    fig = plt.figure(figsize=(26, 13), dpi=300, facecolor=bg)
    ncyc = int(basin.max()) + 1
    for row, (xs, lab, rng_) in enumerate([(vals, "x (linear)", (0, 1)),
                                           (np.log2(np.maximum(vals, vals[vals > 0].min() / 2)), "log2 x", None)]):
        ax = fig.add_axes([0.05, 0.53 - row * 0.45, 0.93, 0.36])
        ax.set_facecolor(bg)
        for b in range(ncyc):
            s = basin == b
            segs = np.stack([np.stack([xs[s], np.zeros(s.sum())], 1), np.stack([xs[s], depth[s]], 1)], 1)
            ax.add_collection(LineCollection(segs, colors=BASIN_COLS[b % len(BASIN_COLS)], linewidths=0.25, alpha=0.8))
        cyc = depth == 0
        ax.plot(xs[cyc], np.zeros(cyc.sum()), "v", color=ink, ms=6)
        ax.set_ylim(-2, depth.max() + 2)
        ax.set_xlim(*(rng_ or (xs.min() - 0.5, xs.max() + 0.5)))
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(ink)
        ax.tick_params(colors=ink, labelsize=11)
        ax.set_xlabel(lab, family=MONO, color=ink, fontsize=13)
        ax.set_ylabel("steps to cycle", family=MONO, color=ink, fontsize=13)
    name = fmt.replace("_", " ")
    labels = []
    for b in range(ncyc):
        c = L[f"c{b}_cycle"]
        labels.append(f"basin of the {len(c)}-cycle at {vals[c].min():.4g}: {len(L[f'c{b}_nodes'])} values, "
                      f"{100 * L['basin_measure'][b]:.1f}% of seeds")
    fig.text(0.05, 0.955, f"Basins on the number line: {name}", family=SERIF, fontsize=34, color=ink)
    for b, t in enumerate(labels):
        fig.text(0.05 + 0.31 * b, 0.915, "■ " + t, family=MONO, fontsize=11, color=BASIN_COLS[b % len(BASIN_COLS)])
    fig.text(0.98, 0.01, STACK + " · categorical basin palette is a declared choice", family=MONO, fontsize=9, color=ink,
             ha="right")
    p = GALLERY / f"basin_line_{fmt}_{style}.png"
    fig.savefig(p, facecolor=bg)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "graph"
    args = sys.argv[2:]
    styles = [a for a in args if a in ("rivers", "plotter", "riso")] or ["rivers", "plotter", "riso"]
    fmts = [a for a in args if a not in styles] or ["float16", "bfloat16"]
    for f in fmts:
        if what == "graph":
            for s in styles:
                graph(f, s)
        else:
            for s in styles[:1] if args else ["plotter"]:
                number_line(f, s)
