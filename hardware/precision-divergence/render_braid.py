"""The braid: one logistic-map orbit (r=4) computed in bfloat16, float16, float32, float64 and a
1024-bit mpmath reference, overlaid.  Plus the measured divergence plate.

    python render_braid.py braid [paper observatory riso]
    python render_braid.py grid                         # many seeds, small multiples
    python render_braid.py measure                      # divergence-step distributions vs prediction

Exact/measured: every orbit value, the per-seed divergence step n*(0.1), the distributions.
Aesthetic: inks, the choice of seed shown, the vertical offset-free overlay, fonts.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
CACHE, GALLERY = HERE / "cache", HERE / "gallery"
GALLERY.mkdir(exist_ok=True)
D = np.load(CACHE / "divergence.npz")
S = json.loads((CACHE / "divergence_stats.json").read_text())
SERIF, MONO = "P052", "Nimbus Mono PS"
FMTS = ["bfloat16", "float16", "float32", "float64"]
MANT = {"bfloat16": 7, "float16": 10, "float32": 23, "float64": 52}
STACK = "GB10 · torch 2.14.0+cu130 (bf16, CPU) · numpy 2.5 (fp16/32/64) · mpmath 1.3 @1024 bits · 2026-09"

STY = {
    "paper": dict(bg="#f3eee2", ink="#1b1a17", faint="#b9b0a0",
                  cols={"ref": "#1b1a17", "float64": "#2f5d8a", "float32": "#3f8f5a", "float16": "#d08a1f", "bfloat16": "#b8322a"}),
    "observatory": dict(bg="#07080c", ink="#e8e4d8", faint="#3a3d48",
                        cols={"ref": "#f4f1e8", "float64": "#6fb7d9", "float32": "#7fd18b", "float16": "#f0b44c", "bfloat16": "#ff5a4f"}),
    "riso": dict(bg="#f4efe6", ink="#1f5fbf", faint="#c9c1b3",
                 cols={"ref": "#1f5fbf", "float64": "#1f5fbf", "float32": "#7a4fb0", "float16": "#ff4f8b", "bfloat16": "#ff4f8b"}),
}


def pick_seed(proto="A"):
    """Seed whose n*(0.1) is closest to the median for every format (a 'typical' seed; declared choice)."""
    score = 0
    for f in FMTS:
        st = D[f"nstar_{f}_{proto}_0.1"]
        score = score + np.abs(st - np.median(st))
    return int(np.argmin(score))


def braid(style, seed=None, proto="A", nmax=64, fname=None):
    st = STY[style]
    seed = pick_seed(proto) if seed is None else seed
    fig = plt.figure(figsize=(24, 10), dpi=300, facecolor=st["bg"])
    ax = fig.add_axes([0.05, 0.1, 0.92, 0.66])
    ax.set_facecolor(st["bg"])
    n = np.arange(nmax + 1)
    ref = D[f"ref{proto}"][seed, : nmax + 1]
    order = ["float64", "float32", "float16", "bfloat16"]
    lw = {"float64": 2.2, "float32": 2.2, "float16": 2.2, "bfloat16": 2.2}
    ax.plot(n, ref, color=st["cols"]["ref"], lw=4.5 if style != "observatory" else 4.0, alpha=0.18, solid_joinstyle="round",
            zorder=1)
    for f in order:
        o = D[f"{f}{proto}"][seed, : nmax + 1]
        ns = int(D[f"nstar_{f}_{proto}_0.1"][seed])
        ax.plot(n, o, color=st["cols"][f], lw=lw[f], alpha=0.9, solid_joinstyle="round",
                dash_capstyle="round", zorder=3)
        ax.axvline(MANT[f], color=st["cols"][f], lw=1.0, ls=(0, (2, 3)), alpha=0.9, zorder=0)
        ha = {"bfloat16": "right", "float16": "left", "float32": "left", "float64": "left"}[f]
        dx = {"bfloat16": -0.4, "float16": 0.4, "float32": 0.4, "float64": 0.4}[f]
        ax.text(MANT[f] + dx, 1.03, f"{f}\npredicted {MANT[f]}\nmeasured {ns}", color=st["cols"][f], family=MONO,
                fontsize=12, ha=ha, va="bottom", transform=ax.get_xaxis_transform())
        ax.plot([ns], [o[ns]], "o", ms=9, mfc="none", mec=st["cols"][f], mew=1.8, zorder=4)
    ax.set_xlim(-0.5, nmax + 0.5)
    ax.set_ylim(-0.03, 1.03)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(st["faint"])
    ax.tick_params(colors=st["ink"], labelsize=12)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family(MONO)
    ax.set_xlabel("step n", family=MONO, color=st["ink"], fontsize=14)
    ax.set_ylabel("x_n", family=MONO, color=st["ink"], fontsize=14)
    fig.text(0.05, 0.94, "Divergence", family=SERIF, fontsize=44, color=st["ink"])
    x0 = D[f"seeds{proto}"][seed]
    fig.text(0.05, 0.9, f"x_(n+1) = 4 x_n (1 - x_n),  x_0 = {x0:.17g}.  The same orbit in four precisions; the wide pale "
                          f"line is the 1024-bit reference. Dashed rules: mantissa bits m. Rings: first step where |x_n - ref| > 0.1.",
             family=SERIF, fontsize=15, style="italic", color=st["ink"])
    fig.text(0.97, 0.02, STACK, family=MONO, fontsize=10, color=st["ink"], ha="right")
    p = GALLERY / (fname or f"braid_{style}.png")
    fig.savefig(p, facecolor=st["bg"])
    plt.close(fig)
    print("wrote", p, "seed", seed)


def grid(style="paper", proto="A", k=36, nmax=60):
    st = STY[style]
    rng = np.random.default_rng(11)
    seeds = rng.choice(len(D[f"seeds{proto}"]), k, replace=False)
    fig = plt.figure(figsize=(24, 24), dpi=200, facecolor=st["bg"])
    rows = int(np.ceil(k / 4))
    for i, s in enumerate(seeds):
        ax = fig.add_axes([0.03 + (i % 4) * 0.24, 0.93 - (i // 4 + 1) * (0.9 / rows), 0.225, 0.9 / rows - 0.012])
        ax.set_facecolor(st["bg"])
        n = np.arange(nmax + 1)
        ax.plot(n, D[f"ref{proto}"][s, : nmax + 1], color=st["cols"]["ref"], lw=3, alpha=0.18)
        for f in ["float64", "float32", "float16", "bfloat16"]:
            o = D[f"{f}{proto}"][s, : nmax + 1]
            ax.plot(n, o, color=st["cols"][f], lw=0.9, alpha=0.9)
            ns = int(D[f"nstar_{f}_{proto}_0.1"][s])
            if ns <= nmax:
                ax.plot([ns, ns], [-0.05, 1.05], color=st["cols"][f], lw=0.7, ls=":")
        ax.set_xlim(0, nmax)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")
        ax.text(0, 1.07, f"x0={D[f'seeds{proto}'][s]:.6f}", family=MONO, fontsize=8, color=st["ink"], transform=ax.transAxes)
    fig.text(0.03, 0.965, "36 seeds, four precisions", family=SERIF, fontsize=40, color=st["ink"])
    fig.text(0.03, 0.945, "bfloat16 (red) · float16 (amber) · float32 (green) · float64 (blue) · 1024-bit reference (pale). "
                          "Dotted: measured divergence step (|x_n - ref| > 0.1). Steps 0-60.",
             family=MONO, fontsize=13, color=st["ink"])
    p = GALLERY / f"braid_grid_{style}.png"
    fig.savefig(p, facecolor=st["bg"])
    plt.close(fig)
    print("wrote", p)


def measure():
    st = STY["paper"]
    fig = plt.figure(figsize=(20, 10), dpi=250, facecolor=st["bg"])
    ax = fig.add_axes([0.05, 0.12, 0.42, 0.72])
    ax2 = fig.add_axes([0.55, 0.12, 0.42, 0.72])
    for a in (ax, ax2):
        a.set_facecolor(st["bg"])
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=11)
    bins = np.arange(0, 80) - 0.5
    for f in FMTS:
        stp = D[f"nstar_{f}_A_0.1"]
        ax.hist(stp, bins=bins, color=STY["paper"]["cols"][f], alpha=0.75, label=f"{f}  (m={MANT[f]}, median {np.median(stp):.0f})",
                histtype="stepfilled", lw=0)
        ax.axvline(MANT[f], color=STY["paper"]["cols"][f], lw=1.5, ls="--")
    ax.set_xlabel("measured divergence step n*  (first n with |x_n - ref_n| > 0.1), 2000 seeds", family=MONO, fontsize=12)
    ax.set_ylabel("seeds", family=MONO, fontsize=12)
    ax.legend(prop=dict(family=MONO, size=11), frameon=False)
    ax.set_title("native dtypes vs prediction m (dashed)", family=SERIF, fontsize=18, loc="left")
    ps = [p for p in list(range(3, 33)) + [40, 48, 53, 56, 64]]
    for d, c in ((0.01, "#8a8a8a"), (0.1, "#1b1a17"), (0.5, "#b8322a")):
        med = [S[f"ideal{p}|A|{d}"]["median"] for p in ps]
        p10 = [S[f"ideal{p}|A|{d}"]["p10"] for p in ps]
        p90 = [S[f"ideal{p}|A|{d}"]["p90"] for p in ps]
        ax2.fill_between(ps, p10, p90, color=c, alpha=0.12, lw=0)
        ax2.plot(ps, med, "o-", color=c, ms=4, lw=1.4, label=f"delta = {d}: median, 10-90%")
    ax2.plot([0, 66], [-1, 65], color="#2f5d8a", lw=1, ls="--", label="n* = p - 1  (= mantissa bits m)")
    for f in FMTS:
        ax2.plot(MANT[f] + 1, S[f"{f}|A|0.1"]["median"], "s", color=STY["paper"]["cols"][f], ms=10, mfc="none", mew=2)
        ax2.annotate(f, (MANT[f] + 1, S[f"{f}|A|0.1"]["median"]), xytext=(8, -14), textcoords="offset points",
                     family=MONO, fontsize=10, color=STY["paper"]["cols"][f])
    ax2.set_xlabel("significand bits p  (ideal float, mpmath round-to-nearest; m = p - 1)", family=MONO, fontsize=12)
    ax2.set_ylabel("divergence step n*", family=MONO, fontsize=12)
    ax2.legend(prop=dict(family=MONO, size=11), frameon=False, loc="upper left")
    ax2.set_title("one more bit, one more step (500 seeds per p)", family=SERIF, fontsize=18, loc="left")
    ax2.set_xlim(0, 66)
    ax2.set_ylim(0, 68)
    fig.text(0.05, 0.93, "Measured: steps of truth per mantissa bit", family=SERIF, fontsize=30, color=st["ink"])
    fig.text(0.97, 0.02, STACK, family=MONO, fontsize=9, ha="right")
    p = GALLERY / "divergence_measured.png"
    fig.savefig(p, facecolor=st["bg"])
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "braid"
    if what == "braid":
        for s in sys.argv[2:] or ["paper", "observatory", "riso"]:
            braid(s)
    elif what == "grid":
        for s in sys.argv[2:] or ["paper", "observatory"]:
            grid(s)
    elif what == "measure":
        measure()


def delta(style="paper", proto="A", k=48, nmax=64, mode="linear"):
    """River delta: each row is one seed; every strand is (x_n^format - ref_n), drawn around the row's centre line.
    While a format agrees with the reference its strand is fused to the centre; after divergence it swings
    across the row. Rows sorted by the bfloat16 divergence step. Markers: orbit reached the fixed point 0 exactly."""
    st = STY[style]
    rng = np.random.default_rng(3)
    seeds = rng.choice(len(D[f"seeds{proto}"]), k, replace=False)
    seeds = seeds[np.argsort(D[f"nstar_bfloat16_{proto}_0.1"][seeds], kind="stable")]
    fig = plt.figure(figsize=(22, 30), dpi=250, facecolor=st["bg"])
    ax = fig.add_axes([0.04, 0.04, 0.93, 0.86])
    ax.set_facecolor(st["bg"])
    n = np.arange(nmax + 1)
    amp = 0.3 if mode == "linear" else 0.5
    for i, s in enumerate(seeds):
        y0 = k - 1 - i
        ref = D[f"ref{proto}"][s, : nmax + 1]
        ax.plot([0, nmax], [y0 - (0.45 if mode == "log" else 0), y0 - (0.45 if mode == "log" else 0)], color=st["faint"], lw=0.5, alpha=0.6, zorder=0)
        for f in ["float64", "float32", "float16", "bfloat16"]:
            o = D[f"{f}{proto}"][s, : nmax + 1]
            dev = o - ref
            if mode == "log":
                # |error| on a log2 scale, rising upward: 2^-56 at the baseline, 1 at the top of the row
                dev = np.clip((np.log2(np.maximum(np.abs(dev), 2.0 ** -80)) + 56) / 56, 0, 1) * 0.9 / amp - 0.45 / amp
            ns = int(D[f"nstar_{f}_{proto}_0.1"][s])
            # fused part: thick shared strand
            m = min(ns, nmax)
            ax.plot(n[: m + 1], y0 + amp * dev[: m + 1], color=st["cols"][f], lw=3.2 if f == "bfloat16" else 2.2,
                    alpha=0.35 if mode == "linear" else 0.8, solid_capstyle="round", zorder=1)
            if ns <= nmax:
                ax.plot(n[ns - 1:], y0 + amp * dev[ns - 1:], color=st["cols"][f], lw=0.9, alpha=0.95, zorder=2,
                        solid_joinstyle="round")
            z = np.flatnonzero((o == 0) & (np.arange(len(o)) > 0))
            if len(z) and np.all(o[z[0]:] == 0):
                ax.plot([z[0]], [y0 + amp * dev[z[0]]], marker="x", ms=6, mew=1.4, color=st["cols"][f], zorder=3)
    for f in FMTS:
        ax.axvline(MANT[f], color=st["cols"][f], lw=0.8, ls=(0, (1, 3)), alpha=0.8, zorder=0)
        ax.text(MANT[f], k - 0.2, f"m={MANT[f]}", color=st["cols"][f], family=MONO, fontsize=12, ha="center", va="bottom")
    ax.set_xlim(-0.5, nmax + 0.5)
    ax.set_ylim(-1, k + 0.6)
    ax.axis("off")
    for j in range(0, nmax + 1, 8):
        ax.text(j, -0.9, str(j), family=MONO, fontsize=12, color=st["ink"], ha="center")
    fig.text(0.04, 0.955, "River delta", family=SERIF, fontsize=54, color=st["ink"])
    fig.text(0.04, 0.935, f"{k} seeds of x -> 4x(1-x). Each strand is x_n(format) - x_n(1024-bit reference): fused to the row's "
                          "centre line while the format tells the truth, loose once it doesn't.", family=SERIF, fontsize=17,
             style="italic", color=st["ink"])
    fig.text(0.04, 0.922, "bfloat16 · float16 · float32 · float64 (inks as in the braid). Thick: before |error| > 0.1. "
                          "x: the orbit landed exactly on the fixed point 0 and died there. Rows sorted by bfloat16's split.",
             family=MONO, fontsize=12, color=st["ink"])
    fig.text(0.97, 0.015, STACK, family=MONO, fontsize=10, color=st["ink"], ha="right")
    if mode == "log":
        fig.texts[1].set_text(f"{k} seeds of x -> 4x(1-x). Each strand is log2 |x_n(format) - x_n(1024-bit reference)|, from 2^-56 "
                              "(row baseline) to 1 (row top): every error climbs one bit per step from its own precision floor.")
    p = GALLERY / f"river_delta_{style}{'_log' if mode == 'log' else ''}.png"
    fig.savefig(p, facecolor=st["bg"])
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "delta":
    for s_ in sys.argv[2:] or ["paper", "observatory", "riso"]:
        delta(s_, k=32)
        delta(s_, k=32, mode="log")
