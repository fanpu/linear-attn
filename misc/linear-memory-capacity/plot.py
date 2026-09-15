#!/usr/bin/env python3
"""
plot.py: results/metrics.json + by_age.json -> results/capacity.png

  (a) relative retrieval error vs n/d, with the random-key predictions
  (b) top-1 codebook accuracy vs n/d
  (c) additive accuracy vs n / (d^2 / 2 ln V): the decoding capacity scale
  (d) accuracy, (e) signal <Sk_j, v_j>, (f) crosstalk energy vs age (writes after the item),
      at n = 2d, d = 64
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

RES = Path(__file__).parent / "results"
COLOR = {"additive": "#2a78d6", "delta": "#eb6834", "lstsq": "#1baf7a"}
LABEL = {"additive": "additive (linear attn)", "delta": "delta rule, β=1", "lstsq": "least squares (best linear)"}
STYLE = {32: (":", "o"), 64: ("--", "s"), 128: ("-", "^")}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def line(ax, x, y, method, d, **kw):
    ls, mk = STYLE[d]
    ax.plot(x, y, color=COLOR[method], linestyle=ls, marker=mk, markersize=4.5, linewidth=2,
            markeredgecolor=SURFACE, markeredgewidth=1, **kw)


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", fontsize=9,
                color=INK, va="center", bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=color, lw=1.2))


def main():
    meta = json.loads((RES / "metrics.json").read_text())
    rows, vocab = meta["rows"], meta["config"]["vocab"]
    by_age = json.loads((RES / "by_age.json").read_text())
    dims = sorted({r["d"] for r in rows})

    def series(method, d, key):
        rs = sorted((r for r in rows if r["method"] == method and r["d"] == d), key=lambda r: r["n"])
        return np.array([r["n"] for r in rs]), np.array([r[key] for r in rs])

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), facecolor=SURFACE)
    (ax_err, ax_acc, ax_scale), (ax_age, ax_sig, ax_noise) = axes

    # (a) error vs n/d
    style_axes(ax_err, "(a) Relative retrieval error  ‖Sk − v‖² / ‖v‖²")
    r = np.linspace(0.01, 4, 200)
    ax_err.plot(r, r, color=MUTED, lw=6, alpha=0.18, solid_capstyle="round")
    ax_err.plot(r, np.maximum(0, 1 - 1 / r), color=MUTED, lw=6, alpha=0.18, solid_capstyle="round")
    ax_err.text(3.05, 2.65, "theory (n−1)/d", color=MUTED, fontsize=8.5)
    ax_err.text(2.9, 0.42, "theory 1 − d/n", color=MUTED, fontsize=8.5)
    for m in COLOR:
        for d in dims:
            n, e = series(m, d, "err")
            line(ax_err, n / d, e, m, d)
    ax_err.axvline(1, color=MUTED, lw=1, ls=(0, (2, 3)))
    ax_err.set_xlabel("items stored / key dim  (n / d)", color=MUTED)
    ax_err.set_ylim(-0.1, 4.2)
    ax_err.text(0.12, 3.95, "delta has the lowest error of the lossy rules\n"
                "but the worst accuracy in (b): it forgets old\n"
                "items (readout → 0) instead of adding noise",
                fontsize=8.5, color=MUTED, va="top")

    # (b) accuracy vs n/d
    style_axes(ax_acc, f"(b) Top-1 recall, decoded against a {vocab}-entry codebook")
    for m in COLOR:
        for d in dims:
            n, a = series(m, d, "acc")
            line(ax_acc, n / d, a, m, d)
    ax_acc.axvline(1, color=MUTED, lw=1, ls=(0, (2, 3)))
    ax_acc.set_xlabel("items stored / key dim  (n / d)", color=MUTED)
    ax_acc.set_ylim(0, 1.1)
    n, a = series("additive", 128, "acc")
    label_end(ax_acc, 4, a[-1], "additive", COLOR["additive"], dy=-12)
    n, a = series("lstsq", 128, "acc")
    label_end(ax_acc, 4, a[-1], "lstsq", COLOR["lstsq"], dy=7)
    n, a = series("delta", 32, "acc")
    label_end(ax_acc, 1.6, 0.14, "delta", COLOR["delta"])
    ax_acc.text(0.1, 0.3, "no collapse in n/d:\nbigger d holds on longer", fontsize=8.5, color=MUTED)

    # (c) additive on the decoding scale
    cap = lambda d: d * d / (2 * np.log(vocab))
    style_axes(ax_scale, "(c) Additive recall collapses on n / (d² / 2 ln V)")
    for d in dims:
        n, a = series("additive", d, "acc")
        line(ax_scale, n / cap(d), a, "additive", d)
    ax_scale.axvline(1, color=MUTED, lw=1, ls=(0, (2, 3)))
    ax_scale.set_xlabel("n / (d² / 2 ln V)", color=MUTED)
    ax_scale.set_ylim(0, 1.04)
    ax_scale.text(1.08, 0.93, "crosstalk per codebook direction ~ √n / d\n"
                  "must stay below 1/√(2 ln V) for argmax to win\n"
                  "(d = 128 only reaches 0.5 on this axis)",
                  fontsize=8.5, color=MUTED, va="top")

    # (d, e, f) by age at n = 2d
    d0 = 64
    ages = [x for x in by_age if x["d"] == d0 and x["ratio"] == 2.0]
    w = 5  # 5-item moving average; 30 trials per position is noisy

    def smooth(x, y):
        return np.asarray(x[w // 2: len(x) - w // 2]) / d0, np.convolve(y, np.ones(w) / w, mode="valid")

    for ax, key, title, ylabel in (
        (ax_age, "acc", f"(d) Recall vs age at n = 2d  (d = {d0})", "top-1 accuracy"),
        (ax_sig, "sig", "(e) Signal left on the stored value  ⟨Sk, v⟩", "signal"),
        (ax_noise, "noise", "(f) Crosstalk  ‖Sk − ⟨Sk, v⟩ v‖²", "noise energy"),
    ):
        style_axes(ax, title)
        for x in ages:
            ax.plot(*smooth(x["age"], x[key]), color=COLOR[x["method"]], lw=2)
        ax.set_xlabel("writes after the item / d  (0 = most recent)", color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)

    ax_age.set_ylim(0, 1.04)
    label_end(ax_age, 0.4, 0.99, "delta", COLOR["delta"], dy=-14)
    label_end(ax_age, 1.5, 0.94, "additive", COLOR["additive"], dy=-16)
    label_end(ax_age, 1.5, 1.0, "lstsq", COLOR["lstsq"], dy=10)

    a = np.linspace(0, 2, 100)
    ax_sig.plot(a, np.exp(-a), color=MUTED, lw=6, alpha=0.18, zorder=0)
    ax_sig.text(0.3, 0.45, "theory e^(−age/d)", color=MUTED, fontsize=8.5)
    ax_sig.text(1.22, 0.56, "lstsq: d/n = 0.5", color=MUTED, fontsize=8.5)
    ax_sig.set_ylim(0, 1.15)

    ax_noise.axhline((2 * d0 - 1) / d0, color=MUTED, lw=6, alpha=0.18, zorder=0)
    ax_noise.text(0.05, 2.1, "theory (n−1)/d", color=MUTED, fontsize=8.5)
    ax_noise.set_ylim(0, 2.4)
    ax_noise.text(0.62, 1.45, "delta writes the newest item exactly;\n"
                  "its noise grows while its signal decays,\n"
                  "so old items drown", color=MUTED, fontsize=8.5, va="top")

    ax_acc.set_ylabel("top-1 accuracy", color=MUTED)
    ax_scale.set_ylabel("top-1 accuracy", color=MUTED)
    ax_err.set_ylabel("relative error", color=MUTED)

    method_h = [Line2D([], [], color=COLOR[m], lw=2, label=LABEL[m]) for m in COLOR]
    dim_h = [Line2D([], [], color=MUTED, ls=STYLE[d][0], marker=STYLE[d][1], lw=1.5, ms=4.5, label=f"d = {d}")
             for d in dims]
    fig.legend(handles=method_h + dim_h, loc="upper center", ncol=6, frameon=False, fontsize=9.5,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Linear associative memory S ∈ ℝ^(d×d): random unit keys, unit values",
                 y=0.955, fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RES / "capacity.png", dpi=150, facecolor=SURFACE)
    print(f"wrote {RES / 'capacity.png'}")


if __name__ == "__main__":
    main()
