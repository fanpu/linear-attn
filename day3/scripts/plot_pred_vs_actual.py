"""Plot scaling-law predicted L(N, D) vs actual final loss across model sizes.

Left: loss vs non-embedding N (log x), actual seeds + both predictions.
Right: parity plot, predicted vs actual, with the y = x line.

    uv run python scripts/plot_pred_vs_actual.py [--out pred_vs_actual.png]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

SIZES = ["30M", "60M", "125M"]
N_TOTAL = np.array([85.7e6, 140.9e6, 204.6e6])
N_NONEMB = np.array([34.1e6, 63.7e6, 127.4e6])
PRED_TOTAL = np.array([4.24, 3.81, 3.48])
PRED_NONEMB = np.array([4.54, 4.02, 3.58])
ACTUAL = np.array([
    [3.7418, 3.7465],  # s0, s1
    [3.4870, 3.4886],
    [3.2558, 3.2616],
])
ACTUAL_AVG = ACTUAL.mean(axis=1)

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#8a8980"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="pred_vs_actual.png")
    args = p.parse_args()

    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "axes.grid": True, "grid.color": "#e6e5df",
        "grid.linewidth": 0.8,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: loss vs model size -------------------------------------------
    x = N_NONEMB
    ax.plot(x, PRED_TOTAL, "-o", color=BLUE, lw=2, ms=8, label="Predicted, total N")
    ax.plot(x, PRED_NONEMB, "-s", color=ORANGE, lw=2, ms=8, label="Predicted, non-emb N")
    # seed spread (<0.006) is smaller than the marker, so plot the average only
    ax.plot(x, ACTUAL_AVG, "-D", color=INK, lw=2, ms=8, label="Actual (2-seed avg)")
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{s}\n({n / 1e6:.1f}M)" for s, n in zip(SIZES, x)])
    ax.minorticks_off()
    ax.set_xlabel("Model size (non-embedding params)")
    ax.set_ylabel("Final loss")
    ax.set_title("Loss vs model size", loc="left", color=INK)
    ax.legend(frameon=False)

    # --- Right: parity plot --------------------------------------------------
    lo = min(ACTUAL.min(), PRED_TOTAL.min(), PRED_NONEMB.min()) - 0.1
    hi = max(ACTUAL.max(), PRED_TOTAL.max(), PRED_NONEMB.max()) + 0.1
    bx.plot([lo, hi], [lo, hi], "--", color=MUTED, lw=1, label="y = x")
    bx.scatter(ACTUAL_AVG, PRED_TOTAL, color=BLUE, marker="o", s=64,
               edgecolor="white", lw=2, zorder=3, label="Total N")
    bx.scatter(ACTUAL_AVG, PRED_NONEMB, color=ORANGE, marker="s", s=64,
               edgecolor="white", lw=2, zorder=3, label="Non-emb N")
    for s, a, pt, pn in zip(SIZES, ACTUAL_AVG, PRED_TOTAL, PRED_NONEMB):
        bx.annotate(s, (a, pn), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
        bx.annotate(f"+{pt - a:.2f}", (a, pt), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=MUTED)
    bx.set_xlim(lo, hi)
    bx.set_ylim(lo, hi)
    bx.set_aspect("equal")
    bx.set_xlabel("Actual loss (seed avg)")
    bx.set_ylabel("Predicted L(N, D)")
    bx.set_title("Predicted vs actual", loc="left", color=INK)
    bx.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for s, a, pt, pn in zip(SIZES, ACTUAL_AVG, PRED_TOTAL, PRED_NONEMB):
        print(f"{s:>5}: actual {a:.4f}  pred_total {pt:.2f} ({pt - a:+.3f})  "
              f"pred_nonemb {pn:.2f} ({pn - a:+.3f})")


if __name__ == "__main__":
    main()
