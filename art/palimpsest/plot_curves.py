"""The measurement, drawn plainly. One row per network/optimiser, one column per octave band.

red    : A-specific ghost in the A->B run, g_A - mean(g_decoys), divided by its step-0 value
         (1 = all of A's letters in that band still in the output, 0 = none)
sepia  : fraction of B written in that band, 1 + g_B
grey   : the same A-specific quantity in the C->B (dashed) and none->B (dotted) runs, divided by
         the A->B run's step-0 value: what the metric reads on networks that never saw A
band   : +-2 sd of a single decoy page around the decoy mean, same normalisation (chance level)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VEL = "#eee2c8"
INK = "#342216"
RED = "#961e16"
SEP = "#8a6a44"


def main():
    root = sys.argv[1]
    groups = sys.argv[2].split(",")
    suffix = sys.argv[3]
    out = sys.argv[4]
    lo_min = float(sys.argv[5]) if len(sys.argv) > 5 else 4
    S = {s["name"]: s for s in json.load(open(os.path.join(root, "summary.json")))}
    bands = S[groups[0] + "_AB" + suffix]["bands"]
    keep = [k for k, b in enumerate(bands) if float(b.split("-")[0]) >= lo_min]
    fig, axs = plt.subplots(len(groups), len(keep), figsize=(1.55 * len(keep) + 0.8, 1.15 * len(groups) + 0.7),
                            sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(VEL)
    for r, g in enumerate(groups):
        A = S[g + "_AB" + suffix]
        C = S.get(g + "_CB" + suffix)
        N = S.get(g + "_noneB" + suffix)
        x = np.maximum(np.array(A["steps"], float), 0.5)
        L = np.array(A["letters"]); L0 = L[0]
        Bw = 1 + np.array(A["gB"])
        sd = np.array(A["gD_std"])
        for j, k in enumerate(keep):
            ax = axs[r, j]
            ax.set_facecolor(VEL)
            ax.fill_between(x, -2 * sd[:, k] / L0[k], 2 * sd[:, k] / L0[k], color=INK, alpha=0.10, lw=0)
            for X, ls in ((C, (0, (3, 2))), (N, (0, (1, 1.5)))):
                if X is not None:
                    ax.plot(np.maximum(np.array(X["steps"], float), 0.5), np.array(X["letters"])[:, k] / L0[k],
                            color=INK, lw=0.7, ls=ls, alpha=0.7)
            ax.plot(x, Bw[:, k], color=SEP, lw=1.1)
            ax.plot(x, L[:, k] / L0[k], color=RED, lw=1.5)
            ax.set_xscale("log")
            ax.axhline(0, color=INK, lw=0.3)
            ax.axhline(1, color=INK, lw=0.3, alpha=0.4)
            ax.set_ylim(-0.3, 1.3)
            ax.set_xlim(0.5, 2500)
            ax.tick_params(colors=INK, labelsize=5.5, length=2)
            for s in ax.spines.values():
                s.set_color(INK); s.set_linewidth(0.4)
        lab = g.replace("siren30", "SIREN").replace("ff32", "FF σ32").replace("ff8", "FF σ8").replace("_w", "\nwidth ").replace("_adam", "\nAdam").replace("_sgd", "\nSGD")
        axs[r, 0].set_ylabel(lab, color=INK, fontsize=6.5, rotation=0, ha="right", va="center", family="serif")
    for j, k in enumerate(keep):
        axs[0, j].set_title(f"{bands[k]} cycles/image", color=INK, fontsize=7, family="serif", style="italic")
    for ax in axs[-1]:
        ax.set_xlabel("step of training on B", color=INK, fontsize=6, family="serif")
    fig.text(0.01, 0.005, "red: page A's letters left in the output (A-specific, 1 = all).  sepia: page B written.  "
             "grey dashed / dotted: the red measure on nets that held C / nothing.  shading: ±2 sd of a decoy page.",
             color=INK, fontsize=6, family="serif", style="italic")
    fig.tight_layout(pad=0.4, rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=220, facecolor=VEL)
    print(out)


if __name__ == "__main__":
    main()
