"""Coordinate-check figure (reads cache/coord_check.json)."""
import json, os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(f"{HERE}/cache/coord_check.json"))
W = np.array(d["widths"])
S.use("light")
STEPR = LinearSegmentedColormap.from_list("t", ["#f3b695", S.RICH, "#7a2408"])
cols = [("attn0", "attention output, block 1"), ("mlp0", "MLP output, block 1"), ("mlp3", "MLP output, block 4"),
        ("logits", "output logits")]
fig, axs = plt.subplots(2, len(cols), figsize=(15, 6.6), sharex=True, gridspec_kw=dict(hspace=0.38, wspace=0.3))
for j, (tap, title) in enumerate(cols):
    lo, hi = np.inf, 0
    for i, param in enumerate(["sp", "mup"]):
        ax = axs[i, j]
        for t in range(1, d["steps"] + 1):
            v = np.array([[d["runs"][f"{param}_{w}_{s}"]["drms"][tap][t] for s in range(3)] for w in W])
            mu = v.mean(1)
            c = STEPR((t - 1) / (d["steps"] - 1))
            ax.plot(W, mu, "-o", color=c, ms=4, lw=1.6, mec=S.PAPER, mew=0.8)
            lo, hi = min(lo, mu.min()), max(hi, mu.max())
            if j == 0 and i == 0:
                ax.text(W[-1] * 1.12, mu[-1], f"t={t}", color=c, fontsize=8.5, va="center")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.minorticks_off()
        ax.set_xticks(W); ax.set_xticklabels([f"{w:,}" for w in W], fontsize=9)
        if i == 1:
            ax.set_xlabel("width d")
        if j == 0:
            ax.set_ylabel(("standard param. (SP)" if param == "sp" else "μP") + "\nRMS of  $x_t - x_0$", color=S.INK)
        if i == 0:
            ax.set_title(title, fontsize=11.5)
    for i in range(2):
        axs[i, j].set_ylim(lo / 1.6, hi * 1.6)
fig.text(0.5, 0.99, "", ha="center")
S.save(fig, f"{HERE}/figures/coord_check.png")
