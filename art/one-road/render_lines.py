"""Becher grid: loss along the straight line init -> trained solution (Goodfellow et al. 2015)."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from atlas import PAPER, INK, RED, FAINT, RULE, TASK_TITLE, ARCH_NAME, save
from models import IMAGE_ARCHS, MODADD_ARCHS

ROOT = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(ROOT, "cache", "lineloss.json")))
A = np.array(J["alpha"])
tasks = ["mnist", "fashion", "cifar", "modadd"]
ncol = 8
fig = plt.figure(figsize=(18, 11.5))
L, R, T, B = 0.07, 0.98, 0.86, 0.1
cw = (R - L) / ncol; rh = (T - B) / 4
for r, t in enumerate(tasks):
    archs = IMAGE_ARCHS if t != "modadd" else MODADD_ARCHS
    for c, a in enumerate(archs):
        key = f"{t}/{a}_{'adamw' if t == 'modadd' and a != 'logreg' else 'adam'}_true_s0"
        if key not in J["runs"]:
            continue
        ax = fig.add_axes([L + c * cw + 0.006, B + (3 - r) * rh + 0.02, cw - 0.012, rh - 0.045])
        ax.set_facecolor(PAPER)
        for x in (0, 1):
            ax.axvline(x, color=RULE, lw=0.8, zorder=1)
        ax.plot(A, np.clip(J["runs"][key]["te"], 1e-4, None), color=RED, lw=1.0, zorder=3)
        ax.plot(A, np.clip(J["runs"][key]["tr"], 1e-4, None), color=INK, lw=1.0, zorder=4)
        ax.set_yscale("log"); ax.set_ylim(1e-4, 300); ax.set_xlim(-0.25, 1.25)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        ax.tick_params(labelsize=6.5, width=0.5, length=2, colors=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["init", "trained"] if r == 3 or t == "modadd" else [])
        if c > 0: ax.set_yticklabels([])
        else: ax.set_ylabel(TASK_TITLE[t], fontsize=10)
        ax.set_title(ARCH_NAME[a] + (", AdamW" if t == "modadd" and a != "logreg" else ""), fontsize=7.5, style="italic", pad=3)
fig.text(0.5, 0.955, "THE STRAIGHT LINE", ha="center", fontsize=22)
fig.text(0.5, 0.92, "cross-entropy along the straight line in weight space from each network's initialisation to its trained "
         "weights (Goodfellow, Vinyals & Saxe 2015); identical axes in every cell", ha="center", fontsize=10, style="italic")
fig.text(L, 0.04, "Ink: 1,000 training examples. Red: 1,000 held-out examples. Log loss axis 1e-4 to 300; alpha from -0.25 to 1.25. "
         "Seed 0, Adam (AdamW with weight decay 1 for the grokking transformers and MLP). BatchNorm statistics interpolated with the weights.",
         fontsize=7.5, color=FAINT)
save(fig, "straight_line_becher.png", dpi=250)
