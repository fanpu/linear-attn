"""Ink-on-cream table plate: how much do runs agree on learning order? (CPU only)"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(ROOT, "cache", "agreement.json")))
CREAM = "#F2EDE1"; INK = "#1E1C1A"; MADDER = "#A3302A"
NAMES = {"mlp_256": "MLP", "cnn": "CNN", "resnetbnb": "ResNet-8", "vit": "ViT", "logreg": "linear"}
ORDER = ["logreg", "mlp_256", "cnn", "vit", "resnetbnb"]
ix = [A["archs"].index(a) for a in ORDER]; archs = ORDER
AA = np.array(A["AA"])[np.ix_(ix, ix)]; AW = np.array(A["AW"])[np.ix_(ix, ix)]
plt.rcParams.update({"font.family": "serif", "text.color": INK, "axes.edgecolor": INK})
fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), facecolor=CREAM, gridspec_kw=dict(width_ratios=[1, 1.1]))
for ax, M, title in [(axs[0], AA, "Spearman of learning order, all 60,000"),
                     (axs[1], AW, "same, within class (class effect removed)")]:
    ax.set_facecolor(CREAM)
    n = len(archs)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=INK, alpha=float(np.clip(v, 0, 1)) ** 2, ec=CREAM, lw=2))
            ax.text(j + 0.5, n - 0.5 - i, f"{v:.2f}", ha="center", va="center", fontsize=13,
                    color=CREAM if v > 0.6 else INK, weight="bold" if i == j else "normal")
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect(1)
    ax.set_xticks(np.arange(n) + 0.5); ax.set_xticklabels([NAMES.get(a, a) for a in archs])
    ax.set_yticks(np.arange(n) + 0.5); ax.set_yticklabels([NAMES.get(a, a) for a in archs][::-1])
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(title, fontsize=12, loc="left")
nulls = [f"class-only null (order by the other run's class means): {A['class_only_null']['cross_arch']:.2f}"]
nulls += [f"shuffled-label memorisation order vs true order ({k[5:].replace('_shuf_s0','')}): {v:+.2f}"
          for k, v in A.items() if k.startswith("null_")]
b = A["baseline_nearest_centroid_margin"]
nulls += ["pixel-space nearest-class-mean margin, no network: " + ", ".join(f"{NAMES.get(k, k).split(chr(10))[0]} {v:.2f}" for k, v in b["per_arch"].items())]
nulls += [f"last-learned 1% shared: seeds {A['last1pct_overlap']['same_arch']:.0%}, architectures "
          f"{A['last1pct_overlap']['cross_arch']:.0%} (chance 1%)"]
fig.text(0.06, 0.015, "Diagonal (bold) = different seeds of one architecture.   " + nulls[0] + "\n" + "\n".join(nulls[1:]),
         fontsize=9, color=INK, linespacing=1.5)
fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.22, wspace=0.3)
p = os.path.join(ROOT, "gallery", "agreement_table.png"); fig.savefig(p, dpi=200, facecolor=CREAM); print("wrote", p)
