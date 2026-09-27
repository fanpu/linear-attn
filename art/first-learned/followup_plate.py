"""FOLLOW-UP plate: agreement matrices before vs after retuning ResNet-8 and ViT (CPU only).
Same treatment as plate_agreement.py: cream ground, ink cells with alpha = value^2, bold diagonal = seed pairs.
Rows: before (shared SGD recipe) / after (ResNet-8 and ViT retuned). Columns: all 60,000 (step clock),
within class (step clock), within class on the speed-normalised progress clock."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(ROOT, "cache", "followup", "agreement_followup.json")))
CREAM = "#F2EDE1"; INK = "#1E1C1A"; MADDER = "#A3302A"
NAMES = {"mlp_256": "MLP", "cnn": "CNN", "resnetbnb": "ResNet-8", "vit": "ViT", "logreg": "linear"}
archs = A["archs"]; n = len(archs)
plt.rcParams.update({"font.family": "serif", "text.color": INK, "axes.edgecolor": INK})
fig, axs = plt.subplots(2, 3, figsize=(17, 11.6), facecolor=CREAM)
cols = [("step", "AA", "Spearman, all 60,000"), ("step", "AW", "within class"), ("progress", "AW", "within class, speed-normalised clock")]
rows = [("before", "BEFORE  one shared recipe (SGD 0.02, 8 ep); ViT at 97.5–98.1% train acc"),
        ("after", "AFTER  ResNet-8 SGD 0.1 cosine, ViT AdamW 1e-3 cosine; MLP, CNN, ResNet-8, ViT all ≥ 99.6% train acc (linear capped at 92.7%)")]
for r, (setname, rlab) in enumerate(rows):
    for c, (clock, m, title) in enumerate(cols):
        ax = axs[r, c]; ax.set_facecolor(CREAM); M = np.array(A[setname][clock][m])
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=INK, alpha=float(np.clip(v, 0, 1)) ** 2, ec=CREAM, lw=2))
                ax.text(j + 0.5, n - 0.5 - i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                        color=CREAM if v > 0.6 else INK, weight="bold" if i == j else "normal")
        ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect(1)
        ax.set_xticks(np.arange(n) + 0.5); ax.set_xticklabels([NAMES[a] for a in archs], fontsize=10)
        ax.set_yticks(np.arange(n) + 0.5); ax.set_yticklabels([NAMES[a] for a in archs][::-1], fontsize=10)
        ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_title(title, fontsize=12, loc="left")
    fig.text(0.04, 0.965 - r * 0.47, rlab, fontsize=13, color=MADDER if r else INK, weight="bold")
lines = []
for setname in ["before", "after"]:
    for clock in ["step", "progress"]:
        S = A[setname][clock]["summary"]; N = np.array(A[setname][clock]["AN"])
        null = np.mean([N[i, j] for i in range(n) for j in range(n) if i != j])
        lines.append(f"{setname:6s} {clock:8s} clock   within-class: seeds {S['AW']['seeds']:.2f} · linear/MLP/CNN cross {S['AW']['group_cross']:.2f} · "
                     f"ResNet/ViT→group {S['AW']['lanes_to_group']:.2f}    all-60k class-only null {null:.2f}    "
                     f"last-1% shared: seeds {S['AL']['seeds']:.0%}, group {S['AL']['group_cross']:.0%}, ResNet/ViT→group {S['AL']['lanes_to_group']:.0%}")
fig.text(0.04, 0.012, "Diagonal (bold) = seed pairs of one architecture (3 seeds each).\n" + "\n".join(lines),
         fontsize=9, color=INK, linespacing=1.55, family="monospace")
fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.14, wspace=0.28, hspace=0.42)
p = os.path.join(ROOT, "gallery", "followup_agreement_before_after.png"); fig.savefig(p, dpi=200, facecolor=CREAM); print("wrote", p)
