"""Quick contact sheet of a film cache (diagnostic, not gallery)."""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fn = sys.argv[1]; d = np.load(fn); gens = [int(x) for x in sys.argv[2].split(",")]
ev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
regs = ["replace", "anchored", "accumulate"]
fig, ax = plt.subplots(3, len(gens), figsize=(1.6 * len(gens), 5), facecolor="k")
for i, r in enumerate(regs):
    D = d[f"{r}/display"]
    for j, g in enumerate(gens):
        a = ax[i, j]; a.set_facecolor("k"); a.set_xticks([]); a.set_yticks([])
        X = D[g // ev].astype(float)
        a.scatter(X[:, 0], X[:, 1], s=0.3, c="w", lw=0, alpha=0.5); a.set_xlim(-1.45, 1.45); a.set_ylim(-1.45, 1.45)
        a.set_title(f"{r[:3]} g{g} sw{d[f'{r}/sw2'][g]:.2f}", color="w", fontsize=6)
plt.savefig(fn.replace("cache/", "logs/").replace(".npz", ".png"), dpi=90, facecolor="k"); print("ok")
