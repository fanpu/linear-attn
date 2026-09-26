import sys; sys.path.insert(0, "/home/fzeng/ml/research/art/one-road")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from atlas import Map
task = sys.argv[1]; metric = sys.argv[2] if len(sys.argv) > 2 else "B"
fig, axs = plt.subplots(2, 3, figsize=(21, 10))
for r, sp in enumerate(("tr", "te")):
    for c, mn in enumerate(("main", "null", "memo")):
        M = Map(task, sp, mn, metric); ax = axs[r, c]
        cols = plt.cm.tab10.colors
        archs = sorted(set(M.runs[i]["arch"] for i in M.run_ids(None)))
        for ri in M.run_ids(None):
            P = M.traj("run", ri); rr = M.runs[ri]
            ax.plot(P[:, 0], P[:, 1], color=cols[archs.index(rr["arch"])], lw=0.8, ls="-" if rr["labels"] == "true" else ":")
            ax.text(P[-1, 0], P[-1, 1], rr["name"], fontsize=6)
        if mn == "null":
            for ri in M.run_ids():
                P = M.traj("perm", ri); ax.plot(P[:, 0], P[:, 1], color="r", lw=0.4, alpha=0.5)
        G = M.points("geo"); ax.plot(G[:, 0], G[:, 1], "k--", lw=0.6)
        ax.plot(*M.X[M.i0, :2], "k^"); ax.plot(*M.X[M.i1, :2], "ko")
        ax.set_title(f"{task} {sp} {mn} stress {np.round(M.stress[:3], 3)}"); ax.set_aspect("equal")
plt.tight_layout(); plt.savefig(f"/home/fzeng/ml/research/art/one-road/diag/{task}_{metric}_all.png", dpi=70)
