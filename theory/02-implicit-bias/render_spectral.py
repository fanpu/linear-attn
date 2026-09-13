"""Build-on figure: which matrix norm's margin does each optimizer maximize?  Reads cache/spectral.npz.

  figures/spectral.png   (a) spectral-norm margin (fraction of the optimum) vs steps, all optimizers
                         (b) end-of-training table: margin in each norm as a fraction of that norm's optimum
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import style as S
from compute_spectral import norms, OPTS

S.light()
D = np.load("cache/spectral.npz")
X, Y, st = D["X"], D["Y"], D["steps"]
Sn, n, d = X.shape
k = D["W_gd"].shape[2]
NORMS = [("fro", "Frobenius"), ("max", "max entry"), ("spec", "spectral"), ("nuc", "nuclear")]
LAB = {"gd": "GD", "ngd": "normalized GD", "sign": "sign GD", "spec": "spectral descent", "spec_dec": "spectral descent, η∝1/√t",
       "muon_svd": "Muon (exact polar)", "muon_ns": "Muon (Newton–Schulz)", "adam": "Adam (ε = 0)"}
COL = {"gd": S.BLUE, "ngd": "#86b6ef", "sign": S.ORANGE, "adam": S.YELLOW, "spec": S.DEPTH_COLORS[3], "spec_dec": S.DEPTH_COLORS[1],
       "muon_svd": S.MAGENTA, "muon_ns": S.RED}
OWN = {"gd": "fro", "ngd": "fro", "sign": "max", "adam": "max", "spec": "spec", "spec_dec": "spec", "muon_svd": "spec", "muon_ns": "spec"}


def min_margin(W):
    L = np.einsum("...skd,snd->...snk", W, X)
    ly = np.take_along_axis(L, np.broadcast_to(Y[..., None], L.shape[:-1] + (1,)), -1)
    M = np.where(np.eye(k)[Y] > 0, np.inf, ly - L)
    return M.min((-2, -1))


opt = {nm: 1 / norms(D["ref_" + nm])[nm] for nm, _ in NORMS}  # optimal normalized margins, per dataset
ratio = {}
for o in OPTS:
    W = D["W_" + o]
    m = min_margin(W)  # T, S
    nr = norms(W)
    ratio[o] = {nm: m / np.maximum(nr[nm], 1e-300) / opt[nm][None] for nm, _ in NORMS}

fig = plt.figure(figsize=(15.5, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.22)
ax = fig.add_subplot(gs[0, 0])
order = ["gd", "ngd", "sign", "adam", "spec_dec", "spec", "muon_svd", "muon_ns"]
ends = []
for o in order:
    r = ratio[o]["spec"]
    med = np.median(r, 1)
    ok = st >= 10
    lw = 2.4 if OWN[o] == "spec" else 1.6
    ax.semilogx(st[ok], med[ok], color=COL[o], lw=lw, alpha=1 if OWN[o] == "spec" else 0.85)
    ends.append((med[-1], o))
ax.axhline(1, color=S.INK, lw=1, ls=(0, (4, 3)))
ends.sort()
prev = -1
for v, o in ends:
    yy = max(v, prev + 0.035)
    prev = yy
    ax.annotate(f"{LAB[o]}  {v:.3f}", (st[-1], v), xytext=(st[-1] * 1.6, yy), textcoords="data", va="center", fontsize=9.5, color=S.INK2,
                arrowprops=dict(arrowstyle="-", color=S.AXIS, lw=0.8))
ax.set_xlim(10, st[-1] * 60)
ax.set_ylim(0.45, 1.04)
ax.set_xlabel("steps t")
ax.set_ylabel("spectral-norm margin / best possible")
ax.set_title("(a) Spectral-norm margin of a linear 4-class classifier")
ax.text(12, 1.012, "1 = spectral-norm max-margin solution (cvxpy)", fontsize=9.5, color=S.INK2)

# ---- table
bx = fig.add_subplot(gs[0, 1])
rows = order
M = np.array([[np.median(ratio[o][nm][-1]) for nm, _ in NORMS] for o in rows])
cmap = LinearSegmentedColormap.from_list("seq", ["#f4f1ea", "#cfc8ea", "#8f80d6", "#4a3aa7", "#241b63"])
bx.imshow(np.clip((M - 0.5) / 0.5, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i, o in enumerate(rows):
    for j, (nm, _) in enumerate(NORMS):
        v = M[i, j]
        bx.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10.5, color="white" if v > 0.82 else S.INK,
                fontweight="bold" if OWN[o] == nm else "normal")
        if OWN[o] == nm:
            bx.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=S.ORANGE, lw=2.2))
bx.set_xticks(range(len(NORMS))); bx.set_xticklabels([lab for _, lab in NORMS])
bx.set_yticks(range(len(rows))); bx.set_yticklabels([LAB[o] for o in rows])
bx.xaxis.tick_top()
bx.grid(False)
for sp in bx.spines.values():
    sp.set_visible(False)
bx.set_title(f"(b) After {st[-1]:.0e} steps: margin in each norm / that norm's optimum", pad=28)
bx.text(1.5, len(rows) - 0.35, "boxed: the norm whose steepest descent the optimizer is.  median over 6 datasets", ha="center", va="top", fontsize=9.5, color=S.INK2)
fig.savefig("figures/spectral.png", bbox_inches="tight")
print("wrote spectral.png")
for o in rows:
    print(f"{o:10s}", " ".join(f"{nm}:{np.median(ratio[o][nm][-1]):.4f} (min {ratio[o][nm][-1].min():.4f})" for nm, _ in NORMS))
