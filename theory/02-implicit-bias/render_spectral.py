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
LAB = {"gd": "GD", "ngd": "normalized GD", "sign": "sign GD", "spec": "spectral descent", "spec_dec": "spectral descent, η = 0.1/√t",
       "muon_svd": "Muon (exact polar)", "muon_ns": "Muon (Newton–Schulz)", "adam": "Adam (ε = 0)"}
COL = {"gd": S.BLUE, "ngd": "#86b6ef", "sign": S.ORANGE, "adam": S.YELLOW, "spec": S.DEPTH_COLORS[3], "spec_dec": S.DEPTH_COLORS[1],
       "muon_svd": S.MAGENTA, "muon_ns": S.RED}
OWN = {"gd": "fro", "ngd": "fro", "sign": "max", "adam": "max", "spec": "spec", "spec_dec": "spec", "muon_svd": "spec", "muon_ns": "spec"}


def min_margin(W):
    L = np.einsum("...skd,snd->...snk", W, X)
    ly = np.take_along_axis(L, np.broadcast_to(Y[..., None], L.shape[:-1] + (1,)), -1)
    M = np.where(np.eye(k)[Y] > 0, np.inf, ly - L)
    return M.min((-2, -1))


import os
NSV = {}
if os.path.exists("cache/ns.npz"):
    NS = np.load("cache/ns.npz")
    for v, lab in [("ns10", "Muon (10 NS steps)"), ("ns20", "Muon (20 NS steps)"), ("ns5_nomom", "NS spectral descent, no momentum")]:
        D_W = NS["W_" + v]
        NSV[v] = D_W
        LAB[v] = lab
        OWN[v] = "spec"
    COL.update({"ns10": "#f08c8c", "ns20": "#f6b8b8", "ns5_nomom": "#b33a3a"})
    LAB["muon_ns"] = "Muon (5 NS steps)"
opt = {nm: 1 / norms(D["ref_" + nm])[nm] for nm, _ in NORMS}  # optimal normalized margins, per dataset
ratio = {}
for o in list(OPTS) + list(NSV):
    W = D["W_" + o] if o in OPTS else NSV[o]
    m = min_margin(W)  # T, S
    nr = norms(W)
    ratio[o] = {nm: m / np.maximum(nr[nm], 1e-300) / opt[nm][None] for nm, _ in NORMS}

fig = plt.figure(figsize=(19, 6.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 0.62], wspace=0.62)
ax = fig.add_subplot(gs[0, 0])
order = ["gd", "ngd", "sign", "adam", "spec_dec", "spec", "muon_svd", "muon_ns"] + [v for v in ["ns10", "ns20"] if v in NSV]
ends = []
PLOT_A = ["gd", "sign", "spec_dec", "spec", "muon_svd", "muon_ns", "ns20"]
for o in [o for o in order if o in PLOT_A]:
    r = ratio[o]["spec"]
    med = np.median(r, 1)
    ok = st >= 10
    lw = 2.4 if OWN[o] == "spec" else 1.6
    ax.semilogx(st[ok], med[ok], color=COL[o], lw=lw, alpha=1 if OWN[o] == "spec" else 0.85)
    ends.append((med[-1], o))
ax.axhline(1, color=S.INK, lw=1, ls=(0, (4, 3)))
ends.sort(key=lambda e: -e[0])
for i_, (v, o) in enumerate(ends):
    yy = 0.735 - 0.04 * i_
    ax.plot([1.2e4, 3.2e4], [yy, yy], color=COL[o], lw=2.4 if OWN[o] == "spec" else 1.6)
    ax.text(4.2e4, yy, f"{LAB[o]}", va="center", fontsize=9.5, color=S.INK2)
ax.text(1.2e4, 0.735 + 0.045, "(end values: table b)", va="center", fontsize=9, color=S.MUTED)
ax.set_xlim(10, 4e6)
ax.set_ylim(0.42, 1.04)
ax.set_xlabel("steps t")
ax.set_ylabel("spectral-norm margin / best possible")
ax.set_title("(a) Spectral-norm margin of a linear 4-class classifier")
ax.text(12, 1.012, "1 = spectral-norm max-margin solution (cvxpy)", fontsize=9.5, color=S.INK2)

# ---- table
bx = fig.add_subplot(gs[0, 1])
rows = order
M = np.array([[np.median(ratio[o][nm][-1]) for nm, _ in NORMS] for o in rows])
cmap = LinearSegmentedColormap.from_list("seq", ["#f4f1ea", "#cfc8ea", "#8f80d6", "#4a3aa7", "#241b63"])
bx.imshow(np.clip((M - 0.7) / 0.3, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i, o in enumerate(rows):
    for j, (nm, _) in enumerate(NORMS):
        v = M[i, j]
        bx.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10.5, color="white" if v > 0.9 else S.INK,
                fontweight="bold" if OWN[o] == nm else "normal")
        if OWN[o] == nm:
            bx.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=S.ORANGE, lw=2.2))
bx.set_xticks(range(len(NORMS))); bx.set_xticklabels([lab for _, lab in NORMS])
bx.set_yticks(range(len(rows))); bx.set_yticklabels([LAB[o] for o in rows])
bx.xaxis.tick_top()
bx.grid(False)
for sp in bx.spines.values():
    sp.set_visible(False)
bx.set_title("(b) After $10^6$ steps: margin / that norm's optimum", pad=28)
bx.text(1.5, len(rows) - 0.35, "boxed: the norm whose steepest descent the method is\nmedian over 6 datasets", ha="center", va="top", fontsize=9.5, color=S.INK2)
# ---- (c) mechanism: Newton-Schulz maps each singular value ratio s = sigma_i/sigma_1 to phi(s), not to 1
from compute_spectral import ns5
cx = fig.add_subplot(gs[0, 2])
ss = np.logspace(-3, 0, 400)


def phi(sv, steps):
    out = []
    for x in sv:
        G = np.zeros((1, 2, 3)); G[0, 0, 0] = 1.0; G[0, 1, 1] = x
        out.append(abs(ns5(G, steps)[0][1, 1]))  # diagonal input stays diagonal: entry (1,1) is the image of x
    return np.array(out)


cx.axhspan(0.68, 1.13, color=S.GRID, alpha=0.6, lw=0)
cx.semilogx(ss, phi(ss, 5), color=S.RED, lw=2.2, label="5 Newton–Schulz steps")
cx.axhline(1, color=S.INK, lw=1, ls=(0, (4, 3)))
cx.text(1.3e-3, 1.02, "exact polar factor", fontsize=9.5, color=S.INK2, va="bottom")
if NSV:
    si = NS["svin_ns5"][-1]  # S x k, descending
    rat = (si[:, 1:3] / si[:, :1]).ravel()
    cx.scatter(rat, phi(rat, 5), s=30, color=S.RED, edgecolor=S.PAPER, lw=1.2, zorder=5)
    cx.text(0.97, 0.06, "dots: singular-value ratios of the actual\nMuon update after $10^6$ steps", transform=cx.transAxes, ha="right", fontsize=9, color=S.INK2)
cx.set_ylim(0.3, 1.3)
cx.set_xlabel(r"$\sigma_i / \sigma_1$ of the matrix fed to Newton–Schulz")
cx.set_ylabel("singular value after orthogonalization")
cx.set_title("(c) Newton–Schulz maps singular values\ninto a band, not onto 1", fontsize=12)
cx.text(1.3e-3, 0.66, "shaded: 0.68–1.13", fontsize=9, color=S.INK2, va="top")
fig.savefig("figures/spectral.png", bbox_inches="tight")
print("wrote spectral.png")
for o in rows:
    print(f"{o:10s}", " ".join(f"{nm}:{np.median(ratio[o][nm][-1]):.4f} (min {ratio[o][nm][-1].min():.4f})" for nm, _ in NORMS))
