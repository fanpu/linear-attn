"""Preview (matplotlib, not a gallery render) of the exact Stage B/C chart and the canyon grids."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "cache", "stageC")
PREV = os.path.join(HERE, "cache", "preview")
c = np.load(os.path.join(C, "coords.npz"))
X, t_ref, t_edge = c["coords"], int(c["t_ref"]), int(c["t_edge"])
lam = np.load(os.path.join(HERE, "cache", "stageB", "log.npz"))
le = lam["evals"][:, 0] * float(lam["eta"]) / 2
T = len(X)
tt = np.arange(T)
stats = {}

fig = plt.figure(figsize=(16, 10))
w = slice(t_ref - 200, t_ref + 200)
ax = fig.add_subplot(2, 3, 1, projection="3d")
Y = X[w]
ax.plot(Y[:, 1], Y[:, 2], Y[:, 0], lw=0.3, c="0.5")
ax.scatter(Y[::2, 1], Y[::2, 2], Y[::2, 0], s=2, c="C0", label="even")
ax.scatter(Y[1::2, 1], Y[1::2, 2], Y[1::2, 0], s=2, c="C3", label="odd")
ax.set(xlabel="pc1", ylabel="pc2", zlabel="u_ref", title=f"exact chart, steps {t_ref - 200}–{t_ref + 199}")
ax.legend(fontsize=7)
ax = fig.add_subplot(2, 3, 2, projection="3d")
Z = X[t_edge:]
sc = ax.scatter(Z[:, 1], Z[:, 2], Z[:, 0], s=0.3, c=le[t_edge:], cmap="RdBu_r", vmin=0.9, vmax=1.1)
ax.set(xlabel="pc1", ylabel="pc2", zlabel="u_ref", title="EoS phase, colour λ₁η/2 (preview map)")
fig.colorbar(sc, ax=ax, shrink=0.6)
ax = fig.add_subplot(2, 3, 3)
ax.plot(tt[w], X[w, 0], lw=0.4, c="k")
ax.set(title="axis 1 = ⟨θ−θ_ref, u_ref⟩", xlabel="step")
for j, name in enumerate(["local", "global"]):
    p = os.path.join(C, f"canyon_{name}.npz")
    if not os.path.exists(p):
        continue
    g = np.load(p)
    V = g["loss"]
    G = V.shape[0]
    stats[name] = {"span": g["span"].tolist(), "loss_min": float(V.min()), "loss_max": float(V.max()),
                   "loss_ref": float(g["loss_ref"]), "argmin": [int(i) for i in np.unravel_index(V.argmin(), V.shape)]}
    ax = fig.add_subplot(2, 3, 4 + j)
    im = ax.imshow(np.log10(V[:, :, G // 2]), origin="lower", aspect="auto", cmap="viridis",
                   extent=[g["ax2"][0], g["ax2"][-1], g["ax1"][0], g["ax1"][-1]], interpolation="nearest")
    sel = (np.abs(tt - t_ref) <= 200) if name == "local" else (tt >= t_edge)
    ax.plot(X[sel, 1], X[sel, 0], lw=0.2, c="w", alpha=0.6)
    ax.set(xlabel="pc1", ylabel="u_ref", title=f"{name}: log10 loss, slice pc2≈0 (+trajectory projected)")
    fig.colorbar(im, ax=ax)
ax = fig.add_subplot(2, 3, 6)
if os.path.exists(os.path.join(C, "canyon_local.npz")):
    g = np.load(os.path.join(C, "canyon_local.npz"))
    V = g["loss"]
    G = V.shape[0]
    ax.plot(g["ax1"], V[:, G // 2, G // 2], "k.-", label="along u_ref")
    ax.plot(g["ax2"], V[G // 2, :, G // 2], "C0.-", label="along pc1")
    ax.plot(g["ax3"], V[G // 2, G // 2, :], "C3.-", label="along pc2")
    ax.set(yscale="log", title="local grid: 1D cuts through the centre", xlabel="offset")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PREV, "stageC_chart.png"), dpi=100)
print(json.dumps(stats, indent=1))
print("extent (max |coord|) EoS:", np.abs(X[t_edge:]).max(0), "local:", np.abs(X[w]).max(0))
