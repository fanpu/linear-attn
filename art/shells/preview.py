"""M1 previews (matplotlib, not gallery pieces).  Reads only cache/.
  python preview.py cache/vol/resnet20_final_random_g27.npz
writes cache/preview/<name>_{slices,atlas,levelsets}.png
Declared: log10(loss) on matplotlib 'magma' (perceptually uniform sequential), nearest-neighbour
pixels (no interpolation), level outlines at the analysis levels in a fixed 3-colour order."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib import CACHE, LL_ROOT
from analyze import LEVELS

path = sys.argv[1]
v = np.load(path); meta = json.loads(str(v["meta"]))
loss, A, B, C = v["loss"], v["a"], v["b"], v["c"]
name = os.path.basename(path).replace(".npz", "")
od = os.path.join(CACHE, "preview"); os.makedirs(od, exist_ok=True)
lg = np.log10(np.clip(np.nan_to_num(loss, nan=np.inf, posinf=1e300), 1e-300, 1e300))
vmin, vmax = np.log10(np.nanmin(loss)), min(np.nanmax(lg), np.quantile(lg, 0.995))
LC = ["#f2f2f2", "#39c0c8", "#2f6fdf"]  # outline colours for LEVELS, fixed order
pca = meta["dirs"] == "pca"
traj = None
if pca:
    import torch
    traj = torch.load(os.path.join(CACHE, "dirs", f"{meta['model']}_pca3.pt"), weights_only=False)["coords"]


def ext(u, w):
    du, dw = u[1] - u[0], w[1] - w[0]
    return [u[0] - du / 2, u[-1] + du / 2, w[0] - dw / 2, w[-1] + dw / 2]


def panel(ax, img, u, w, xl, yl, title, tr=None):
    im = ax.imshow(img, origin="lower", extent=ext(u, w), cmap="magma", vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    for L, c in zip(LEVELS, LC):
        if img.min() <= np.log10(L) <= img.max():
            ax.contour(u, w, img, levels=[np.log10(L)], colors=[c], linewidths=1.0, linestyles='solid')
    if tr is not None:
        ax.plot(tr[:, 0], tr[:, 1], "-o", color="#7fdc7f", ms=2.5, lw=0.8)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=9)
    return im


ka, kb, kc = [int(np.argmin(np.abs(x))) for x in (A, B, C)]
ncol = 5 if meta["dirs"] == "random" else 3
fig, axs = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.4), constrained_layout=True)
im = panel(axs[0], lg[kc], A, B, "a (d1)", "b (d2)", f"c = {C[kc]:+.2f} slab", traj[:, [0, 1]] if pca else None)
panel(axs[1], lg[:, kb, :], A, C, "a (d1)", "c (d3)", f"b = {B[kb]:+.2f} slab", traj[:, [0, 2]] if pca else None)
panel(axs[2], lg[:, :, ka], B, C, "b (d2)", "c (d3)", f"a = {A[ka]:+.2f} slab", traj[:, [1, 2]] if pca else None)
if not pca:
    g = np.load(os.path.join(LL_ROOT, "cache", "surf", f"{meta['model']}_final_g51.npz"))
    gl = np.log10(g["loss"])
    panel(axs[3], gl, g["xs"], g["ys"], "a", "b", "loss-landscape g51 plate (reference)")
    gx = np.round(g["xs"], 6)
    rel = np.full((len(B), len(A)), np.nan)
    for j, bb in enumerate(B):
        for i, aa in enumerate(A):
            if round(aa, 6) in gx and round(bb, 6) in gx:
                r = g["loss"][np.where(gx == round(bb, 6))[0][0], np.where(gx == round(aa, 6))[0][0]]
                rel[j, i] = abs(loss[kc, j, i] - r) / r
    im2 = axs[4].imshow(np.log10(rel), origin="lower", extent=ext(A, B), cmap="viridis", interpolation="nearest")
    axs[4].set_title(f"log10 |rel diff| vs g51 (max {np.nanmax(rel):.1e})", fontsize=9)
    axs[4].set_xlabel("a"); axs[4].set_ylabel("b")
    fig.colorbar(im2, ax=axs[4], shrink=0.8)
fig.colorbar(im, ax=axs[:3] if pca else axs[:4], shrink=0.8, label="log10 loss")
fig.suptitle(f"{name}: centre slabs; outlines at loss {', '.join(f'{L:.3g}' for L in LEVELS)}"
             + ("; green = checkpoint trajectory projected" if pca else ""), fontsize=10)
fig.savefig(os.path.join(od, f"{name}_slices.png"), dpi=110); plt.close(fig)

n = len(C); nc = 7; nr = int(np.ceil(n / nc))
fig, axs = plt.subplots(nr, nc, figsize=(2.1 * nc, 2.1 * nr), constrained_layout=True)
for k, ax in enumerate(axs.flat):
    if k >= n:
        ax.axis("off"); continue
    panel(ax, lg[k], A, B, "", "", f"c={C[k]:+.2f}")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"{name}: every c slab, log10 loss (magma, shared scale {vmin:.2f}..{vmax:.2f})", fontsize=10)
fig.savefig(os.path.join(od, f"{name}_atlas.png"), dpi=100); plt.close(fig)

an = json.load(open(path.replace(".npz", ".analysis.json")))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
proj = [(0, A, B, "a", "b", [0, 1]), (1, A, C, "a", "c", [0, 2]), (2, B, C, "b", "c", [1, 2])]
for ax, (drop, u, w, xl, yl, cols) in zip(axs, proj):
    ax.set_facecolor("#111")
    for L, c, st in zip(LEVELS, LC, an["anisotropy_native_grid"]):
        m = loss <= L
        sil = m.any(axis=drop)  # silhouette of the sublevel set, projected along the dropped axis
        ax.contourf(u, w, sil.astype(float), levels=[0.5, 1.5], colors=[c], alpha=0.25)
        ax.contour(u, w, sil.astype(float), levels=[0.5], colors=[c], linewidths=1.2)
    st = an["anisotropy"][-1]
    mu = np.array(st["centroid_abc"])
    for r, vec in zip(st["semi_axes"], st["axes_abc"]):
        vec = np.array(vec)
        ax.plot([mu[cols[0]] - r * vec[cols[0]], mu[cols[0]] + r * vec[cols[0]]],
                [mu[cols[1]] - r * vec[cols[1]], mu[cols[1]] + r * vec[cols[1]]], color="#ff8c42", lw=1)
    if pca:
        ax.plot(traj[:, cols[0]], traj[:, cols[1]], "-o", color="#7fdc7f", ms=2.5, lw=0.8)
    ax.set_xlim(u[0], u[-1]); ax.set_ylim(w[0], w[-1]); ax.set_aspect("equal")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f"projection along {'cba'[drop]}", fontsize=9)
fig.suptitle(f"{name}: silhouettes of {{loss <= L}} for L = " + ", ".join(f"{L:.3g}" for L in LEVELS)
             + " (white, cyan, blue); orange = principal semi-axes at the largest level", fontsize=10)
fig.savefig(os.path.join(od, f"{name}_levelsets.png"), dpi=110); plt.close(fig)
print("wrote", od, name)
