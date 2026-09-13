"""Neuron textile: each MLP neuron's post-ReLU activation over the full 113 x 113 (a, b) grid at the final
checkpoint, neurons grouped by their dominant frequency.  Also the 'unscrambled' companion where rows/columns
are re-indexed by k*a mod 113.  Output: gallery/textile_*.png"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from styles import STYLES, fig_canvas, riso_composite, save_png, hex2rgb, P

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--per_cluster", type=int, default=24)
ap.add_argument("--rows_per_cluster", type=int, default=2)
args = ap.parse_args()
A = dict(np.load("cache/analysis.npz"))
s = args.seed
tag = f"seed{int(A['init_seeds'][s])}"
acts = np.load("cache/acts_final.npy", mmap_mode="r")[s].astype(np.float32)   # (P,P,M)
K = int(A["nkeys"][s]); keys = sorted(A["keys"][s][:K].tolist())
nk = A["neuron_k"][s]; nf = A["neuron_frac"][s]; nv = A["neuron_var"][s]

clusters = []
for k in keys:
    idx = np.where((nk == k) & (nv > 1e-6))[0]
    idx = idx[np.argsort(-nv[idx])]                     # strongest first
    clusters.append((k, idx))
print({k: len(i) for k, i in clusters})


def tile(m, unscramble_k=None):
    x = acts[:, :, m]
    if unscramble_k is not None:
        inv = pow(int(unscramble_k), -1, P)
        perm = (np.arange(P) * inv) % P                  # position i holds token a with k*a = i (mod p)
        x = x[perm][:, perm]
    return x / max(x.max(), 1e-9)


def mosaic(unscramble=False, gap=10, cgap=34, px=113):
    cols = args.per_cluster; rpc = args.rows_per_cluster
    Wpx = cols * px + (cols - 1) * gap
    Hpx = K * (rpc * px + (rpc - 1) * gap) + (K - 1) * cgap
    canvas = np.zeros((Hpx, Wpx), np.float32)
    owner = np.full((Hpx, Wpx), -1, np.int16)
    y = 0
    for ci, (k, idx) in enumerate(clusters):
        for r in range(rpc):
            for c in range(cols):
                n = r * cols + c
                if n >= len(idx):
                    continue
                x0 = c * (px + gap); y0 = y + r * (px + gap)
                canvas[y0:y0 + px, x0:x0 + px] = tile(idx[n], k if unscramble else None)
                owner[y0:y0 + px, x0:x0 + px] = ci
        y += rpc * px + (rpc - 1) * gap + cgap
    return canvas, owner


def render(style, unscramble=False, scale=2):
    st = STYLES[style]
    canvas, owner = mosaic(unscramble)
    canvas = np.kron(canvas, np.ones((scale, scale), np.float32))   # nearest-neighbour upscale (no interpolation)
    owner = np.kron(owner, np.ones((scale, scale), np.int16))
    m = 120 * scale
    H, W = canvas.shape
    full = np.zeros((H + 2 * m, W + 2 * m), np.float32); full[m:-m, m:-m] = canvas
    own = np.full(full.shape, -1, np.int16); own[m:-m, m:-m] = owner
    v = np.sqrt(np.clip(full, 0, 1))                                  # declared sqrt tone map
    if style == "nocturne":
        rgb = plt.get_cmap("magma")(v)[..., :3]
        rgb[own < 0] = hex2rgb(st["bg"])
        img = (rgb * 255).astype(np.uint8)
    elif style == "plate":
        cm = LinearSegmentedColormap.from_list("sep", [st["bg"], "#b89b72", "#6b4e2f", "#1f150b"])
        img = (cm(v)[..., :3] * 255).astype(np.uint8)
    elif style == "plotter":
        cm = LinearSegmentedColormap.from_list("ink", [st["bg"], st["ink"]])
        img = (cm(v)[..., :3] * 255).astype(np.uint8)
    else:  # riso: alternate clusters between the two inks
        l0 = np.where((own >= 0) & (own % 2 == 0), v, 0); l1 = np.where((own >= 0) & (own % 2 == 1), v, 0)
        img = riso_composite([l0 * 0.95, l1 * 0.95], [st["ink"], st["ink2"]], st["bg"], offsets=[(0, 0), (3, -4)], grain=0.18)
    # captions via matplotlib overlay
    Hh, Ww = img.shape[:2]
    fig = plt.figure(figsize=(Ww / 200, Hh / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img, interpolation="nearest"); ax.axis("off")
    y = m
    rpc = args.rows_per_cluster
    for ci, (k, idx) in enumerate(clusters):
        ax.text(m - 18 * scale, y + (rpc * 113 + (rpc - 1) * 10) * scale / 2, f"k={k}\n{len(idx)} neurons", ha="right", va="center",
                fontsize=8, color=st["muted"], family="DejaVu Sans Mono")
        y += (rpc * 113 + (rpc - 1) * 10 + 34) * scale
    head = ("unscrambled: rows and columns re-indexed by k·a mod 113" if unscramble else
            "MLP neurons over the full (a, b) grid, grouped by frequency")
    ax.text(m, m * 0.45, head, fontsize=12, color=st["ink"], va="center", style="italic" if style == "plate" else "normal")
    ax.text(Ww - m, m * 0.45, f"{tag} · final checkpoint · a ↓ b → · per-neuron max-normalised, sqrt tone",
            fontsize=8, color=st["muted"], va="center", ha="right")
    fig.savefig(f"gallery/textile{'_unscrambled' if unscramble else ''}_{tag}_{style}.png", dpi=200)
    plt.close(fig)


for style in STYLES:
    render(style)
    print("textile", style, flush=True)
for style in ["nocturne", "plate", "plotter"]:
    render(style, unscramble=True)
