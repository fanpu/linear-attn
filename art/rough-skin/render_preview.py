"""Matplotlib preview slices (M1 check images, not gallery pieces).

    render_preview.py <width> <R>

cache/preview/slices_w<n>_r<R>.png: rows = activation, columns = depth L. Top half of each tile: the central
axial z-slice of the volume; bottom: exact oblique slice 0 (256^3 spacing). Two tones = sign of T - u at the
volume median u (declared palette: pale/dark grey), nearest-neighbour pixels.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

n, R = int(sys.argv[1]), int(sys.argv[2])
V = np.load(f"cache/field_w{n}_r{R}.npy", mmap_mode="r")
S = np.load(f"cache/slices_exact_w{n}.npy")
fig, axs = plt.subplots(4, LMAX, figsize=(3.2 * LMAX, 13), dpi=110)
for ia, a in enumerate(ACTS):
    for l in range(LMAX):
        f = np.asarray(V[ia, l])
        u = median_level(f)
        ax = axs[2 * ia, l]
        ax.imshow((f[:, :, R // 2] > u).T, cmap="Greys", vmin=-0.3, vmax=1.3, origin="lower", interpolation="nearest")
        ax.set_title(f"{a} L={l+1}  z-slice {R}²", fontsize=9)
        ax = axs[2 * ia + 1, l]
        ax.imshow((S[ia, l, 0] > u).T, cmap="Greys", vmin=-0.3, vmax=1.3, origin="lower", interpolation="nearest")
        ax.set_title(f"{a} L={l+1}  oblique exact 176²", fontsize=9)
for ax in axs.flat:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Rough Skin M1 preview: width {n}, grid {R}³ over a 0.5 rad exp-map cube on S³ (sign of T − median)", fontsize=11)
fig.tight_layout()
out = f"cache/preview/slices_w{n}_r{R}.png"
fig.savefig(out); print(out)
