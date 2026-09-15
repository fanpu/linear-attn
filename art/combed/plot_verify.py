"""Combed M2 verification plate: 3D box counting of the N = 16 basin label boundary vs the planar Voronoi null,
at 128^3 and 256^3 (matplotlib 2D plate; reads cache/basin_boxcount.json).

  art/.venv/bin/python art/combed/plot_verify.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import combed_common as C

bc = json.loads((C.CACHE / "basin_boxcount.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
for res, mk in (("128", "o"), ("256", "s")):
    for name, col, lab in (("labels", "#b2182b", "closed-form basins"), ("null_voronoi_x0", "#2166ac", "null: Voronoi of x0")):
        r = bc[res][name]
        eps = np.array(r["eps_world"]); cnt = np.array(r["counts"])
        axes[0].loglog(eps, cnt, mk + "-", c=col, alpha=0.9 if res == "256" else 0.5, ms=4,
                       label=f"{lab}, {res}^3: D = {r['D_fit']:.2f} (eps 1-{r['D_fit_range_vox'][1]} vox)")
        mid = np.sqrt(eps[1:] * eps[:-1])
        axes[1].semilogx(mid, r["local_slopes"], mk + "-", c=col, alpha=0.9 if res == "256" else 0.5, ms=4, label=f"{lab}, {res}^3")
e = np.array(bc["256"]["labels"]["eps_world"])
axes[0].loglog(e, bc["256"]["labels"]["counts"][0] * (e / e[0]) ** -2.0, "k:", lw=1, label="slope -2 (a surface)")
axes[0].set_xlabel("box side eps (world units; cube side 5)"); axes[0].set_ylabel("boxes meeting the label boundary")
axes[0].legend(fontsize=6.5); axes[0].set_title("3D box counting, boundary = voxels with a differing 6-neighbour", fontsize=9)
axes[1].axhline(2, c="k", lw=0.8, ls=":")
axes[1].set_ylim(1.8, 3.05); axes[1].set_xlabel("eps (world units)"); axes[1].set_ylabel("local slope -dlogN/dlog eps")
axes[1].set_title("local slopes: basins track the planar null; the rise at large eps is box saturation", fontsize=9)
axes[1].legend(fontsize=6.5)
ratio = bc.get("boundary_voxel_ratio_256_over_128")
fig.suptitle(f"Combed basin companion, N = 16, t = 1 - 1e-6.  Boundary voxels 256^3 / 128^3 = {ratio:.2f} "
             "(surface ~4, volume-filling ~8).  No fractal claim.", fontsize=9)
fig.tight_layout()
out = C.ROOT / "gallery" / "verify" / "basin_boxcount.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print("wrote", out)
