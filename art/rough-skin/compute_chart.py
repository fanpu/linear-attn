"""CPU: measure the distortion of the declared chart (exp map of the tangent cube [-0.25, 0.25]^3 rad at BASE).
Numerical Jacobian (central differences) at every node of a 33^3 sub-grid spanning the cube, plus the
geodesic lengths of the 256^3 grid's edges along the cube's edges and diagonals. Output cache/chart.json."""
import json
import numpy as np
from common import *

E = tangent_frame()
a = np.linspace(-HALF, HALF, 33)
T = np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)
eps = 1e-6
J = np.stack([(expmap(T + eps * e, E=E) - expmap(T - eps * e, E=E)) / (2 * eps) for e in np.eye(3)], -1)
sv = np.linalg.svd(J, compute_uv=False)
_, s_an = stretch_singular_values(T)
h = 2 * HALF / 256
ax = grid_coords(256)
def geo(p, q): return np.arccos(np.clip(np.sum(p * q, -1), -1, 1))
# voxel edge lengths along x at the centre row and along the cube corner edge
cen = expmap(np.stack([ax, 0 * ax, 0 * ax], -1), E=E)
cor = expmap(np.stack([ax, 0 * ax + ax[-1], 0 * ax + ax[-1]], -1), E=E)
out = dict(half_side_rad=HALF, base=BASE.tolist(), grid_spacing_rad=h,
           sv_max=float(sv.max()), sv_min=float(sv.min()),
           max_stretch=float(sv.max() - 1), max_compression=float(1 - sv.min()),
           analytic_corner_compression=float(1 - np.sin(np.sqrt(3) * HALF) / (np.sqrt(3) * HALF)),
           max_anisotropy=float((sv.max(1) / sv.min(1)).max()),
           numeric_vs_analytic_max_err=float(np.abs(sv.min(1) - s_an).max()),
           edge_len_centre_row=[float(geo(cen[:-1], cen[1:]).min()), float(geo(cen[:-1], cen[1:]).max())],
           edge_len_corner_row=[float(geo(cor[:-1], cor[1:]).min()), float(geo(cor[:-1], cor[1:]).max())])
json.dump(out, open("cache/chart.json", "w"), indent=1)
print(json.dumps(out, indent=1))
