"""Quantify convexity / roughness of cached 2-D surfaces (CPU).  python analyze_surf.py g51
Principal curvatures of the loss L(α,β) by central finite differences on the grid (Li et al. 2018 Fig. 7
use |λmin/λmax|); local minima (strict, 8-neighbour); roughness = RMS of the discrete Laplacian of log10 L
relative to that of a 2-D quadratic fit.  Writes cache/surf_stats_<tag>.json and cache/curv_<model>_<tag>.npz."""
import json, os, sys, numpy as np
from scipy import ndimage
from common import CACHE
tag = sys.argv[1] if len(sys.argv) > 1 else "g51"
out = {}
for m in ["resnet20", "resnet20_noshort", "resnet56", "resnet56_noshort"]:
    p = f"{CACHE}/surf/{m}_final_{tag}.npz"
    if not os.path.exists(p):
        continue
    d = np.load(p); L = d["loss"]; h = d["xs"][1] - d["xs"][0]
    Lyy, Lyx = np.gradient(np.gradient(L, h, axis=0), h)
    Lxy, Lxx = np.gradient(np.gradient(L, h, axis=1), h)
    B = 0.5 * (Lxy + Lyx); tr = Lxx + Lyy; det = Lxx * Lyy - B ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    lmax, lmin = tr / 2 + disc, tr / 2 - disc
    ratio = lmin / (np.abs(lmax) + 1e-12)
    inner = np.zeros_like(L, bool); inner[1:-1, 1:-1] = True
    Z = np.log10(L)
    mins = (Z == ndimage.minimum_filter(Z, size=3, mode="nearest")) & inner
    basin = (L < np.log(10)) & inner
    lap = ndimage.laplace(Z)[1:-1, 1:-1] / h ** 2
    A = np.linalg.lstsq(np.c_[np.ones(L.size), *[f.ravel() for f in np.meshgrid(d["xs"], d["ys"])],
                              *[g.ravel() for g in (lambda X, Y: (X * X, X * Y, Y * Y))(*np.meshgrid(d["xs"], d["ys"]))]],
                        Z.ravel(), rcond=None)[0]
    X, Y = np.meshgrid(d["xs"], d["ys"])
    res = Z - (A[0] + A[1] * X + A[2] * Y + A[3] * X * X + A[4] * X * Y + A[5] * Y * Y)
    np.savez(f"{CACHE}/curv_{m}_{tag}.npz", lmin=lmin, lmax=lmax, ratio=ratio, xs=d["xs"], ys=d["ys"])
    out[m] = dict(grid=L.shape[0], spacing=float(h), centre_loss=float(L[L.shape[0] // 2, L.shape[1] // 2]),
                  min_loss=float(L.min()), max_loss=float(L.max()), basin_area_frac=float(basin.mean()),
                  local_minima=int(mins.sum()), local_minima_in_basin=int((mins & basin).sum()),
                  frac_negcurv_all=float((lmin[inner] < 0).mean()),
                  frac_negcurv_basin=float((lmin[basin] < 0).mean()) if basin.any() else None,
                  mean_abs_ratio_negcurv=float(np.abs(ratio[inner & (lmin < 0)]).mean()) if (inner & (lmin < 0)).any() else 0,
                  rms_laplacian_log10=float(np.sqrt(np.mean(lap ** 2))),
                  rms_quadfit_residual_log10=float(np.sqrt(np.mean(res ** 2))))
json.dump(out, open(f"{CACHE}/surf_stats_{tag}.json", "w"), indent=1)
for m, v in out.items():
    print(m, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
