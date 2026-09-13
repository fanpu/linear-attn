"""Verification for the finite-width frontier (fractals doc section 11). CPU only.

For a boundary-centred zoom chain (compute_zoom_chain.py) it reports, per level:
  * local box-counting slope of the frontier (edge cells of L_avg > tau), fitted over box sizes
    2 ... R/8 pixels of that level's native grid;
  * precision: fraction of pixels whose ordered/chaotic label differs between float64 and
    (a) float32, (b) float64 with inputs perturbed by 1e-13 relative (roundoff proxy);
  * resolution check: box counts at matched physical box sizes for 256 / 512 / 1024 grids;
  * tau dependence: local slope for tau in 1e-9 ... 1 (the paper reports the max over tau);
  * stitched multi-level box count N(eps) over the full zoom range;
  * null model: the infinite-width (mean-field) L^(D) rendered through the same pipeline, with its
    own boundary-centred zoom chain (closed form, so it is cheap).

  python analyze_fractal.py --tag A --N 100
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_core import edge_cells, box_counts, grid_axes, meanfield_erf_L, pick_zoom_center

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="A")
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--D", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tau", type=float, default=1e-5)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--null_levels", type=int, default=9)
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
C = lambda f: os.path.join(HERE, "cache", f)


def load(tag, dtype, res):
    f = C(f"zoom_{tag}_N{args.N}_D{args.D}_s{args.seed}_{dtype}_r{res}.npz")
    return np.load(f) if os.path.exists(f) else None


def local_slope(B, lo=2, hi_frac=8):
    R = B.shape[0]
    E = edge_cells(B)
    sizes = [s for s in (1, 2, 4, 8, 16, 32, 64, 128, 256) if lo <= s <= R // hi_frac]
    n = box_counts(E, sizes)
    ok = n > 0
    if ok.sum() < 2:
        return np.nan, sizes, n
    sl = np.polyfit(np.log(1 / np.array(sizes)[ok]), np.log(n[ok]), 1)[0]
    return float(sl), sizes, n


rep = {"tag": args.tag, "N": args.N, "D": args.D, "tau": args.tau, "levels": []}
Z = load(args.tag, "f64", args.res)
Z32 = load(args.tag, "f32", args.res)
Zp = load(args.tag + "pert", "f64", args.res)
wins = Z["windows"]
nlev = Z["L_avg"].shape[0]
for k in range(nlev):
    L = Z["L_avg"][k]
    B = L > args.tau
    side = wins[k][1] - wins[k][0]
    sl, sizes, n = local_slope(B)
    d = dict(level=k, window=[float(x) for x in wins[k]], side=float(side), zoom=float(4.0 / side),
             chaotic_frac=float(B.mean()), edge_cells=int(edge_cells(B).sum()), slope=sl,
             sizes=sizes, counts=n.tolist(), mf_chaotic_frac=float((Z["L_mf"][k] > args.tau).mean()))
    if Z32 is not None and k < Z32["L_avg"].shape[0]:
        d["mismatch_f32"] = float(((Z32["L_avg"][k] > args.tau) != B).mean())
    if Zp is not None and k < Zp["L_avg"].shape[0]:
        d["mismatch_pert"] = float(((Zp["L_avg"][k] > args.tau) != B).mean())
        d["slope_pert"] = local_slope(Zp["L_avg"][k] > args.tau)[0]
    taus = np.logspace(-9, 0, 19)
    d["tau_scan"] = [[float(t), local_slope(L > t)[0]] for t in taus]
    # resolution check
    rc = {}
    for r in (512, 1024):
        Zr = load(args.tag + "res", "f64", r)
        if Zr is None or k >= Zr["L_avg"].shape[0]:
            continue
        Br = Zr["L_avg"][k] > args.tau
        f = r // args.res
        # label agreement after block-subsampling the fine grid at the coarse pixel centres is not
        # exact (centres differ); compare box counts at matched physical box sizes instead
        Er = edge_cells(Br)
        phys = [s for s in (2, 4, 8, 16, 32) if s * f <= r // 4]
        rc[str(r)] = dict(sizes_coarse_px=phys, counts_coarse=box_counts(edge_cells(B), phys).tolist(),
                          counts_fine=box_counts(Er, [s * f for s in phys]).tolist(),
                          slope_fine=local_slope(Br)[0], chaotic_frac=float(Br.mean()),
                          fine_counts_small=box_counts(Er, [1, 2, 4, f]).tolist())
    d["rescheck"] = rc
    rep["levels"].append(d)

# stitched N(eps): level k+1's window is one box of side (side_k / step) inside level k.
# N_total(eps) ~ N_k(box = R/step) * N_{k+1}(eps) / 1 ; upper-biased because windows are chosen at
# maximal mixing (declared). eps in units of the full [0,4] window side.
step = wins[0][1] / (wins[1][1] - wins[1][0]) if nlev > 1 else 4
R = args.res
st_eps, st_logN = [], []
mult = 0.0
for k in range(nlev):
    B = Z["L_avg"][k] > args.tau
    E = edge_cells(B)
    side = wins[k][1] - wins[k][0]
    sizes = [2, 4, 8, 16, 32, 64]
    n = box_counts(E, sizes)
    for s, c in zip(sizes, n):
        if c > 0:
            st_eps.append(side * s / R / 4.0)
            st_logN.append(mult + np.log(c))
    nb = box_counts(E, [int(R / step)])[0]
    mult += np.log(max(nb, 1))
st_eps, st_logN = np.array(st_eps), np.array(st_logN)
order = np.argsort(-st_eps)
rep["stitched"] = dict(eps=st_eps[order].tolist(), logN=st_logN[order].tolist())
sel = st_eps > 0
rep["stitched_slope"] = float(np.polyfit(np.log(1 / st_eps), st_logN, 1)[0])
rep["stitched_eps_range"] = [float(st_eps.min()), float(st_eps.max())]

# null model: mean-field chain with its own boundary-centred zooms, same pipeline
nw = [(0.0, 4.0, 0.0, 4.0)]
null = []
for k in range(args.null_levels):
    x0, x1, y0, y1 = nw[k]
    xs, ys = grid_axes(x0, x1, y0, y1, R)
    SW, SB = np.meshgrid(xs, ys)
    _, Lmf = meanfield_erf_L(SW, SB, args.D)
    B = Lmf > args.tau
    sl, sizes, n = local_slope(B)
    null.append(dict(level=k, window=list(map(float, nw[k])), slope=sl, counts=n.tolist(), sizes=sizes,
                     chaotic_frac=float(B.mean())))
    np.save(C(f"null_mf_level{k}_r{R}.npy"), Lmf.astype(np.float64))
    cx, cy = pick_zoom_center(B, 1 / step, margin=1 / step / 2 + 0.02)
    wx = (x1 - x0) / step
    mx, my = x0 + cx * (x1 - x0), y0 + cy * (y1 - y0)
    nw.append((mx - wx / 2, mx + wx / 2, my - wx / 2, my + wx / 2))
rep["null"] = null
json.dump(rep, open(C(f"fractal_report_{args.tag}_N{args.N}.json"), "w"), indent=1)
print(f"{'lev':>3} {'zoom':>9} {'chaos':>6} {'edges':>6} {'slope':>6} {'f32mis':>7} {'pertmis':>8} {'mf':>5} {'null':>5}")
for d, nl in zip(rep["levels"], null):
    print(f"{d['level']:>3} {d['zoom']:>9.3g} {d['chaotic_frac']:>6.3f} {d['edge_cells']:>6} {d['slope']:>6.2f} "
          f"{d.get('mismatch_f32', np.nan):>7.4f} {d.get('mismatch_pert', np.nan):>8.4f} {d['mf_chaotic_frac']:>5.2f} {nl['slope']:>5.2f}")
print("stitched slope", rep["stitched_slope"], "eps range", rep["stitched_eps_range"])
for d in rep["levels"]:
    print(d["level"], "tau scan:", " ".join(f"{t:.0e}:{s:.2f}" for t, s in d["tau_scan"][::2]))
