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
ap.add_argument("--label", default="tau", help="tau (L_avg > tau) or sync (pair never reached L < 1e-10 within D)")
ap.add_argument("--plates", default="", help="tag:level,level;tag:level,level for 1024 re-renders (4x check)")
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


def adj_corr(B):
    """Pearson correlation of the binary label between horizontally and vertically adjacent pixels
    (1 = resolved smooth regions, 0 = neighbouring pixels independent, i.e. unresolved / area-filling)."""
    b = B.astype(float)
    cs = []
    for a, c in ((b[:, :-1], b[:, 1:]), (b[:-1, :], b[1:, :])):
        a, c = a.ravel() - a.mean(), c.ravel() - c.mean()
        den = np.sqrt((a * a).sum() * (c * c).sum())
        cs.append(float((a * c).sum() / den) if den > 0 else np.nan)
    return float(np.nanmean(cs))


def lab(Zf, k):
    if args.label == "sync":
        return Zf["t_hit"][k] > args.D
    return Zf["L_avg"][k] > args.tau


def lab_mf(Lmf_D):
    return (Lmf_D >= 1e-10) if args.label == "sync" else (Lmf_D > args.tau)


rep = {"label": args.label, "tag": args.tag, "N": args.N, "D": args.D, "tau": args.tau, "levels": []}
Z = load(args.tag, "f64", args.res)
Z32 = load(args.tag, "f32", args.res)
Zp = load(args.tag + "pert", "f64", args.res)
wins = Z["windows"]
nlev = Z["L_avg"].shape[0]
for k in range(nlev):
    L = Z["L_avg"][k]
    B = lab(Z, k)
    side = wins[k][1] - wins[k][0]
    sl, sizes, n = local_slope(B)
    d = dict(level=k, window=[float(x) for x in wins[k]], side=float(side), zoom=float(4.0 / side),
             chaotic_frac=float(B.mean()), edge_cells=int(edge_cells(B).sum()), slope=sl,
             sizes=sizes, counts=n.tolist(), mf_chaotic_frac=float(lab_mf(Z["L_mf"][k]).mean()),
             adj_corr=adj_corr(B))
    if Z32 is not None and k < Z32["L_avg"].shape[0]:
        d["mismatch_f32"] = float((lab(Z32, k) != B).mean())
        d["slope_f32"] = local_slope(lab(Z32, k))[0]
    if Zp is not None:
        jj = [j for j in range(Zp["windows"].shape[0]) if np.allclose(Zp["windows"][j], wins[k], rtol=0, atol=1e-15)]
        if jj:
            d["mismatch_pert"] = float((lab(Zp, jj[0]) != B).mean())
    taus = np.logspace(-12, 0, 25)
    d["tau_scan"] = [[float(t), local_slope(L > t)[0]] for t in taus]
    # resolution check
    rc = {}
    cands = []
    for f in sorted(glob.glob(C(f"zoom_{args.tag}*_N{args.N}_D{args.D}_s{args.seed}_f64_r*.npz"))):
        r = int(f.split("_r")[-1].split(".")[0])
        if r == args.res:
            continue
        Zr = np.load(f)
        jj = [j for j in range(Zr["windows"].shape[0]) if np.allclose(Zr["windows"][j], wins[k], rtol=0, atol=1e-15)]
        if jj:
            cands.append((r, Zr, jj[0]))
    for r, Zr, kk in cands:
        if Zr is None:
            continue
        Br = lab(Zr, kk)
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
    B = lab(Z, k)
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
null, null_B = [], []
for k in range(args.null_levels):
    x0, x1, y0, y1 = nw[k]
    xs, ys = grid_axes(x0, x1, y0, y1, R)
    SW, SB = np.meshgrid(xs, ys)
    LmfD, Lmf = meanfield_erf_L(SW, SB, args.D)
    B = lab_mf(LmfD) if args.label == "sync" else (Lmf > args.tau)
    sl, sizes, n = local_slope(B)
    null.append(dict(level=k, window=list(map(float, nw[k])), slope=sl, counts=n.tolist(), sizes=sizes,
                     chaotic_frac=float(B.mean()), adj_corr=adj_corr(B)))
    null_B.append(B)
    np.save(C(f"null_mf_level{k}_r{R}.npy"), Lmf.astype(np.float64))
    cx, cy = pick_zoom_center(B, 1 / step, margin=1 / step / 2 + 0.02)
    wx = (x1 - x0) / step
    mx, my = x0 + cx * (x1 - x0), y0 + cy * (y1 - y0)
    nw.append((mx - wx / 2, mx + wx / 2, my - wx / 2, my + wx / 2))
rep["null"] = null
np.savez_compressed(C(f"null_labels_{args.label}_r{R}.npz"), B=np.stack(null_B), windows=np.array(nw[:len(null_B)]))
json.dump(rep, open(C(f"fractal_report_{args.tag}_N{args.N}_{args.label}.json"), "w"), indent=1)
print(f"{'lev':>3} {'zoom':>9} {'chaos':>6} {'edges':>6} {'slope':>6} {'f32mis':>7} {'pertmis':>8} {'mf':>5} {'null':>5} {'adj':>5}")
for d, nl in zip(rep["levels"], null):
    print(f"{d['level']:>3} {d['zoom']:>9.3g} {d['chaotic_frac']:>6.3f} {d['edge_cells']:>6} {d['slope']:>6.2f} "
          f"{d.get('mismatch_f32', np.nan):>7.4f} {d.get('mismatch_pert', np.nan):>8.4f} {d['mf_chaotic_frac']:>5.2f} {nl['slope']:>5.2f} {d['adj_corr']:>5.2f}")
    print("    rescheck:", {r: round(v["slope_fine"], 3) for r, v in d["rescheck"].items()})
print("stitched slope", rep["stitched_slope"], "eps range", rep["stitched_eps_range"])
for d in rep["levels"]:
    print(d["level"], "tau scan:", " ".join(f"{t:.0e}:{s:.2f}" for t, s in d["tau_scan"][::2]))
