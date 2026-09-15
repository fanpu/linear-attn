"""CPU: dimension tables for the net fields, calibrated against cache/calibration3d.json.

    measure_dims.py            -> cache/dims.json, cache/dims_tables.md

Fields: width 1024 on 128^3 (toy), width 4096 on 256^3, and width 4096 on 128^3 (its even-node subsample,
the resolution-doubling partner). Level: the median of each volume (declared).
"""
import json, os
import numpy as np
from common import *

CALS = {"window": json.load(open("cache/calibration3d_window.json")),      # primary
        "periodic": json.load(open("cache/calibration3d.json"))}           # comparison
cal = CALS["window"]
MATCH_K = {("w1024", 128): 0, ("w4096", 256): 1, ("w4096", 128): 2}
ALT_K = {("w1024", 128): [1, 2], ("w4096", 256): [0], ("w4096", 128): [0, 1]}


def curve(T, k, what, proto="window"):
    """Mean and sd of the estimator on exact-dimension power-law fields, sorted by D_true."""
    rows = {}
    for v in CALS[proto].values():
        if v["T"] != T or v["k"] != k or v["H"] == "smooth":
            continue
        x = v["D3"] if what == "D3" else float(np.mean(v["slice_1pD"]))
        rows.setdefault(v["D_true"], []).append(x)
    Dt = np.array(sorted(rows)); m = np.array([np.mean(rows[d]) for d in Dt]); s = np.array([np.std(rows[d]) for d in Dt])
    return Dt, m, s, np.array([len(rows[d]) for d in Dt])


def smooth_null(T, k, what, proto="window"):
    xs = [v["D3"] if what == "D3" else float(np.mean(v["slice_1pD"])) for v in CALS[proto].values()
          if v["T"] == T and v["k"] == k and v["H"] == "smooth"]
    return float(np.mean(xs)), float(np.std(xs))


def invert(x, T, k, what, proto="window"):
    Dt, m, s, _ = curve(T, k, what, proto)
    if not np.all(np.diff(m) > 0):
        return float("nan"), float("nan"), "non-monotone"
    if x < m[0]:
        return float("nan"), float("nan"), f"below range (<{Dt[0]:.3f})"
    if x > m[-1]:
        return float("nan"), float("nan"), f"above range (>{Dt[-1]:.4f})"
    d = float(np.interp(x, m, Dt))
    slope = np.gradient(m, Dt)
    sd = float(np.interp(x, m, s) / np.interp(x, m, slope))
    return d, sd, ""


def slices_from_volume(f, stride, lvl):
    out = []
    for pl in slice_planes():
        P = slice_points(pl, SLICE_NPX[stride], stride * 2.0 / 256)
        out.append(1 + dim2(sample_trilinear(f, P, stride=stride), level=lvl, fit=FIT2_BY_STRIDE[stride])[0])
    return np.array(out)


res = {}
sources = [("w1024", 128, "cache/field_w1024_r128.npy", 1), ("w4096", 256, "cache/field_w4096_r256.npy", 1),
           ("w4096", 128, "cache/field_w4096_r256.npy", 2)]
for wtag, T, path, sub in sources:
    if not os.path.exists(path):
        print("missing", path); continue
    V = np.load(path, mmap_mode="r")
    stride = 256 // T
    fit3 = (2, 64) if T == 256 else (2, 32)
    k = MATCH_K[(wtag, T)]
    S_exact = np.load(f"cache/slices_exact_{wtag}.npy") if os.path.exists(f"cache/slices_exact_{wtag}.npy") else None
    for ia, a in enumerate(ACTS):
        for l in range(LMAX):
            f = np.ascontiguousarray(V[ia, l][::sub, ::sub, ::sub]).astype(np.float64)
            lvl = median_level(f)
            c = boxcount_mask(boundary_mask(f, lvl), SIZES3)
            D3, se = fit_slope(SIZES3, c, *fit3)
            sl = slices_from_volume(f, stride, lvl)
            row = dict(width=wtag, T=T, act=a, L=l + 1, level=lvl, counts=c.tolist(), D3_raw=D3, D3_fit_se=se,
                       theory=theory(a, l + 1), slice_raw=sl.tolist(), slice_raw_mean=float(sl.mean()),
                       slice_raw_sd=float(sl.std()), k=k)
            row["D3_cal"], row["D3_cal_sd"], row["D3_cal_note"] = invert(D3, T, k, "D3")
            row["slice_cal"], row["slice_cal_sd"], row["slice_cal_note"] = invert(float(sl.mean()), T, k, "slice")
            row["D3_cal_altk"] = {str(kk): invert(D3, T, kk, "D3")[0] for kk in ALT_K[(wtag, T)]}
            row["D3_cal_periodic"] = invert(D3, T, k, "D3", "periodic")[0]
            if S_exact is not None:
                ex = np.array([1 + dim2(S_exact[ia, l, j].astype(np.float64), level=lvl, fit=FIT2_BY_STRIDE[1])[0]
                               for j in range(12)])
                row["slice_exact256_raw_mean"] = float(ex.mean()); row["slice_exact256_raw_sd"] = float(ex.std())
            res[f"{wtag}_T{T}_{a}_L{l+1}"] = row
            print(f"{wtag} T{T} {a} L{l+1}: D3 raw {D3:.3f} cal {row['D3_cal']:.3f} ({row['D3_cal_note']}) "
                  f"slice raw {sl.mean():.3f} cal {row['slice_cal']:.3f} theory {row['theory']:.4f}", flush=True)

# float32 vs float64
flips = {}
for wtag, T in [("w1024", 128), ("w4096", 256)]:
    fn = f"cache/slab64_{wtag}_r{T}.npz"
    if not os.path.exists(fn):
        continue
    z = np.load(fn)
    V = np.load(f"cache/field_{wtag}_r{T}.npy", mmap_mode="r")
    for ia, a in enumerate(ACTS):
        for l in range(LMAX):
            lvl = median_level(np.asarray(V[ia, l]))
            s32 = z["o32"][ia, l] > lvl; s64 = z["o64"][ia, l] > lvl
            B32 = boundary_mask(z["o32"][ia, l].astype(np.float64), lvl); B64 = boundary_mask(z["o64"][ia, l], lvl)
            flips[f"{wtag}_{a}_L{l+1}"] = dict(
                voxels=int(s32.size), sign_flips=int((s32 != s64).sum()), sign_flip_rate=float((s32 != s64).mean()),
                boundary_voxels_differing=int((B32 != B64).sum()),
                hidden_code_flip_rate=float(z["code_flips"][ia, l] / z["codes_total"]),
                max_abs_diff=float(np.abs(z["o32"][ia, l] - z["o64"][ia, l]).max()),
                slab_seconds=float(z["seconds"]))

# draw-to-draw scatter at width 1024, 128^3 (seed 7 = main draw, seeds 8-11 extra)
draws = {}
seed_files = [(7, "cache/field_w1024_r128.npy")] + [(sd, f"cache/field_w1024_r128_s{sd}.npy") for sd in (8, 9, 10, 11)]
seed_files = [(sd, fn) for sd, fn in seed_files if os.path.exists(fn)]
for ia, a in enumerate(ACTS):
    for l in range(LMAX):
        raws, cals = [], []
        for sd, fn in seed_files:
            f = np.asarray(np.load(fn, mmap_mode="r")[ia, l]).astype(np.float64)
            D3 = fit_slope(SIZES3, boxcount_mask(boundary_mask(f, median_level(f)), SIZES3), 2, 32)[0]
            raws.append(D3); cals.append(invert(D3, 128, 0, "D3")[0])
        draws[f"{a}_L{l+1}"] = dict(seeds=[sd for sd, _ in seed_files], D3_raw=raws, D3_cal=cals)

# draw statistics at width 4096, 256^3 (M2 ruling: seeds 7, 8, 9)
draws4 = {}
seed_files4 = [(7, "cache/field_w4096_r256.npy")] + [(sd, f"cache/field_w4096_r256_s{sd}.npy") for sd in (8, 9)]
seed_files4 = [(sd, fn) for sd, fn in seed_files4 if os.path.exists(fn)]
for ia, a in enumerate(ACTS):
    for l in range(LMAX):
        rows = []
        for sd, fn in seed_files4:
            f = np.asarray(np.load(fn, mmap_mode="r")[ia, l]).astype(np.float64)
            lvl = median_level(f)
            D3 = fit_slope(SIZES3, boxcount_mask(boundary_mask(f, lvl), SIZES3), 2, 64)[0]
            sl = slices_from_volume(f, 1, lvl)
            rows.append(dict(seed=sd, D3_raw=D3, D3_cal=invert(D3, 256, 1, "D3")[0], D3_cal_k0=invert(D3, 256, 0, "D3")[0],
                             D3_cal_periodic=invert(D3, 256, 1, "D3", "periodic")[0],
                             slice_raw=float(sl.mean()), slice_cal=invert(float(sl.mean()), 256, 1, "slice")[0]))
        draws4[f"{a}_L{l+1}"] = rows

calrows = {}
for proto in ["window", "periodic"]:
    for T, k in [(256, 0), (256, 1), (128, 0), (128, 1), (128, 2)]:
        for what in ["D3", "slice"]:
            Dt, m, s, cnt = curve(T, k, what, proto)
            calrows[f"{proto}_T{T}_k{k}_{what}"] = dict(D_true=Dt.tolist(), mean=m.tolist(), sd=s.tolist(), n=cnt.tolist(),
                                                        smooth_null=smooth_null(T, k, what, proto))
json.dump(dict(dims=res, flips=flips, calibration=calrows, draws_w1024_r128=draws, draws_w4096_r256=draws4), open("cache/dims.json", "w"), indent=1)

# markdown tables
L = []
L.append("### Estimator calibration (power-law Gaussian fields; mean ± sd over seeds: window 4, periodic 6)\n")
L.append("| protocol | grid T | sub-voxel octaves k | estimator | " + " | ".join(f"D={d:.4g}" for d in calrows["window_T256_k0_D3"]["D_true"]) + " | smooth null (D=2) |")
L.append("|---|---|---|---|" + "---|" * (len(calrows["window_T256_k0_D3"]["D_true"]) + 1))
for key, v in calrows.items():
    proto, T, k, what = key.split("_")
    L.append(f"| {proto} | {T[1:]} | {k[1:]} | {'3D box' if what == 'D3' else '1 + slice'} | " +
             " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in zip(v["mean"], v["sd"])) +
             f" | {v['smooth_null'][0]:.3f} ± {v['smooth_null'][1]:.3f} |")
L.append("\n### Dimension table (level = volume median)\n")
L.append("| width | grid | act | L | theory | 3D raw | 3D calibrated (window, k) | 3D cal, other k | 3D cal, periodic protocol | 1+slice raw (12 oblique) | 1+slice calibrated | 1+exact slice raw (256 spacing) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in res.values():
    cal3 = f"{r['D3_cal']:.3f} ± {r['D3_cal_sd']:.3f}" if np.isfinite(r["D3_cal"]) else r["D3_cal_note"]
    cals = f"{r['slice_cal']:.3f} ± {r['slice_cal_sd']:.3f}" if np.isfinite(r["slice_cal"]) else r["slice_cal_note"]
    alt = ", ".join(f"k={kk}: {d:.3f}" for kk, d in r["D3_cal_altk"].items())
    ex = f"{r['slice_exact256_raw_mean']:.3f} ± {r['slice_exact256_raw_sd']:.3f}" if "slice_exact256_raw_mean" in r else "-"
    L.append(f"| {r['width'][1:]} | {r['T']}³ | {r['act']} | {r['L']} | {r['theory']:.4f} | {r['D3_raw']:.3f} | {cal3} (k={r['k']}) | {alt} | {r['D3_cal_periodic']:.3f} | "
             f"{r['slice_raw_mean']:.3f} ± {r['slice_raw_sd']:.3f} | {cals} | {ex} |")
L.append("\n### float32 vs float64 (32 central z-planes)\n")
L.append("| field | voxels | output sign flips | rate | boundary voxels differing | hidden-code flip rate (layer L) | max abs diff |")
L.append("|---|---|---|---|---|---|---|")
for key, v in flips.items():
    L.append(f"| {key} | {v['voxels']} | {v['sign_flips']} | {v['sign_flip_rate']:.2e} | {v['boundary_voxels_differing']} | "
             f"{v['hidden_code_flip_rate']:.2e} | {v['max_abs_diff']:.2e} |")
L.append("\n### Draw-to-draw scatter, width 1024 on 128³ (calibrated with k = 0)\n")
L.append("| act | L | seeds | 3D raw per draw | raw mean ± sd | calibrated mean ± sd |")
L.append("|---|---|---|---|---|---|")
for key, v in draws.items():
    a, l = key.split("_L")
    c = np.array(v["D3_cal"], float); fin = c[np.isfinite(c)]
    cs = f"{fin.mean():.3f} ± {fin.std():.3f} ({len(fin)}/{len(c)} in range)" if len(fin) else "out of range"
    L.append(f"| {a} | {l} | {len(v['seeds'])} | {', '.join(f'{x:.3f}' for x in v['D3_raw'])} | "
             f"{np.mean(v['D3_raw']):.3f} ± {np.std(v['D3_raw']):.3f} | {cs} |")
L.append("\n### Draw statistics, width 4096 on 256³ (window calibration, k = 1; systematics: k = 0 and periodic protocol)\n")
L.append("| act | L | draws | 3D raw per draw | 3D calibrated mean ± sd | 3D cal k=0 mean | 3D cal periodic mean | 1+slice raw mean | 1+slice calibrated mean ± sd |")
L.append("|---|---|---|---|---|---|---|---|---|")
def ms(xs):
    x = np.array(xs, float); x = x[np.isfinite(x)]
    return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f} (n={len(x)})" if len(x) else "out of range"
for key, rows in draws4.items():
    a, l = key.split("_L")
    L.append(f"| {a} | {l} | {len(rows)} | {', '.join(f'{r['D3_raw']:.3f}' for r in rows)} | {ms([r['D3_cal'] for r in rows])} | "
             f"{ms([r['D3_cal_k0'] for r in rows])} | {ms([r['D3_cal_periodic'] for r in rows])} | "
             f"{np.mean([r['slice_raw'] for r in rows]):.3f} | {ms([r['slice_cal'] for r in rows])} |")
open("cache/dims_tables.md", "w").write("\n".join(L) + "\n")
print("wrote cache/dims.json, cache/dims_tables.md")
