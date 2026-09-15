"""Verification numbers for a loss volume (CPU).
  python analyze.py cache/vol/resnet20_final_random_g27.npz [--levels 0.5 1.0 2.302585]
 * slice reproduction (random dirs only): c = 0 slab vs loss-landscape resnet20_final_g51 at
   coincident (a, b); b = c = 0 row vs resnet20_final_line (401 points) at coincident a.
 * shell anisotropy at declared loss levels: sublevel set {loss <= level}, the 26-connected
   component containing the grid minimum, on the grid supersampled s x per axis by trilinear
   interpolation of log(loss) (declared), second moments in physical (a, b, c) units.
   For a solid ellipsoid var = r^2 / 5, so semi-axes r_i = sqrt(5 lambda_i).  fill = voxel volume /
   (4/3 pi r1 r2 r3) (1 for an ellipsoid).  touches = component reaches the box face (shell clipped).
 * null: an analytic rotated ellipsoidal onion with known semi-axes through the same function.
writes <vol>.analysis.json"""
import argparse, json, os
import numpy as np
from scipy import ndimage
from lib import LL_ROOT

LEVELS = [0.5, 1.0, float(np.log(10))]


def upsample_log(loss, axes, s):
    n = [len(a) for a in axes]
    lg = np.log(np.clip(np.nan_to_num(loss, nan=1e300, posinf=1e300), 1e-300, None))
    if s == 1:
        return lg, axes
    q = [np.linspace(0, k - 1, (k - 1) * s + 1) for k in n]
    Z, Y, X = np.meshgrid(*q, indexing="ij")
    up = ndimage.map_coordinates(lg, [Z, Y, X], order=1)
    ax_up = [np.interp(qq, np.arange(k), a) for qq, k, a in zip(q, n, axes)]
    return up, ax_up


def shell_stats(loss, axes, level, s=4):
    """loss[c, b, a]; axes = [c_axis, b_axis, a_axis] (physical coordinates)."""
    lg, ax = upsample_log(loss, axes, s)
    mask = lg <= np.log(level)
    if not mask.any():
        return dict(level=level, empty=True)
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    seed = np.unravel_index(np.argmin(lg), lg.shape)
    comp = lab == lab[seed]
    n_other = int(mask.sum() - comp.sum())
    idx = np.nonzero(comp)
    dv = np.prod([abs(a[1] - a[0]) for a in ax])
    pts = np.stack([ax[2][idx[2]], ax[1][idx[1]], ax[0][idx[0]]], 1)  # (a, b, c)
    mu = pts.mean(0)
    cov = np.cov(pts.T, bias=True) + np.diag([abs(ax[2][1]-ax[2][0]), abs(ax[1][1]-ax[1][0]), abs(ax[0][1]-ax[0][0])]) ** 2 / 12
    lam, vec = np.linalg.eigh(cov)
    lam, vec = lam[::-1], vec[:, ::-1]
    r = np.sqrt(5 * lam)
    vol = comp.sum() * dv
    touches = bool(comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()
                   or comp[:, :, 0].any() or comp[:, :, -1].any())
    return dict(level=level, semi_axes=r.tolist(), ratio_r1_r3=float(r[0] / r[2]), ratio_r2_r3=float(r[1] / r[2]),
                ratio_r1_r2=float(r[0] / r[1]), axes_abc=vec.T.tolist(), centroid_abc=mu.tolist(),
                volume=float(vol), fill=float(vol / (4 / 3 * np.pi * np.prod(r))), touches_box=touches,
                voxels_native=int((loss <= level).sum()), other_components_voxels=n_other, supersample=s)


def null_test(axes, s=4):
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    half = min((ax[-1] - ax[0]) / 2 for ax in axes)
    ctr = np.array([(axes[2][0] + axes[2][-1]) / 2, (axes[1][0] + axes[1][-1]) / 2, (axes[0][0] + axes[0][-1]) / 2])
    r_true = np.array([0.9, 0.5, 0.3]) * half / 1.04  # scaled to the box: unchanged on the random-direction grid
    C, B, A = np.meshgrid(*axes, indexing="ij")
    P = (np.stack([A, B, C], -1) - ctr) @ Q  # coordinates in the ellipsoid frame
    f = 0.1 + ((P / r_true) ** 2).sum(-1)  # level 0.1 + 1 is the ellipsoid with semi-axes r_true
    st = shell_stats(f, axes, 1.1, s)
    cosines = [abs(float(np.dot(st["axes_abc"][i], Q[:, i]))) for i in range(3)]
    return dict(r_true=r_true.tolist(), r_est=st["semi_axes"], fill=st["fill"], axis_abs_cos=cosines,
                ratio_true=float(r_true[0] / r_true[2]), ratio_est=st["ratio_r1_r3"])


def plate_repro(loss, A, B, C, plate):
    """c = 0 slab vs a loss-landscape 2D plate at coincident (a, b)."""
    g = np.load(plate)
    gx = np.round(g["xs"], 6); gy = np.round(g["ys"], 6)
    kc = int(np.where(np.round(C, 6) == 0)[0][0])
    ia = [i for i, a in enumerate(np.round(A, 6)) if a in gx]
    jb = [j for j, b in enumerate(np.round(B, 6)) if b in gy]
    ref = np.array([[g["loss"][np.where(gy == round(B[j], 6))[0][0], np.where(gx == round(A[i], 6))[0][0]]
                     for i in ia] for j in jb])
    rel = np.abs(loss[kc][np.ix_(jb, ia)] - ref) / ref
    return dict(plate=os.path.basename(plate), n_points=int(rel.size), max_rel=float(rel.max()),
                median_rel=float(np.median(rel)), p99_rel=float(np.quantile(rel, 0.99)),
                ref_range=[float(ref.min()), float(ref.max())])


def vol_repro(loss, A, B, C, other):
    """coincident grid points of two volumes (e.g. the ep040 17^3 film volume vs the final 27^3 volume)."""
    o = np.load(other)
    def idx(u, w):
        u6, w6 = np.round(u, 6), np.round(w, 6)
        return [i for i in range(len(u)) if u6[i] in w6], [int(np.where(w6 == u6[i])[0][0]) for i in range(len(u)) if u6[i] in w6]
    (ia, oa), (jb, ob), (kc, oc) = idx(A, o["a"]), idx(B, o["b"]), idx(C, o["c"])
    x = loss[np.ix_(kc, jb, ia)]; y = o["loss"][np.ix_(oc, ob, oa)]
    rel = np.abs(x - y) / y
    return dict(other=os.path.basename(other), n_points=int(rel.size), max_rel=float(rel.max()), median_rel=float(np.median(rel)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("vol")
    p.add_argument("--levels", type=float, nargs="*", default=LEVELS)
    p.add_argument("--rel-levels", type=float, nargs="*", default=[2.0, 4.0, 8.0],
                   help="also levels = centre loss x these factors (declared, for the epoch film)")
    p.add_argument("--plates", nargs="*", default=None, help="2D plates for slice reproduction (default: g51 and g101 of the model if present)")
    p.add_argument("--compare-vol", default=None)
    p.add_argument("--s", type=int, default=4)
    args = p.parse_args()
    v = np.load(args.vol); meta = json.loads(str(v["meta"]))
    loss, A, B, C = v["loss"], v["a"], v["b"], v["c"]
    axes = [C, B, A]
    ka, kb, kc = [int(np.argmin(np.abs(x))) for x in (A, B, C)]
    res = dict(vol=args.vol, meta=meta, center_loss=float(loss[kc, kb, ka]),
               loss_min=float(np.nanmin(loss)), loss_max=float(np.nanmax(loss)),
               nonfinite=int((~np.isfinite(loss)).sum()),
               argmin_abc=[float(A[np.unravel_index(np.nanargmin(loss), loss.shape)[2]]),
                           float(B[np.unravel_index(np.nanargmin(loss), loss.shape)[1]]),
                           float(C[np.unravel_index(np.nanargmin(loss), loss.shape)[0]])])
    if meta["dirs"] == "random" and meta.get("epoch") in (None, 40):
        plates = args.plates
        if plates is None:
            plates = [os.path.join(LL_ROOT, "cache", "surf", f"{meta['model']}_final_{t}.npz") for t in ("g51", "g101")]
            plates = [q for q in plates if os.path.exists(q)]
        res["slice_repro"] = [plate_repro(loss, A, B, C, q) for q in plates]
        if res["slice_repro"]:
            res["slice_repro_g51"] = res["slice_repro"][0]
        lnf = os.path.join(LL_ROOT, "cache", "surf", f"{meta['model']}_final_line.npz")
        if os.path.exists(lnf):
            ln = np.load(lnf)
            lx = np.round(ln["xs"], 6)
            pairs = [(i, int(np.where(lx == round(a, 6))[0][0])) for i, a in enumerate(A) if round(a, 6) in lx]
            rl = np.array([abs(loss[kc, kb, i] - ln["loss"][k]) / ln["loss"][k] for i, k in pairs])
            res["line_repro"] = dict(n_points=len(pairs), max_rel=float(rl.max()), median_rel=float(np.median(rl)))
    if args.compare_vol:
        res["vol_repro"] = vol_repro(loss, A, B, C, args.compare_vol)
    levels = list(args.levels) + [res["center_loss"] * f for f in args.rel_levels]
    res["level_kind"] = ["absolute"] * len(args.levels) + [f"centre x {f:g}" for f in args.rel_levels]
    res["anisotropy"] = [shell_stats(loss, axes, L, args.s) for L in levels]
    res["anisotropy_native_grid"] = [shell_stats(loss, axes, L, 1) for L in levels]
    res["null_ellipsoid"] = null_test(axes, args.s)
    json.dump(res, open(args.vol.replace(".npz", ".analysis.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("center_loss", "loss_min", "loss_max", "nonfinite", "argmin_abc") }))
    for k in ("slice_repro", "line_repro", "vol_repro"):
        if k in res:
            print(k, res[k])
    for kind, d in zip(res["level_kind"], res["anisotropy"]):
        if d.get("empty"):
            print(f"  L={d['level']:.4g} ({kind}): empty"); continue
        print(f"  L={d['level']:.4g} ({kind}): r={np.round(d['semi_axes'], 3).tolist()} r1/r3={d['ratio_r1_r3']:.3f} "
              f"r2/r3={d['ratio_r2_r3']:.3f} fill={d['fill']:.3f} touches={d['touches_box']} vox={d['voxels_native']}")
    print("null", round(res["null_ellipsoid"]["ratio_est"], 3), "of", res["null_ellipsoid"]["ratio_true"])
