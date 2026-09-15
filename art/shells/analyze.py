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
    r_true = np.array([0.9, 0.5, 0.3])
    C, B, A = np.meshgrid(*axes, indexing="ij")
    P = np.stack([A, B, C], -1) @ Q  # coordinates in the ellipsoid frame
    f = 0.1 + ((P / r_true) ** 2).sum(-1)  # level 0.1 + 1 is the ellipsoid with semi-axes r_true
    st = shell_stats(f, axes, 1.1, s)
    cosines = [abs(float(np.dot(st["axes_abc"][i], Q[:, i]))) for i in range(3)]
    return dict(r_true=r_true.tolist(), r_est=st["semi_axes"], fill=st["fill"], axis_abs_cos=cosines,
                ratio_true=float(r_true[0] / r_true[2]), ratio_est=st["ratio_r1_r3"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("vol")
    p.add_argument("--levels", type=float, nargs="*", default=LEVELS)
    p.add_argument("--s", type=int, default=4)
    args = p.parse_args()
    v = np.load(args.vol); meta = json.loads(str(v["meta"]))
    loss, A, B, C = v["loss"], v["a"], v["b"], v["c"]
    axes = [C, B, A]
    res = dict(vol=args.vol, meta=meta, center_loss=float(loss[len(C)//2, len(B)//2, len(A)//2]) if meta["dirs"] == "random" else None,
               loss_min=float(np.nanmin(loss)), loss_max=float(np.nanmax(loss)),
               nonfinite=int((~np.isfinite(loss)).sum()),
               argmin_abc=[float(A[np.unravel_index(np.nanargmin(loss), loss.shape)[2]]),
                           float(B[np.unravel_index(np.nanargmin(loss), loss.shape)[1]]),
                           float(C[np.unravel_index(np.nanargmin(loss), loss.shape)[0]])])
    if meta["dirs"] == "random":
        g = np.load(os.path.join(LL_ROOT, "cache", "surf", f"{meta['model']}_final_g51.npz"))
        gx = np.round(g["xs"], 6)
        kc = int(np.where(np.round(C, 6) == 0)[0][0])
        ia = [i for i, a in enumerate(np.round(A, 6)) if a in gx]
        ref = np.array([[g["loss"][np.where(gx == round(B[j], 6))[0][0], np.where(gx == round(A[i], 6))[0][0]]
                         for i in ia] for j in ia])
        ours = loss[kc][np.ix_(ia, ia)]
        rel = np.abs(ours - ref) / ref
        res["slice_repro_g51"] = dict(n_points=int(rel.size), max_rel=float(rel.max()), median_rel=float(np.median(rel)),
                                      p99_rel=float(np.quantile(rel, 0.99)),
                                      argmax_ab=[float(A[ia[np.unravel_index(rel.argmax(), rel.shape)[1]]]),
                                                 float(B[ia[np.unravel_index(rel.argmax(), rel.shape)[0]]])],
                                      ref_range=[float(ref.min()), float(ref.max())])
        ln = np.load(os.path.join(LL_ROOT, "cache", "surf", f"{meta['model']}_final_line.npz"))
        lx = np.round(ln["xs"], 6); jb = int(np.where(np.round(B, 6) == 0)[0][0])
        pairs = [(i, int(np.where(lx == round(a, 6))[0][0])) for i, a in enumerate(A) if round(a, 6) in lx]
        rl = np.array([abs(loss[kc, jb, i] - ln["loss"][k]) / ln["loss"][k] for i, k in pairs])
        res["line_repro"] = dict(n_points=len(pairs), max_rel=float(rl.max()), median_rel=float(np.median(rl)))
    res["anisotropy"] = [shell_stats(loss, axes, L, args.s) for L in args.levels]
    res["anisotropy_native_grid"] = [shell_stats(loss, axes, L, 1) for L in args.levels]
    res["null_ellipsoid"] = null_test(axes, args.s)
    json.dump(res, open(args.vol.replace(".npz", ".analysis.json"), "w"), indent=1)
    out = {k: v for k, v in res.items() if k not in ("meta",)}
    for key in ("anisotropy", "anisotropy_native_grid"):
        out[key] = [{k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in d.items()
                     if k in ("level", "semi_axes", "ratio_r1_r3", "ratio_r2_r3", "fill", "touches_box", "voxels_native", "empty")}
                    for d in res[key]]
    print(json.dumps(out, indent=1))
