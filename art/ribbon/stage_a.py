"""Stage A (CPU): a 3D ribbon chart in count-sketch space from art/edge-of-stability/cache/main4.npz.

Chart (declared). The count-sketch S: R^P -> R^1024 (random bucket + random sign per parameter) is a
random linear projection that preserves inner products only approximately: for unit u and any v,
<Sv, Su> - <v, u> has standard deviation about |v| / sqrt(1024).
  axis 1  a1_t = <S(theta_t - theta_0) - S(theta_ref - theta_0), s_ref>, s_ref = S u1(t_ref) / |S u1(t_ref)|
  axes 2-3  top-2 principal components of the 21-step centred mean of S(theta_t - theta_0) over the
          EoS phase, after projecting out s_ref; coordinates are <S(theta_t - theta_ref), pc_j>.

Validation against the stored exact series (main4 has no exact projection onto a FIXED vector; it has
x_t = <theta_t - thetabar_t, u1(t)> and dtheta_u1_t = <theta_t - theta_{t-1}, u1(t)> along the CURRENT
u1(t)). On steps where |cos(u1(t), u1(t_ref))| >= thr the two agree up to the overlap, so:
  coordinate test: detrended a1 (same 21-step centred mean) vs sign-aligned x_t
  chord (flip) test: a1_t - a1_{t-1} vs sign-aligned dtheta_u1_t
The per-step overlap is the sketch cosine of u1_sketch; it is checked against the exact 100-step
Gram matrix u1_gram.

Usage: stage_a.py [t_ref]   -> cache/stageA.npz, cache/stageA_verify.json, cache/preview/stageA_*.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/home/fzeng/ml/research/art/edge-of-stability/cache/main4.npz"
PREV = os.path.join(HERE, "cache", "preview")
os.makedirs(PREV, exist_ok=True)
M, HALF = 1, 10  # model index of 2/eta = 80 in main4; centred mean half-width (source)
T_EDGE = 405  # main4_verify.json, 2/eta = 80

d = np.load(SRC)
assert float(d["invs"][M]) == 80.0
eta = float(d["eta"][M])
T = int(d["steps_done"])
s = d["sketch"][:T, M].astype(np.float64)
us = d["u1_sketch"][:T, M].astype(np.float64)
x = d["x"][:T, M].astype(np.float64)
dth = d["dtheta_u1"][:T, M].astype(np.float64)
lam = d["evals"][:T, M, 0].astype(np.float64)
snap_t, G = d["snap_t"], d["u1_gram"][M]
t_ref = int(sys.argv[1]) if len(sys.argv) > 1 else 3200
eos = np.arange(T) >= T_EDGE
W = 2 * HALF + 1


def centred_mean(a):
    """21-step centred mean; NaN where the window is incomplete (as x in the source)."""
    out = uniform_filter1d(a, size=W, axis=0, mode="nearest")
    out[:HALF] = np.nan
    out[T - HALF:] = np.nan
    return out


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


# ---- per-step overlap with the reference eigenvector (sketch cosine), checked against the exact Gram ----
un = us / np.linalg.norm(us, axis=1, keepdims=True)
Gs = np.abs(un[snap_t] @ un[snap_t].T)
iu = np.triu_indices(len(snap_t), 1)
gram_check = {"corr_exact_vs_sketch": pearson(G[iu], Gs[iu]),
              "mean_abs_diff": float(np.abs(G[iu] - Gs[iu]).mean()),
              "p95_abs_diff": float(np.percentile(np.abs(G[iu] - Gs[iu]), 95))}


def validate(tr, thr_list=(0.8, 0.9, 0.95)):
    sref = us[tr] / np.linalg.norm(us[tr])
    a1 = (s - s[tr]) @ sref
    cos = un @ (us[tr] / np.linalg.norm(us[tr]))
    sg = np.sign(cos)
    a1_det = a1 - centred_mean(a1[:, None])[:, 0]
    da1 = np.r_[np.nan, np.diff(a1)]
    xe, de = x * sg, dth * sg
    out = {}
    for thr in thr_list:
        ok = eos & (np.abs(cos) >= thr)
        okc = ok & np.isfinite(a1_det) & np.isfinite(xe)
        okd = ok & np.isfinite(da1) & np.isfinite(de)
        r = {"n_steps_coord": int(okc.sum()), "n_steps_chord": int(okd.sum()),
             "coverage_of_eos": float(okc.sum() / eos.sum())}
        if okc.sum() >= 10:
            r["r_coord"] = pearson(a1_det[okc], xe[okc])
            r["sign_coord"] = float(np.mean(np.sign(a1_det[okc]) == np.sign(xe[okc])))
            big = okc & (np.abs(xe) > np.nanmedian(np.abs(xe[okc])) * 0.25)
            r["sign_coord_above_quarter_median_amp"] = float(np.mean(np.sign(a1_det[big]) == np.sign(xe[big])))
        if okd.sum() >= 10:
            r["r_chord"] = pearson(da1[okd], de[okd])
            r["sign_chord"] = float(np.mean(np.sign(da1[okd]) == np.sign(de[okd])))
        out[f"thr{thr}"] = r
    return out, a1, a1_det, cos


res_ref, a1, a1_det, cos = validate(t_ref)
# piecewise references: every snapshot step in the EoS phase, each validated on its own window
pw = {}
for thr in (0.9,):
    xs, ys, xd, yd, cover = [], [], [], [], np.zeros(T, bool)
    for tr in snap_t[snap_t >= T_EDGE]:
        _, a1r, a1dr, cr = validate(int(tr), thr_list=())
        ok = eos & (np.abs(cr) >= thr) & (np.abs(np.arange(T) - tr) <= 50)  # nearest-reference window
        sg = np.sign(cr)
        okc = ok & np.isfinite(a1dr) & np.isfinite(x)
        xs.append(a1dr[okc]); ys.append((x * sg)[okc])
        da = np.r_[np.nan, np.diff(a1r)]
        okd = ok & np.isfinite(da) & np.isfinite(dth)
        xd.append(da[okd]); yd.append((dth * sg)[okd])
        cover |= okc
    X1, Y1, X2, Y2 = map(np.concatenate, (xs, ys, xd, yd))
    pw[f"thr{thr}"] = {"n": int(len(X1)), "coverage_of_eos": float(cover[eos].mean()),
                       "r_coord": pearson(X1, Y1), "sign_coord": float(np.mean(np.sign(X1) == np.sign(Y1))),
                       "r_chord": pearson(X2, Y2), "sign_chord": float(np.mean(np.sign(X2) == np.sign(Y2)))}

# ---- fraction of EoS steps close to ANY single reference (how long a fixed u1 stays u1) ----
cover_by_ref = {}
for tr in snap_t[snap_t >= T_EDGE][::5]:
    c = np.abs(un @ un[tr])
    cover_by_ref[int(tr)] = float((c[eos] >= 0.9).mean())

# ---- axes 2-3: PCs of the smoothed sketch trajectory over the EoS phase, orthogonal to s_ref ----
sref = us[t_ref] / np.linalg.norm(us[t_ref])
sb = centred_mean(s)
Z = sb[eos & np.isfinite(sb[:, 0])]
Z = Z - Z.mean(0)
Zp = Z - np.outer(Z @ sref, sref)
_, sv, Vt = np.linalg.svd(Zp, full_matrices=False)
pc = Vt[:2]
var_total = float((Z ** 2).sum())
var_pc = (sv[:2] ** 2 / var_total).tolist()
var_axis1 = float(((Z @ sref) ** 2).sum() / var_total)
coords = np.stack([a1, (s - s[t_ref]) @ pc[0], (s - s[t_ref]) @ pc[1]], 1)

# ---- decision ----
main = res_ref["thr0.9"]
passes = bool(main.get("r_coord", 0) >= 0.9 and main.get("sign_chord", 0) >= 0.95
              and main.get("sign_coord", 0) >= 0.95 and main["coverage_of_eos"] >= 0.5)
V = {"source": SRC, "model": "2/eta=80", "t_ref": t_ref, "t_edge": T_EDGE, "eos_steps": int(eos.sum()),
     "gram_check": gram_check, "single_reference": res_ref, "piecewise_references_100": pw,
     "coverage_by_reference_thr0.9": cover_by_ref,
     "axes23_variance_fraction_of_smoothed_eos_sketch": var_pc, "axis1_variance_fraction": var_axis1,
     "decision_stageA_is_M2_source": passes,
     "rule": "r_coord>=0.9 and sign(coord,chord)>=0.95 over thr0.9 windows, windows covering >=50% of EoS"}
json.dump(V, open(os.path.join(HERE, "cache", "stageA_verify.json"), "w"), indent=1)
np.savez(os.path.join(HERE, "cache", "stageA.npz"), coords=coords.astype(np.float32), a1_det=a1_det,
         cos_ref=cos, sref=sref, pc=pc, t_ref=t_ref, lam_eta_2=lam * eta / 2, var_pc=var_pc)
print(json.dumps({k: V[k] for k in ["gram_check", "single_reference", "piecewise_references_100",
                                    "coverage_by_reference_thr0.9", "axes23_variance_fraction_of_smoothed_eos_sketch",
                                    "axis1_variance_fraction", "decision_stageA_is_M2_source"]}, indent=1))

# ---- previews (matplotlib; not gallery renders) ----
okc = eos & (np.abs(cos) >= 0.9) & np.isfinite(a1_det) & np.isfinite(x)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].plot(np.arange(T), np.abs(cos), lw=0.3, c="k")
ax[0].plot(snap_t, G[np.searchsorted(snap_t, t_ref)], "o", ms=3, c="C3", label="exact Gram (100-step snaps)")
ax[0].axhline(0.9, c="C0", lw=0.8)
ax[0].set(title=f"|cos(u1(t), u1({t_ref}))| (sketch)", xlabel="step")
ax[0].legend(fontsize=8)
sgn = np.sign(cos)
ax[1].scatter((x * sgn)[okc], a1_det[okc], s=2, c="k")
ax[1].set(title=f"coordinate: r={main.get('r_coord', np.nan):.3f}, sign={main.get('sign_coord', np.nan):.3f}",
          xlabel="exact <θ-θ̄, u1(t)> (sign-aligned)", ylabel="sketch axis 1, detrended")
da1 = np.r_[np.nan, np.diff(a1)]
okd = eos & (np.abs(cos) >= 0.9) & np.isfinite(da1) & np.isfinite(dth)
ax[2].scatter((dth * sgn)[okd], da1[okd], s=2, c="k")
ax[2].set(title=f"chord: r={main.get('r_chord', np.nan):.3f}, sign={main.get('sign_chord', np.nan):.3f}",
          xlabel="exact <θt-θt-1, u1(t)> (sign-aligned)", ylabel="Δ sketch axis 1")
fig.tight_layout()
fig.savefig(os.path.join(PREV, f"stageA_validation_ref{t_ref}.png"), dpi=110)
plt.close(fig)

w = slice(t_ref - 200, t_ref + 200)
tt = np.arange(T)[w]
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 3, 1, projection="3d")
C = coords[w]
ax.plot(C[:, 1], C[:, 2], C[:, 0], lw=0.3, c="0.6")
ax.scatter(C[::2, 1], C[::2, 2], C[::2, 0], s=2, c="C0")
ax.scatter(C[1::2, 1], C[1::2, 2], C[1::2, 0], s=2, c="C3")
ax.set(xlabel="pc1", ylabel="pc2", zlabel="axis 1", title=f"sketch chart, steps {tt[0]}-{tt[-1]}")
ax = fig.add_subplot(1, 3, 2)
ax.plot(tt, C[:, 0], lw=0.4, c="k")
ax.set(title="axis 1 (not detrended) — slow part carries sketch cross-talk", xlabel="step")
ax = fig.add_subplot(1, 3, 3)
ax.plot(tt, a1_det[w], lw=0.4, c="k", label="sketch, detrended")
ax.plot(tt, (x * sgn)[w], lw=0.4, c="C3", alpha=0.7, label="exact x (moving u1)")
ax.legend(fontsize=8)
ax.set(xlabel="step", title="detrended axis 1 vs exact x")
fig.tight_layout()
fig.savefig(os.path.join(PREV, f"stageA_chart_ref{t_ref}.png"), dpi=110)
plt.close(fig)
print("wrote previews")
