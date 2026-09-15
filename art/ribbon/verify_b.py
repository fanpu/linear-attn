"""Verify Stage B (CPU) and build the exact M2 chart axes.

Reads cache/stageB/{log.npz, replay.npz, bank_vecs.npy, theta_f32.npy, thetabar_f16.npy} and
art/edge-of-stability/cache/main4.npz. Writes cache/stageB_verify.json, cache/stageB/axes.npz and
previews in cache/preview/.

  1. lambda_1*eta/2 trace vs main4 (2/eta = 80): median pointwise relative difference, hover medians.
  2. replay exactness (anchor mismatch), cold-start audit, bank eigenvector rotation (bank Gram).
  3. Stage A's sketch method re-tested against EXACT fixed-vector projections over the whole EoS phase.
  4. Axes (declared chart): axis 1 = bank u1 at t_ref (nearest bank step to mid-EoS);
     axes 2-3 = top-2 PCs of the 21-step centred mean theta-bar (every 10 steps, EoS phase),
     orthogonalised against axis 1. Unit vectors in weight space, float32.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
B = os.path.join(HERE, "cache", "stageB")
PREV = os.path.join(HERE, "cache", "preview")
L = np.load(os.path.join(B, "log.npz"))
R = np.load(os.path.join(B, "replay.npz"))
m4 = np.load("/home/fzeng/ml/research/art/edge-of-stability/cache/main4.npz")
T = int(L["steps_done"])
eta, inv = float(L["eta"]), float(L["inv"])
lamB = L["evals"][:T, 0].astype(float)
lamM = m4["evals"][:T, 1, 0].astype(float)
k = int(R["k"])


def t_edge_of(lam):  # verify.py definition
    below = np.flatnonzero(lam < inv)
    fb = int(below[0]) if len(below) else 0
    above = np.flatnonzero((lam >= inv) & (np.arange(len(lam)) > fb))
    return int(above[0])


teB, teM = t_edge_of(lamB), t_edge_of(lamM)
post = np.arange(T) >= max(teB, teM) + 200
rel = np.abs(lamB - lamM) / lamM
out = {"t_edge_B": teB, "t_edge_main4": teM,
       "median_rel_diff_lam_all_steps": float(np.median(rel)),
       "median_rel_diff_lam_after_edge200": float(np.median(rel[post])),
       "p95_rel_diff_lam_after_edge200": float(np.percentile(rel[post], 95)),
       "hover_median_B": float(np.median(lamB[post] * eta / 2)),
       "hover_median_main4": float(np.median(lamM[post] * eta / 2)),
       "hover_p5_p95_B": np.percentile(lamB[post] * eta / 2, [5, 95]).tolist(),
       "hover_p5_p95_main4": np.percentile(lamM[post] * eta / 2, [5, 95]).tolist(),
       "loss_final_B": float(L["loss"][T - 1]), "loss_final_main4": float(m4["loss"][T - 1, 1]),
       "replay_anchor_rel_err_max": float(np.nanmax(R["anchor_err"])),
       "cold_checks": [[int(t), float(v[0]), float(lamB[t])] for t, v in zip(L["checks_t"], L["checks_v"]) if t >= 0],
       "wall_train_s": json.loads(str(L["meta"])).get("wall_train_s"),
       "wall_replay_s": float(R["wall_replay_s"])}
# pointwise divergence of the two trajectories (same seed, float32 nondeterminism) via sketches
sk_div = np.linalg.norm(L["sketch"][:T] - m4["sketch"][:T, 1], axis=1) / np.maximum(np.linalg.norm(m4["sketch"][:T, 1], axis=1), 1e-9)
out["sketch_rel_diff_B_vs_main4_at"] = {int(t): float(sk_div[t]) for t in [10, 100, 405, 1000, 3000, 5999]}

bank_t = L["bank_t"]
bank = np.load(os.path.join(B, "bank_vecs.npy"), mmap_mode="r")
NB = len(bank_t)
U1 = np.asarray(bank[:, 0])
Gb = np.abs(U1 @ U1.T)
out["bank_u1_overlap_lag1_median_after_edge"] = float(np.median(np.diag(Gb, 1)[bank_t[:-1] >= teB]))
out["bank_top3_subspace_overlap_lag1_median_after_edge"] = float(np.median(
    [np.linalg.norm(np.asarray(bank[i]) @ np.asarray(bank[i + 1]).T) ** 2 / 3 for i in range(NB - 1) if bank_t[i] >= teB]))
out["bank_resid_rel_median"] = np.median(L["bank_resid"] / np.abs(L["bank_evals"]), axis=0).tolist()

# ---- 3. Stage A method vs exact, over the whole EoS phase ----
mid = (teB + T) / 2
iref = int(np.argmin(np.abs(bank_t - mid)))
t_ref = int(bank_t[iref])
col = iref * k  # u1 of bank entry iref
proj = R["proj_bank"].astype(np.float64)
exact1 = proj[:, col] - proj[t_ref, col]  # <theta_t - theta_ref, u_ref>
g = torch.Generator().manual_seed(1234)
import eosnet as N  # noqa: E402

idx = torch.randint(0, 1024, (N.P,), generator=g).numpy()
sgn = (torch.randint(0, 2, (N.P,), generator=g) * 2 - 1).numpy()
su = np.zeros(1024)
np.add.at(su, idx, U1[iref].astype(np.float64) * sgn)
su /= np.linalg.norm(su)
sk = L["sketch"][:T].astype(np.float64)
a1 = (sk - sk[t_ref]) @ su
from scipy.ndimage import uniform_filter1d  # noqa: E402


def detr(v):
    o = v - uniform_filter1d(v, 21, mode="nearest")
    o[:10] = np.nan
    o[-10:] = np.nan
    return o


eos = np.arange(T) >= teB
okw = eos & np.isfinite(detr(a1))
d_ex, d_sk = detr(exact1), detr(a1)
c_ex, c_sk = np.r_[np.nan, np.diff(exact1)], np.r_[np.nan, np.diff(a1)]
okc = eos & np.isfinite(c_ex)
near = okw & (np.abs(np.arange(T) - t_ref) <= 200)


def pr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt(a @ a * b @ b))


out["stageA_method_vs_exact_fixed_u_ref"] = {
    "t_ref": t_ref,
    "raw_r_eos": pr(a1[eos], exact1[eos]),
    "raw_rms_err_eos": float(np.sqrt(np.mean((a1[eos] - exact1[eos]) ** 2))),
    "raw_rms_exact_eos": float(np.sqrt(np.mean(exact1[eos] ** 2))),
    "detrended_r_eos": pr(d_sk[okw], d_ex[okw]),
    "detrended_sign_eos": float(np.mean(np.sign(d_sk[okw]) == np.sign(d_ex[okw]))),
    "chord_r_eos": pr(c_sk[okc], c_ex[okc]),
    "chord_sign_eos": float(np.mean(np.sign(c_sk[okc]) == np.sign(c_ex[okc]))),
    "detrended_r_tref_pm200": pr(d_sk[near], d_ex[near]),
    "detrended_sign_tref_pm200": float(np.mean(np.sign(d_sk[near]) == np.sign(d_ex[near]))),
}

# ---- 4. axes ----
TE = 10
tb = np.load(os.path.join(B, "thetabar_f16.npy"), mmap_mode="r")
ts = np.arange(tb.shape[0]) * TE
rows = np.flatnonzero((ts >= teB) & (ts >= 10) & (ts <= T - 1 - 10))
Zs = np.asarray(tb[rows]).astype(np.float32)
u = U1[iref].astype(np.float64)
Zm = Zs.mean(0)
Z = (Zs - Zm).astype(np.float64)
del Zs
a_u = Z @ u
tot = float((Z ** 2).sum())
Z -= np.outer(a_u, u)
Gz = Z @ Z.T
w, Q = np.linalg.eigh(Gz)
w, Q = w[::-1], Q[:, ::-1]
pcs = (Q[:, :2].T @ Z) / np.sqrt(w[:2])[:, None]
pcs -= np.outer(pcs @ u, u)  # re-orthogonalise (numerical)
pcs[0] /= np.linalg.norm(pcs[0])
pcs[1] -= (pcs[1] @ pcs[0]) * pcs[0]
pcs[1] /= np.linalg.norm(pcs[1])
axes = np.stack([u, pcs[0], pcs[1]]).astype(np.float32)
theta_ref = np.asarray(np.load(os.path.join(B, "theta_f32.npy"), mmap_mode="r")[t_ref // TE])
out["axes"] = {"t_ref": t_ref, "axis1": f"bank u1 at step {t_ref} (lambda={float(L['bank_evals'][iref, 0]):.2f})",
               "pc_variance_fraction_of_smoothed_eos_theta": (w[:2] / tot).tolist(),
               "axis1_variance_fraction_of_smoothed_eos_theta": float((a_u ** 2).sum() / tot),
               "gram_of_axes": (axes.astype(np.float64) @ axes.T.astype(np.float64)).round(6).tolist(),
               "n_thetabar_rows": int(len(rows))}
np.savez(os.path.join(B, "axes.npz"), axes=axes, theta_ref=theta_ref, t_ref=t_ref, t_edge=teB, eta=eta)
json.dump(out, open(os.path.join(HERE, "cache", "stageB_verify.json"), "w"), indent=1)
print(json.dumps({kk: v for kk, v in out.items() if kk != "cold_checks"}, indent=1))

# ---- previews ----
fig, ax = plt.subplots(3, 1, figsize=(14, 9))
tt = np.arange(T)
ax[0].plot(tt, lamM * eta / 2, lw=0.3, c="k", label="main4 (2/η=80)")
ax[0].plot(tt, lamB * eta / 2, lw=0.3, c="C3", alpha=0.7, label="Stage B")
ax[0].axhline(1, c="C0", lw=0.6)
ax[0].set(ylim=(0.8, 1.2), title="λ₁·η/2", xlabel="step")
ax[0].legend(fontsize=8)
w_ = slice(t_ref - 200, t_ref + 200)
ax[1].plot(tt[w_], d_ex[w_], lw=0.5, c="k", label="exact ⟨θ−θ̄, u_ref⟩ (fixed)")
ax[1].plot(tt[w_], d_sk[w_], lw=0.5, c="C3", alpha=0.7, label="sketch estimate")
ax[1].legend(fontsize=8)
ax[1].set(title=f"detrended axis 1 near t_ref={t_ref}")
ax[2].plot(tt[teB:], exact1[teB:], lw=0.3, c="k", label="exact ⟨θ−θ_ref, u_ref⟩")
ax[2].plot(tt[teB:], a1[teB:], lw=0.3, c="C3", alpha=0.7, label="sketch estimate")
ax[2].legend(fontsize=8)
ax[2].set(title="axis 1, not detrended, EoS phase", xlabel="step")
fig.tight_layout()
fig.savefig(os.path.join(PREV, "stageB_verify.png"), dpi=110)
plt.close(fig)
print("wrote", os.path.join(PREV, "stageB_verify.png"))
