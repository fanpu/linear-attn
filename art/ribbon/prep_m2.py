"""M2 data preparation (CPU): chart coordinates for the hero window, the honesty panel and the context
plate, plus the canyon boxes for canyon_box.py. Writes cache/m2/{hero.npz, context.npz, boxes.json}.

Hero chart (fixed frame, declared): axis 1 = <theta_t - theta_ref, u_ref>, u_ref = bank u1 at t_ref = 3250;
axes 2-3 = top-2 PCs of the 21-step centred-mean trajectory (Stage B, verify_b.py). Exact per-step
values from cache/stageC/coords.npz.
Honesty panel (declared): the oscillation part of axis 1 is replaced by the stored moving-frame
x_t = <theta_t - thetabar_t, u1(t)>; the slow part <thetabar_t - theta_ref, u_ref> and axes 2-3 are kept,
so the two panels differ only in which eigenvector the flip is measured along.
Context chart (piecewise frames, declared): axis 1 = sum_i w_i(t) <theta_t - thetabar_t, v_i>, where
v_i is the principal oscillation direction inside the bank top-3 subspace B_i at bank step i (top
eigenvector of the 3x3 covariance of the detrended projections over the 250 steps centred on it),
sign-aligned to v_{i-1} by the exact weight-space inner product, and w_i triangular weights between
adjacent bank centres. Axes 2-3 as in the hero.
"""
import json
import os

import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "m2")
os.makedirs(OUT, exist_ok=True)
L = np.load(os.path.join(HERE, "cache", "stageB", "log.npz"))
R = np.load(os.path.join(HERE, "cache", "stageB", "replay.npz"))
C = np.load(os.path.join(HERE, "cache", "stageC", "coords.npz"))
X, t_ref, t_edge = C["coords"].astype(np.float64), int(C["t_ref"]), int(C["t_edge"])
T = len(X)
eta = float(L["eta"])
lam = L["evals"][:, 0].astype(np.float64) * eta / 2
bank_t = L["bank_t"]
NB, k = len(bank_t), int(R["k"])
P = R["proj_bank"].astype(np.float64)


def cmean(v):
    return uniform_filter1d(v, 21, axis=0, mode="nearest")


# ---------------- hero window ----------------
W0, W1 = 3050, 3450  # Decision logged in NOTES: centred on t_ref, three clear bursts
w = np.arange(W0, W1)
iref = int(np.flatnonzero(bank_t == t_ref)[0])
Dref = P[:, iref * k:(iref + 1) * k] - cmean(P[:, iref * k:(iref + 1) * k])
osc_fixed = X[:, 0] - cmean(X[:, 0])
slow = cmean(X[:, 0])
x_moving = L["xyz"][:, 0].astype(np.float64)
Xm = X.copy()
Xm[:, 0] = slow + np.nan_to_num(x_moving)
capture = float((Dref[w, 0] ** 2).sum() / (Dref[w] ** 2).sum())
u1_swaps = float(np.mean(L["u1_overlap_prev"][w] < 0.95))
np.savez(os.path.join(OUT, "hero.npz"), t=w, fixed=X[w], moving=Xm[w], lam=lam[w], t_ref=t_ref,
         capture_u_ref_of_top3=capture)

# ---------------- context: piecewise frames ----------------
bank = np.load(os.path.join(HERE, "cache", "stageB", "bank_vecs.npy"), mmap_mode="r")
V = []
frac = []
for i in range(NB):
    lo, hi = max(t_edge, bank_t[i] - 125), min(T - 10, bank_t[i] + 125)
    Di = P[:, i * k:(i + 1) * k] - cmean(P[:, i * k:(i + 1) * k])
    if hi <= lo:
        V.append(np.array([1.0, 0, 0]))
        frac.append(np.nan)
        continue
    Cv = Di[lo:hi].T @ Di[lo:hi]
    ev, Q = np.linalg.eigh(Cv)
    V.append(Q[:, -1])
    frac.append(float(ev[-1] / ev.sum()))
V = np.array(V)  # coefficients in bank frame i
align = [np.nan]
for i in range(1, NB):
    Bi, Bp = np.asarray(bank[i], np.float64), np.asarray(bank[i - 1], np.float64)
    s = V[i] @ (Bi @ Bp.T) @ V[i - 1]
    if s < 0:
        V[i] = -V[i]
    align.append(float(abs(s)))
centres = bank_t.astype(np.float64)
a_pw = np.zeros(T)
wsum = np.zeros(T)
tt = np.arange(T)
for i in range(NB):
    Di = P[:, i * k:(i + 1) * k] - cmean(P[:, i * k:(i + 1) * k])
    wi = np.clip(1 - np.abs(tt - centres[i]) / 250.0, 0, None)
    if i == 0:
        wi[tt < centres[0]] = 1
    if i == NB - 1:
        wi[tt > centres[-1]] = 1
    a_pw += wi * (Di @ V[i])
    wsum += wi
a_pw /= wsum
ctx_t = np.arange(t_edge, T - 10)
Xc = X[ctx_t].copy()
Xc[:, 0] = a_pw[ctx_t]
np.savez(os.path.join(OUT, "context.npz"), t=ctx_t, coords=Xc, lam=lam[ctx_t], frame_frac=np.array(frac),
         frame_align=np.array(align), bank_t=bank_t)

# ---------------- canyon boxes (chart offsets from theta_ref along u_ref, pc1, pc2) ----------------
MARGIN = 0.30  # declared: each side, fraction of the trajectory's range
hlo, hhi = X[w].min(0), X[w].max(0)
rng = hhi - hlo
boxes = {"hero32": {"lo": (hlo - MARGIN * rng).tolist(), "hi": (hhi + MARGIN * rng).tolist(), "n": 32},
         "hero64": {"lo": (hlo - MARGIN * rng).tolist(), "hi": (hhi + MARGIN * rng).tolist(), "n": 64}}
clo, chi = X[ctx_t].min(0), X[ctx_t].max(0)
crng = chi - clo
glo, ghi = clo - 0.10 * crng, chi + 0.10 * crng
amax = np.abs(a_pw[ctx_t]).max()
glo[0], ghi[0] = -1.3 * amax, 1.3 * amax  # the fixed u_ref axis, centred on theta_ref
boxes["context32"] = {"lo": glo.tolist(), "hi": ghi.tolist(), "n": 32}
json.dump(boxes, open(os.path.join(OUT, "boxes.json"), "w"), indent=1)
info = {"hero_window": [W0, W1], "capture_u_ref_of_top3_detrended_energy": capture,
        "u1_swap_fraction_in_window": u1_swaps, "lam_range_window": [float(lam[w].min()), float(lam[w].max())],
        "hero_range": {"lo": hlo.tolist(), "hi": hhi.tolist()},
        "context_range": {"lo": clo.tolist(), "hi": chi.tolist()}, "context_amax": float(amax),
        "frame_frac_median": float(np.nanmedian(frac)), "frame_align_median": float(np.nanmedian(align)),
        "frame_align_min": float(np.nanmin(align)), "boxes": boxes}
json.dump(info, open(os.path.join(OUT, "prep_info.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
