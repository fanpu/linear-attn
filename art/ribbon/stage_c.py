"""Stage C (GPU): exact per-step chart coordinates and the 32^3 canyon loss volumes.

Inputs: cache/stageB/axes.npz (from verify_b.py), cache/stageB/theta_f32.npy (float32 anchors).
  1. coords (T, 3) = <theta_t - theta_ref, axis_j> for EVERY step, replayed from the float32 anchors
     (<= 9 gradient steps each, as in stage_b.py); also coords of the 21-step centred mean.
  2. Loss volumes L(theta_ref + a e1 + b e2 + c e3) on 32^3 grids (cell-centred linspace including
     both ends), full-batch MSE on the 5000 training images, float32 forward, float64 sums.
       global: span +-1.5 x max|coord_j| over the EoS phase (t >= t_edge)
       local : span +-1.5 x max|coord_j| over t_ref +- 200 steps
     The first layer is linear in the grid offset, so X W0^T is precomputed per axis; a fused-path
     check against the plain forward is logged.
Checkpoint: each grid is written per 32x32 slab (cache/stageC/<grid>_partial.npz); reruns resume.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eosnet as N  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.10)
dev, dt = "cuda", torch.float32
B = os.path.join(HERE, "cache", "stageB")
OUT = os.path.join(HERE, "cache", "stageC")
os.makedirs(OUT, exist_ok=True)
G, BATCH, TE, HALF = 32, 64, 10, 10
A = np.load(os.path.join(B, "axes.npz"))
t_ref, t_edge, eta = int(A["t_ref"]), int(A["t_edge"]), float(A["eta"])
E = torch.tensor(A["axes"], device=dev)  # (3, P)
th_ref = torch.tensor(A["theta_ref"], device=dev)[None]
th_mm = np.load(os.path.join(B, "theta_f32.npy"), mmap_mode="r")
NA = th_mm.shape[0]
T = NA * TE
X, Y, _ = N.load_data(5000, dt, dev)
meta = {"t_ref": t_ref, "t_edge": t_edge, "grid": G, "dtype": "float32 forward, float64 loss sums",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

# ---- 1. coordinates for every step ----
cpath = os.path.join(OUT, "coords.npz")
t0 = time.time()
if os.path.exists(cpath):
    coords = np.load(cpath)["coords"]
else:
    coords = np.full((T, 3), np.nan, np.float64)
    for a in range(NA):
        th = torch.tensor(np.asarray(th_mm[a]), device=dev)[None]
        for j in range(TE):
            t = a * TE + j
            coords[t] = (E @ (th - th_ref)[0]).double().cpu().numpy()
            if j < TE - 1:
                th = th - eta * N.grad_hvps(th, None, X, Y)[2]
    np.savez(cpath, coords=coords, t_ref=t_ref)
meta["wall_coords_s"] = time.time() - t0
torch.cuda.empty_cache()
from scipy.ndimage import uniform_filter1d  # noqa: E402

cbar = uniform_filter1d(coords, 2 * HALF + 1, axis=0, mode="nearest")
cbar[:HALF] = np.nan
cbar[T - HALF:] = np.nan
tt = np.arange(T)
ext = {"global": np.abs(coords[tt >= t_edge]).max(0),
       "local": np.abs(coords[np.abs(tt - t_ref) <= 200]).max(0)}

# ---- 2. loss grids ----
ps_ref = N.unflatten(th_ref)
ps_e = N.unflatten(E)  # each (3, ...)
H1_ref = X @ ps_ref[0][0].T + ps_ref[1][0]  # (n, 200)
H1_e = torch.stack([X @ ps_e[0][j].T + ps_e[1][j] for j in range(3)])  # (3, n, 200)


def loss_fused(C):  # C (b, 3) float64 offsets -> (b,) loss
    Ct = torch.tensor(C, dtype=dt, device=dev)
    z1 = H1_ref[None] + torch.einsum("bj,jnh->bnh", Ct, H1_e)
    h = torch.tanh(z1)
    W2 = ps_ref[2] + torch.einsum("bj,jkl->bkl", Ct, ps_e[2])
    b2 = ps_ref[3] + Ct @ ps_e[3]
    h = torch.tanh(h @ W2.transpose(1, 2) + b2[:, None])
    W4 = ps_ref[4] + torch.einsum("bj,jkl->bkl", Ct, ps_e[4])
    b4 = ps_ref[5] + Ct @ ps_e[5]
    o = h @ W4.transpose(1, 2) + b4[:, None]
    return (0.5 * ((o - Y) ** 2).double().sum(-1).mean(-1)).cpu().numpy()


def loss_plain(C):
    th = th_ref + torch.tensor(C, dtype=dt, device=dev) @ E
    o = N.forward(th, X)
    return (0.5 * ((o - Y) ** 2).double().sum(-1).mean(-1)).cpu().numpy()


rng = np.random.default_rng(0)
Ctest = rng.uniform(-1, 1, (4, 3)) * ext["global"]
fused_check = float(np.max(np.abs(loss_fused(Ctest) - loss_plain(Ctest)) / loss_plain(Ctest)))
meta["fused_vs_plain_max_rel"] = fused_check
print(f"fused vs plain max rel diff {fused_check:.2e}", flush=True)
assert fused_check < 1e-4
for name in ("local", "global"):
    span = 1.5 * ext[name]
    ax = [np.linspace(-s, s, G) for s in span]
    part = os.path.join(OUT, f"{name}_partial.npz")
    vol = np.full((G, G, G), np.nan)
    i0 = 0
    if os.path.exists(part):
        Pp = np.load(part)
        vol, i0 = Pp["vol"], int(Pp["i_next"])
    tg = time.time()
    for i in range(i0, G):
        bb, cc = np.meshgrid(ax[1], ax[2], indexing="ij")
        C = np.stack([np.full(G * G, ax[0][i]), bb.ravel(), cc.ravel()], 1)
        vals = np.concatenate([loss_fused(C[s:s + BATCH]) for s in range(0, len(C), BATCH)])
        vol[i] = vals.reshape(G, G)
        np.savez(part + ".tmp.npz", vol=vol, i_next=i + 1)
        os.replace(part + ".tmp.npz", part)
    meta[f"wall_{name}_s"] = time.time() - tg
    ref_loss = float(loss_plain(np.zeros((1, 3)))[0])
    np.savez(os.path.join(OUT, f"canyon_{name}.npz"), loss=vol.astype(np.float32), ax1=ax[0], ax2=ax[1],
             ax3=ax[2], span=span, extent=ext[name], t_ref=t_ref, loss_ref=ref_loss)
    print(f"{name}: span {span} loss range {np.nanmin(vol):.4f}..{np.nanmax(vol):.4f} ref {ref_loss:.4f} "
          f"{meta[f'wall_{name}_s']:.0f}s", flush=True)
    torch.cuda.empty_cache()
np.savez(os.path.join(OUT, "coords.npz"), coords=coords, coords_bar=cbar, t_ref=t_ref, t_edge=t_edge)
json.dump(meta | {k: v.tolist() for k, v in ext.items()}, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("stage_c done", flush=True)
