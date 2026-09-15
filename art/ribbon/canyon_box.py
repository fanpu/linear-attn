"""Canyon loss volumes on arbitrary boxes (GPU), for M2.

L(theta_ref + a u_ref + b pc1 + c pc2) on an n^3 grid of chart offsets (a, b, c) spanning the boxes in
cache/m2/boxes.json (lo/hi are the grid's corner-voxel centres, as in r3d). Full-batch MSE on the 5000
training images; float32 forward, float64 sums. Same fused first layer as stage_c.py (the first layer is
linear in the offsets), checked against the plain forward. Output cache/m2/canyon_<name>.npz with
loss[k, j, i] at (a_k, b_j, c_i) and a per-slab checkpoint.
Usage: [CANYON_DEVICE=cpu] canyon_box.py name [name ...]
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

dev, dt = os.environ.get("CANYON_DEVICE", "cuda"), torch.float32
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.10)
else:
    torch.set_num_threads(4)
OUT = os.path.join(HERE, "cache", "m2")
BATCH = 64
A = np.load(os.path.join(HERE, "cache", "stageB", "axes.npz"))
E = torch.tensor(A["axes"], device=dev)
th_ref = torch.tensor(A["theta_ref"], device=dev)[None]
X, Y, _ = N.load_data(5000, dt, dev)
boxes = json.load(open(os.path.join(OUT, "boxes.json")))
ps_ref, ps_e = N.unflatten(th_ref), N.unflatten(E)
H1_ref = X @ ps_ref[0][0].T + ps_ref[1][0]
H1_e = torch.stack([X @ ps_e[0][j].T + ps_e[1][j] for j in range(3)])


def loss_fused(C):
    Ct = torch.tensor(C, dtype=dt, device=dev)
    h = torch.tanh(H1_ref[None] + torch.einsum("bj,jnh->bnh", Ct, H1_e))
    W2 = ps_ref[2] + torch.einsum("bj,jkl->bkl", Ct, ps_e[2])
    h = torch.tanh(h @ W2.transpose(1, 2) + (ps_ref[3] + Ct @ ps_e[3])[:, None])
    W4 = ps_ref[4] + torch.einsum("bj,jkl->bkl", Ct, ps_e[4])
    o = h @ W4.transpose(1, 2) + (ps_ref[5] + Ct @ ps_e[5])[:, None]
    return (0.5 * ((o - Y) ** 2).double().sum(-1).mean(-1)).cpu().numpy()


def loss_plain(C):
    o = N.forward(th_ref + torch.tensor(C, dtype=dt, device=dev) @ E, X)
    return (0.5 * ((o - Y) ** 2).double().sum(-1).mean(-1)).cpu().numpy()


for name in sys.argv[1:]:
    bx = boxes[name]
    n = bx["n"]
    ax = [np.linspace(bx["lo"][j], bx["hi"][j], n) for j in range(3)]
    Ct = np.random.default_rng(1).uniform(bx["lo"], bx["hi"], (4, 3))
    chk = float(np.max(np.abs(loss_fused(Ct) - loss_plain(Ct)) / loss_plain(Ct)))
    assert chk < 1e-4, chk
    part = os.path.join(OUT, f"canyon_{name}_partial.npz")
    vol, i0 = np.full((n, n, n), np.nan), 0
    if os.path.exists(part):
        Pp = np.load(part)
        vol, i0 = Pp["vol"], int(Pp["i_next"])
    t0 = time.time()
    bb, cc = np.meshgrid(ax[1], ax[2], indexing="ij")
    for i in range(i0, n):
        C = np.stack([np.full(n * n, ax[0][i]), bb.ravel(), cc.ravel()], 1)
        vol[i] = np.concatenate([loss_fused(C[s:s + BATCH]) for s in range(0, len(C), BATCH)]).reshape(n, n)
        if (i + 1) % 4 == 0 or i == n - 1:
            np.savez(part + ".tmp.npz", vol=vol, i_next=i + 1)
            os.replace(part + ".tmp.npz", part)
    wall = time.time() - t0
    np.savez(os.path.join(OUT, f"canyon_{name}.npz"), loss=vol.astype(np.float32), a=ax[0], b=ax[1], c=ax[2],
             lo=bx["lo"], hi=bx["hi"], fused_check=chk, wall_s=wall, device=(torch.cuda.get_device_name(0) if dev == "cuda" else "cpu (GB10 Grace, 4 threads)"),
             torch=torch.__version__)
    print(f"{name}: n={n} loss {vol.min():.4f}..{vol.max():.4f} fused_check {chk:.1e} {wall:.0f}s", flush=True)
    torch.cuda.empty_cache()
print("canyon_box done", flush=True)
