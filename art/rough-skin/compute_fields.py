"""GPU: exact finite-width Heaviside and ReLU nets (same weights) on the exp-map cube patch of S^3.

    compute_fields.py <width> <stride> [seed]     (seed 7 is the main draw; other seeds get a _s<seed> suffix)
      stride 2 -> 128^3 grid (even nodes of the 256^3 grid), stride 1 -> 256^3 grid

Writes (resumable, one checkpoint per z-plane):
    cache/chunks/w<n>_r<R>/z<iz>.npy     float32 [2 acts, L, R, R]
    cache/field_w<n>_r<R>.npy            float32 [2, L, R, R, R]  (assembled; axis order x, y, z)
    cache/slab64_w<n>_r<R>.npz           float32 and float64 outputs on 32 central z-planes, hidden-code flip counts
    cache/slices_exact_w<n>.npy          float32 [2, L, 12, 176, 176], exact net on the 12 oblique planes (256^3 spacing)
    cache/timing_w<n>_r<R>.json
"""
import json, os, sys, time
import numpy as np
import torch
from common import *

torch.cuda.set_per_process_memory_fraction(0.10)
dev = torch.device("cuda")
n = int(sys.argv[1]); stride = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
R = 256 // stride
tag = f"w{n}_r{R}" + (f"_s{seed}" if seed != 7 else "")
os.makedirs(f"cache/chunks/{tag}", exist_ok=True)
ax = grid_coords(256)[::stride]
E = tangent_frame()

W0, v, Ws = make_weights(n, seed)
W32 = [W0.float().to(dev), v.float().to(dev), [W.float().to(dev) for W in Ws]]
print(f"[{tag}] weights ready, n={n}, L={LMAX}, grid {R}^3, h={ax[1]-ax[0]:.3e} rad", flush=True)


def plane_points(iz):
    Xg, Yg = np.meshgrid(ax, ax, indexing="ij")
    t = np.stack([Xg, Yg, np.full_like(Xg, ax[iz])], -1).reshape(-1, 3)
    return expmap(t, E=E)


@torch.no_grad()
def run_plane(iz):
    X = torch.as_tensor(plane_points(iz), dtype=torch.float32, device=dev)
    out = torch.stack([forward_all(X, W32[0], W32[1], W32[2], a) for a in ACTS])   # [2, L, R*R]
    torch.cuda.synchronize()
    return out.view(2, LMAX, R, R).cpu().numpy()


# ---- main float32 sweep ----
t_start = time.time(); done_new = 0; t_new = 0.0
for iz in range(R):
    fn = f"cache/chunks/{tag}/z{iz:03d}.npy"
    if os.path.exists(fn):
        continue
    t0 = time.time()
    arr = run_plane(iz)
    np.save(fn + ".tmp.npy", arr); os.replace(fn + ".tmp.npy", fn)
    dt = time.time() - t0
    if done_new > 0:          # skip the warm-up plane in the rate
        t_new += dt
    done_new += 1
    if iz % max(1, R // 16) == 0:
        rate = (done_new - 1) * R * R / t_new if t_new > 0 else float("nan")
        print(f"[{tag}] z {iz}/{R} {dt:.2f}s/plane, {rate:.3g} vox/s (both acts, all L)", flush=True)
sweep_s = time.time() - t_start
torch.cuda.empty_cache()

fld = f"cache/field_{tag}.npy"
if not os.path.exists(fld):
    V = np.lib.format.open_memmap(fld + ".tmp.npy", mode="w+", dtype=np.float32, shape=(2, LMAX, R, R, R))
    for iz in range(R):
        V[:, :, :, :, iz] = np.load(f"cache/chunks/{tag}/z{iz:03d}.npy")
    V.flush(); del V
    os.replace(fld + ".tmp.npy", fld)
print(f"[{tag}] field assembled", flush=True)

# ---- float64 slab: 32 central z-planes, rows in half-plane chunks ----
slabfn = f"cache/slab64_{tag}.npz"
if not os.path.exists(slabfn) and seed == 7:          # float64 check only on the main draw
    W64 = [W0.to(dev), v.to(dev), [W.to(dev) for W in Ws]]
    nz = 32 // stride
    z0 = R // 2 - nz // 2
    o32 = np.zeros((2, LMAX, nz, R, R), np.float32)
    o64 = np.zeros((2, LMAX, nz, R, R), np.float64)
    flips = np.zeros((2, LMAX), np.int64); codes_total = 0
    half = (R * R) // 2
    t0 = time.time()
    with torch.no_grad():
        for k in range(nz):
            P = torch.as_tensor(plane_points(z0 + k), dtype=torch.float64, device=dev)
            for ia, a in enumerate(ACTS):
                r32, r64 = [], []
                for c0 in range(0, R * R, half):
                    Xb = P[c0:c0 + half]
                    oa, ca = forward_codes(Xb.float(), W32[0], W32[1], W32[2], a)
                    ob, cb = forward_codes(Xb, W64[0], W64[1], W64[2], a)
                    for l in range(LMAX):
                        flips[ia, l] += int((ca[l] != cb[l]).sum())
                    if ia == 0:
                        codes_total += ca[0].numel()
                    r32.append(oa.cpu().numpy()); r64.append(ob.cpu().numpy())
                    del oa, ob, ca, cb
                o32[ia, :, k] = np.concatenate(r32, 1).reshape(LMAX, R, R)
                o64[ia, :, k] = np.concatenate(r64, 1).reshape(LMAX, R, R)
    np.savez(slabfn, z0=z0, o32=o32, o64=o64, code_flips=flips, codes_total=codes_total, seconds=time.time() - t0)
    del W64
    torch.cuda.empty_cache()
    print(f"[{tag}] float64 slab done: hidden code flips per layer {flips.tolist()} of {codes_total}", flush=True)

# ---- exact oblique slices at the 256^3 spacing ----
slfn = f"cache/slices_exact_w{n}" + (f"_s{seed}" if seed != 7 else "") + ".npy"
if not os.path.exists(slfn):
    npx = SLICE_NPX[1]
    S = np.zeros((2, LMAX, 12, npx, npx), np.float32)
    with torch.no_grad():
        for k, pl in enumerate(slice_planes()):
            t = slice_points(pl, npx, 2.0 / 256) * HALF
            X = torch.as_tensor(expmap(t, E=E), dtype=torch.float32, device=dev).view(-1, 4)
            for ia, a in enumerate(ACTS):
                S[ia, :, k] = forward_all(X, W32[0], W32[1], W32[2], a).view(LMAX, npx, npx).cpu().numpy()
    np.save(slfn, S)
    print(f"[{tag}] exact slices done", flush=True)

tj = f"cache/timing_{tag}.json"
if done_new > 1:
    json.dump(dict(width=n, R=R, planes_new=done_new, sweep_seconds=sweep_s,
                   vox_per_s=(done_new - 1) * R * R / t_new,
                   note="rate = voxels per second for both activations and all L=1..4 outputs in one pass, float32"),
              open(tj, "w"), indent=1)
print(f"[{tag}] all done in {time.time()-t_start:.0f}s", flush=True)
