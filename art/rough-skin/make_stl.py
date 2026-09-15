"""STL (M3): Heaviside L = 1 and ReLU L = 1 zero-set solids, printable casts.

Declared:
  - level = volume median (the piece's level throughout every measurement and render: the literal T = 0 level
    is not used because the kernel's constant component means it can miss the 0.5 rad patch entirely, see
    NOTES.md's level decision), so "zero-set" here means "the declared level set", not literally T = 0.
  - grid: Heaviside L = 1 at 96^3 (native evaluation, same weights, seed 7 -- the 128^3 mesh is 584k faces,
    about 29 MB, over the 20 MB budget; decimation libraries are not installed, so per the M3 ruling this uses
    the 96^3 fallback instead, declared here). ReLU L = 1 uses the 128^3 even-node subsample of the cached
    256^3 field (7.7 MB, no fallback needed), the same field used by gallery/cast_relu_L1_*.
  - r3d.marching_cubes(..., closed=True) smooths below the grid scale (spec 0.5): sub-voxel roughness below
    roughly 4/n =~ 1e-3 rad (the finite-width cutoff) does not appear in the printed surface, and is coarser
    still at 96^3 than at 128^3.
  - closed=True pads with the boundary value mirrored about the level, so faces on the cube's outer wall are
    flat caps and the mesh is watertight even where the excursion set touches the box.

python make_stl.py
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d
from common import *

SIZE_MM = 80.0
MAX_MB = 20.0
GAL = "gallery"
SEED = 7
os.makedirs(GAL, exist_ok=True)


def field_from_cache(ia):
    """128^3 even-node subsample of the cached 256^3 field, L = 1."""
    V = np.load("cache/field_w4096_r256.npy", mmap_mode="r")
    f = np.asarray(V[ia, 0]).astype(np.float64)[::2, ::2, ::2]
    G = grid_coords(256)[::2]
    lo, hi = (float(G[0]),) * 3, (float(G[-1]),) * 3
    return f, lo, hi, 128


def field_native(ia, R):
    """Fresh evaluation of L = 1 on a native R^3 grid (same weights as the cache, seed 7): used when the
    128^3 mesh from the cache is too large to keep the STL under budget."""
    ax = grid_coords(R)
    E = tangent_frame()
    W0, v, Ws = make_weights(4096, SEED, L=1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W0, v = W0.float().to(dev), v.float().to(dev)
    act = ACTS[ia]
    out = np.empty((R, R, R), np.float64)
    with torch.no_grad():
        for iz in range(R):
            Xg, Yg = np.meshgrid(ax, ax, indexing="ij")
            t = np.stack([Xg, Yg, np.full_like(Xg, ax[iz])], -1).reshape(-1, 3)
            X = torch.as_tensor(expmap(t, E=E), dtype=torch.float32, device=dev)
            o = forward_all(X, W0, v, [], act)[0].view(R, R).cpu().numpy()
            out[:, :, iz] = o.astype(np.float64)
    lo, hi = (float(ax[0]),) * 3, (float(ax[-1]),) * 3
    return out, lo, hi, R


def make_one(name, ia, out_path):
    f, lo, hi, R = field_from_cache(ia)
    u = median_level(f)
    verts, faces = r3d.marching_cubes(np.ascontiguousarray(f.transpose(2, 1, 0)), u, lo, hi, closed=True)
    est_mb = len(faces) * 50 / 1e6
    note = f"128^3 cache subsample"
    if est_mb > MAX_MB:
        R2 = 96
        print(f"{name}: 128^3 mesh ~{est_mb:.1f} MB (faces {len(faces)}) > {MAX_MB} MB; falling back to native {R2}^3", flush=True)
        f, lo, hi, R = field_native(ia, R2)
        u = median_level(f)
        verts, faces = r3d.marching_cubes(np.ascontiguousarray(f.transpose(2, 1, 0)), u, lo, hi, closed=True)
        note = f"{R2}^3 native (128^3 was {est_mb:.1f} MB, over the {MAX_MB} MB budget)"
    wt = r3d.is_watertight(faces)
    vol = r3d.mesh_volume(verts, faces)
    vmm = r3d.scale_to_mm(verts, SIZE_MM)
    r3d.write_stl(out_path, vmm, faces, header=f"rough-skin {name} L1 seed{SEED} {R}^3")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"{name}: grid={R}^3 ({note}) verts={len(verts)} faces={len(faces)} watertight={wt} "
          f"mesh_volume={vol:.4g} size_mm={np.ptp(vmm,0).round(2).tolist()} file={size_mb:.2f}MB -> {out_path}")
    return dict(name=name, grid=R, note=note, verts=len(verts), faces=len(faces), watertight=bool(wt),
                mesh_volume=vol, size_mm=np.ptp(vmm, 0).round(3).tolist(), file_mb=size_mb, path=out_path)


if __name__ == "__main__":
    torch.set_num_threads(4)
    results = []
    t0 = time.time()
    results.append(make_one("heaviside", 0, f"{GAL}/cast_heaviside_L1_zeroset.stl"))
    results.append(make_one("relu", 1, f"{GAL}/cast_relu_L1_zeroset.stl"))
    import json
    json.dump(results, open("cache/stl_report.json", "w"), indent=1)
    print(f"done in {time.time()-t0:.1f}s")
