"""Spectral split fog: the signed field T - u in a thin slab of the cube, volume-rendered with r3d.

    render_fog.py [--size 2048] [--sub 1] [--device cuda] [--only name,...]

Declared transfer function (Sohl-Dickstein 'Spectral' split, as in art/depth-roughness §2.7 'seam' mapping):
  * split at the level u (volume median). Each side is rank (CDF) normalised on its own over the slab voxels:
    T <= u -> q in [-1, 0) (q -> 0 at the level), T > u -> q in (0, 1].
  * colour: x = (q + 1)/2. Below side runs pale yellow (far) -> green -> blue -> purple #5e4fa2 (at the level),
    matplotlib Spectral(0.5 + x); above side runs deep red #9e0142 (at the level) -> orange -> pale yellow (far),
    Spectral(x - 0.5). The two dark ends meet at the level set, so the skin is a dark seam.
  * opacity(x) = 0.10 + 0.90 (1 - |2x - 1|)^24, extinction = 25 * opacity per world unit (cube side 2): dense
    within ~3 % of rank from the level, translucent pastel elsewhere. An aesthetic choice.
  * sampling: nearest voxel (mode="nearest"; fractal data at voxel resolution, spec §0.5), step = h/2, jittered.
  * slab: z-voxels 116..139 (24 voxels = 0.047 rad) of the 256^3 cube; full x, y. Orthographic camera.
  * ground: paper (0.955); the fog is composited with r3d.over. (A dark ground was tried: low-opacity emission
    reads muddy on black, so it was dropped.)
The ReLU null (same weights) goes through the identical function and parameters.
"""
import argparse, os, sys, time
import numpy as np
import torch
import matplotlib
from matplotlib.colors import ListedColormap
sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=2048)
ap.add_argument("--sub", type=int, default=1)
ap.add_argument("--device", default="cuda")
ap.add_argument("--only", default="")
ap.add_argument("--out", default="gallery")
args = ap.parse_args()
dev = args.device
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.10)
else:
    torch.set_num_threads(4)
S = args.size
os.makedirs(args.out, exist_ok=True)
V = np.load("cache/field_w4096_r256.npy", mmap_mode="r")
Z0, Z1 = 116, 140
coords = grid_coords(256) / HALF
sp = matplotlib.colormaps["Spectral"]
xs = np.linspace(0, 1, 2048)
SPLIT = ListedColormap(np.where((xs < 0.5)[:, None], sp(np.clip(0.5 + xs, 0, 1)), sp(np.clip(xs - 0.5, 0, 1))))
TF = r3d.TransferFunction(SPLIT, -1.0, 1.0, opacity=lambda x: 0.10 + 0.90 * (1 - (2 * x - 1).abs()) ** 24, density=25.0)


def split_rank(f, u):
    q = np.zeros_like(f, dtype=np.float32)
    lo = f <= u
    for side, sign in [(lo, -1.0), (~lo, 1.0)]:
        v = f[side]
        r = np.argsort(np.argsort(v * sign, kind="stable"), kind="stable").astype(np.float32)   # 0 = nearest the level
        q[side] = sign * (r + 0.5) / len(v)
    return q


def render(ia, l, ground):
    vol = np.asarray(V[ia, l]).astype(np.float64)
    u = median_level(vol)
    f = vol[::args.sub, ::args.sub, Z0:Z1:args.sub]
    q = split_rank(f, u)
    data = torch.from_numpy(np.ascontiguousarray(q.transpose(2, 1, 0))).to(dev)       # [z, y, x]
    xy = coords[::args.sub]
    lo = (xy[0], xy[0], coords[Z0]); hi = (xy[-1], xy[-1], coords[Z0 + ((Z1 - Z0 - 1) // args.sub) * args.sub])
    cam = r3d.orbit((0, 0, coords[(Z0 + Z1) // 2]), 6.0, az_deg=-35, el_deg=58, width=S, height=S, ortho_height=2.75)
    h = 2.0 / 256 * args.sub
    rgb, a = r3d.render_volume(data, lo, hi, cam, TF, step=h / 2, mode="nearest", jitter=True, seed=3, device=dev)
    back = torch.full((S, S, 3), ground, dtype=rgb.dtype, device=rgb.device)
    return r3d.over(rgb, a, back).cpu()


JOBS = {}
for ia, l, nm in [(0, 0, "heaviside_L1"), (0, 1, "heaviside_L2"), (1, 0, "relu_L1")]:
    for g, gn in [(0.955, "paper")]:
        JOBS[f"fog_split_{nm}_{gn}"] = (ia, l, g)
only = [s for s in args.only.split(",") if s]
for name, (ia, l, g) in JOBS.items():
    if only and name not in only:
        continue
    out = f"{args.out}/{name}.png"
    if os.path.exists(out):
        continue
    t0 = time.time()
    r3d.save_png(out, render(ia, l, g))
    print(out, f"{time.time()-t0:.1f}s", flush=True)
    if dev == "cuda":
        torch.cuda.empty_cache()
