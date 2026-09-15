"""Plaster casts of the excursion set {T <= u} (u = volume median) with r3d: crisp voxels for Heaviside (and for the
ReLU null through the identical path), linear isosurface for ReLU only; exterior and quadrant cutaway (see below).

    render_casts.py [--size 2048] [--sub 1] [--device cuda] [--seed 7] [--only name,...]

Declared choices (also in NOTES.md):
  chart      exp map of the 0.5 rad tangent cube at p (world = tangent coords / 0.25 rad, cube [-1, 1]^3), orthographic camera
  solid      T <= u (the side below the median); u = median of the 256^3 volume
  lighting   matte plaster, one raking directional light + hard shadow + ambient occlusion: form only, not data
  section    faces lying on the cube boundary or on the cutaway planes are a darker plaster (they are cuts, not skin)
  cutaway    quadrant x > 0, y < 0 removed through the full height (the two octants facing the camera; a single
             top octant misses most of the L = 1 skin, which lies near z ~ 0). Voxels: cells dropped; iso: CSG min with
             the quadrant's signed distance (linear interpolation rounds the cut edge within one voxel)
"""
import argparse, os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=2048)
ap.add_argument("--sub", type=int, default=1)
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--only", default="")
ap.add_argument("--out", default="gallery")
args = ap.parse_args()
dev = args.device
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.10)
else:
    torch.set_num_threads(4)
S = args.size
fn = "cache/field_w4096_r256.npy" if args.seed == 7 else f"cache/field_w4096_r256_s{args.seed}.npy"
V = np.load(fn, mmap_mode="r")
E = 127.5 * 2 / 256
LO, HI = (-E,) * 3, (E,) * 3
CAM = dict(radius=6.0, az_deg=-35, el_deg=32, ortho_height=3.35)
LIGHT = (0.45, -1.0, 0.75)
PLASTER = torch.tensor([0.95, 0.93, 0.89], dtype=torch.float64)
SECTION = torch.tensor([0.70, 0.68, 0.65], dtype=torch.float64)
GROUND = 0.975
os.makedirs(args.out, exist_ok=True)


def field(ia, l):
    f = np.asarray(V[ia, l]).astype(np.float32)
    u = median_level(f)
    f = f[::args.sub, ::args.sub, ::args.sub]
    return torch.from_numpy(np.ascontiguousarray(f.transpose(2, 1, 0))), u      # [z, y, x]


def shade(pos, nrm, occ, section):
    ao = r3d.ambient_occlusion(pos, nrm, occ, n_rays=32, radius=0.12, seed=1)
    sh = r3d.hard_shadow(pos, nrm, LIGHT, occ)
    lum = (0.30 + 0.70 * r3d.lambert(nrm, LIGHT, ambient=0.0) * sh) * (0.35 + 0.65 * ao)
    base = PLASTER.to(pos.device).expand(len(pos), 3).clone()
    base[section] = SECTION.to(pos.device)
    return base * lum[:, None].double()


def render_voxel(ia, l, cut):
    f, u = field(ia, l)
    solid = (f <= u)
    R = solid.shape[0]; c = R // 2
    if cut:
        solid[:, :c, c:] = False
    solid = solid.to(dev)
    cam = r3d.orbit((0, 0, 0), CAM["radius"], CAM["az_deg"], CAM["el_deg"], width=S, height=S, ortho_height=CAM["ortho_height"])
    hit = r3d.render_voxels(solid.to(torch.int64), LO, HI, cam, solid=solid, device=dev)
    m = hit["mask"]
    pos, nrm, cell = hit["pos"][m], hit["normal"][m], hit["cell"][m]
    occ = r3d.voxel_occluder(solid, LO, HI)
    sec = torch.zeros(len(cell), dtype=torch.bool, device=pos.device)
    for a in range(3):
        sec |= ((cell[:, a] == 0) & (nrm[:, a] < -0.5)) | ((cell[:, a] == R - 1) & (nrm[:, a] > 0.5))
    if cut:   # faces on the cutaway planes: cells adjacent to the removed octant, facing into it
        i, j = cell[:, 0], cell[:, 1]
        sec |= (i == c - 1) & (nrm[:, 0] > 0.5) & (j < c)
        sec |= (j == c) & (nrm[:, 1] < -0.5) & (i >= c)
    img = torch.full((S, S, 3), GROUND, dtype=torch.float64, device=pos.device)
    img[m.to(pos.device)] = shade(pos, nrm, occ, sec)
    return img.cpu()


def render_iso(ia, l, cut):
    f, u = field(ia, l)
    g = (u - f).double()                                   # solid = g >= 0
    # normalise to ~signed distance in world units: divide by the median |grad| near the surface
    h = 2 * E / (g.shape[0] - 1)
    gz, gy, gx = torch.gradient(g, spacing=h)
    gm = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    near = g.abs() < 2 * h * gm
    g = g / gm[near].median()
    R = g.shape[0]
    t = torch.linspace(-E, E, R, dtype=torch.float64)
    Z, Y, X = torch.meshgrid(t, t, t, indexing="ij")
    if cut:
        o = torch.minimum(X, -Y)                             # >= 0 inside the removed quadrant
        g = torch.minimum(g, -o)
    g = g.float().to(dev)
    cam = r3d.orbit((0, 0, 0), CAM["radius"], CAM["az_deg"], CAM["el_deg"], width=S, height=S, ortho_height=CAM["ortho_height"])
    hit = r3d.render_iso(g, LO, HI, cam, 0.0, step=h / 2, device=dev)
    m = hit["mask"]
    pos, nrm = hit["pos"][m], hit["normal"][m]
    occ = r3d.iso_occluder(g, LO, HI, 0.0, h)
    tol = 0.3 * h
    sec = ((pos.abs() > E - tol).any(1))
    if cut:
        sec |= ((pos[:, 0].abs() < tol) & (pos[:, 1] < 0)) | ((pos[:, 1].abs() < tol) & (pos[:, 0] > 0))
    img = torch.full((S, S, 3), GROUND, dtype=torch.float64, device=pos.device)
    img[m.to(pos.device)] = shade(pos, nrm.to(pos.dtype), occ, sec)
    return img.cpu()


JOBS = {
    "cast_heaviside_L1_voxel": (render_voxel, 0, 0, False),
    "cast_heaviside_L1_voxel_cutaway": (render_voxel, 0, 0, True),
    "cast_heaviside_L2_voxel": (render_voxel, 0, 1, False),
    "cast_heaviside_L2_voxel_cutaway": (render_voxel, 0, 1, True),
    "cast_relu_L1_voxel": (render_voxel, 1, 0, False),
    "cast_relu_L1_voxel_cutaway": (render_voxel, 1, 0, True),
    "cast_relu_L1_iso": (render_iso, 1, 0, False),
    "cast_relu_L1_iso_cutaway": (render_iso, 1, 0, True),
}
only = [s for s in args.only.split(",") if s]
sfx = "" if args.seed == 7 else f"_s{args.seed}"
for name, (fnc, ia, l, cut) in JOBS.items():
    if only and name not in only:
        continue
    out = f"{args.out}/{name}{sfx}.png"
    if os.path.exists(out):
        continue
    t0 = time.time()
    img = fnc(ia, l, cut)
    r3d.save_png(out, img)
    print(out, f"{time.time()-t0:.1f}s", flush=True)
    if dev == "cuda":
        torch.cuda.empty_cache()
