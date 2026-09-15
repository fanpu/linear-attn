"""Combed M2 basin companion: every noise voxel of a cube in [-2.5, 2.5]^3 labelled by the training point its
closed-form flow (N = 16) reaches at t = 1 - 1e-6. Crisp voxels (no interpolation) with an octant cutaway, the same
render of the null (Voronoi cell of the start point itself), and a slice atlas.

  art/.venv/bin/python art/combed/render_basin.py --res 128 --size 1024          # CPU
  gpu1.sh art/.venv/bin/python art/combed/render_basin.py --res 128 --size 2400 --device cuda

Measured: labels (and their boundary), training-point positions.
Declared: chart (identity R^3: noise coordinates and data coordinates share the axes), orthographic camera az -60 el 40,
categorical palette colorcet glasbey_category10 (label order = training-set index), one Lambert light at ambient 0.5
(form only), octant cutaway removing x > 0, y < 0, z > 0 (the octant facing the camera), training points drawn as
spheres (r 0.07) in their basin's colour with a dark rim.
"""
from __future__ import annotations

import argparse
import time

import colorcet
import numpy as np
import torch
from PIL import Image

import combed_common as C
import render_common as R
from render_common import r3d

LIGHT = (0.35, -0.55, 1.0)
PT_R = 0.07


def palette(n=16):
    hexes = colorcet.glasbey_category10[:n]
    return np.array([list(h) if not isinstance(h, str) else [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in hexes], float)


def render(labels: np.ndarray, data: np.ndarray, size: int, device: str, cut: bool, show_points: bool):
    res = labels.shape[0]
    lab = torch.tensor(labels.astype(np.int64) + 1, device=device)
    g = torch.linspace(-2.5, 2.5, res, device=device)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    solid = torch.ones_like(lab, dtype=torch.bool)
    if cut:
        solid &= ~((X > 0) & (Y < 0) & (Z > 0))
    cam = R.camera(size, ortho_height=9.2)
    hit = r3d.render_voxels(lab, (-2.5,) * 3, (2.5,) * 3, cam, solid=solid, device=device)
    pal = torch.tensor(palette(), dtype=torch.float64, device=device)
    img = torch.full((size, size, 3), 0.965, dtype=torch.float64, device=device)
    m = hit["mask"]
    img[m] = pal[hit["label"][m] - 1] * (0.5 + 0.5 * r3d.lambert(hit["normal"][m], LIGHT, ambient=0.0))[:, None]
    if show_points:
        P = torch.tensor(data, dtype=torch.float64, device=device)
        sp = r3d.splat_spheres(P, PT_R, cam, attrs=torch.arange(len(P), device=device, dtype=torch.float64)[:, None])
        rim = r3d.splat_spheres(P, PT_R * 1.35, cam)
        front_rim = rim["mask"] & (rim["depth"] < hit["depth"])
        img[front_rim] = 0.08
        front = sp["mask"] & (sp["depth"] < hit["depth"])
        idx = sp["attr"][front][:, 0].round().long()
        img[front] = pal[idx] * (0.55 + 0.45 * r3d.lambert(sp["normal"][front], LIGHT, ambient=0.0))[:, None]
    return img


def slice_atlas(z, res, out, names=("labels", "null_voronoi_x0"), ks=None):
    pal = (palette() * 255).astype(np.uint8)
    g = np.linspace(-2.5, 2.5, res)
    ks = ks or [int(round(v)) for v in np.linspace(res * 0.15, res * 0.85, 6)]
    scale = max(1, 256 // res * 2)
    rows = []
    for name in names:
        tiles = []
        for k in ks:
            sl = pal[z[name][k]][::-1]  # row 0 = top = max y
            tiles.append(np.pad(np.kron(sl, np.ones((scale, scale, 1), np.uint8)), ((6, 6), (6, 6), (0, 0)), constant_values=246))
        rows.append(np.concatenate(tiles, 1))
    img = np.concatenate(rows, 0)
    zs = ", ".join(f"{g[k]:+.2f}" for k in ks)
    img = R.caption_strip(img, [f"Slice atlas, z = {zs} (left to right), x right, y up, one pixel block per voxel ({res}^3, nearest, no interpolation).",
                                "Top: basin labels (training point reached by the closed-form flow from that noise voxel, N = 16, t = 1 - 1e-6). "
                                "Bottom: null, Voronoi cell of the noise point itself. Palette colorcet glasbey_category10 (declared)."],
                          bg=(246, 246, 246), fg=(30, 30, 30), scale=1.6)
    R.save(out, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()
    z = np.load(C.CACHE / f"basin_N16_R{a.res}.npz")
    import json
    bc = json.loads((C.CACHE / "basin_boxcount.json").read_text()) if (C.CACHE / "basin_boxcount.json").exists() else {}
    D = bc.get(str(a.res), {}).get("labels", {}).get("D_fit")
    Dn = bc.get(str(a.res), {}).get("null_voronoi_x0", {}).get("D_fit")
    dtxt = (f"3D box counting of the label boundary (eps 1-{a.res // 16} voxels): D = {D:.2f}, planar-Voronoi null through the same "
            f"pipeline {Dn:.2f}; a surface, no fractal claim.") if D else ""
    for name, title in (("labels", "closed-form basins"), ("null_voronoi_x0", "NULL: Voronoi cell of the noise point")):
        for cut in (False, True):
            img = R.to_u8(render(z[name], z["data"], a.size, a.device, cut, show_points=cut))
            lines = [f"{title}, N = 16: each voxel of a {a.res}^3 grid of starting noise in [-2.5, 2.5]^3 coloured by the training point "
                     + ("its closed-form flow reaches at t = 1 - 1e-6." if name == "labels" else "nearest to it (same render path as the basins)."),
                     ("Octant x>0, y<0, z>0 removed; training points drawn as spheres in their basin colour. " if cut else "Exterior. ")
                     + "Crisp voxels, categorical palette glasbey_category10, one Lambert light (form only), orthographic az -60 el 40. " + dtxt]
            img = R.caption_strip(img, lines, bg=(246, 246, 246), fg=(30, 30, 30), scale=0.8)
            stem = "basin" if name == "labels" else "basin_null_voronoi"
            R.save(R.GALLERY / "basin" / f"{stem}_R{a.res}{'_cutaway' if cut else ''}{a.tag}.png", img)
    slice_atlas(z, a.res, R.GALLERY / "basin" / f"basin_slices_R{a.res}{a.tag}.png")
    print(f"[basin render] R={a.res} {a.size}px {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
