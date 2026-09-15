"""The four README recipes, runnable. Each block between '# --- recipe N' markers is copied verbatim into README.md.

  cd art/_shared && ../.venv/bin/python r3d/examples/recipes.py [--size 128] [--out /tmp/r3d_recipes]
"""
import argparse, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import r3d  # noqa: E402


def plaster(S, OUT):
    # --- recipe 1: plaster iso with AO and a hard shadow
    t = torch.linspace(-1, 1, 97)
    Z, Y, X = torch.meshgrid(t, t, t, indexing="ij")                       # data[k, j, i] = f(x_i, y_j, z_k)
    field = torch.maximum(0.5 - torch.sqrt(X**2 + Y**2 + (Z - 0.1)**2),     # ball resting above a slab;
                          -0.45 - Z)                                       # solid is field >= level
    lo, hi, level = (-1,) * 3, (1,) * 3, 0.0
    cam = r3d.orbit((0, 0, 0), 5.0, az_deg=35, el_deg=35, width=S, height=S, ortho_height=3.0)
    hit = r3d.render_iso(field, lo, hi, cam, level)
    m = hit["mask"]
    pos, nrm = hit["pos"][m], hit["normal"][m]
    occ = r3d.iso_occluder(field, lo, hi, level, 0.01)                     # reused by shadows and AO
    light = (-0.4, -0.6, 1.2)
    ao = r3d.ambient_occlusion(pos, nrm, occ, n_rays=32, radius=0.5)
    sh = r3d.hard_shadow(pos, nrm, light, occ)
    lum = (0.3 + 0.7 * r3d.lambert(nrm, light, ambient=0.0) * sh) * (0.4 + 0.6 * ao)   # light shows form only
    img = torch.full((S, S, 3), 0.75)
    img[m] = torch.tensor([0.93, 0.91, 0.87]) * lum[:, None]
    r3d.save_png(f"{OUT}/recipe_plaster.png", img)
    # --- end


def voxels(S, OUT):
    # --- recipe 2: crisp voxels with a clip cutaway
    k = torch.arange(24)
    Z, Y, X = torch.meshgrid(k, k, k, indexing="ij")
    labels = (X // 6 + Y // 6 + Z // 6) % 3 + 1                           # 0 = empty, > 0 = category
    lo, hi = (-1,) * 3, (1,) * 3                                           # centres of the corner cells
    clip = [((0, 0, 0), (1, 1, 1))]                                        # (point, normal INTO the removed half)
    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=40, el_deg=30, width=S, height=S, ortho_height=3.6)
    hit = r3d.render_voxels(labels, lo, hi, cam, clip=clip)
    m = hit["mask"]
    pal = torch.tensor([[0, 0, 0], [0.85, 0.45, 0.25], [0.25, 0.45, 0.75], [0.9, 0.85, 0.6]], dtype=torch.float64)
    img = torch.zeros(S, S, 3, dtype=torch.float64)
    img[m] = pal[hit["label"][m]] * r3d.lambert(hit["normal"][m], (0.3, -0.5, 1.0), ambient=0.35)[:, None]
    r3d.save_png(f"{OUT}/recipe_voxels.png", img)
    # --- end


def fog_over_tubes(S, OUT):
    # --- recipe 3: volume fog composited over opaque tubes
    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=30, el_deg=30, width=S, height=S, ortho_height=3.0)
    t = torch.linspace(0, 2 * math.pi, 2001, dtype=torch.float64)
    P = torch.stack([torch.cos(t), torch.sin(t), 0.3 * torch.sin(3 * t)], 1)
    pts, _ = r3d.sample_polyline(P, 0.005)
    tube = r3d.splat_spheres(pts, 0.05, cam)                               # depth = view-axis depth, inf = empty
    tm = tube["mask"]
    opaque = torch.full((S, S, 3), 0.03, dtype=torch.float64)
    opaque[tm] = 0.85 * r3d.lambert(tube["normal"][tm], (0.3, -0.5, 1.0))[:, None]
    g = torch.linspace(-1.5, 1.5, 64)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    fog = torch.exp(-(X**2 + Y**2 + Z**2) / 0.5)
    tf = r3d.TransferFunction("magma", 0.0, 1.0, opacity=lambda x: x, density=2.0)   # declared transfer function
    rgb, a = r3d.render_volume(fog, (-1.5,) * 3, (1.5,) * 3, cam, tf, step=0.02, depth=tube["depth"])
    r3d.save_png(f"{OUT}/recipe_fog_tubes.png", r3d.over(rgb, a, opaque))  # premultiplied fog over the tubes
    # --- end


def hairlines(S, OUT):
    # --- recipe 4: glow hairlines and a hidden-line SVG
    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=25, el_deg=30, width=S, height=S, ortho_height=2.6)
    t = torch.linspace(0, 2 * math.pi, 1001, dtype=torch.float64)
    curves = [torch.stack([torch.cos(t), torch.sin(t) * math.cos(b), torch.sin(t) * math.sin(b)], 1)
              for b in torch.linspace(0, math.pi, 7)[:-1].tolist()]           # six great circles
    spacing = 0.004
    pts = torch.cat([r3d.sample_polyline(P, spacing)[0] for P in curves])
    glow = r3d.splat_additive(pts, cam, sigma_px=0.8, weight=spacing / cam.pixel_scale())   # line brightness
    img = r3d.glow_tonemap(glow, 1.5)[..., None] * torch.tensor([1.0, 0.8, 0.5], dtype=torch.float64)  # ~ independent of spacing
    r3d.save_png(f"{OUT}/recipe_hairlines.png", img)
    R = 0.02                                                               # occluder: the curves as thin tubes
    depth = r3d.splat_spheres(pts, R, cam)["depth"]
    runs = [run for P in curves for run in r3d.visible_runs(P, cam, depth, eps=3 * R)]   # eps > R, see README
    r3d.write_svg(f"{OUT}/recipe_hairlines.svg", runs, S, S, stroke="#1f1d1b", stroke_width=1.0, background="#f3efe6")
    # --- end


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="/tmp/r3d_recipes")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(4)
    for fn in (plaster, voxels, fog_over_tubes, hairlines):
        fn(a.size, a.out); print("done", fn.__name__, flush=True)
