"""Analytic nulls through the r3d pipeline (spec art/ml-art-3d.md §11.5). CPU, ~10 s at 480 px (4 threads).

  cd art/_shared && ../.venv/bin/python r3d/examples/nulls.py [--size 480]
"""
import argparse, math, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import r3d  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "gallery")
PLASTER, NIGHT = torch.tensor([0.93, 0.91, 0.87]), torch.tensor([0.035, 0.035, 0.05])


def grid(n, a=1.5, dtype=torch.float32):
    t = torch.linspace(-a, a, n, dtype=dtype)
    Z, Y, X = torch.meshgrid(t, t, t, indexing="ij")
    return X, Y, Z


def plaster_torus(S):
    lo, hi = (-1.5, -1.5, -0.75), (1.5, 1.5, 0.75)
    xy, z = torch.linspace(-1.5, 1.5, 129), torch.linspace(-0.75, 0.75, 65)          # h = 3/128 on every axis
    Z, Y, X = torch.meshgrid(z, xy, xy, indexing="ij")
    torus = 0.35 - torch.sqrt((torch.sqrt(X ** 2 + Y ** 2) - 0.9) ** 2 + (Z - 0.1) ** 2)   # R 0.9, r 0.35, bottom at z = -0.25
    f = torch.maximum(torus, -0.35 - Z)                                                # plinth: solid below z = -0.35
    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=35, el_deg=35, width=S, height=S, ortho_height=4.6)
    hit = r3d.render_iso(f, lo, hi, cam, 0.0)
    m = hit["mask"]
    occ = r3d.iso_occluder(f, lo, hi, 0.0, 0.01)
    light = (-0.35, -0.6, 1.4)
    pos, nrm = hit["pos"][m], hit["normal"][m]
    ao = r3d.ambient_occlusion(pos, nrm, occ, n_rays=24, radius=0.6)
    sh = r3d.hard_shadow(pos, nrm, light, occ)
    lum = (0.3 + 0.7 * r3d.lambert(nrm, light, ambient=0.0) * sh) * (0.4 + 0.6 * ao)
    img = PLASTER.expand(S, S, 3).clone() * 0.8
    img[m] = PLASTER * lum[:, None]
    r3d.save_png(os.path.join(OUT, "null_plaster_torus.png"), img)


def voxel_checker(S):
    n = 16
    k = torch.arange(n)
    Z, Y, X = torch.meshgrid(k, k, k, indexing="ij")
    labels = ((X // 4 + Y // 4 + Z // 4) % 2 + 1).long()
    lo, hi = (-1.0,) * 3, (1.0,) * 3
    clip = [((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=40, el_deg=30, width=S, height=S, ortho_height=3.6)
    hit = r3d.render_voxels(labels, lo, hi, cam, clip=clip)
    m = hit["mask"]
    pal = torch.tensor([[0, 0, 0], [0.85, 0.45, 0.25], [0.25, 0.45, 0.75]], dtype=torch.float64)
    shade = r3d.lambert(hit["normal"][m], (0.3, -0.5, 1.0), ambient=0.35)
    img = NIGHT.double().expand(S, S, 3).clone()
    img[m] = pal[hit["label"][m]] * shade[:, None]
    r3d.save_png(os.path.join(OUT, "null_voxels_checker.png"), img)


def volume_split(S):
    X, Y, Z = grid(128)
    v = torch.sqrt((X / 0.8) ** 2 + (Y / 0.6) ** 2 + (Z / 0.45) ** 2) - 1.0   # smooth two-sided field, seam at v = 0
    # Declared TF: Spectral over v in [-0.8, 0.8] (red inside, blue outside); opacity |u|^4 (1 - |u|), u = v / 0.8,
    # so the seam and everything beyond |v| = 0.8 are transparent.
    op = lambda x: (2 * x - 1).abs().pow(4) * (1 - (2 * x - 1).abs()) / 0.08192
    tf = r3d.TransferFunction("Spectral", -0.8, 0.8, opacity=op, density=10.0)
    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=30, el_deg=30, width=S, height=S, ortho_height=3.0)
    rgb, a = r3d.render_volume(v, (-1.5,) * 3, (1.5,) * 3, cam, tf, step=0.01, clip=[((0, 0, 0), (-0.2, 1.0, 0.6))])
    r3d.save_png(os.path.join(OUT, "null_volume_split.png"), r3d.over(rgb, a, NIGHT))


def knot(S):
    import matplotlib
    t = torch.linspace(0, 2 * math.pi, 4001, dtype=torch.float64)
    p, q = 2, 3
    r = 0.9 + 0.35 * torch.cos(q * t)
    P = torch.stack([r * torch.cos(p * t), r * torch.sin(p * t), 0.35 * torch.sin(q * t)], 1)   # (2,3) torus knot
    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=20, el_deg=65, width=S, height=S, ortho_height=3.0)
    pts, s = r3d.sample_polyline(P, 0.004)
    R = 0.03
    hit = r3d.splat_spheres(pts, R, cam, attrs=(s / (len(P) - 1))[:, None])
    m = hit["mask"]
    # glow halo on the centre line, depth-tested against the tubes so strands behind a crossing stay dark
    glow = r3d.splat_additive(pts, cam, sigma_px=S / 80, weight=6.0, depth=hit["depth"], eps=3 * R)
    col = torch.tensor(matplotlib.colormaps["twilight_shifted"](hit["attr"][..., 0].numpy())[..., :3])   # colour = knot parameter t
    lum = r3d.lambert(hit["normal"], (-0.5, -0.7, 0.6), ambient=0.3)
    img = NIGHT.double().expand(S, S, 3).clone() + r3d.glow_tonemap(glow, 1.0)[..., None] * torch.tensor([0.95, 0.7, 0.4], dtype=torch.float64)
    img[m] = col[m] * lum[m][:, None]
    r3d.save_png(os.path.join(OUT, "null_glow_knot.png"), img)
    runs = r3d.visible_runs(P, cam, hit["depth"], eps=3 * R)     # eps > R: a tube hides its own centre line by R/cos(tilt)
    r3d.write_svg(os.path.join(OUT, "null_plotter_knot.svg"), runs, S, S, stroke="#1f1d1b", stroke_width=1.2, background="#f3efe6")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(); S = args.size
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)
    for fn in (plaster_torus, voxel_checker, volume_split, knot):
        if args.only and fn.__name__ not in args.only:
            continue
        t0 = time.time(); fn(S); print(f"done {fn.__name__} {time.time() - t0:.1f}s", flush=True)
