"""M2 renders for Number Knot: helix, knot and depth towers, measured beside the shuffled-label null.
Reads cache/geom_M2.npz only (geometry.py computes it). Writes gallery/ (PNG, SVG).

  python render_m2.py --size 700 --device cpu --out cache/preview/m2      # proofs (CPU)
  python render_m2.py --size 2400 --device cuda                           # heroes (via gpu1.sh)

Declared choices (also in README captions):
- colour: residues and calendar position use cyclic maps (colorcet `cyclic_rygcbmr_50_90_c64_s25` (L* 55-84, near-constant lightness) on the dark
  ground, cmcrameri `romaO` for plaster pigment), sampled at K equal steps;
- light: one Lambert key light (glow) or a raking light plus ambient occlusion (plaster) shows form only;
- connectors (tubes, rings, threads) are declared: they join consecutive integers, the class means of one layer in
  calendar order, and one class's means across layers; beads are the measured points;
- plaster pigment = 45 % white + 55 % data colour; the fitted K&T curve is drawn as a thin opaque line (knot: a < 100 only); glow halos are additive Gaussians on connector centre lines.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d  # noqa: E402

import common as C  # noqa: E402
import colorcet  # noqa: E402,F401  (registers cet maps)
import cmcrameri.cm  # noqa: E402,F401

NIGHT = (0.035, 0.035, 0.045)
PAPER = "#f3efe6"
INK = "#1f1d1b"
PLASTER_BG = (0.80, 0.79, 0.76)
DARK_MAP, LIGHT_MAP = "cet_cyclic_rygcbmr_50_90_c64_s25", "cmc.romaO"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def cyc(values, K, cmap):
    return np.asarray(matplotlib.colormaps[cmap]((np.asarray(values) % K) / K))[..., :3]


# ---------------------------------------------------------------- scenes (pure geometry + declared connectors)

def scene_numbers(G, kind, which):
    """which = helix | knot. Beads: measured points; connector: the 0..999 polyline."""
    P = G[f"{which}_{kind}"].copy()
    c = P.mean(0)
    P -= c
    fit = G[f"{which}fit_{kind}"] - c
    a = G["a"]
    if which == "knot":
        fit = fit[: len(fit) // 10]          # declared: the fitted knot drawn once, for 0 <= a < 100
    K = 100 if which == "helix" else 10          # colour = residue of the period drawn (helix T=100, knot minor T=10)
    af = np.arange(len(fit)) / 20.0                # geometry.py samples the fit at 20 points per integer
    return dict(beads=P, value=a, K=K, bead_r=0.04 if which == "helix" else 0.035, glow=0.5,
                lines=[dict(P=P, value=a, K=K, r=0.006, closed=False, opaque=False),               # consecutive integers
                       dict(P=fit, value=af, K=K, r=0.012 if which == "helix" else 0.022, closed=False, opaque=True, fit=True)])  # fitted curve


def scene_tower(G, name, kind):
    if name == "numbers":
        tw = G[f"tower_numbers_{kind}"]                      # (L, N, 2)
        L, N, _ = tw.shape
        dz, K, value = 0.32, 100, np.tile(G["a"], L)
        lab_used = G["a"] if kind == "measured" else G["a_null"]
        means = np.stack([[tw[l][(lab_used % 100) == k].mean(0) for k in range(100)] for l in range(L)])
        bead_r, ring_value, thread = 0.014, np.arange(100), False
    else:
        tw = G[f"tower_{name}_{kind}"]
        means = G[f"tower_{name}_{kind}_means"]
        L, N, _ = tw.shape
        K = means.shape[1]
        dz, value = 0.2, np.tile(G[f"tower_{name}_labels"], L)
        bead_r, ring_value, thread = (0.032 if K == 7 else 0.026), np.arange(K), True
    z = (np.arange(L) - (L - 1) / 2) * dz
    beads = np.concatenate([np.c_[tw[l], np.full(N, z[l])] for l in range(L)])
    lines = []
    for l in range(L):
        ring = np.c_[means[l], np.full(K, z[l])]
        lines.append(dict(P=np.r_[ring, ring[:1]], value=np.r_[ring_value, ring_value[:1]], K=K, r=bead_r * 0.22, closed=True))
    if thread:
        for k in range(K):
            lines.append(dict(P=np.c_[means[:, k], z], value=np.full(L, k), K=K, r=bead_r * 0.22, closed=False))
    return dict(beads=beads, value=value, K=K, bead_r=bead_r, lines=lines, glow=0.35 if name == "numbers" else 1.0,
                plot_bead=0.5)


# ---------------------------------------------------------------- rasterisation

def cam_light(cam, right, up, toward):
    """Light direction in the camera frame (declared, form only): right/up/toward-viewer weights."""
    f, r, u = cam.basis()
    L = right * r + up * u - toward * f
    return tuple((L / np.linalg.norm(L)).tolist())


def gather(scene, cmap, dev, spacing):
    """All spheres (beads + sampled connectors) with colour and a connector flag."""
    b = torch.tensor(scene["beads"], dtype=torch.float64)
    cols = [torch.tensor(cyc(scene["value"], scene["K"], cmap))]
    cen, rad, flag, lpts, lcol = [b], [torch.full((len(b),), scene["bead_r"], dtype=torch.float64)], [torch.zeros(len(b))], [], []
    for ln in scene["lines"]:
        P = torch.tensor(ln["P"], dtype=torch.float64)
        pts, s = r3d.sample_polyline(P, min(spacing, 0.5 * ln["r"]))   # spacing < radius: no string-of-pearls
        vcol = torch.tensor(cyc(ln["value"], ln["K"], cmap))
        i0 = s.floor().long().clamp(max=len(P) - 1); i1 = (i0 + 1).clamp(max=len(P) - 1); f = (s - i0)[:, None]
        c = vcol[i0] * (1 - f) + vcol[i1] * f
        if ln.get("opaque", True):
            cen.append(pts); rad.append(torch.full((len(pts),), ln["r"], dtype=torch.float64)); flag.append(torch.ones(len(pts)))
            cols.append(c)
        lpts.append(pts); lcol.append(c)
    return (torch.cat(cen).to(dev), torch.cat(rad).to(dev), torch.cat(cols).to(dev), torch.cat(flag).to(dev),
            torch.cat(lpts).to(dev) if lpts else None, torch.cat(lcol).to(dev) if lcol else None)


def style_glow(scene, cam, dev, S):
    cen, rad, col, flag, lpts, lcol = gather(scene, DARK_MAP, dev, spacing=scene["bead_r"] * 0.25)
    hit = r3d.splat_spheres(cen, rad, cam, attrs=torch.cat([col, flag[:, None]], 1))
    m = hit["mask"]
    img = torch.tensor(NIGHT, dtype=torch.float64, device=dev).expand(cam.height, cam.width, 3).clone()
    if lpts is not None:
        sp = scene["bead_r"] * 0.25
        glow = r3d.splat_additive(lpts, cam, color=lcol, sigma_px=max(1.0, S / 700), weight=scene.get("glow", 1.0) * sp / cam.pixel_scale(),
                                  depth=hit["depth"], eps=3 * scene["bead_r"])
        img = img + r3d.glow_tonemap(glow, 1.2)
    lum = r3d.lambert(hit["normal"][m], cam_light(cam, -0.5, 0.6, 0.6), ambient=0.35)
    img[m] = hit["attr"][m][:, :3] * lum[:, None]
    return img.clamp(0, 1)


def sphere_field(cen, rad, G, pad):
    c = cen.cpu().numpy(); r = rad.cpu().numpy()
    lo = c.min(0) - pad; hi = c.max(0) + pad
    n = np.maximum(8, np.round((hi - lo) / (hi - lo).max() * G)).astype(int)
    axes = [np.linspace(lo[i], hi[i], n[i]) for i in range(3)]
    Z, Y, X = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    q = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    d, idx = cKDTree(c).query(q, k=8, workers=4)
    f = (r[idx] - d).max(1).reshape(Z.shape)                  # union of balls: field >= 0 inside
    return torch.tensor(f), tuple(lo), tuple(hi), float(((hi - lo) / (n - 1)).min())


def style_plaster(scene, cam, dev, S, G=None):
    G = G or (192 if S < 1200 else 256)
    cen, rad, col, flag, _, _ = gather(scene, LIGHT_MAP, dev, spacing=scene["bead_r"] * 0.25)
    hit = r3d.splat_spheres(cen, rad, cam, attrs=col)
    m = hit["mask"]
    ray_o, ray_d = cam.rays(dev, torch.float64)
    t = cam.depth_to_t(hit["depth"][m], ray_d[m])
    pos = ray_o[m] + ray_d[m] * t[:, None]
    nrm = hit["normal"][m]
    field, lo, hi, h = sphere_field(cen, rad, G, pad=4 * scene["bead_r"])
    occ = r3d.iso_occluder(field.to(dev), lo, hi, 0.0, h * 0.5)
    ao = r3d.ambient_occlusion(pos, nrm, occ, n_rays=24, radius=12 * scene["bead_r"], bias=max(0.3 * scene["bead_r"], 2 * h))
    print(f"  plaster: grid h={h:.4f} bead_r={scene['bead_r']} ao mean={ao.mean().item():.2f} p10={ao.quantile(0.1).item():.2f}", flush=True)
    light = cam_light(cam, -0.85, 0.35, 0.25)                   # raking key light from the upper left
    lum = (0.45 + 0.6 * r3d.lambert(nrm, light, ambient=0.0)) * (0.5 + 0.5 * ao)
    pigment = 0.45 + 0.55 * hit["attr"][m]
    img = torch.tensor(PLASTER_BG, dtype=torch.float64, device=dev).expand(cam.height, cam.width, 3).clone()
    img[m] = pigment * lum[:, None]
    return img.clamp(0, 1)


def style_plotter(scene, cam, dev):
    cen, rad, _, _, _, _ = gather(scene, DARK_MAP, dev, spacing=scene["bead_r"] * 0.25)
    depth = r3d.splat_spheres(cen, rad, cam)["depth"]
    runs = []
    for ln in scene["lines"]:
        if not ln.get("opaque", True):
            continue                              # declared: plotter omits the consecutive-integer hairlines
        runs += r3d.visible_runs(torch.tensor(ln["P"], dtype=torch.float64), cam, depth, eps=3 * scene["bead_r"])
    b = torch.tensor(scene["beads"], dtype=torch.float64)
    pix, z = cam.project(b.to(dev))
    pix, z = pix.cpu().numpy(), z.cpu().numpy()
    D = depth.cpu().numpy()
    rpx = scene.get("plot_bead", 1.0) * scene["bead_r"] / cam.pixel_scale()
    ang = np.linspace(0, 2 * np.pi, 9)
    for (x, y), zz in zip(pix, z):
        ci, ri = int(round(x)), int(round(y))
        if 0 <= ri < D.shape[0] and 0 <= ci < D.shape[1] and zz <= D[ri, ci] + 1.5 * scene["bead_r"]:
            runs.append(np.c_[x + rpx * np.cos(ang), y + rpx * np.sin(ang)])
    return runs


# ---------------------------------------------------------------- composition

def shared_camera(scenes, W, H, az, el, margin=1.12):
    allp = np.concatenate([s["beads"] for s in scenes])
    tgt = tuple(((allp.max(0) + allp.min(0)) / 2).tolist())
    probe = r3d.orbit(tgt, 50.0, az_deg=az, el_deg=el, width=W, height=H, ortho_height=2.0)
    pix, _ = probe.project(torch.tensor(allp))
    ext_w = (pix[:, 0].max() - pix[:, 0].min()).item() * probe.pixel_scale()
    ext_h = (pix[:, 1].max() - pix[:, 1].min()).item() * probe.pixel_scale()
    oh = margin * max(ext_h, ext_w * H / W)
    return r3d.orbit(tgt, 50.0, az_deg=az, el_deg=el, width=W, height=H, ortho_height=oh)


def label(img_np, texts, S, fg, W):
    im = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, max(14, S // 45))
    for i, t in enumerate(texts):
        d.text((i * W + S // 40, S // 40), t, fill=fg, font=f)
    return im


def render_pair(name, scenes, W, H, az, el, dev, out, S, texts, styles):
    cam = shared_camera(scenes, W, H, az, el)
    times = {}
    for st in styles:
        t0 = time.time()
        if st == "plotter":
            runs = []
            for i, sc in enumerate(scenes):
                runs += [r + np.array([i * W, 0]) for r in style_plotter(sc, cam, dev)]
            r3d.write_svg(str(out / f"{name}_plotter.svg"), runs, W * len(scenes), H, stroke=INK,
                          stroke_width=max(0.6, S / 1600), background=PAPER)
        else:
            fn = style_glow if st == "glow" else style_plaster
            img = torch.cat([fn(sc, cam, dev, S) for sc in scenes], 1).cpu().numpy()
            fg = (235, 235, 235) if st == "glow" else (40, 38, 36)
            label(img, texts, S, fg, W).save(out / f"{name}_{st}.png")
        times[st] = round(time.time() - t0, 1)
        print(f"{name} {st} {times[st]}s", flush=True)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=700)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="gallery")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--styles", nargs="*", default=["glow", "plaster", "plotter"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.10)
    out = C.ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    G = dict(np.load(C.CACHE / "geom_M2.npz"))
    meta = json.loads((C.CACHE / "geom_M2.json").read_text())
    S = args.size
    jobs = {
        "tower_days": lambda: render_pair("tower_days", [scene_tower(G, "days", "measured"), scene_tower(G, "days", "null")],
                                          S * 2 // 3, S, 30, 28, dev, out, S,
                                          ["Qwen3-0.6B days, measured", "point-label shuffle (null)"], args.styles),
        "tower_months": lambda: render_pair("tower_months", [scene_tower(G, "months", "measured"), scene_tower(G, "months", "null")],
                                            S * 2 // 3, S, 30, 28, dev, out, S,
                                            ["Qwen3-0.6B months, measured", "point-label shuffle (null)"], args.styles),
        "tower_numbers": lambda: render_pair("tower_numbers", [scene_tower(G, "numbers", "measured"), scene_tower(G, "numbers", "null")],
                                             S * 2 // 3, S, 30, 28, dev, out, S,
                                             ["OLMo-2 numbers T=100 plane, measured", "shuffled labels (null)"], args.styles),
        "hero_days": lambda: render_pair("hero_tower_days", [scene_tower(G, "days", "measured")], S * 2 // 3, S, 30, 40,
                                         dev, out, S, [""], args.styles),
        "hero_months": lambda: render_pair("hero_tower_months", [scene_tower(G, "months", "measured")], S * 2 // 3, S, 30, 40,
                                           dev, out, S, [""], args.styles),
        "helix": lambda: render_pair("helix", [scene_numbers(G, "measured", "helix"), scene_numbers(G, "null", "helix")],
                                     S * 3 // 4, S, 35, 35, dev, out, S,
                                     ["OLMo-2 helix T=100, measured", "shuffled labels (null)"], args.styles),
        "knot": lambda: render_pair("knot", [scene_numbers(G, "measured", "knot"), scene_numbers(G, "null", "knot")],
                                    S, S, 25, 50, dev, out, S,
                                    ["OLMo-2 knot T=100 x T=10, measured", "shuffled labels (null)"], args.styles),
    }
    timings = {}
    for k, fn in jobs.items():
        if args.only and k not in args.only:
            continue
        timings[k] = fn()
    (out / f"timings_{S}.json").write_text(json.dumps(dict(size=S, device=str(dev), timings=timings), indent=1))


if __name__ == "__main__":
    main()
