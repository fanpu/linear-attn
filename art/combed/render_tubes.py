"""Combed M2 tubes: 400 trajectories as tubes with endpoint spheres (plaster and dark styles, plus a cutaway), and a
hidden-line plotter SVG of 2,000 trajectories.

  art/.venv/bin/python art/combed/render_tubes.py tubes --size 512 --ns 64                  # CPU toy
  gpu1.sh art/.venv/bin/python art/combed/render_tubes.py tubes svg --size 2400 --device cuda --ns 64

Measured: trajectory positions (RK4 states to t = 1-1e-6), endpoint positions.
Declared: chart (identity R^3, orthographic az -60 el 40), tube radius, sphere radius, light (Lambert, one raking
light; shows form only), plaster colour, dark style hue = t (colorcet bmy), the cutaway plane (removes every sample on
the camera side of the plane through the origin facing the camera), SVG stroke, occluder radius for hidden lines.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import combed_common as C
import render_common as R
from render_common import r3d

N_TUBE, N_SVG = 400, 2000
LIGHT = (-0.9, -0.2, 0.55)  # raking, from the left
OH = 3.4
TUBE_R, END_R = 0.0075, 0.022
SVG_OH, SVG_OCC_R, SVG_EPS, SVG_T0, SVG_MIN_RUN_PX = 2.6, 0.004, 10.0, 0.5, 3.0


def tube_hits(kind, n, cam, device, cut=False):
    z = R.load_dense(kind, n)
    st = torch.tensor(z["states"][:, :N_TUBE], dtype=torch.float64, device=device)
    t = torch.tensor(z["t"], dtype=torch.float64, device=device)
    pts, tp, _ = R.densify(st, t, TUBE_R * 0.5)
    end = st[-1]
    if cut:
        f = torch.tensor(np.asarray(cam.target, float) - np.asarray(cam.eye, float), dtype=torch.float64, device=device)
        f = f / f.norm()
        keep = (pts @ f) >= 0  # remove the half-space nearer the camera
        pts, tp = pts[keep], tp[keep]
        end = end[(end @ f) >= 0]
    tube = r3d.splat_spheres(pts, TUBE_R, cam, attrs=tp[:, None])
    sph = r3d.splat_spheres(end, END_R, cam, attrs=torch.full((len(end), 1), 2.0, dtype=torch.float64, device=device))
    use_s = sph["depth"] < tube["depth"]
    depth = torch.where(use_s, sph["depth"], tube["depth"])
    normal = torch.where(use_s[..., None], sph["normal"], tube["normal"])
    attr = torch.where(use_s[..., None], sph["attr"], tube["attr"])[..., 0]
    mask = tube["mask"] | sph["mask"]
    return dict(depth=depth, normal=normal, attr=attr, mask=mask, is_end=use_s & sph["mask"])


def shade(hit, style, size, device):
    m = hit["mask"]
    lam = r3d.lambert(hit["normal"][m], LIGHT, ambient=0.0)
    if style == "plaster":
        img = torch.tensor([0.80, 0.785, 0.76], dtype=torch.float64, device=device).expand(size, size, 3).clone()
        base = torch.tensor([0.95, 0.93, 0.89], dtype=torch.float64, device=device)
        endc = torch.tensor([0.62, 0.20, 0.12], dtype=torch.float64, device=device)
        col = torch.where(hit["is_end"][m][:, None], endc, base)
        img[m] = col * (0.62 + 0.38 * lam)[:, None]
    else:
        img = torch.tensor([0.02, 0.02, 0.03], dtype=torch.float64, device=device).expand(size, size, 3).clone()
        col = R.cmap_lut("cet_bmy", hit["attr"][m].clamp(0, 1))
        endc = torch.tensor([1.0, 0.95, 0.85], dtype=torch.float64, device=device)
        col = torch.where(hit["is_end"][m][:, None], endc, col)
        img[m] = col * (0.30 + 0.70 * lam)[:, None]
    return img


def tubes(n, kind, size, device, tag=""):
    t0 = time.time()
    cam = R.camera(size, OH)
    mem, null = R.memo_numbers(kind, n)
    name = "closed-form v*" if kind == "closed" else f"trained {R.TRAIN_DECL}"
    for cut in (False, True):
        hit = tube_hits(kind, n, cam, device, cut)
        for style in ("plaster", "dark"):
            img = R.to_u8(shade(hit, style, size, device))
            what = "CUTAWAY: samples on the camera side of the plane through the origin removed" if cut else "exterior"
            lines = [f"N = {n}, {name}; first {N_TUBE} of the 20,000 shared seeds as tubes (r {TUBE_R}), endpoints at {R.T_END_LABEL} as spheres (r {END_R}). {what}.",
                     f"memorised {mem:.3f} (all 20k seeds, d1 < d2/3), null (fresh knot points) {null:.3f}.  Declared: " +
                     ("matte plaster, endpoints terracotta, one raking Lambert light (form only; no AO, no shadow)" if style == "plaster"
                      else "hue = t (colorcet bmy), endpoints warm white, one Lambert light (form only)") + ", orthographic az -60 el 40."]
            bg = (204, 200, 194) if style == "plaster" else (8, 8, 10)
            fg = (40, 38, 36) if style == "plaster" else (215, 210, 200)
            img = R.caption_strip(img, lines, bg=bg, fg=fg, scale=0.75)
            R.save(R.GALLERY / "tubes" / f"tubes_{style}_{kind}_N{n}{'_cutaway' if cut else ''}{tag}.png", img)
    print(f"[tubes] {kind} N={n} {size}px {time.time() - t0:.0f}s", flush=True)


def svg(n, kind, size, device, tag=""):
    t0 = time.time()
    cam = R.camera(size, SVG_OH)
    z = R.load_dense(kind, n)
    sel = z["t"] >= SVG_T0  # declared time window
    st = torch.tensor(z["states"][sel, :N_SVG], dtype=torch.float64, device=device)
    t = torch.tensor(z["t"][sel], dtype=torch.float64, device=device)
    px = cam.pixel_scale()
    occ_r = max(SVG_OCC_R, 2.5 * px)
    pts, _, tid = R.densify(st, t, min(occ_r * 0.5, px))
    depth = r3d.splat_spheres(pts, occ_r, cam)["depth"]
    order = torch.argsort(tid, stable=True)
    pts, tid = pts[order], tid[order]
    counts = torch.bincount(tid, minlength=st.shape[1]).tolist()
    runs = []
    for P in torch.split(pts, counts):
        for r in r3d.visible_runs(P, cam, depth, eps=SVG_EPS * occ_r):
            if len(r) > 2:  # thin to ~1 px vertex spacing
                step = max(1, int(round(px / (occ_r * 0.5))) if px > occ_r * 0.5 else 1)
                r = np.concatenate([r[::step], r[-1:]])
            if np.linalg.norm(np.diff(r, axis=0), axis=1).sum() >= SVG_MIN_RUN_PX:
                runs.append(r)
    out = R.GALLERY / "plotter" / f"plotter_{kind}_N{n}{tag}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    r3d.write_svg(out, runs, size, size, stroke="#1f1d1b", stroke_width=0.35 * size / 1600, background="#f3efe6")
    print(f"[svg] {kind} N={n} {len(runs)} runs, {out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=["tubes", "svg"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ns", type=int, nargs="*", default=[64])
    ap.add_argument("--kinds", nargs="*", default=["closed", "mlp"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    for n in a.ns:
        for kind in a.kinds:
            if "tubes" in a.what:
                tubes(n, kind, a.size, a.device, a.tag)
            if "svg" in a.what:
                svg(n, kind, a.size, a.device, a.tag)
