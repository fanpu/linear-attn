"""Combed M2 hair: additive glow of 20k trajectories coloured by t (closed-form | trained diptych), and a
cross-eye stereo pair.

  gpu1.sh art/.venv/bin/python art/combed/render_hair.py diptych --size 2048 --device cuda --ns 16 64 256 1024 4096
  art/.venv/bin/python art/combed/render_hair.py diptych --size 512 --ns 64          # CPU toy

Measured: trajectory positions (RK4 states, t in [0, 1-1e-6]), endpoints, memorised fractions.
Declared: chart (identity R^3, orthographic, az -60 el 40, ortho height 4.4), colour = t through colorcet 'bmy',
additive Gaussian splats sigma 0.7 px with weight spacing/pixel (line brightness ~ hair density in the image plane),
per-channel tonemap 1 - exp(-exposure x), endpoint glow sigma 1.6 px in warm white, black ground.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

import combed_common as C
import render_common as R
from render_common import r3d

HAIR_CMAP = "cet_bmy"
END_RGB = (1.0, 0.93, 0.78)
OH = 3.4


def glow_panel(kind: str, n: int, size: int, device: str, k: float, wref: float, gamma: float, end_exposure: float,
               n_hair: int | None = None, cam=None, ortho_height: float = OH):
    """Hue = density-weighted mean t of the hairs through a pixel (colorcet bmy).
    Brightness = (log1p(W/k) / log1p(wref/k))^gamma, W = hair length density per 2048-px-equivalent pixel
    (so the mapping does not change with image size). Endpoints: separate glow, screen-blended."""
    z = R.load_dense(kind, n)
    states = torch.tensor(z["states"][:, :n_hair], dtype=torch.float64, device=device)
    t = torch.tensor(z["t"], dtype=torch.float64, device=device)
    cam = cam or R.camera(size, ortho_height)
    px = cam.pixel_scale()
    spacing = 0.7 * px
    B = states.shape[1]
    W = torch.zeros(size, size, dtype=torch.float64, device=device)
    Wt = torch.zeros(size, size, dtype=torch.float64, device=device)
    for b0 in range(0, B, 2000):  # bounded memory
        pts, tp, _ = R.densify(states[:, b0:b0 + 2000], t, spacing)
        W += r3d.splat_additive(pts, cam, weight=spacing / px, sigma_px=0.7)
        Wt += r3d.splat_additive(pts, cam, weight=(spacing / px) * tp, sigma_px=0.7)
    tmean = Wt / W.clamp_min(1e-12)
    W = W * (20000 / B) * (size / 2048)
    lum = (torch.log1p(W / k) / math.log1p(wref / k)).clamp(0, 1) ** gamma
    img = R.cmap_lut(HAIR_CMAP, tmean) * lum[..., None]
    e = r3d.splat_additive(states[-1], cam, weight=1.0, sigma_px=1.6 * size / 2048) * (20000 / B)
    eg = r3d.glow_tonemap(e, end_exposure)[..., None] * torch.tensor(END_RGB, dtype=torch.float64, device=device)
    return 1 - (1 - img) * (1 - eg)  # screen blend (declared)


def diptych(n: int, size: int, device: str, P: dict, n_hair=None, tag=""):
    t0 = time.time()
    panels = [R.to_u8(glow_panel(kd, n, size, device, n_hair=n_hair, **P)) for kd in ("closed", "mlp")]
    gap = np.zeros((size, max(4, size // 128), 3), np.uint8)
    img = np.concatenate([panels[0], gap, panels[1]], 1)
    mc, null = R.memo_numbers("closed", n)
    mm, _ = R.memo_numbers("mlp", n)
    lines = [f"N = {n} points on a trefoil.  left: closed-form optimal field v*   right: trained {R.TRAIN_DECL}",
             f"memorised (d1 < d2/3) at {R.T_END_LABEL}:  closed-form {mc:.3f}   trained {mm:.3f}   null (fresh knot points) {null:.3f}",
             "20,000 shared noise seeds, RK4 256 + 96 tail steps, float64.  Declared: hue = mean t (colorcet bmy),",
             f"brightness = log-compressed hair density, endpoint glow; orthographic az -60 el 40.  {R.STACK}"]
    img = R.caption_strip(img, lines, scale=1.0 * (2 * size) / 2048 / 2)
    R.save(R.GALLERY / "hair" / f"hair_diptych_N{n}{tag}.png", img)
    print(f"[diptych] N={n} {size}px {time.time() - t0:.0f}s", flush=True)


def stereo(n: int, kind: str, size: int, device: str, P: dict, sep: float = 0.9):
    cam = R.camera(size)
    left, right = r3d.stereo_pair(cam, sep)
    # cross-eye: the right-eye image goes on the left
    pr = R.to_u8(glow_panel(kind, n, size, device, cam=right, **P))
    pl = R.to_u8(glow_panel(kind, n, size, device, cam=left, **P))
    gap = np.zeros((size, size // 32, 3), np.uint8)
    img = np.concatenate([pr, gap, pl], 1)
    mem, null = R.memo_numbers(kind, n)
    name = "closed-form v*" if kind == "closed" else "trained MLP"
    img = R.caption_strip(img, [f"CROSS-EYE stereo pair (right-eye view on the left), parallel axes, separation {sep} world units, orthographic.",
                                f"N = {n}, {name}; memorised {mem:.3f} at {R.T_END_LABEL}, null (fresh knot points) {null:.3f}. "
                                f"colour = t (colorcet bmy), additive glow."], scale=0.9)
    R.save(R.GALLERY / "stereo" / f"hair_stereo_crosseye_{kind}_N{n}.png", img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["diptych", "stereo"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ns", type=int, nargs="*", default=[64])
    ap.add_argument("--kind", default="closed")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--wref", type=float, default=150.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--end-exposure", type=float, default=0.05)
    ap.add_argument("--n-hair", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    P = dict(k=a.k, wref=a.wref, gamma=a.gamma, end_exposure=a.end_exposure)
    for n in a.ns:
        if a.what == "diptych":
            diptych(n, a.size, a.device, P, a.n_hair, a.tag)
        else:
            stereo(n, a.kind, a.size, a.device, P)
