"""Combed M3 film: the N sweep itself, N in {16, 64, 256, 1024, 4096}, as a diptych of closed-form v* (left) and
trained MLP (right), same shared noise seeds, same camera (slow continuous rotation across the whole film; only the
content hard-cuts at each N). Each segment prints its N and the memorised fraction for both fields with the
fresh-knot null for that N.

  OMP_NUM_THREADS=4 art/.venv/bin/python art/combed/render_film.py --test    # 3 frames, checks the pipeline
  OMP_NUM_THREADS=4 art/.venv/bin/python art/combed/render_film.py           # full film (CPU; ~25-30 min)

Measured: trajectory positions (RK4 states, t in [0, 1-1e-6]), endpoints, memorised fractions, the fresh-knot null.
Declared: chart (identity R^3, orthographic, el 40, az sweeping -60 -> +60 across the whole film), colour = mean t
(colorcet bmy), log-compressed hair-density brightness, endpoint glow (render_hair.glow_panel, unchanged). Hard cuts
between N (no interpolation of flows). For frame-rate budget, each frame shows 4,000 of the shared 20,000 seeds
(declared); the static diptychs (gallery/hair/hair_diptych_N*.png) show all 20,000.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import combed_common as C
import render_common as R
import render_hair as RH
from render_common import r3d

PANEL, GAP = 520, 24
FPS = 10
FRAMES_PER_N = 50           # ~5 s per N segment at FPS=10 -> 25 s total for 5 Ns
AZ0, AZ_SWEEP = -60.0, 120.0  # degrees, continuous over the whole film (same camera pattern for every N)
N_HAIR_FILM = 4000           # of the shared 20,000 seeds (speed budget; declared)
P = dict(k=1.0, wref=150.0, gamma=0.9, end_exposure=0.05)
FRAMES_DIR = C.CACHE / "frames_film"


def cam_at(fr: int, total: int, size: int):
    az = AZ0 + AZ_SWEEP * fr / max(1, total - 1)
    return R.camera(size, RH.OH, az_deg=az)


def make_frame(n: int, cam, device: str, seg_frac: float):
    panels = [R.to_u8(RH.glow_panel(kd, n, PANEL, device, n_hair=N_HAIR_FILM, cam=cam, **P)) for kd in ("closed", "mlp")]
    gap = np.zeros((PANEL, GAP, 3), np.uint8)
    img = np.concatenate([panels[0], gap, panels[1]], 1)
    mc, null = R.memo_numbers("closed", n)
    mm, _ = R.memo_numbers("mlp", n)
    lines = [f"N = {n} points on a trefoil.  left: closed-form optimal field v*   right: trained {R.TRAIN_DECL}",
             f"memorised (d1 < d2/3) at {R.T_END_LABEL}:  closed-form {mc:.3f}   trained {mm:.3f}   null (fresh knot points) {null:.3f}",
             f"{N_HAIR_FILM} of the 20,000 shared seeds shown per frame (declared, film frame-rate budget); hue = mean t, log-density brightness."]
    return R.caption_strip(img, lines, scale=0.62)


def film(device: str = "cpu", test: bool = False):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    ns = list(C.NS) if not test else [16, 4096]
    fpn = FRAMES_PER_N if not test else 2
    total = len(ns) * fpn
    t0 = time.time()
    fr = 0
    for seg, n in enumerate(ns):
        for local in range(fpn):
            path = FRAMES_DIR / f"{fr:05d}.png"
            if not path.exists():
                cam = cam_at(fr, total, PANEL)
                img = make_frame(n, cam, device, local / max(1, fpn - 1))
                R.save(path, img)
            fr += 1
        print(f"[film] segment N={n} done, {fr}/{total} frames, {time.time() - t0:.0f}s elapsed", flush=True)
    if test:
        print(f"[film] test done in {time.time() - t0:.0f}s, frames in {FRAMES_DIR}")
        return
    mp4 = R.GALLERY / "film_N_sweep.mp4"
    gif = R.GALLERY / "film_N_sweep.gif"
    r3d.write_film(str(FRAMES_DIR / "%05d.png"), str(mp4), fps=FPS, gif=str(gif), gif_width=480)
    print(f"[film] wrote {mp4} and {gif}, {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    film(a.device, a.test)
