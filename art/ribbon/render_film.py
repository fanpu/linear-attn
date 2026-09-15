"""Ribbon M3 film: the camera follows the 21-step centred mean ("central flow") path through the hero
window (Stage B steps 3050-3449), with a trailing window of the last ~150 steps of the ribbon drawn, and
the fixed local canyon (hero32, M3 valley-floor transfer function) translucent behind, no cutaway (so the
whole canyon reads through as the camera moves past it).

Declared:
  path      the camera target is the 21-step centred mean (uniform_filter1d, mode="nearest") of the
            window's own fixed-frame coordinates, i.e. the oscillation-free "central flow" of the window
            in the same (u_ref, pc1, pc2) chart as the hero plate. Camera az/el/radius/zoom are constant
            (the hero's), so this is a pure translation alongside the path, not an orbit.
  trailing  at output frame i (window index), the ribbon shows steps [max(0, i-TRAIL+1), i] only, growing
            from nothing at the start of the window to a constant TRAIL=150-step tail.
  canyon    identical box, TF and axes as the hero plate (cache/m2/canyon_hero32.npz), rendered fresh every
            frame (parallax as the camera moves), never cut away.
Usage: render_film.py [--size S] [--device cpu|cuda] [--fps F] [--start I] [--end J] [--test]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import matplotlib
import cmcrameri.cm  # noqa: F401
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d  # noqa: E402

import render_ribbon as rr  # reuse Chart, ribbon_layers, canyon_layer, lam_rgb, BG, HERO_FACT, tt, annotate

HERE = os.path.dirname(os.path.abspath(__file__))
M2 = os.path.join(HERE, "cache", "m2")
GAL = os.path.join(HERE, "gallery")
FRAMES = os.path.join(HERE, "cache", "film_frames")
os.makedirs(FRAMES, exist_ok=True)
os.makedirs(GAL, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=800)
ap.add_argument("--device", default="cpu")
ap.add_argument("--fps", type=int, default=16)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--test", action="store_true", help="render frames --start:--end only, no ffmpeg")
args = ap.parse_args()

rr.DEV = args.device  # reuse render_ribbon's functions without going through its own argparse/CLI
rr.DT = torch.float32 if args.device == "cuda" else torch.float64
torch.set_num_threads(4)

TRAIL = 150
AZ, EL = -120.0, 38.0
R_TUBE = 0.006

h = np.load(os.path.join(M2, "hero.npz"))
C, lam, tsteps = h["fixed"], h["lam"], h["t"]
N = len(C)
smooth = uniform_filter1d(C, 21, axis=0, mode="nearest")  # the 21-step centred mean ("central flow")

bx = json.load(open(os.path.join(M2, "boxes.json")))["hero32"]
chart = rr.Chart(bx["lo"], bx["hi"], rr.HERO_FACT)
target_w = chart(smooth)  # (N, 3) world-space camera targets, one per step

end = args.end if args.end is not None else N
S = args.size


def frame_cam(i):
    t = target_w[i]
    return r3d.orbit(tuple(t), 8.0, AZ, EL, width=S, height=S,
                      ortho_height=1.35 * (chart.hi_w - chart.lo_w).max())


def render_frame(i):
    cam = frame_cam(i)
    lo = max(0, i - TRAIL + 1)
    Pw = chart(C[lo:i + 1])
    lam_w = lam[lo:i + 1]
    img, depth, glow = rr.ribbon_layers(Pw, lam_w, cam, R_TUBE, clip_y=None)
    vol = rr.canyon_layer("hero32", chart, cam, depth, None, 6.0 / (chart.hi_w - chart.lo_w).max(), step_div=120)
    out = rr.composite(img, glow, vol[:2], expo=1.3).cpu().numpy()
    return out


t0 = time.time()
for i in range(args.start, end):
    path = os.path.join(FRAMES, f"frame_{i:05d}.png")
    if os.path.exists(path):
        continue
    out = render_frame(i)
    r3d.save_png(path, torch.as_tensor(out))
    if i % 20 == 0 or args.test:
        print(f"frame {i}/{end} {time.time()-t0:.1f}s", flush=True)

print(f"rendered {args.start}:{end} in {time.time()-t0:.1f}s", flush=True)

if not args.test:
    pattern = os.path.join(FRAMES, "frame_%05d.png")
    mp4 = os.path.join(GAL, "film_hero.mp4")
    gif = os.path.join(GAL, "film_hero.gif")
    r3d.write_film(pattern, mp4, fps=args.fps, gif=gif, gif_width=540)
    sz = os.path.getsize(mp4) / 1e6
    print(f"wrote {mp4} ({sz:.1f} MB), {gif}", flush=True)
    if sz > 20:
        tmp = mp4 + ".tmp.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "27", tmp], check=True)
        os.replace(tmp, mp4)
        print(f"re-encoded at crf 27: {os.path.getsize(mp4)/1e6:.1f} MB", flush=True)
