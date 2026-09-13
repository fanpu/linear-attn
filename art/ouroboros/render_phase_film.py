"""Phase map over (n, lambda) evolving over generations: film.

usage: python render_phase_film.py ring gmm [--regime replace] [--tau 0.25] [--sub 5] [--pairing sd_spectral]

Measured per cell, per generation g: excess(g) = sliced W2(g) - sliced W2(0) of one chain (seed 0, CRN).
Signed field x = excess(g) - tau: x > 0 (red side) = currently more than tau worse than its own generation-0 fit,
x < 0 (purple side) = not. Each side rank-normalised against ALL frames (fixed colour scale over time).
Between integer generations the field is linearly interpolated for smooth motion (declared).
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw

import style as S

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("model")
ap.add_argument("--regime", default="replace"); ap.add_argument("--tau", type=float, default=0.25)
ap.add_argument("--sub", type=int, default=5); ap.add_argument("--pairing", default="sd_spectral")
ap.add_argument("--hold", type=int, default=90); ap.add_argument("--tag", default="")
a = ap.parse_args()

d = np.load(f"cache/phase_{a.target}_{a.model}_{a.regime}.npz")
lams, ns = d["lams"], d["ns"]
sw = d["sw2"][0].astype(float)  # [L, N, G+1]
G = sw.shape[-1] - 1
X = (sw - sw[..., :1]) - a.tau  # [L, N, G+1]
ref = X[..., 1:].ravel()
L_, N_ = len(lams), len(ns)
frac = (X > 0).mean((0, 1))  # share of the (lambda, n) grid above tau per generation

W, H = 1920, 1080
cy = 9; cx = 19
pw, ph = N_ * cx, L_ * cy
ml, mt = 140, (H - ph) // 2 - 20
fg, dim = S.INK["bone"], (140, 132, 118)
ft, fs, fa, fb = S.font("serif", 40), S.font("serif_it", 21), S.font("serif", 22), S.font("serif", 64)
rx = ml + pw + 55  # right column

tmp = f"/tmp/ouro_phasefilm_{a.regime}{a.tag}"
shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
frames = [g + k / a.sub for g in range(G) for k in range(a.sub)] + [float(G)] * a.hold
# precompute colour for integer generations, interpolate RGB? no: interpolate the field, then colour
for fi, t in enumerate(frames):
    g0 = int(np.floor(t)); g1 = min(g0 + 1, G); w = t - g0
    x = X[..., g0] * (1 - w) + X[..., g1] * w
    rgb = S.P.render_split(x, a.pairing, near_boundary="small", ref=ref) * 255
    img = np.zeros((H, W, 3), np.uint8); img[:] = S.INK["night"]
    img[mt:mt + ph, ml:ml + pw] = np.kron(np.flipud(rgb), np.ones((cy, cx, 1))).astype(np.uint8)
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    dr.rectangle([ml - 1, mt - 1, ml + pw, mt + ph], outline=dim)
    for nt in [8, 16, 32, 64, 128, 256, 512]:
        xx = ml + (np.log(nt) - np.log(ns[0])) / (np.log(ns[-1]) - np.log(ns[0])) * (pw - cx) + cx / 2
        dr.line([xx, mt + ph, xx, mt + ph + 10], fill=dim, width=2)
        dr.text((xx - 14, mt + ph + 14), str(nt), font=fa, fill=dim)
    dr.text((ml + pw / 2 - 150, mt + ph + 44), "n, samples per generation (log)", font=fa, fill=dim)
    for lt in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = mt + ph - lt * (ph - cy) - cy / 2
        dr.line([ml - 10, yy, ml, yy], fill=dim, width=2)
        dr.text((ml - 62, yy - 12), f"{lt:.2f}", font=fa, fill=dim)
    dr.text((20, mt - 44), "λ, real fraction kept", font=fa, fill=dim)
    dr.text((rx, 90), "Self-consumption", font=ft, fill=fg)
    dr.text((rx, 140), "phase map", font=ft, fill=fg)
    body = [f"{a.regime}: each generation refits a K = 8", "Gaussian mixture to λ·n real samples", "and (1−λ)·n samples of its own",
            "previous fit (ring of 8 modes).", "", f"red: sliced W2 now more than τ = {a.tau}", "above its own generation-0 fit.",
            "purple: not (yet). Each side rank-", "normalised over all frames (declared).", "", "one chain per cell, seed 0,",
            "common random numbers across cells;", "motion interpolates between generations."]
    for i, ln in enumerate(body):
        dr.text((rx, 210 + i * 29), ln, font=fs, fill=dim)
    dr.text((rx, 640), f"generation {int(round(t)) if t == int(t) else int(t)}", font=fb, fill=fg)
    # mini curve: share of grid above tau
    cx0, cy0, cw, chh = rx, 820, 400, 150
    dr.line([cx0, cy0 + chh, cx0 + cw, cy0 + chh], fill=dim); dr.line([cx0, cy0, cx0, cy0 + chh], fill=dim)
    pts = [(cx0 + g / G * cw, cy0 + chh - frac[g] / (frac.max() * 1.1) * chh) for g in range(G + 1)]
    dr.line(pts, fill=(90, 84, 76), width=2)
    gi = min(int(round(t)), G)
    dr.line(pts[:gi + 1], fill=(244, 109, 67), width=3) if gi > 0 else None
    dr.ellipse([pts[gi][0] - 6, pts[gi][1] - 6, pts[gi][0] + 6, pts[gi][1] + 6], fill=(244, 109, 67))
    dr.text((cx0, cy0 + chh + 10), f"share of grid above τ: {frac[gi]:.0%}", font=fa, fill=dim)
    im.save(f"{tmp}/{fi:05d}.png")
S.video(tmp, f"film_phase_{a.target}_{a.model}_{a.regime}_{a.pairing}{a.tag}", fps=30, gif_width=720, gif_fps=12)
Image.open(f"{tmp}/{len(frames) - 1:05d}.png").save(f"/tmp/ouro/phasefilm_last.png")
shutil.rmtree(tmp)
