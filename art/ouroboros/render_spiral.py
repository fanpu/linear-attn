"""Nested-generations spiral: generation g is drawn at scale r^g on a logarithmic spiral, so the
chain's history coils inward toward the most-retrained generation at the eye.

usage: python render_spiral.py <cache/film_*.npz> <style: dark|paper|riso> [--arms replace,accumulate]
       [--G 150] [--size 3000] [--r 0.972] [--dtheta 0.30]
Measured: each tile is the model's own samples at that generation (display pool).
Aesthetic: spiral layout (log spiral, scale r per generation), density normalised per tile area, inks.
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import style as S

ap = argparse.ArgumentParser()
ap.add_argument("cache"); ap.add_argument("style")
ap.add_argument("--arms", default="replace"); ap.add_argument("--G", type=int, default=150)
ap.add_argument("--size", type=int, default=3000); ap.add_argument("--r", type=float, default=0.972)
ap.add_argument("--dtheta", type=float, default=0.30); ap.add_argument("--tile", type=float, default=0.155)
ap.add_argument("--cmap", default="klimt_gold"); ap.add_argument("--name", default="")
ap.add_argument("--nocap", action="store_true"); ap.add_argument("--stride", type=int, default=1)
a = ap.parse_args()

d = np.load(a.cache)
meta = eval(str(d["meta"]))
target, model, n, Gmax, K, every = meta[0], meta[1], int(meta[2]), int(meta[3]), int(meta[4]), int(meta[6])
arms = a.arms.split(",")
G = min(a.G, Gmax)
Wd = Hd = a.size
C = np.array([Wd / 2, Hd / 2])
R0 = 0.40 * Wd
canvas = [np.zeros((Hd, Wd)) for _ in arms]

for ai, arm in enumerate(arms):
    D = d[f"{arm}/display"].astype(np.float64)
    phase = np.pi * ai  # second arm is the point reflection of the first
    for i, g in enumerate(range(0, G + 1, a.stride)):
        s = a.r ** i
        th = np.pi / 2 + phase - a.dtheta * i  # start at top, coil clockwise inward
        c = C + R0 * s * np.array([np.cos(th), -np.sin(th)])
        half = a.tile * Wd * s / 2  # tile half-size in px, data window EXT maps to it
        X = D[g // every]
        px = c[0] + X[:, 0] / S.EXT[1] * half
        py = c[1] - X[:, 1] / S.EXT[1] * half
        ok = (px >= 0) & (px < Wd - 1) & (py >= 0) & (py < Hd - 1)
        # bilinear splat, weight normalised by tile area so ink density per unit data area is constant
        w = (60.0 / max(half, 1.0)) ** 2 / len(X) * 4000
        x0, y0 = np.floor(px[ok]).astype(int), np.floor(py[ok]).astype(int)
        fx, fy = px[ok] - x0, py[ok] - y0
        for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(canvas[ai], (y0 + dy, x0 + dx), w * ww)

sig = a.size / 3000 * 1.0
dens = [gaussian_filter(cv, sig) for cv in canvas]
ref = np.percentile(np.concatenate([x[x > 1e-9] for x in dens]), 70)
V = [np.clip(np.log1p(x / ref) / np.log1p(40), 0, 1) for x in dens]

if a.style == "dark":
    L = S.lut(a.cmap)
    img = np.tile(np.array(S.INK["night"], float), (Hd, Wd, 1))
    for v in V:
        al = np.clip(v * 1.5, 0, 1)[..., None]
        img = img * (1 - al) + S.apply_lut(v, L) * al
    fg, dim = S.INK["bone"], (120, 114, 100)
elif a.style == "paper":
    img = S.paper_texture(Hd, Wd)
    inks = [np.array(S.INK["sepia_ink"], float), np.array(S.INK["vermilion"], float)]
    for v, ink in zip(V, inks):
        al = np.clip(v, 0, 1)[..., None]
        img = img * (1 - al) + ink * al
    fg, dim = S.INK["iron_gall"], (120, 110, 96)
else:  # riso: two spot inks multiplied (declared), slight misregistration of the second drum
    paper = np.array(P_ := S.P.RISO_PAPER if hasattr(S.P, "RISO_PAPER") else "#f4efe4")
    base = np.array(S.P.hex2rgb(paper) if isinstance(paper.item(), str) else paper, float)
    base = base * (255 if base.max() <= 1 else 1)
    inks = [np.array([255, 72, 176], float), np.array([0, 120, 191], float)]
    img = np.tile(base, (Hd, Wd, 1))
    for i, (v, ink) in enumerate(zip(V, inks)):
        if i == 1:
            v = np.roll(v, (int(3 * sig), int(-4 * sig)), (0, 1))
        cov = np.clip(v * 1.1, 0, 1)[..., None]
        img = img * (1 - cov + cov * ink / 255)
    fg, dim = (40, 36, 60), (110, 104, 120)

im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
if not a.nocap:
    dr = ImageDraw.Draw(im)
    u = a.size / 3000
    MODEL = {"gmm": f"Gaussian mixture (K = {K}, EM)", "kde": "Gaussian KDE (LOO-CV bandwidth)"}[model]
    lab = {"replace": "replace (λ = 0)", "anchored": "anchored (λ = 0.25)", "accumulate": "accumulate"}
    dr.text((80 * u, 70 * u), "OUROBOROS", font=S.font("serif", int(64 * u)), fill=fg)
    dr.text((80 * u, 150 * u), f"{MODEL} retrained on its own samples, n = {n}", font=S.font("serif_it", int(34 * u)), fill=dim)
    dr.text((80 * u, 195 * u), f"generation 0 at top; one tile every {a.stride} generation(s), ≈{2*np.pi/a.dtheta:.0f} tiles per turn, scale ×{a.r} per tile, "
            f"to generation {G} at the eye", font=S.font("serif_it", int(28 * u)), fill=dim)
    dr.text((80 * u, Hd - 110 * u), " · ".join(lab[x] for x in arms) + ("   (second arm: point reflection)" if len(arms) > 1 else ""),
            font=S.font("serif", int(30 * u)), fill=fg)
name = a.name or f"spiral_{target}_{model}_{'-'.join(arms)}_{a.style}.png"
S.save(im, name)
