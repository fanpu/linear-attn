"""Tail-loss ridgeline: one row per generation, the distribution (per log-unit) of each sample's distance
to the nearest centre of the model's own mixture components, in units of the true per-mode sigma.
On a log axis a Gaussian's shape is scale-free, so shrinking variance appears as the ridge sliding left.

usage: python render_ridgeline.py <cache/film_ring_gmm_*.npz> <regime> <style: night|paper> [--G 160] [--step 2]
Measured: histogram of log10(r/sigma) of the display samples at each generation; the share beyond 2 sigma
(truth: e^-2 = 13.5%). Aesthetic: 0.01-decade bins, Gaussian smoothing (0.04 decades), occluding fills.
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
import style as S
from common import RING_CENTERS, RING_SIGMA

ap = argparse.ArgumentParser()
ap.add_argument("cache"); ap.add_argument("regime"); ap.add_argument("style")
ap.add_argument("--G", type=int, default=160); ap.add_argument("--step", type=int, default=2)
ap.add_argument("--lo", type=float, default=-3.0); ap.add_argument("--hi", type=float, default=0.8)
ap.add_argument("--W", type=int, default=2400); ap.add_argument("--name", default="")
a = ap.parse_args()
d = np.load(a.cache)
meta = eval(str(d["meta"]))
model, n, every = meta[1], int(meta[2]), int(meta[6])
D = d[f"{a.regime}/display"].astype(np.float64)
MU = d[f"{a.regime}/disp_mu"]
gens = list(range(0, min(a.G, (D.shape[0] - 1) * every) + 1, a.step))

bins = np.arange(a.lo, a.hi + 1e-9, 0.01)
xc = 0.5 * (bins[1:] + bins[:-1])
rows, tails = [], []
for g in gens:
    X = D[g // every]
    r = np.sqrt(((X[:, None, :] - MU[g // every][None]) ** 2).sum(-1)).min(1) / RING_SIGMA
    h, _ = np.histogram(np.log10(np.maximum(r, 1e-12)), bins)
    rows.append(gaussian_filter1d(h / len(r) / 0.01, 4))
    tails.append((r > 2).mean())
rows = np.array(rows)
rr = 10 ** xc  # truth: 2D Gaussian radius r/sigma ~ Rayleigh; density per log10 unit = ln10 * r * f(r)
truth = np.log(10) * rr * rr * np.exp(-rr ** 2 / 2)

W = a.W
H = int(W * 1.3)
u = W / 2400
ml, mr, mt, mb = int(0.17 * W), int(0.15 * W), int(0.2 * W), int(0.08 * W)
pw = W - ml - mr
nrow = len(rows)
dy = (H - mt - mb) / (nrow + 4)
amp = dy * 7 / truth.max()
night = a.style == "night"
if night:
    ground, line, fg, dim, accent = S.INK["night"], (236, 230, 214), S.INK["bone"], (120, 114, 100), (220, 90, 60)
    im = Image.new("RGB", (W, H), ground)
else:
    ground, line, fg, dim, accent = S.INK["paper"], S.INK["iron_gall"], S.INK["iron_gall"], (112, 102, 90), S.INK["vermilion"]
    im = Image.fromarray(S.paper_texture(H, W).astype(np.uint8))
dr = ImageDraw.Draw(im)
xs = ml + (xc - a.lo) / (a.hi - a.lo) * pw
fnt, fit = S.font("serif", int(26 * u)), S.font("serif_it", int(26 * u))


def ridge(y0, h, col, width, dashed=False):
    ys = y0 - amp * np.maximum(h, 0)
    pts = list(zip(xs.tolist(), ys.tolist()))
    if not dashed:
        dr.polygon(pts + [(xs[-1], y0 + 1), (xs[0], y0 + 1)], fill=ground)
        dr.line(pts, fill=col, width=width)
    else:
        for j in range(0, len(pts) - 5, 10):
            dr.line(pts[j:j + 5], fill=col, width=width)


y_truth = mt + 2 * dy
ridge(y_truth, truth, accent, max(1, int(3 * u)), dashed=True)
dr.text((ml - 130 * u, y_truth - 22 * u), "truth", font=fit, fill=accent)
dr.text((ml + pw + 30 * u, y_truth - 22 * u), "13.5%", font=fnt, fill=accent)
x2 = ml + (np.log10(2) - a.lo) / (a.hi - a.lo) * pw
for yy in range(int(mt), int(H - mb), int(18 * u)):
    dr.line([x2, yy, x2, yy + 8 * u], fill=dim, width=1)
for i, (g, h) in enumerate(zip(gens, rows)):
    y0 = mt + (i + 4) * dy
    ridge(y0, h, line, max(1, int(2 * u)))
    if i % max(1, 10 // a.step) == 0:
        dr.text((ml - 130 * u, y0 - 22 * u), f"{g:>3d}", font=fnt, fill=dim)
        dr.text((ml + pw + 30 * u, y0 - 22 * u), f"{100 * tails[i]:4.1f}%", font=fnt, fill=dim)
M = {"gmm": "Gaussian mixture (K = 8, EM)", "kde": "Gaussian KDE"}[model]
R = {"replace": "replace, λ = 0", "anchored": "replace, λ = 0.25", "accumulate": "accumulate"}[a.regime]
dr.text((ml, 0.05 * W), "The tails go first", font=S.font("serif", int(76 * u)), fill=fg)
for k, t in enumerate([f"{M} trained on its own samples ({R}, n = {n}). One line per generation: how far each sample",
                       "lands from the nearest centre of the model's own components, in units of the true σ (log axis).",
                       "Right: share of samples beyond 2σ (dotted rule). Top, dashed: the true distribution."]):
    dr.text((ml, 0.05 * W + (100 + 40 * k) * u), t, font=S.font("serif_it", int(29 * u)), fill=dim)
dr.text((ml - 130 * u, H - mb + 20 * u), "gen.", font=fit, fill=dim)
for e in range(int(np.ceil(a.lo)), int(np.floor(a.hi)) + 1):
    xx = ml + (e - a.lo) / (a.hi - a.lo) * pw
    lab = {-3: "0.001σ", -2: "0.01σ", -1: "0.1σ", 0: "1σ"}.get(e, f"10^{e}σ")
    dr.text((xx - 30 * u, H - mb + 20 * u), lab, font=fnt, fill=dim)
S.save(im, a.name or f"ridgeline_ring_{model}_{a.regime}_{a.style}.png")
print("tail share at gens 0,10,50,last:", tails[0], tails[5 if a.step == 2 else 0], tails[min(25, len(tails) - 1)], tails[-1])
