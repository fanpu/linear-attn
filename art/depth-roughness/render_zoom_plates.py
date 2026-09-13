"""Zoom plates: nested windows (each 1/4 the side of the previous) centred on one point of the
level set of a depth-L Heaviside GP draw, 0.785 rad down to 1.9e-7 rad. Multi-scale exact-SHT +
flat-sky band sampler (zoom_engine). usage: render_zoom_plates.py <kernel> <seed> <style>"""
import sys, numpy as np
from render_common import *
from common import fit_dim, theory
import cmcrameri.cm as cmc

kernel, seed, style = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(f"cache/multiscale_{kernel}_s{seed}.npz")
u = float(d["u"]); js = list(d["j"]); Fs = d["F"]
L = int(kernel.split("_L")[1])
PL, GAP, COLS = 1024, 70, 4
rows = (len(js) + COLS - 1) // COLS
W = COLS * PL + (COLS + 1) * GAP
TOP = 330
H = TOP + rows * (PL + 130) + 215
bg = DARK if style == "dark" else PAPER
fg = (226, 222, 212) if style == "dark" else (27, 27, 34)
dimc = (150, 146, 138) if style == "dark" else (115, 110, 100)
img = to_img(np.ones((H, W, 3)) * bg * (1 if style == "dark" else grain((H, W), 3, 0.01)[..., None]))
dr = ImageDraw.Draw(img)
dr.text((GAP, 60), f"Twelve windows into one coastline", font=font(84), fill=fg)
dr.text((GAP, 170), f"Heaviside network, infinite width, depth L = {L} (dimH = {2-2.0**-L:.4g}). Each window is 1/4 the side of the last;"
        f" the square marks the next.", font=font(36, "italic"), fill=dimc)
dr.text((GAP, 220), f"From {Fs[0]:.3g} rad to {Fs[-1]:.2g} rad across: 6.6 decades. Level u = {u:.4f}, fixed for every window.",
        font=font(36, "italic"), fill=dimc)
for i, j in enumerate(js):
    f = d[f"f{j}"].astype(np.float64)
    x = GAP + (i % COLS) * (PL + GAP); y = TOP + (i // COLS) * (PL + 130)
    if style == "plotter":
        cov = ink_coverage(f, u, 2, weight=1)
        rgb = mix(PAPER, INK, np.clip(cov * 1.7, 0, 1) ** 0.9)
    elif style == "dark":
        g = downsample(f.astype(np.float32), 2)
        sd = max(np.percentile(np.abs(g - u), 98), 1e-12)
        t = np.clip(0.5 + 0.5 * (g - u) / sd, 0, 1)            # diverging about the level, per-plate stretch
        rgb = cmc.berlin(t)[..., :3]
        rgb = mix(rgb, np.array([1.0, 0.95, 0.85]), np.clip(ink_coverage(f, u, 2) * 1.5, 0, 0.9))
    elif style == "riso":
        up = downsample((f > u).astype(np.float32), 2)
        ln = ink_coverage(f, u, 2, weight=2)
        rgb = multiply_ink(PAPER, [(RISO_TEAL, 0.75 * up), (RISO_PINK, 0.95 * np.clip(ln * 1.3, 0, 1))])
    im = to_img(rgb)
    di = ImageDraw.Draw(im)
    if i < len(js) - 1:
        a = PL * 3 // 8
        di.rectangle([a, a, PL - a - 1, PL - a - 1], outline=(200, 40, 40) if style != "dark" else (255, 200, 120), width=3)
    img.paste(im, (x, y))
    s, cnt = d["sizes"], d["counts"][i]
    D = fit_dim(s, cnt, 8, 64)[0]
    dr.text((x, y + PL + 16), f"{chr(65+i)}   {Fs[i]:.2e} rad   x{4**i:,}", font=font(34, "mono"), fill=fg)
    dr.text((x, y + PL + 60), f"box D {D:.2f} (8-64 px)   fraction above u {d['frac'][i]:.2f}", font=font(26, "mono"), fill=dimc)
dr.multiline_text((GAP, H - 190),
    "Windows A-B: exact spherical-harmonic sample (l <= 8192) on the sphere (A resolves to 1/1020 of its side, the rest to 1/256). From C on, flat-sky Gaussian bands\n"
    "(l up to 8192*4^10) are added, one per window, from the flat-sky limit of the same kernel spectrum.\n"
    "Every later window resolves wavelengths down to 1/256 of its side (4 px here), so the drawing is band-limited at every scale in the same relative way.\n"
    "float64 throughout (coordinates ~1e-10 rad resolved with 6 digits to spare); no precision floor is reached at 1.9e-7 rad.",
    font=font(28), fill=dimc, spacing=10)
out = f"gallery/zoom_plates_{kernel}_{style}.png"
img.save(out); print(out, img.size)
