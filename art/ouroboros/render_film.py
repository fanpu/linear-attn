"""Generations film: three self-consuming regimes side by side, one generation per beat.

usage: python render_film.py <cache/film_*.npz> <style: dark|paper> [--sub 3] [--gmax G] [--still g1,g2,...]
Measured: each frame is the model's own samples (display pool, CRN) at that generation, splatted to a
density; sparkline = sliced W2 to the true distribution. Aesthetic: splat blur, log tone knee, palette,
crossfade between generations (--sub frames).
"""
import argparse, os, shutil, tempfile
import numpy as np
from PIL import Image, ImageDraw
import style as S

ap = argparse.ArgumentParser()
ap.add_argument("cache"); ap.add_argument("style")
ap.add_argument("--sub", type=int, default=3); ap.add_argument("--gmax", type=int, default=0)
ap.add_argument("--still", default=""); ap.add_argument("--cmap", default="klimt_gold")
ap.add_argument("--name", default="")
a = ap.parse_args()

d = np.load(a.cache)
meta = eval(str(d["meta"]))
target, model, n, G, K = meta[0], meta[1], int(meta[2]), int(meta[3]), int(meta[4])
anchor = float(meta[7])
G = min(G, a.gmax) if a.gmax else G
REG = [("replace", "replace", "λ = 0 · each generation trains only on the last one's samples"),
       ("anchored", "replace", f"λ = {anchor:g} · a fixed {anchor:.0%} of the real data kept"),
       ("accumulate", "accumulate", "accumulate · every sample ever made is kept")]
MODEL = {"gmm": f"Gaussian mixture, K = {K}, refit by EM", "kde": "Gaussian KDE, bandwidth by leave-one-out CV"}[model]
TGT = {"ring": "a ring of eight Gaussians", "spiral": "a noisy spiral", "fern": "Barnsley's fern"}[target]

W, H = 1920, 1080
PS, GAP, X0, Y0 = 560, 60, 70, 170
dark = a.style == "dark"
ground = np.array(S.INK["night"] if dark else S.INK["paper"], float)
fg = S.INK["bone"] if dark else S.INK["iron_gall"]
dim = (130, 124, 112) if dark else (120, 110, 96)
L = S.lut(a.cmap) if dark else None
ink = np.array(S.INK["sepia_ink"], float)
disp = {k: d[f"{k}/display"].astype(np.float64) for k, _, _ in REG}
sw2 = {k: d[f"{k}/sw2"] for k, _, _ in REG}
every = int(meta[6])
ext_y = S.EXT
if target == "fern":
    ext_y = (-1.45, 1.45)

c0 = S.splat(disp["replace"][0], PS, sigma=1.1)
knee = np.percentile(c0[c0 > 0], 60)
top = np.log1p(np.percentile(c0, 99.9) / knee)
base_tex = S.paper_texture(H, W) if not dark else None

allsw = np.concatenate([sw2[k][:G + 1] for k, _, _ in REG])
lo, hi = np.log10(max(allsw.min() * 0.8, 1e-3)), np.log10(allsw.max() * 1.25)
f_t, f_s, f_l, f_big = S.font("serif", 34), S.font("serif_it", 22), S.font("serif", 21), S.font("serif", 60)
f_m = S.font("mono", 16)


def density(k, gi):
    c = S.splat(disp[k][gi], PS, sigma=1.1)
    return np.clip(np.log1p(c / knee) / top, 0, 1)


def frame(dens, gf):
    img = np.tile(ground, (H, W, 1)) if dark else base_tex.copy()
    for i, (k, _, lab) in enumerate(REG):
        x = X0 + i * (PS + GAP)
        v = dens[i]
        if dark:
            rgb = S.apply_lut(v, L).astype(float)
            a_ = np.clip(v * 1.6, 0, 1)[..., None]
            img[Y0:Y0 + PS, x:x + PS] = ground * (1 - a_) + rgb * a_
        else:
            a_ = np.clip(v ** 0.9, 0, 1)[..., None]
            img[Y0:Y0 + PS, x:x + PS] = img[Y0:Y0 + PS, x:x + PS] * (1 - a_) + ink * a_
    im = Image.fromarray(img.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    dr.text((X0, 50), f"Ouroboros — {TGT}, learned by a {MODEL}, fed back to itself", font=f_t, fill=fg)
    dr.text((X0, 100), f"n = {n} samples per generation · each panel is the model's own output at that generation",
            font=f_s, fill=dim)
    g = int(np.floor(gf))
    for i, (k, _, lab) in enumerate(REG):
        x = X0 + i * (PS + GAP)
        dr.rectangle([x - 1, Y0 - 1, x + PS, Y0 + PS], outline=dim if not dark else (40, 38, 34), width=1)
        dr.text((x, Y0 + PS + 14), lab, font=f_l, fill=fg)
        # sparkline of sliced W2 (log scale, shared)
        sy0, sh = Y0 + PS + 60, 150
        dr.line([x, sy0 + sh, x + PS, sy0 + sh], fill=dim, width=1)
        s = sw2[k][:g + 1]
        xs = x + np.arange(len(s)) / max(G, 1) * PS
        ys = sy0 + sh - (np.log10(np.maximum(s, 1e-3)) - lo) / (hi - lo) * sh
        if len(s) > 1:
            dr.line(list(zip(xs.tolist(), ys.tolist())), fill=fg, width=2)
        dr.ellipse([xs[-1] - 4, ys[-1] - 4, xs[-1] + 4, ys[-1] + 4], fill=(196, 64, 40))
        y00 = sy0 + sh - (np.log10(max(sw2[k][0], 1e-3)) - lo) / (hi - lo) * sh
        for xx in range(x, x + PS, 12):
            dr.line([xx, y00, xx + 5, y00], fill=dim, width=1)
        dr.text((x + PS - 170, sy0 - 26), f"sliced W₂ {s[-1]:.3f}", font=f_m, fill=dim)
        dr.text((x, sy0 + sh + 6), f"sliced W₂ to truth (log) · gens 0–{G} · - - gen 0", font=f_m, fill=dim)
    dr.text((W - 470, H - 95), f"generation {g:>3d}", font=f_big, fill=fg)
    return im


if a.still:
    gens = [int(x) for x in a.still.split(",")]
    for g in gens:
        frame([density(k, g // every) for k, _, _ in REG], g).save(f"{S.GAL}/film_{target}_{model}_{a.style}_g{g:03d}.png")
        print("still", g)
else:
    tmp = tempfile.mkdtemp(prefix="ouro_film_")
    fi = 0
    prev = [density(k, 0) for k, _, _ in REG]
    hold = 20
    for _ in range(hold):
        frame(prev, 0).save(f"{tmp}/{fi:05d}.png"); fi += 1
    for g in range(1, G // every + 1):
        cur = [density(k, g) for k, _, _ in REG]
        for s in range(1, a.sub + 1):
            t = s / a.sub
            frame([(1 - t) * p + t * c for p, c in zip(prev, cur)], (g - 1 + t) * every).save(f"{tmp}/{fi:05d}.png"); fi += 1
        prev = cur
    for _ in range(45):
        frame(prev, G).save(f"{tmp}/{fi:05d}.png"); fi += 1
    S.video(tmp, a.name or f"film_{target}_{model}_{a.style}")
    shutil.rmtree(tmp)
