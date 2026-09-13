"""Botanical-engraving plates of Barnsley's fern as learned, and re-learned, by a Gaussian mixture.

usage: python render_fern.py <cache/film_fern_gmm_*.npz> sheet|hero [--gens 0,10,40,100,200] [--regimes replace,anchored,accumulate]
Measured: stipple dots = the model's own samples at that generation; engraved ellipses = the fitted mixture
components (1 and 2 sigma contours; line darkness ~ component weight). First column = the true fern (chaos game).
Aesthetic: paper, sepia ink, dot size, 4-line hatching inside each ellipse, plate typography, 2x supersampling.
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw
import style as S
from common import reference_sample

ap = argparse.ArgumentParser()
ap.add_argument("cache"); ap.add_argument("mode")
ap.add_argument("--gens", default="0,10,40,100,200"); ap.add_argument("--regimes", default="replace,anchored,accumulate")
ap.add_argument("--size", type=int, default=3000); ap.add_argument("--name", default="")
ap.add_argument("--ink", default="sepia_ink")
a = ap.parse_args()
d = np.load(a.cache)
meta = eval(str(d["meta"]))
n, K, every = int(meta[2]), int(meta[4]), int(meta[6])
gens = [int(g) for g in a.gens.split(",")]
regs = a.regimes.split(",")
SS = 2
INK = np.array(S.INK[a.ink])
truth = reference_sample("fern", 20000)
# fern occupies x in ~[-0.66, 0.66], y in [-1.35, 1.35] in data units
XR, YR = (-0.75, 0.75), (-1.4, 1.4)


def draw_panel(dr, box, X, comps, ink, dot):
    x0, y0, x1, y1 = box
    sx = (x1 - x0) / (XR[1] - XR[0]); sy = (y1 - y0) / (YR[1] - YR[0])
    s = min(sx, sy)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    to = lambda p: (cx + p[..., 0] * s, cy - p[..., 1] * s)
    if comps is not None:
        pi, mu, cov = comps
        t = np.linspace(0, 2 * np.pi, 90)
        circ = np.stack([np.cos(t), np.sin(t)], 1)
        for k in np.argsort(pi):
            L = np.linalg.cholesky(cov[k] + 1e-9 * np.eye(2))
            w = float(np.clip(np.sqrt(pi[k] * K), 0.15, 1.0))
            col = tuple(int(255 - (255 - c) * w) for c in (0.55 * ink + 0.45 * np.array([242, 236, 222])))
            for rad, width in ((2.0, 1), (1.0, 2)):
                P = mu[k] + rad * circ @ L.T
                px, py = to(P)
                dr.line(list(zip(px.tolist(), py.tolist())), fill=col, width=width * SS)
            # engraved hatching inside the 1-sigma ellipse, along the minor axis direction
            for hh in np.linspace(-0.8, 0.8, 5):
                seg = mu[k] + np.array([[hh, -np.sqrt(1 - hh * hh)], [hh, np.sqrt(1 - hh * hh)]]) @ L.T
                px, py = to(seg)
                dr.line(list(zip(px.tolist(), py.tolist())), fill=col, width=SS)
    px, py = to(X)
    r = dot * SS
    colp = tuple(int(c) for c in ink)
    for x, y in zip(px, py):
        if x0 <= x <= x1 and y0 <= y <= y1:
            dr.ellipse([x - r, y - r, x + r, y + r], fill=colp)


def comps_at(reg, g):
    key = f"{reg}/disp_pi"
    if key not in d.files:
        return None
    return d[key][g // every], d[f"{reg}/disp_mu"][g // every], d[f"{reg}/disp_cov"][g // every]


serif, ital = lambda z: S.font("school", int(z * SS)), lambda z: S.font("school_it", int(z * SS))
paper = S.paper_texture
lab = {"replace": "replaced each generation (λ = 0)", "anchored": "a quarter of the real sample kept (λ = 0.25)",
       "accumulate": "all samples accumulated"}
if a.mode == "sheet":
    cols = 1 + len(gens)
    pw, ph = 520, 900
    gx, gy = 40, 170
    ml, mt = 240, 420
    W = ml + cols * pw + (cols - 1) * gx + 120
    H = mt + len(regs) * ph + (len(regs) - 1) * gy + 260
    im = Image.fromarray(paper(H * SS, W * SS, amp=0.8).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    Z = lambda v: v * SS
    dr.rectangle([Z(50), Z(50), Z(W - 50), Z(H - 50)], outline=tuple(INK), width=Z(3))
    dr.rectangle([Z(64), Z(64), Z(W - 64), Z(H - 64)], outline=tuple(INK), width=Z(1))
    dr.text((Z(ml), Z(110)), "Filix ouroborum", font=ital(84), fill=tuple(INK))
    dr.text((Z(ml), Z(215)), f"Barnsley's fern, learned by a {K}-component Gaussian mixture from n = {n} samples, "
            "then retrained on its own samples, generation after generation", font=ital(34), fill=tuple(INK))
    dr.text((Z(W - 260), Z(110)), "Pl. VII", font=serif(48), fill=tuple(INK))
    for i, reg in enumerate(regs):
        yb = mt + i * (ph + gy)
        dr.text((Z(ml), Z(yb - 70)), lab[reg], font=ital(36), fill=tuple(INK))
        for j in range(cols):
            xb = ml + j * (pw + gx)
            box = (Z(xb), Z(yb), Z(xb + pw), Z(yb + ph))
            if j == 0:
                if i == 0:
                    draw_panel(dr, box, truth[:12000], None, INK, 0.9)
                    dr.text((Z(xb + 150), Z(yb + ph + 20)), "ex natura", font=ital(34), fill=tuple(INK))
                continue
            g = gens[j - 1]
            X = d[f"{reg}/display"][g // every].astype(float)[:12000]
            draw_panel(dr, box, X, comps_at(reg, g), INK, 0.9)
            sw = d[f"{reg}/sw2"][g]
            dr.text((Z(xb + 120), Z(yb + ph + 20)), f"gen. {g}", font=ital(34), fill=tuple(INK))
            dr.text((Z(xb + 120), Z(yb + ph + 62)), f"W₂ {sw:.3f}", font=S.font("mono", 22 * SS), fill=tuple(INK))
    for j in range(1):
        pass
    im = im.resize((W, H), Image.LANCZOS)
    S.save(im, a.name or f"fern_sheet_{a.ink}.png")
else:  # hero: two large panels, generation gens[0] and gens[-1] of the first regime
    reg = regs[0]
    W, H = a.size, int(a.size * 0.78)
    im = Image.fromarray(paper(H * SS, W * SS, amp=0.8).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    Z = lambda v: int(v * SS)
    u = W / 3000
    dr.rectangle([Z(40 * u), Z(40 * u), Z(W - 40 * u), Z(H - 40 * u)], outline=tuple(INK), width=Z(3))
    g0, g1 = gens[0], gens[-1]
    for j, g in enumerate([g0, g1]):
        xb = 150 * u + j * (W - 300 * u) / 2
        box = (Z(xb), Z(200 * u), Z(xb + (W - 300 * u) / 2), Z(H - 230 * u))
        X = d[f"{reg}/display"][g // every].astype(float)
        draw_panel(dr, box, X, comps_at(reg, g), INK, 1.1)
        dr.text((Z(xb + 500 * u), Z(H - 200 * u)), f"generation {g}", font=ital(46 * u), fill=tuple(INK))
    dr.text((Z(150 * u), Z(90 * u)), f"Filix ouroborum — the fern and its {g1}th retelling", font=ital(64 * u), fill=tuple(INK))
    dr.text((Z(150 * u), Z(H - 120 * u)), f"{K}-component Gaussian mixture, n = {n}, {lab[reg]}. Dots: the model's samples; "
            "ellipses: its components at 1σ and 2σ.", font=ital(34 * u), fill=tuple(INK))
    im = im.resize((W, H), Image.LANCZOS)
    S.save(im, a.name or f"fern_hero_{reg}_{a.ink}.png")
