"""Depth strata (brainstorm pick): the value of the depth-L field along one fixed 150-degree arc of a
great circle, L = 1 (top) ... 12 (bottom), same random draw; hidden-line ridgelines, one ink.
Declared: each ridge is centred on its median and scaled by its own robust spread (deep fields are a
large random constant plus small fluctuations), so the *shape* is faithful and the amplitude is not."""
import sys, numpy as np
from render_common import *
from common import *
from globe import make_alm

LMAX = 4096
NPT = 9000
arc = np.radians(150)
e1, e2, c = rot_frame(np.array([0.3, -0.5, 0.8]))
t = np.linspace(-arc / 2, arc / 2, NPT)
V = np.cos(t)[:, None] * c + np.sin(t)[:, None] * e1
th, ph = vec_to_thetaphi(V)
z = white_alm(LMAX, 11)
Wd, Hd = 3600, 4200
AMP, LW = 1.1, 2
def strata(act, Ls, ink, name, title):
    img = np.ones((Hd, Wd, 3)) * PAPER * grain((Hd, Wd), 5, 0.01)[..., None]
    pil = to_img(img)
    dr = ImageDraw.Draw(pil, "RGBA")
    top, bottom, left, right = 700, Hd - 350, 260, Wd - 260
    gap = (bottom - top) / (len(Ls) - 1 + 1.6)
    amp = gap * AMP
    xs = np.linspace(left, right, NPT)
    sp = np.load("cache/spectra.npz"); names = list(sp["names"])
    curves = []
    for L in Ls:
        f = synth(alm_from_white(z, sp["C"][names.index(f"{act}_L{L}")][:LMAX + 1], LMAX), LMAX, th, ph, nthreads=6)
        f = f - np.median(f)
        s = np.percentile(np.abs(f), 95) + 1e-12
        curves.append(f / s)
    # draw from back (top) to front (bottom); each ridge occludes the ones above it
    for i, (L, f) in enumerate(zip(Ls, curves)):
        y0 = top + (i + 1) * gap
        ys = y0 - amp * 0.5 * f
        poly = list(zip(xs, ys)) + [(right, Hd), (left, Hd)]
        dr.polygon(poly, fill=tuple(int(255 * x) for x in PAPER) + (255,))
        dr.line(list(zip(xs, ys)), fill=tuple(int(255 * x) for x in ink) + (255,), width=LW, joint="curve")
        dr.text((left - 200, y0 - 20), f"L={L}", font=font(40, "mono"), fill=(27, 27, 34))
    # repaint footer band
    dr.rectangle([0, Hd - 330, Wd, Hd], fill=tuple(int(255 * x) for x in PAPER) + (255,))
    dr.text((left, 180), title, font=font(110), fill=(27, 27, 34))
    dr.text((left, 340), f"{ACT_LABEL[act]} network, infinite width: its value along one 150-degree arc of a great circle, depth 1 to {Ls[-1]}.",
            font=font(46, "italic"), fill=(95, 90, 82))
    dr.text((left, 410), "Same random draw throughout. Each stratum centred and scaled to its own spread (declared).",
            font=font(46, "italic"), fill=(95, 90, 82))
    dr.multiline_text((left, Hd - 290), "Exact band-limited samples (l <= 4096) of the depth-L Gaussian-process limit (Di Lillo et al. 2025), 9000 points along the arc.\n"
                      "Hidden-line ridgeline rendering: each stratum masks the ones above it.", font=font(38), fill=(95, 90, 82), spacing=12)
    pil.save(f"gallery/strata_{name}.png"); print(name)

strata("heaviside", list(range(1, 9)), INK, "heaviside", "Strata")
AMP, LW = 1.9, 3
strata("relu", list(range(1, 13)), np.array([0.45, 0.22, 0.1]), "relu", "Strata, regular")
