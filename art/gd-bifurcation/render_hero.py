"""Hero plates of the GD cascade (prod4: GD on 1/2(x1x2x3x4-1)^2, all four coordinates, float64).

Measured: count of visited iterates P_t = x1x2x3x4 per pixel (8192 iterates per eta after 20000 burn-in,
16000 etas), max Lyapunov exponent per eta.  Aesthetic: tone curve, palette, paper, misregistration.

Styles:  dark (colorcet fire)  |  paper (single ink)  |  riso (two inks, Lyapunov strip in pink)
"""
import sys
import numpy as np
from PIL import Image, ImageDraw
from render_lib import *

LO, HI = 0.45, 0.99
YLO, YHI = -0.03, 1.80


def load(lo=LO, hi=HI, W=8000, H=2000, ylo=YLO, yhi=YHI, name="prod4"):
    d = np.load(f"{CACHE}/bif_{name}_hi.npz")
    e = d["etas"]
    m = (e >= lo) & (e <= hi)
    c = density(e[m], d["P"][m], lo, hi, ylo, yhi, W, H)
    d2 = np.load(f"{CACHE}/bif_{name}.npz")  # lyapunov from the 2048-record run (same grid, same init)
    return c, e, d2["lyap"], d2


def tone(c, sat_pct=99.0, gamma=0.8):
    nz = c[c > 0]
    c0 = np.percentile(nz, 30)
    sat = np.percentile(nz, sat_pct)
    t = np.log1p(c / c0) / np.log1p(sat / c0)
    return np.clip(t, 0, 1) ** gamma


def dark(c, out, lut="cc:fire"):
    t = tone(c)
    img = apply_lut(t, cmap_lut(lut))
    save(img, out)
    return img


def paper_ink(c, out, ink=(28, 30, 52), base=(243, 238, 226)):
    H, W = c.shape
    t = tone(c, sat_pct=98.5, gamma=0.9)
    pap = paper(H, W, base=base, grain=5)
    img = ink_multiply(pap, 0.93 * t, ink)
    save(img, out)
    return img


def lyap_strip_mask(e, lyap, lo, hi, W, H, vmin=-0.9, vmax=0.7):
    pos, neg, y0, lam, lmin = lyap_raster(e, lyap, lo, hi, W, H, vmin, vmax)
    return pos, neg, int(round(y0))


def riso(c, e, lyap, out, lo=LO, hi=HI, strip_h=520, gap=60, pad=(140, 160), ink1=(0, 120, 191), ink2=(255, 72, 176),
         offset=(5, -4)):
    H, W = c.shape
    Ht = pad[0] + H + gap + strip_h + pad[0]
    Wt = W + 2 * pad[1]
    pap = paper(Ht, Wt, base=(246, 242, 232), grain=7, seed=3)
    t = tone(c, sat_pct=98.0, gamma=0.85)
    a1 = np.zeros((Ht, Wt))
    a1[pad[0]:pad[0] + H, pad[1]:pad[1] + W] = 0.95 * t
    # Lyapunov strip, second ink, deliberately misregistered by `offset` px
    pos, neg, y0 = lyap_strip_mask(e, lyap, lo, hi, W, strip_h)
    a2 = np.zeros((Ht, Wt))
    ys = pad[0] + H + gap + offset[1]
    xs = pad[1] + offset[0]
    rng = np.random.default_rng(7)
    halftone = (rng.random((strip_h, W)) < 0.55)  # stochastic screen for the negative (periodic) part
    a2[ys:ys + strip_h, xs:xs + W] = np.where(pos, 0.95, 0) + np.where(neg & halftone, 0.8, 0)
    a2[ys + y0 - 1:ys + y0 + 2, xs:xs + W] = 0.9  # lambda = 0 rule
    img = ink_multiply(ink_multiply(pap, a1, ink1), a2, ink2)
    save(img, out)
    return img


def main():
    which = sys.argv[1:] or ["dark", "paper", "riso"]
    c, e, lyap, d2 = load()
    if "dark" in which:
        dark(c, f"{GAL}/hero_dark_fire.png")
        to_img(dark(c, "/tmp/gdb/_x.png"))
    if "paper" in which:
        paper_ink(c, f"{GAL}/hero_paper_ink.png")
    if "riso" in which:
        riso(c, e, lyap, f"{GAL}/hero_riso_lyapunov.png")


if __name__ == "__main__":
    main()
