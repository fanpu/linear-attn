"""Hero plates of the GD cascade (prod4: GD on 1/2(x1x2x3x4-1)^2, all four coordinates, float64).

Measured: count of visited iterates P_t = x1x2x3x4 per pixel (8192 iterates per eta after 20000 burn-in,
16000 etas); Lyapunov exponent per eta; iterate phase (t - t0) mod 16.
Aesthetic (declared): tone curve (log density, floor = median density of the chaotic band, ceiling = 99.7th
percentile of the band so only caustic folds reach white), minimum stroke weight (pixels above the ceiling
are dilated to 5 px so period-1/2/4 branches survive print scale), palettes, paper grain, misregistration.

Styles:
  hero_dark_fire_full      eta 0.45-0.99 (honest full range incl. the stable regime)
  hero_dark_fire           eta 0.48-0.99 (crop from just before the first doubling)
  hero_paper_ink           single ink on paper
  hero_riso_lyapunov       two inks: cascade in blue, Lyapunov strip in fluorescent pink (lambda<0 dips in a 40% tint)
  hero_phase_braid         hue = circular mean of bit-reversed iterate phase (t - t0) mod 16 (cyclic map), brightness = density
"""
import sys
import numpy as np
from scipy.ndimage import maximum_filter
from render_lib import *

YLO, YHI = -0.03, 1.80
BAND = (0.80, 0.95)  # eta range used to calibrate the tone floor/ceiling


def load(lo, hi, W=8000, H=2000, ylo=YLO, yhi=YHI, name="prod4"):
    d = np.load(f"{CACHE}/bif_{name}_hi.npz")
    e = d["etas"]
    m = (e >= lo) & (e <= hi)
    c = density(e[m], d["P"][m], lo, hi, ylo, yhi, W, H)
    return c, d


def tone(c, lo, hi, floor_pct=50, ceil_pct=99.7, gamma=1.0, stroke=5):
    W = c.shape[1]
    a = int((BAND[0] - lo) / (hi - lo) * W)
    b = int((BAND[1] - lo) / (hi - lo) * W)
    band = c[:, a:b]
    nz = band[band > 0]
    c0 = np.percentile(nz, floor_pct)
    sat = np.percentile(nz, ceil_pct)
    t = np.clip(np.log1p(c / c0) / np.log1p(sat / c0), 0, 1) ** gamma
    if stroke:
        # minimum stroke weight only for periodic columns (few occupied rows), so chaotic-band density
        # peaks are not inflated into dots
        occ = (c > 0).mean(0)
        heavy = (c > sat) & (occ < 0.02)[None, :]
        t = np.maximum(t, 0.92 * maximum_filter(heavy.astype(float), size=stroke))
    return t


def dark(c, lo, hi, out, lut="cc:fire"):
    img = apply_lut(tone(c, lo, hi), cmap_lut(lut))
    save(img, out)


def paper_ink(c, lo, hi, out, ink=(24, 26, 46), base=(243, 238, 226)):
    H, W = c.shape
    t = tone(c, lo, hi, floor_pct=40, ceil_pct=99.0, gamma=0.85)
    pap = paper(H, W, base=base, grain=4, seed=1)
    save(ink_multiply(pap, 0.95 * t, ink), out)


def riso(c, d, lo, hi, out, strip_h=560, gap=70, pad=(150, 170), ink1=(0, 120, 191), ink2=(255, 72, 176),
         offset=(6, -5)):
    H, W = c.shape
    Ht = pad[0] + H + gap + strip_h + pad[0]
    Wt = W + 2 * pad[1]
    pap = paper(Ht, Wt, base=(246, 242, 232), grain=5, seed=3)
    t = tone(c, lo, hi, floor_pct=40, ceil_pct=99.3, gamma=0.9)
    a1 = np.zeros((Ht, Wt))
    a1[pad[0]:pad[0] + H, pad[1]:pad[1] + W] = 0.95 * t
    e = d["etas"]
    lam = d["lyap_bal"]  # exponent of the oscillating mode, 8192 steps
    m = (e >= lo) & (e <= hi)
    vmax, vmin = 0.7, -1.6
    pos, neg, y0, lmax, lmin = lyap_raster(e[m], lam[m], lo, hi, W, strip_h, vmin, vmax)
    y0 = int(round(y0))
    a2 = np.zeros((Ht, Wt))
    ys = pad[0] + H + gap + offset[1]
    xs = pad[1] + offset[0]
    a2[ys:ys + strip_h, xs:xs + W] = np.where(pos, 0.95, 0) + np.where(neg, 0.38, 0)
    a2[ys + y0 - 1:ys + y0 + 2, xs:xs + W] = 0.95
    img = ink_multiply(ink_multiply(pap, a1, ink1), a2, ink2)
    save(img, out)


def bitrev(k, bits):
    return int(format(k, f"0{bits}b")[::-1], 2)


def phase_braid(d, lo, hi, out, W=8000, H=2000, bits=4, ylo=YLO, yhi=YHI):
    """hue = circular mean over visits of the bit-reversed phase (t mod 16): the first period doubling splits
    the circle in half, the next in quarters, ... so the branch tree maps onto the hue circle."""
    nph = 2 ** bits
    e = d["etas"]
    m = (e >= lo) & (e <= hi)
    P = d["P"][m]
    R = P.shape[1]
    # phase is measured relative to the iterate with the largest P among the first 16 recorded, so that
    # the same branch gets the same phase in every eta column (otherwise the transient sets an arbitrary shift)
    t0 = np.argmax(np.nan_to_num(P[:, :nph], nan=-np.inf), axis=1)
    ph = ((np.arange(R)[None, :] - t0[:, None]) % nph).astype(np.int8)
    Z = np.zeros((H, W), complex)
    Ctot = np.zeros((H, W))
    for k in range(nph):
        ck = density(e[m], np.where(ph == k, P, np.nan), lo, hi, ylo, yhi, W, H)
        Z += ck * np.exp(2j * np.pi * bitrev(k, bits) / nph)
        Ctot += ck
    Rbar = np.abs(Z) / np.maximum(Ctot, 1)
    hue = (np.angle(Z) / (2 * np.pi)) % 1.0
    t = tone(Ctot, lo, hi, stroke=0)
    lut = cmap_lut("cc:cyclic_mygbm_30_95_c78")
    col = apply_lut(hue, lut)
    grey = np.full_like(col, 200.0)
    chroma = np.clip(Rbar, 0, 1)[..., None]
    rgb = (chroma * col + (1 - chroma) * grey) * t[..., None]
    occ = (Ctot > 0).mean(0)
    a = int((BAND[0] - lo) / (hi - lo) * W)
    b = int((BAND[1] - lo) / (hi - lo) * W)
    bnd = Ctot[:, a:b]
    sat = np.percentile(bnd[bnd > 0], 99.7)
    heavy = maximum_filter(((Ctot > sat) & (occ < 0.02)[None, :]).astype(float), size=5)
    # thicken strokes using the colour of the nearest heavy pixel
    colh = maximum_filter(rgb, size=(5, 5, 1))
    rgb = np.where(heavy[..., None] > 0, np.maximum(rgb, 0.95 * colh), rgb)
    save(rgb, out)


def main():
    which = sys.argv[1:] or ["dark", "crop", "paper", "riso", "phase"]
    if "dark" in which:
        c, d = load(0.45, 0.99)
        dark(c, 0.45, 0.99, f"{GAL}/hero_dark_fire_full.png")
    lo, hi = 0.48, 0.99
    c, d = load(lo, hi)
    if "crop" in which:
        dark(c, lo, hi, f"{GAL}/hero_dark_fire.png")
    if "paper" in which:
        paper_ink(c, lo, hi, f"{GAL}/hero_paper_ink.png")
    if "riso" in which:
        riso(c, d, lo, hi, f"{GAL}/hero_riso_lyapunov.png")
    if "phase" in which:
        phase_braid(d, lo, hi, f"{GAL}/hero_phase_braid.png")


if __name__ == "__main__":
    main()
