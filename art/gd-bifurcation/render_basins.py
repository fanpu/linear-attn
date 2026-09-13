"""Initialisation basin maps (Zhu et al. degree-4 model, eta = 0.2).

Measured per pixel: converge / diverge, first step with loss < 1e-12 (converged pixels), smooth escape
value nu (diverged pixels), sharpness of the minimum reached.
Aesthetic (declared): palettes; log scaling of times; the smooth escape value itself is the standard
continuous colouring (fractals doc Sec. 11.5) and is declared as such.

Styles: dark (escape time in fire, convergence time in a cool ramp), paper (single-ink boundary drawing),
riso (two inks), plus the zoom-sequence plate with the Sec. 11 verification panels (render_basin_plate).
"""
import json
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter
from render_lib import *
from analyze_boxcount import boundary


def load(tag, res=4096, name="zhu4"):
    d = np.load(f"{CACHE}/basin_{name}_{tag}_{res}.npz")
    return {k: d[k] for k in d.files}


def dark(d):
    s = d["status"]
    nu = d["nu"]
    t = d["tev"].astype(float)
    img = np.zeros(s.shape + (3,))
    div = s == 2
    conv = s == 1
    # escape: fast escape = dark, slow (near the boundary) = bright
    v = nu[div]
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.7)
    te = np.clip((np.log(nu) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
    img[div] = apply_lut(te[div] ** 1.1, cmap_lut("cc:fire"))
    # convergence: time to reach loss < 1e-12, cool ramp
    lc = np.log(t)
    c = lc[conv]
    lo, hi = np.percentile(c, 0.5), np.percentile(c, 99.5)
    tc = np.clip((lc - lo) / (hi - lo), 0, 1)
    img[conv] = apply_lut(0.15 + 0.7 * tc[conv], cmap_lut("cmc:oslo"))
    return img


def paper_line(d, ink=(24, 26, 46), weight=1):
    s = d["status"]
    b = boundary(s == 2)
    if weight > 1:
        b = maximum_filter(b, size=weight)
    H, W = s.shape
    pap = paper(H, W, base=(243, 238, 226), grain=3, seed=2)
    a = 0.95 * b + 0.07 * (s == 1)
    return ink_multiply(pap, a, ink)


def riso(d, ink1=(0, 95, 160), ink2=(255, 40, 150), offset=(3, -2)):
    s = d["status"]
    H, W = s.shape
    rng = np.random.default_rng(4)
    nu = d["nu"]
    div = s == 2
    v = nu[div]
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    te = np.zeros_like(nu)
    te[div] = np.clip((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
    t = d["tev"].astype(float)
    conv = s == 1
    lc = np.log(t)
    c = lc[conv]
    tc = np.zeros_like(lc)
    tc[conv] = np.clip((lc[conv] - np.percentile(c, 0.5)) / (np.percentile(c, 99.5) - np.percentile(c, 0.5)), 0, 1)
    # stochastic screens (declared)
    a1 = (rng.random((H, W)) < (0.25 + 0.7 * tc)) * conv * 0.9
    a2 = np.zeros((H, W))
    a2[:] = (rng.random((H, W)) < te ** 1.3) * div * 0.95
    a2 = np.roll(a2, offset, axis=(1, 0))
    pap = paper(H, W, base=(246, 242, 232), grain=3, seed=6)
    return ink_multiply(ink_multiply(pap, a1, ink1), a2, ink2)


def spectral(d):
    """Declared: Sohl-Dickstein Spectral split. Converged side ranked by convergence speed (1/steps), purple at
    the boundary; diverged side ranked by escape speed (1/nu), deep red at the boundary."""
    s = d["status"]
    t = d["tev"].astype(float)
    nu = d["nu"].astype(float)
    return spectral_split(1 / np.maximum(t, 1), 1 / np.nan_to_num(nu, nan=1e9), s == 1, s == 2)


def main():
    args = sys.argv[1:]
    only = None
    if args and args[0].startswith("--only="):
        only = args[0].split("=")[1]
        args = args[1:]
    which = args or ["wideT", "z1T", "z2T"]
    for tag in which:
        d = load(tag)
        for style, fn in [("dark", dark), ("paper", paper_line), ("riso", riso), ("spectral", spectral)]:
            if only and style != only:
                continue
            img = fn(d)
            save(img[::-1], f"{GAL}/basin_{tag.rstrip('T')}_{style}.png")  # row 0 = top = largest y


if __name__ == "__main__":
    main()
