"""Window atlas: every periodic window found in the chaotic band, each a miniature cascade.

Measured: per-plate count rasters from the full 4-coordinate GD (cache/atlas/pXX.npz), Lyapunov exponent
of the oscillating mode per eta.  Aesthetic: tone curve, palette, layout, typography.
Outputs: gallery/atlas_dark.png (grid), gallery/atlas_paper.png (survey sheet), gallery/atlas/pXX.png
"""
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter
from render_lib import *

HERO = (0.45, 0.99, -0.03, 1.80)


def tone_plate(C, stroke=3):
    nz = C[C > 0]
    c0 = np.percentile(nz, 50)
    sat = np.percentile(nz, 99.5)
    t = np.clip(np.log1p(C / c0) / np.log1p(sat / c0), 0, 1)
    occ = (C > 0).mean(0)
    heavy = (C > sat) & (occ < 0.03)[None, :]
    return np.maximum(t, 0.92 * maximum_filter(heavy.astype(float), size=stroke))


def fmt(v, span):
    digs = max(3, int(np.ceil(-np.log10(span))) + 2)
    return f"{v:.{digs}f}"


def strip(etas, lam, W, H, rect):
    lo, hi = rect[0], rect[1]
    pos, neg, y0, _, _ = lyap_raster(etas, lam, lo, hi, W, H, -1.5, 0.7)
    return pos, neg, int(round(y0))


def plate_img(d, W, H, style="dark", strip_h=None):
    C = d["C"].astype(float)
    t = tone_plate(C)
    im = Image.fromarray(np.uint8(t * 255)).resize((W, H), Image.LANCZOS)
    t = np.asarray(im, float) / 255
    sh = strip_h or H // 7
    pos, neg, y0 = strip(d["etas"], d["lyap_bal"], W, sh, d["rect"])
    if style == "dark":
        img = apply_lut(t, cmap_lut("cc:fire"))
        s = np.zeros((sh, W, 3))
        s[pos] = (235, 235, 235)
        s[neg] = (90, 90, 110)
        s[y0] = (160, 160, 160)
        return np.vstack([img, np.zeros((6, W, 3)), s])
    else:
        pap = paper(H + 6 + sh, W, base=(243, 238, 226), grain=3, seed=int(d["period"]))
        a = np.zeros((H + 6 + sh, W))
        a[:H] = 0.95 * t
        a[H + 6:][pos] = 0.9
        a[H + 6:][neg] = 0.35
        a[H + 6 + y0] = 0.8
        return ink_multiply(pap, a, (24, 26, 46))


def main():
    files = sorted(glob.glob(f"{CACHE}/atlas/p*.npz"))
    D = [dict(np.load(f)) for f in files]
    D.sort(key=lambda d: int(d["period"]))
    os.makedirs(f"{GAL}/atlas", exist_ok=True)
    # individual plates (full res) with coordinates
    for d in D:
        img = plate_img(d, 2400, 1500, "dark")
        pil = to_img(img)
        dr = ImageDraw.Draw(pil)
        lo, hi, ylo, yhi = d["rect"]
        f1, f2 = font(44, "mono-bold"), font(26, "mono")
        dr.text((40, 30), f"period {int(d['period'])}", fill=(255, 240, 220), font=f1)
        dr.text((40, 90), f"η ∈ [{fmt(lo, hi - lo)}, {fmt(hi, hi - lo)}]   P ∈ [{fmt(ylo, yhi - ylo)}, {fmt(yhi, yhi - ylo)}]",
                fill=(220, 200, 180), font=f2)
        dr.text((40, 125), f"zoom ×{(HERO[1] - HERO[0]) / (hi - lo):,.0f} in η, ×{(HERO[3] - HERO[2]) / (yhi - ylo):,.0f} in P"
                f"   ·   full 4-coordinate GD, float64", fill=(170, 150, 140), font=f2)
        pil.save(f"{GAL}/atlas/p{int(d['period']):02d}.png", optimize=True)
    # grids
    for style in ["dark", "paper"]:
        cols = 4
        pw, ph = 1200, 750
        sh = ph // 7
        cell_h = ph + 6 + sh + 110
        rows = int(np.ceil(len(D) / cols))
        margin = 120
        head = 260
        Wt = cols * pw + (cols - 1) * 50 + 2 * margin
        Ht = head + rows * cell_h + (rows - 1) * 40 + margin
        bg = (8, 8, 10) if style == "dark" else (243, 238, 226)
        fg = (240, 225, 205) if style == "dark" else (24, 26, 46)
        fg2 = (160, 145, 135) if style == "dark" else (110, 100, 95)
        canvas = paper(Ht, Wt, base=bg, grain=0 if style == "dark" else 3, seed=11) if style == "paper" else np.full((Ht, Wt, 3), bg, float)
        pil = to_img(canvas)
        dr = ImageDraw.Draw(pil)
        dr.text((margin, 70), "WINDOW ATLAS  ·  every periodic window of gradient descent on ½(x₁x₂x₃x₄ − 1)² is a small copy of the whole cascade",
                fill=fg, font=font(46, "serif"))
        dr.text((margin, 140), "each plate: widest window of that base period found in η ∈ [η∞, 0.99]; branch nearest the critical point; strip below: Lyapunov exponent (λ>0 up, λ<0 down)",
                fill=fg2, font=font(28, "serif"))
        for i, d in enumerate(D):
            r, c = divmod(i, cols)
            x = margin + c * (pw + 50)
            y = head + r * (cell_h + 40)
            img = plate_img(d, pw, ph, style)
            pil.paste(to_img(img), (x, y))
            lo, hi, ylo, yhi = d["rect"]
            dr.text((x, y + ph + 6 + sh + 12), f"p = {int(d['period'])}", fill=fg, font=font(34, "mono-bold"))
            dr.text((x + 160, y + ph + 6 + sh + 10), f"η {fmt(lo, hi - lo)} … {fmt(hi, hi - lo)}", fill=fg2, font=font(24, "mono"))
            dr.text((x + 160, y + ph + 6 + sh + 42), f"P {fmt(ylo, yhi - ylo)} … {fmt(yhi, yhi - ylo)}   ×{(HERO[1] - HERO[0]) / (hi - lo):,.0f}",
                    fill=fg2, font=font(24, "mono"))
        out = f"{GAL}/atlas_{style}.png"
        pil.save(out, optimize=True)
        print("wrote", out, pil.size)


if __name__ == "__main__":
    main()
