"""Nested tower plate sequence: full cascade -> period 3 -> 9 -> 27 -> 81 -> 243.

Measured: count rasters of full 4-coordinate GD (cache/atlas/tower_pXXX.npz; hero sweep for the full view),
Lyapunov exponent of the oscillating mode.  Each plate marks the next plate's rectangle in red.
Aesthetic: tone curve, palette, layout.  Outputs gallery/tower_plates.png (dark) and tower_plates_paper.png.
"""
import glob
import numpy as np
from PIL import Image, ImageDraw
from render_lib import *
from render_atlas import plate_img, fmt, tone_plate

FULL = (0.45, 0.99, -0.03, 1.80)


def full_plate(W, H):
    d = np.load(f"{CACHE}/bif_prod4_hi.npz")
    e = d["etas"]
    m = (e >= FULL[0]) & (e <= FULL[1])
    C = density(e[m], d["P"][m], FULL[0], FULL[1], FULL[2], FULL[3], 2400, 1500)
    return dict(C=C, etas=e[m], lyap_bal=d["lyap_bal"][m], rect=FULL, period=1)


def main():
    files = sorted(glob.glob(f"{CACHE}/atlas/tower_p*.npz"), key=lambda f: int(f.split("_p")[-1][:-4]))
    D = [full_plate(2400, 1500)] + [dict(np.load(f)) for f in files]
    pw, ph = 1500, 940
    for style in ["dark", "paper"]:
        cols = 3
        sh = ph // 4
        cell_h = ph + 6 + sh + 120
        rows = 2
        margin, head, gap = 120, 300, 70
        Wt = cols * pw + (cols - 1) * gap + 2 * margin
        Ht = head + rows * cell_h + (rows - 1) * 40 + margin
        bg = (8, 8, 10) if style == "dark" else (243, 238, 226)
        fg = (240, 225, 205) if style == "dark" else (24, 26, 46)
        fg2 = (165, 150, 140) if style == "dark" else (105, 95, 90)
        red = (255, 70, 60) if style == "dark" else (179, 38, 30)
        canvas = np.full((Ht, Wt, 3), bg, float) if style == "dark" else paper(Ht, Wt, base=bg, grain=3, seed=12)
        pil = to_img(canvas)
        dr = ImageDraw.Draw(pil)
        dr.text((margin, 70), "THE TOWER  ·  a period-3 window inside a period-3 window inside a period-3 window …",
                fill=fg, font=font(58, "serif"))
        dr.text((margin, 150), "each plate is a fresh full-GD computation of the red rectangle in the previous one; "
                "ratio of successive superstable gaps → 59.50, 54.99, 55.265 (period-tripling constant ≈ 55.25)",
                fill=fg2, font=font(30, "serif"))
        for i, d in enumerate(D):
            r, c = divmod(i, cols)
            x = margin + c * (pw + gap)
            y = head + r * (cell_h + 40)
            img = plate_img(d, pw, ph, style)
            pim = to_img(img)
            pd = ImageDraw.Draw(pim)
            lo, hi, ylo, yhi = [float(v) for v in d["rect"]]
            if i + 1 < len(D):
                nlo, nhi, nylo, nyhi = [float(v) for v in D[i + 1]["rect"]]
                X0 = (nlo - lo) / (hi - lo) * pw
                X1 = (nhi - lo) / (hi - lo) * pw
                Y0 = (yhi - nyhi) / (yhi - ylo) * ph
                Y1 = (yhi - nylo) / (yhi - ylo) * ph
                if X1 - X0 < 12:
                    cx = 0.5 * (X0 + X1)
                    X0, X1 = cx - 6, cx + 6
                if Y1 - Y0 < 12:
                    cy = 0.5 * (Y0 + Y1)
                    Y0, Y1 = cy - 6, cy + 6
                pd.rectangle([X0, Y0, X1, Y1], outline=red, width=4)
            pil.paste(pim, (x, y))
            zx = (FULL[1] - FULL[0]) / (hi - lo)
            name = "full cascade" if i == 0 else f"period-{int(d['period'])} window"
            dr.text((x, y + ph + 6 + sh + 14), name, fill=fg, font=font(40, "serif"))
            dr.text((x, y + ph + 6 + sh + 66), f"η {fmt(lo, hi - lo)} … {fmt(hi, hi - lo)}    zoom ×{zx:,.0f}",
                    fill=fg2, font=font(26, "mono"))
        out = f"{GAL}/tower_plates{'' if style == 'dark' else '_paper'}.png"
        pil.save(out, optimize=True)
        print("wrote", out, pil.size)


if __name__ == "__main__":
    main()
