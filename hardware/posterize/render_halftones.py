"""Render the halftone pieces (cache/halftones.npz, cache/halftone_spectra.npz).

  halftone_basin_*   Floyd-Steinberg | Bayer 256 | blue noise on the gd-bifurcation basin image, full + 4x crops
  halftone_zone_*    the same four ways (+ Bayer 8) on a zone plate: halftone moire
  halftone_spectra_* 2-D power spectra (log) of each method on flat gray 1/3 and 1/4, plus radial profiles
  halftone_overprint_riso   blue-noise halftone (blue ink) over Bayer halftone (pink ink) of the same basin, misregistered
Styles: riso | paper | dark.  1-bit pixels are exact outputs of each algorithm; inks and crops are declared.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import common as c

H = np.load(f"{c.CACHE}/halftones.npz")
SP = np.load(f"{c.CACHE}/halftone_spectra.npz")
NAMES = {"fs": "Floyd-Steinberg (error diffusion)", "bayer8": "Bayer 8x8 (ordered)", "bayer256": "Bayer 256x256 (ordered, recursive)",
         "blue": "blue noise (void-and-cluster 128)", "white": "white-noise threshold (null)"}


def bits(key):
    shp = H[f"{key}_shape"]
    return np.unpackbits(H[key], count=int(np.prod(shp))).reshape(shp)


def ink(b, st, which=0):
    if st == "dark":
        return np.where(b[..., None] > 0, np.array([0.95, 0.93, 0.88]), np.array([0.03, 0.03, 0.035]))
    if st == "paper":
        return c.ink_on_paper(1 - b.astype(float))   # ink where the halftone is OFF (dark regions of the source)
    inks = [c.RISO_BLUE, c.RISO_PINK, c.RISO_TEAL]
    return c.multiply_layers([(1 - b.astype(float), inks[which % 3])])


def up(a, s):
    return np.repeat(np.repeat(a, s, 0), s, 1)


def triptych(target, methods, st, title, crop_xy):
    bg = (0.03, 0.03, 0.035) if st == "dark" else tuple(c.PAPER)
    fg = (0.88, 0.86, 0.82) if st == "dark" else (tuple(c.INK) if st == "paper" else tuple(c.RISO_BLUE * 0.75))
    S, gap, L, T, cs = 1024, 50, 70, 200, 256
    n = len(methods)
    W = L * 2 + n * S + (n - 1) * gap
    Ht = T + S + 90 + S + 170
    cv = c.canvas(W, Ht, bg)
    c.text(cv, (L, 50), title, 52, fg, c.FONT_SERIF)
    cx, cy = crop_xy
    for i, m in enumerate(methods):
        b = bits(f"{target}_{m}")
        x0 = L + i * (S + gap)
        full = c.box_down(ink(b, st, i), 2048 // S)          # exact 2x2 average
        c.paste(cv, full, x0, T)
        c.line(cv, [(x0 + cx * S // 2048, T + cy * S // 2048), (x0 + (cx + cs) * S // 2048, T + cy * S // 2048),
                    (x0 + (cx + cs) * S // 2048, T + (cy + cs) * S // 2048), (x0 + cx * S // 2048, T + (cy + cs) * S // 2048),
                    (x0 + cx * S // 2048, T + cy * S // 2048)], (0.9, 0.2, 0.1), 3)
        c.text(cv, (x0, T + S + 20), NAMES[m], 36, fg, c.FONT_SERIF)
        crop = b[cy:cy + cs, cx:cx + cs]
        c.paste(cv, ink(up(crop, 4), st, i), x0, T + S + 90)
    if target == "zone":
        c.text(cv, (L, Ht - 105), "Halftone moire, predicted: a threshold matrix with period P mixes with cos(k r^2) and creates ghost zone plates "
               "centred at offsets pi m/(k P) = 2 R_N m/P = 724 m px (P = 4) and 362 m px (P = 8).", 26, fg, c.FONT_SANS)
    c.text(cv, (L, Ht - 60), "top: full 2048^2 1-bit halftone, 2x2-averaged for display (averaging is itself a filter and adds faint moire of its "
           "own; the print has none).  bottom: the red square at 1 dot = 4x4 px, no smoothing (exact dots).  " + c.STACK, 26, fg, c.FONT_SANS)
    return cv


def render_basin(st):
    cv = triptych("basin", ["fs", "bayer256", "blue"], st,
                  "Three ways to print a fractal in one bit  (source: gd-bifurcation basin, luminance)", (1180, 860))
    c.save_png(cv, f"{c.GAL}/halftone_basin_{st}.png")
    print("basin", st)


def render_zone(st):
    cv = triptych("zone", ["fs", "bayer8", "bayer256", "blue"], st,
                  "A zone plate in one bit: each halftone adds its own moire", (1600, 1600))
    c.save_png(cv, f"{c.GAL}/halftone_zone_{st}.png")
    print("zone", st)


def render_spectra(st):
    dark = st == "dark"
    bg = (0.03, 0.03, 0.035) if dark else tuple(c.PAPER)
    fg = (0.88, 0.86, 0.82) if dark else tuple(c.INK)
    methods = ["fs", "bayer8", "bayer256", "blue", "white"]
    fig = plt.figure(figsize=(30, 17), dpi=100, facecolor=bg)
    for r, g in enumerate(("0.3333", "0.2500")):
        for j, m in enumerate(methods):
            ax = fig.add_axes([0.03 + j * 0.152, 0.52 - r * 0.42, 0.142, 0.36])
            S2 = SP[f"{g}_{m}_2d"].astype(np.float32)
            S2 = np.log10((10.0 ** S2).reshape(256, 4, 256, 4).mean((1, 3)))   # 4x4 mean-pool in power (declared)
            v = np.clip((S2 - 2.0) / 6.0, 0, 1)                  # log10 power [2, 8]; white-noise level ~5.3
            if st == "riso":
                rgb = c.multiply_layers([(c.floyd_steinberg(v), c.RISO_BLUE)])
            elif st == "paper":
                rgb = c.ink_on_paper(v, gamma=0.8)
            else:
                rgb = c.cmap_rgb(v, "magma")
            ax.imshow(rgb, extent=[-0.5, 0.5, -0.5, 0.5], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(fg)
            if r == 0:
                ax.set_title(NAMES[m], color=fg, fontsize=15, family="DejaVu Serif")
            if j == 0:
                ax.set_ylabel(f"gray {'1/3' if g == '0.3333' else '1/4'}", color=fg, fontsize=20, family="DejaVu Serif")
    ax = fig.add_axes([0.815, 0.12, 0.17, 0.76], facecolor=bg)
    cols = {"fs": "#e7298a", "bayer8": "#66a61e", "bayer256": "#1b9e77", "blue": "#3b8bd9", "white": "#999999"} if dark else \
        {"fs": "#c51b7d", "bayer8": "#4d9221", "bayer256": "#1b7837", "blue": "#2166ac", "white": "#777777"}
    f = np.arange(181) / 128 * 0.5
    for m in methods:
        ax.semilogy(f, np.maximum(SP[f"0.3333_{m}"], 1.0), color=cols[m], lw=2.2, label=m)
    ax.axvline(0.1, color=fg, lw=1, ls=":")
    ax.set_xlim(0, 0.7); ax.set_ylim(1, 1e9); ax.set_xlabel("radial frequency [cycles/px]", color=fg, fontsize=15)
    ax.set_ylabel("radially averaged power (gray 1/3)", color=fg, fontsize=15)
    ax.tick_params(colors=fg, labelsize=12)
    for s in ax.spines.values():
        s.set_color(fg)
    ax.legend(frameon=False, labelcolor=fg, fontsize=14)
    fig.text(0.03, 0.95, "Halftone spectra: log10 |FFT|^2 of a flat gray, 1024^2, 4x4 power-averaged, zero frequency at centre (colour [2, 8], declared)", fontsize=34, color=fg, family="DejaVu Serif")
    fig.text(0.03, 0.02, "energy below 0.1 cyc/px, gray 1/3: FS 0.0001, Bayer8 0, Bayer256 0.0008, blue noise 0.0006, white noise 0.031 "
             "(= its area share 0.031).\nBayer: isolated lines on a dyadic lattice (at gray 1/4 they sit on the Nyquist edge).  Blue noise: a dark "
             "low-frequency disc.  Floyd-Steinberg: at these rational grays it locks into periodic worms, so its spectrum is lines + ridges.", fontsize=15, color=fg)
    out = f"{c.GAL}/halftone_spectra_{st}.png"
    fig.savefig(out, facecolor=bg); plt.close(fig)
    c.save_png(Image.open(out).convert("RGB"), out)
    print("spectra", st)


def overprint():
    b1 = bits("basin_blue").astype(float)
    b2 = bits("basin_bayer256").astype(float)
    rgb = c.multiply_layers([(1 - b1, c.RISO_BLUE), (c.shift(1 - b2, 3, 2), c.RISO_PINK)])
    cv = c.canvas(2048, 2048 + 110, c.PAPER)
    c.paste(cv, rgb, 0, 0)
    c.text(cv, (30, 2048 + 35), "overprint: blue ink = blue-noise halftone, pink ink = Bayer-256 halftone of the same basin "
           "luminance, pink shifted (3, 2) px.  Every dot exact; inks and shift declared.", 28, c.RISO_BLUE * 0.7, c.FONT_SANS)
    c.save_png(cv, f"{c.GAL}/halftone_overprint_riso.png")
    print("overprint")


if __name__ == "__main__":
    which = sys.argv[1:] or ["basin", "zone", "spectra", "overprint"]
    styles = [s for s in which if s in ("riso", "paper", "dark")] or ["riso", "paper", "dark"]
    for w in which:
        if w in ("basin", "zone", "spectra"):
            for st in styles:
                globals()[f"render_{w}"](st)
        elif w == "overprint":
            overprint()
