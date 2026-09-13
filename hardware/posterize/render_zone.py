"""Zone-plate error fields: sampling aliasing vs amplitude quantization, separated and combined.

For each regime (ws: well sampled, al: aliased) and format:
  row A  signal as sampled f_point, and its sampling-alias part f_point - f_bl     (SAMPLING only; no quantization)
  row B  e_bl: band-limited continuous quantization error                         (QUANTIZATION only)
  row C  e_point = Q(f_point) - f_point: what the quantized tensor actually holds  (COMBINED)
  row D  e_point - e_bl: quantization harmonics folded by the pixel grid          (their interaction)
Styles:
  spectral  Sohl-Dickstein cdf_img (signed, readout='loss', buffer 0.25) on the Spectral map (primary, declared)
  seam      the brief's split-at-zero variant: dark ends meet at zero (declared)
  diverge   linear diverging map (cmcrameri vik / berlin), symmetric limits = 99.5th pct |e|  (the metric version)
  riso      two inks: pink = e > 0 (rounded up), blue = e < 0, 1-bit Floyd-Steinberg of |e|/lim
Run: OMP_NUM_THREADS=4 python render_zone.py [grid] [hero] [styles...]
"""
import json
import sys
import numpy as np
from PIL import Image
import cmcrameri  # noqa: F401  (registers cmc.* colormaps)
import common as c

ST = json.load(open(f"{c.CACHE}/zone_stats.json"))
FMT_LAB = {"int2": "int2 (3 levels)", "int3": "int3 (7)", "int4": "int4 (15)", "fp4": "FP4 E2M1 (15)", "fp8": "FP8 E4M3 (253)"}


def load(name):
    return np.load(f"{c.CACHE}/zone_{name}.npy").astype(np.float64)


def colour(e, st, ref=None, crop=False):
    """crop=True: colour only the display crop, normalising against the full field (subsampled 2x2 for the CDF)."""
    if crop:
        ref = (e if ref is None else ref)[::2, ::2]
        e = e[2048:3072, 2048:3072]
    if st == "spectral":
        return c.spectral_sd(e, ref)
    if st == "seam":
        return c.spectral_seam(e, ref)
    lim = np.percentile(np.abs(ref if ref is not None else e), 99.5) + 1e-12
    if st == "diverge":
        return c.diverging(e, lim, "cmc.vik")
    if st == "diverge_dark":
        return c.diverging(e, lim, "cmc.berlin")
    pos = c.floyd_steinberg(np.clip(e / lim, 0, 1))
    neg = c.floyd_steinberg(np.clip(-e / lim, 0, 1))
    return c.multiply_layers([(pos.astype(float), c.RISO_PINK), (c.shift(neg.astype(float), 2, 1), c.RISO_BLUE)])


def down(img, f=None):
    return img[2048:3072, 2048:3072]


def grid(reg, st):
    """5 formats x 4 rows. Colour normalisation uses the whole 4096^2 field; each panel shows the quadrant crop
    x, y in [2048, 3072) at 1 pixel = 1 sample (image centre at the panel's top-left corner). Downsampling a zone plate for
    display would add aliasing of its own, so no resampling is done."""
    S, gap, L, T = 1024, 36, 330, 230
    fmts = c.FORMATS
    rows = ["f", "bl", "pt", "al"]
    dark = st in ("diverge_dark",)
    bg = (0.03, 0.03, 0.035) if dark else tuple(c.PAPER)
    fg = (0.88, 0.86, 0.82) if dark else tuple(c.INK)
    W = L + len(fmts) * (S + gap) + 40
    H = T + len(rows) * (S + gap + 10) + 160
    cv = c.canvas(W, H, bg)
    k = ST[f"{reg}_k"]; RN = ST[f"{reg}_R_N"]
    c.text(cv, (60, 40), f"POSTERIZE  -  zone plate cos(k r^2), {'well sampled' if reg == 'ws' else 'aliased'} (Nyquist radius {RN:.0f} px of a 4096 px image)",
           58, fg, c.FONT_SERIF)
    c.text(cv, (60, 130), "Which structure comes from where:  row 1 = sampling aliasing of the signal alone;  row 2 = quantization error alone "
           "(continuous, band-limited);  row 3 = the actual error of the quantized pixels;  row 4 = row 3 - row 2.", 30, fg, c.FONT_SANS)
    labels = {"f": "1  SAMPLING ONLY\nsignal alias\nf_point - f_bl", "bl": "2  QUANTIZATION ONLY\ncontinuous error\nQ(f) - f, band-limited",
              "pt": "3  COMBINED\nerror of the pixels\nQ(f_point) - f_point", "al": "4  INTERACTION\nquantization harmonics\nfolded by the grid (3 - 2)"}
    fa = load(f"{reg}_f_point") - load(f"{reg}_f_bl")
    for j, fmt in enumerate(fmts):
        x0 = L + j * (S + gap)
        c.text(cv, (x0, T - 50), FMT_LAB[fmt], 40, fg, c.FONT_SERIF)
        ept = load(f"{reg}_{fmt}_e_point")
        ebl = load(f"{reg}_{fmt}_e_bl")
        panels = {"f": fa if j == 0 else None, "bl": ebl, "pt": ept, "al": ept - ebl}
        for i, r in enumerate(rows):
            y0 = T + i * (S + gap + 10)
            if r == "f":
                if j == 0:
                    if np.abs(fa[2048:3072, 2048:3072]).max() < 1e-12:
                        c.text(cv, (x0 + 20, y0 + S // 2), "identically 0 in this window:\nthe signal is below Nyquist here", 34, fg, c.FONT_SERIF)
                        c.line(cv, [(x0, y0), (x0 + S, y0), (x0 + S, y0 + S), (x0, y0 + S), (x0, y0)], fg, 2)
                    else:
                        c.paste(cv, colour(fa, st, crop=True), x0, y0)
                    c.text(cv, (x0 + S + 20, y0 + 40), f"format-independent: the signal's own aliasing\nrelative RMS {ST[f'{reg}_signal_alias_rel_rms']:.3f}"
                           + ("\n(all of it from the taper zone near R_N:\nthe signal itself is essentially alias-free)" if reg == "ws" else
                              "\n(ghost zone plates, centred every 1024 px;\nodd ones carry a checkerboard sign flip)"), 30, fg, c.FONT_SANS)
                continue
            ref = None   # each panel normalised against its own full field (declared)
            img = colour(panels[r], st, ref, crop=True)
            c.paste(cv, img, x0, y0)
            if r == "al":
                c.text(cv, (x0 + 10, y0 + S - 40), f"aliased share of error RMS {ST[f'{reg}_{fmt}_alias_part_rel_rms']:.2f}", 26,
                       (0.05, 0.05, 0.05) if not dark else (0.95, 0.95, 0.95), c.FONT_MONO)
    for i, r in enumerate(rows):
        y0 = T + i * (S + gap + 10)
        for q, ln in enumerate(labels[r].split("\n")):
            c.text(cv, (40, y0 + 30 + q * 44), ln, 34 if q == 0 else 28, fg, c.FONT_SERIF if q == 0 else c.FONT_SANS)
    style_txt = {"spectral": "colour: Sohl-Dickstein cdf_img (sign-preserving rank normalisation, buffer 0.25) on matplotlib Spectral; red = Q(f) > f",
                 "seam": "colour: each sign rank-normalised separately, Spectral dark ends meeting at zero (purple: Q(f) > f, red: Q(f) < f)",
                 "diverge": "colour: cmcrameri vik, linear, symmetric limits at the 99.5th percentile of |error| per panel pair",
                 "diverge_dark": "colour: cmcrameri berlin, linear, symmetric limits at the 99.5th percentile of |error|",
                 "riso": "riso: pink ink = error > 0 (rounded up), blue ink = error < 0, 1-bit Floyd-Steinberg of |error|/(99.5th pct)"}[st]
    c.text(cv, (60, H - 110), style_txt + ".  Colour is a declared aesthetic mapping; the fields are exact float64.", 28, fg, c.FONT_SANS)
    c.text(cv, (60, H - 60), f"4096^2 field, k = {k:.3e} rad/px^2, panels: crop x, y in [2048, 3072) at 1 px = 1 sample (centre at top-left), colour normalised on the full field.  int-b: symmetric abs-max, clipped; "
           "FP: nearest, ties-to-even.  " + c.STACK, 24, fg, c.FONT_MONO)
    c.save_png(cv, f"{c.GAL}/zone_grid_{reg}_{st}.png")
    print("grid", reg, st)


def hero(reg, fmt, st, crop=None):
    e = load(f"{reg}_{fmt}_e_point")
    if crop:
        y0, x0, n = crop
        e = e[y0:y0 + n, x0:x0 + n]
    img = colour(e, st)
    tag = f"_crop{crop[2]}" if crop else ""
    strip = 90
    h, w = img.shape[:2]
    bg = tuple(c.PAPER) if st in ("riso", "diverge", "spectral", "seam") else (0.03, 0.03, 0.035)
    fg = tuple(c.INK) if bg == tuple(c.PAPER) else (0.85, 0.83, 0.8)
    cv = c.canvas(w, h + strip, bg)
    c.paste(cv, img, 0, 0)
    c.text(cv, (30, h + 25), f"Q(f) - f for f = cos(k r^2) on a {'4096' if not crop else str(crop[2])}^2 pixel grid, {FMT_LAB[fmt]}, "
           f"{'well sampled' if reg == 'ws' else 'aliased'} (R_N = {ST[f'{reg}_R_N']:.0f} px).  {st} colour (declared).  numpy float64, 2026-09-13",
           30 if not crop else 20, fg, c.FONT_MONO)
    c.save_png(cv, f"{c.GAL}/zone_hero_{reg}_{fmt}{tag}_{st}.png")
    print("hero", reg, fmt, st, tag)


if __name__ == "__main__":
    args = sys.argv[1:]
    styles = [a for a in args if a in ("spectral", "seam", "diverge", "diverge_dark", "riso")] or ["spectral", "seam", "diverge", "riso"]
    if "grid" in args or not args:
        for reg in ("ws", "al"):
            for st in styles:
                grid(reg, st)
    if "hero" in args or not args:
        for st in styles:
            hero("ws", "int3", st)
            hero("ws", "int3", st, crop=(1024, 1024, 2048))
            hero("al", "fp4", st, crop=(1024, 1024, 2048))
            hero("ws", "fp8", st, crop=(0, 0, 1536))
