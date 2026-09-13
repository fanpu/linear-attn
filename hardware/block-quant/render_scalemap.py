"""Scale maps (per-block scales as a mosaic), FP4 code images, and NVFP4-vs-MXFP4 block-error difference.
   python render_scalemap.py [tags]
"""
import sys

import numpy as np

import styles as S

LEVELS = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])


def level_image(nib):
    return LEVELS[nib & 7] * np.where(nib >> 3, -1, 1)


def true_geom(m, bs, width):
    return np.repeat(m, bs, axis=1)[:, :width]


def scalemap_night(tag, d, maxrows=2048):
    W = d["W"]
    H, Wd = min(W.shape[0], maxrows), W.shape[1]
    nv = np.log2(d["NVFP4_scale"][:H])
    mx = np.log2(d["MXFP4_scale"][:H])
    lo, hi = np.percentile(nv, 0.2) - 0.25, max(np.percentile(nv, 99.95), mx.max()) + 0.1
    cm = S.cmap("magma")
    side, top, gap, bot = 70, 200, 60, 210
    img = S.canvas(top + H + bot, side * 2 + 2 * Wd + gap, S.NIGHT)
    S.paste(img, S.to_rgb(true_geom(nv, 16, Wd), cm, lo, hi), top, side)
    S.paste(img, S.to_rgb(true_geom(mx, 32, Wd), cm, lo, hi), top, side + Wd + gap)
    S.paste(img, S.colorbar(18, 600, cm), top + H + 70, side)
    fg, mu = (0.93, 0.91, 0.86), (0.55, 0.55, 0.6)
    n_mx = len(np.unique(d["MXFP4_scale_code"][:H]))
    n_nv = len(np.unique(d["NVFP4_scale_code"][:H]))
    img = S.text(img, [
        (side, 30, "Scale Map", 56, "Bold", fg),
        (side, 105, f"{d['name']}  rows 0:{H}   every block's scale drawn over the weights it covers (block = 16 or 32 px wide, 1 px tall)", 24, "Sans", mu),
        (side, top - 44, f"NVFP4  E4M3 x FP32   {n_nv} distinct codes used", 26, "Mono", fg),
        (side + Wd + gap, top - 44, f"MXFP4  E8M0 = 2^k   {n_mx} distinct exponents used", 26, "Mono", fg),
        (side, top + H + 24, f"log2(scale), shared axis {lo:.2f} .. {hi:.2f}; the power-of-two scale posterizes the same structure into {n_mx} octaves", 22, "Sans", mu),
        (side, top + H + 100, f"{lo:.1f}", 20, "Mono", mu), (side + 600, top + H + 100, f"{hi:.1f}", 20, "Mono", mu, "ra"),
        (side, top + H + 150, S.STACK, 20, "Sans", mu),
    ])
    return S.save(img, f"scalemap_night_{tag}.png")


def scalemap_mosaic(tag, d, rows=(0, 72), blocks=(0, 64), k=26, grout=3):
    """Tesserae: one square tile per NVFP4 block (geometry declared: blocks are really 16x1), tile colour =
    log2 scale on a declared stone palette; grout gaps are ornament."""
    nv = np.log2(d["NVFP4_scale"])
    lo, hi = np.percentile(nv, 0.5), np.percentile(nv, 99.9)
    sub = nv[rows[0]:rows[1], blocks[0]:blocks[1]]
    from matplotlib.colors import LinearSegmentedColormap
    stone = LinearSegmentedColormap.from_list("stone", ["#1f1d2b", "#3b3a5a", "#5f6f7a", "#9aa38b", "#d9c48f", "#f1e6c8"])
    tiles = S.to_rgb(sub, stone, lo, hi)
    h, w = sub.shape
    rng = np.random.default_rng(1)
    side, top = 70, 190
    img = S.canvas(top + h * k + 170, side * 2 + w * k, np.array([0.86, 0.84, 0.80]))
    for i in range(h):
        for j in range(w):
            jitter = rng.integers(-1, 2, 2)     # declared hand-laid jitter (<= 1 px)
            y = top + i * k + grout // 2 + jitter[0]
            x = side + j * k + grout // 2 + jitter[1]
            img[y:y + k - grout, x:x + k - grout] = tiles[i, j] * (0.94 + 0.06 * rng.random())
    fg, mu = (0.15, 0.13, 0.12), (0.40, 0.38, 0.35)
    img = S.text(img, [
        (side, 30, "Tesserae", 56, "Serif", fg),
        (side, 105, f"{d['name']}  rows {rows[0]}:{rows[1]}, NVFP4 blocks {blocks[0]}:{blocks[1]} (input dims {16 * blocks[0]}:{16 * blocks[1]}); one tile = one 16-weight block", 22, "Serif", mu),
        (side, 140, "tile colour = log2 of the block scale (data). Square tiles, grout, <=1 px jitter and brightness wobble (+-3%) are ornament.", 22, "Serif", mu),
        (side, top + h * k + 40, S.STACK, 20, "Serif", mu),
    ])
    return S.save(img, f"scalemap_tesserae_{tag}.png")


def scalemap_survey(tag, d, maxrows=1024):
    """Single ink on paper: contour lines at every octave (MXFP4 exponent boundaries, left) and every
    half-octave of the NVFP4 scale (right). Ink where a block's level differs from its neighbour."""
    H = min(d["W"].shape[0], maxrows)
    Wd = d["W"].shape[1]
    nv = true_geom(np.floor(2 * np.log2(d["NVFP4_scale"][:H])), 16, Wd)
    mx = true_geom(np.log2(d["MXFP4_scale"][:H]), 32, Wd)

    def edges(L):
        e = np.zeros(L.shape, bool)
        e[:, 1:] |= L[:, 1:] != L[:, :-1]
        e[1:, :] |= L[1:, :] != L[:-1, :]
        return e
    side, top, gap = 70, 190, 60
    img = S.canvas(top + H + 160, side * 2 + 2 * Wd + gap, S.PAPER)
    for j, (L, lab) in enumerate([(mx, "MXFP4: boundaries between E8M0 exponents (octaves)"),
                                   (nv, "NVFP4: boundaries between half-octave bands of the E4M3 x FP32 scale")]):
        e = edges(L).astype(float)
        hi = (L > np.percentile(L, 97)).astype(float) * 0.35      # filled wash for the top 3% (data)
        S.paste(img, S.ink(np.maximum(e, hi), S.INK_DARK), top, side + j * (Wd + gap))
    fg, mu = (0.1, 0.1, 0.12), (0.40, 0.38, 0.35)
    img = S.text(img, [
        (side, 30, "SURVEY OF SCALES", 56, "Bold", fg),
        (side, 105, f"{d['name']}  rows 0:{H}; one pixel per weight; lines mark where the block scale steps; grey wash = top 3% of scales", 22, "Sans", mu),
        (side, top - 36, "MXFP4  octave contours", 24, "Mono", fg), (side + Wd + gap, top - 36, "NVFP4  half-octave contours", 24, "Mono", fg),
        (side, top + H + 30, S.STACK, 20, "Sans", mu),
    ])
    return S.save(img, f"scalemap_survey_{tag}.png", palette=32)


def scalemap_riso(tag, d, maxrows=1024):
    H = min(d["W"].shape[0], maxrows)
    Wd = d["W"].shape[1]
    nv = true_geom(np.log2(d["NVFP4_scale"][:H]), 16, Wd)
    mx = true_geom(np.log2(d["MXFP4_scale"][:H]), 32, Wd)
    lo, hi = np.percentile(nv, 1), np.percentile(nv, 99.9)
    dn = np.clip((nv - lo) / (hi - lo), 0, 1) ** 1.6
    dm = np.clip((mx - lo) / (hi - lo), 0, 1) ** 1.6
    side, top = 70, 200
    img = S.canvas(top + H + 160, side * 2 + Wd, S.PAPER)
    S.paste(img, S.ink_multi([(dn, S.RISO_BLUE), (dm, S.RISO_PINK)], shift=[(0, 0), (2, 3)]), top, side)
    img *= S.grain(img.shape[:2], 0.035, seed=2)
    img = S.text(img, [
        (side, 30, "SCALES / overprint", 52, "Bold", S.RISO_BLUE),
        (side, 100, f"{d['name']}  rows 0:{H}.  blue = NVFP4 block scale, pink = MXFP4 shared exponent; ink density = log2 scale, shared", 21, "Sans", (0.4, 0.38, 0.35)),
        (side, 132, "(declared: 2-3 px misregistration of the pink drum and paper grain)", 21, "Sans", (0.4, 0.38, 0.35)),
        (side, top + H + 40, S.STACK, 18, "Sans", (0.4, 0.38, 0.35)),
    ])
    return S.save(img, f"scalemap_riso_{tag}.png")


def code_image(tag, d, maxrows=1024, style="night"):
    """The FP4 codes themselves: 15 signed E2M1 levels, posterized diverging palette (declared 15-step vik)."""
    H = min(d["W"].shape[0], maxrows)
    Wd = d["W"].shape[1]
    from matplotlib.colors import ListedColormap
    base = S.cmap("cmc.vik")(np.linspace(0.03, 0.97, 15))[:, :3]
    signed = np.array([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6])
    def paint(nib):
        lv = level_image(nib[:H])
        idx = np.searchsorted(signed, lv)
        return base[idx]
    side, top, gap = 70, 200, 60
    bg = S.NIGHT if style == "night" else S.PAPER
    img = S.canvas(top + H + 230, side * 2 + 2 * Wd + gap, bg)
    S.paste(img, paint(d["NVFP4_nea_nib"]), top, side)
    S.paste(img, paint(d["MXFP4_nea_nib"]), top, side + Wd + gap)
    sw = 44
    for i in range(15):
        img[top + H + 70: top + H + 70 + 30, side + i * sw: side + i * sw + sw - 4] = base[i]
    fg, mu = ((0.93, 0.91, 0.86), (0.55, 0.55, 0.6)) if style == "night" else ((0.1, 0.1, 0.12), (0.4, 0.38, 0.35))
    items = [(side, 30, "What the hardware stores", 56, "Bold", fg),
             (side, 105, f"{d['name']}  rows 0:{H}; each pixel = one 4-bit E2M1 code (sign x level), before the block scale is applied", 24, "Sans", mu),
             (side, top - 44, "NVFP4 codes (16-blocks)", 26, "Mono", fg), (side + Wd + gap, top - 44, "MXFP4 codes (32-blocks)", 26, "Mono", fg),
             (side, top + H + 24, "15 signed levels  -6 -4 -3 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 3 4 6  (posterized diverging palette, declared)", 22, "Sans", mu),
             (side, top + H + 150, S.STACK, 20, "Sans", mu)]
    for i, v in enumerate(signed):
        items.append((side + i * sw + sw // 2 - 2, top + H + 106, f"{v:g}", 16, "Mono", mu, "ma"))
    h_nv = np.bincount(d["NVFP4_nea_nib"].ravel() & 7, minlength=8) / d["NVFP4_nea_nib"].size
    h_mx = np.bincount(d["MXFP4_nea_nib"].ravel() & 7, minlength=8) / d["MXFP4_nea_nib"].size
    items.append((side + 15 * sw + 40, top + H + 70, "magnitude-level use  0 .5 1 1.5 2 3 4 6:", 20, "Mono", mu))
    items.append((side + 15 * sw + 40, top + H + 98, "NVFP4 " + " ".join(f"{x * 100:4.1f}" for x in h_nv) + " %", 20, "Mono", fg))
    items.append((side + 15 * sw + 40, top + H + 124, "MXFP4 " + " ".join(f"{x * 100:4.1f}" for x in h_mx) + " %", 20, "Mono", fg))
    img = S.text(img, items)
    return S.save(img, f"codes_{style}_{tag}.png")


def nvmx_difference(tag, d, maxrows=None):
    """Per 32-weight window: log2(RMSE_MXFP4 / RMSE_NVFP4) as a map, and as a density against the fractional
    octave of the block max.  Dashed line: the fine-rounding approximation log2(1.5) - frac (no fit): the
    MXFP4 step is 2^floor(log2 m)/4 while an ideal amax/6 step is m/6, ratio 1.5 * 2^-frac; above
    frac = log2 1.5 = 0.585 MXFP4 clips the block max instead."""
    H = d["W"].shape[0] if maxrows is None else min(d["W"].shape[0], maxrows)
    Wd = d["W"].shape[1] // 32 * 32
    en = d["NVFP4_nea_err"][:H, :Wd].astype(np.float64).reshape(H, -1, 32)
    em = d["MXFP4_nea_err"][:H, :Wd].astype(np.float64).reshape(H, -1, 32)
    r = 0.5 * np.log2((em ** 2).mean(-1) / (en ** 2).mean(-1))
    amax = d["MXFP4_amax"][:H, : Wd // 32].astype(np.float64)
    frac = np.log2(amax) - np.floor(np.log2(amax))
    Hm = min(H, 1024)
    cm = S.cmap("cmc.vik")
    side, top, gap = 70, 200, 90
    P = 1024
    img = S.canvas(top + max(Hm, P) + 260, side * 2 + Wd + gap + P + 60, S.NIGHT)
    S.paste(img, S.to_rgb(true_geom(r[:Hm], 32, Wd), cm, -2, 2), top, side)
    S.paste(img, S.colorbar(18, 500, cm), top + Hm + 70, side)
    # density panel
    x0 = side + Wd + gap + 60
    yr = (-1.5, 2.5)
    hist, _, _ = np.histogram2d(np.clip(r.ravel(), *yr), frac.ravel(), bins=(P // 4, P // 4), range=(yr, (0, 1)))
    dens = S.up((np.log1p(hist[::-1]) / np.log1p(hist.max())) ** 0.8, 4)
    S.paste(img, S.to_rgb(dens, S.cmap("magma")), top, x0)
    fx = np.arange(P) / P
    ln = np.log2(1.5) - fx
    for j in range(0, P, 6):
        yy = int((yr[1] - ln[j]) / (yr[1] - yr[0]) * P)
        if 0 <= yy < P:
            img[top + yy:top + yy + 2, x0 + j:x0 + j + 3] = (0.6, 0.85, 1.0)
    y0 = int((yr[1] - 0) / (yr[1] - yr[0]) * P)
    img[top + y0, x0:x0 + P] = (0.45, 0.45, 0.5)
    fg, mu = (0.93, 0.91, 0.86), (0.55, 0.55, 0.6)
    items = [
        (side, 30, "Where the power of two hurts", 56, "Bold", fg),
        (side, 105, f"{d['name']}; one value per 32-weight block = log2(RMSE MXFP4 / RMSE NVFP4)", 24, "Sans", mu),
        (side, top - 40, f"map, rows 0:{Hm}   (red = MXFP4 worse, blue = NVFP4 worse; -2..+2)", 22, "Mono", fg),
        (side, top + Hm + 24, f"mean {r.mean():+.2f} octaves;  MXFP4 worse in {100 * (r > 0).mean():.0f}% of {r.size} blocks", 22, "Sans", mu),
        (x0, top - 40, "density vs frac(log2 max|w| of the block)  (log count)", 22, "Mono", fg),
        (x0, top + P + 14, "0", 20, "Mono", mu), (x0 + P, top + P + 14, "1", 20, "Mono", mu, "ra"),
        (x0 + P // 2, top + P + 14, "fractional octave of the block max", 20, "Mono", mu, "ma"),
        (x0 - 12, top, f"{yr[1]:+.1f}", 20, "Mono", mu, "ra"), (x0 - 12, top + P - 20, f"{yr[0]:+.1f}", 20, "Mono", mu, "ra"),
        (x0 - 12, top + y0 - 10, "0", 20, "Mono", mu, "ra"),
        (x0, top + P + 50, "dashed: log2(1.5) - frac, the step-size ratio (no fit). Right of 0.585 the MXFP4 rule clips the block max.", 20, "Sans", mu),
        (side, top + max(Hm, P) + 200, S.STACK, 20, "Sans", mu),
    ]
    img = S.text(img, items)
    return S.save(img, f"nvmx_diff_{tag}.png")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["L26_q_proj", "L16_k_proj", "L27_gate_proj", "embed_rare"]
    for t in tags:
        d = S.load(t)
        scalemap_night(t, d)
        scalemap_mosaic(t, d)
        scalemap_survey(t, d)
        scalemap_riso(t, d)
        code_image(t, d, style="night")
        code_image(t, d, style="paper")
        nvmx_difference(t, d)
