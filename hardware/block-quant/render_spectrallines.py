"""Spectral lines: FP4 levels as spectral lines over the value histograms of the blocks that use them.

Exact grouping: every NVFP4 block with the same E4M3 scale code has *identical* representable values
(level * code * s2), so one row per scale code = one exact 'spectrum': the histogram of all weights in
those blocks (emission density) with its 15 FP4 levels drawn as lines. Rows ordered by scale: the lines fan
out as the scale grows (outlier blocks = widely spaced lines).  MXFP4: one row per E8M0 exponent.
Also a block-by-block plate (individual 16 weights as marks) for blocks sampled across scale quantiles.
   python render_spectrallines.py [tags]
"""
import sys

import numpy as np

import styles as S

LV = np.array([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6])


def groups(d, fmt):
    W = d["W"].astype(np.float64)
    bs = 16 if fmt == "NVFP4" else 32
    H, Wd = W.shape
    B = W[:, : Wd // bs * bs].reshape(H, -1, bs)
    code = d[f"{fmt}_scale_code"][:, : Wd // bs]
    scale = d[f"{fmt}_scale"][:, : Wd // bs]
    out = []
    for c in np.unique(code):
        m = code == c
        out.append((int(c), float(scale[m][0]), B[m].ravel(), int(m.sum())))
    return out


def plate(tag, fmt="NVFP4", style="plate", width=2400, row_h=34, min_blocks=1):
    d = S.load(tag)
    gs = [g for g in groups(d, fmt) if g[3] >= min_blocks]
    xmax = 6 * max(g[1] for g in gs) * 1.04
    nb = 1200
    side, top = 150, 230
    H = top + len(gs) * row_h + 230
    img_w = width + 2 * side
    if style == "plate":      # observatory glass negative: dark emission on pale sepia glass
        bg, dens_col, line_col, fg, mu = np.array([0.90, 0.86, 0.76]), np.array([0.16, 0.12, 0.09]), np.array([0.55, 0.12, 0.08]), np.array([0.2, 0.15, 0.1]), np.array([0.45, 0.38, 0.3])
    elif style == "night":
        bg, dens_col, line_col, fg, mu = S.NIGHT, np.array([0.95, 0.92, 0.85]), np.array([1.0, 0.45, 0.2]), np.array([0.93, 0.91, 0.86]), np.array([0.55, 0.55, 0.6])
    else:                     # plotter line: outline histograms + ticks in one ink
        bg, dens_col, line_col, fg, mu = S.PAPER, S.INK_DARK, S.INK_DARK, S.INK_DARK, np.array([0.4, 0.38, 0.35])
    img = S.canvas(H, img_w, bg)
    edges = np.linspace(-xmax, xmax, nb + 1)
    px = width / nb
    items = [(side, 40, {"plate": "Spectrograph plate", "night": "Spectral lines", "line": "SPECTRAL LINES"}[style], 60,
              "Serif" if style == "plate" else "Bold", fg),
             (side, 118, f"{d['name']}  {fmt}: one row per {'E4M3 block-scale code' if fmt == 'NVFP4' else 'E8M0 shared exponent'} ({len(gs)} rows, smallest scale at top).", 22, "Sans", mu),
             (side, 150, "Emission = histogram of every weight in the blocks with that scale; red lines = the 15 values those blocks can represent.", 22, "Sans", mu),
             (side, 182, "x axis: weight value, linear, shared by all rows.  Row darkness = log count, normalized per row (declared).", 22, "Sans", mu)]
    for i, (c, s, vals, n) in enumerate(gs):
        y = top + i * row_h
        h, _ = np.histogram(vals, edges)
        t = np.log1p(h) / np.log1p(max(h.max(), 1))
        if style == "line":
            base = y + row_h - 4
            prev = base
            for j in range(nb):
                yy = int(base - t[j] * (row_h - 8))
                x = side + int(j * px)
                lo_, hi_ = min(prev, yy), max(prev, yy)
                img[lo_:hi_ + 1, x] = dens_col
                img[yy, x:x + int(px) + 1] = dens_col
                prev = yy
            for L in LV:
                x = side + int((L * s + xmax) / (2 * xmax) * width)
                img[y + 2:y + 8, x] = line_col
        else:
            strip = (np.repeat(t, int(np.ceil(px)))[:width])[None, :, None]
            band = bg * (1 - strip) + dens_col * strip
            img[y + 3:y + row_h - 3, side:side + width] = band
            for L in LV:
                x = side + int((L * s + xmax) / (2 * xmax) * width)
                img[y:y + row_h, x:x + 2] = line_col
        if i % max(1, len(gs) // 16) == 0 or i == len(gs) - 1:
            items.append((side - 14, y + row_h // 2 - 9, f"{s:.1e}", 16, "Mono", mu, "ra"))
            items.append((side + width + 14, y + row_h // 2 - 9, f"{n} blk", 16, "Mono", mu))
    yb = top + len(gs) * row_h + 20
    for v in np.linspace(-xmax, xmax, 9):
        x = side + int((v + xmax) / (2 * xmax) * width)
        img[yb:yb + 10, x] = mu
        items.append((x, yb + 14, f"{v:+.2f}", 18, "Mono", mu, "ma"))
    items += [(side - 14, top - 30, "scale", 16, "Mono", mu, "ra"),
              (side, yb + 70, "Levels = +-{0, 0.5, 1, 1.5, 2, 3, 4, 6} x scale.  Top rows (small scales, most blocks) have tight combs; the bottom rows are the outlier blocks.", 22, "Sans", mu),
              (side, yb + 110, S.STACK, 20, "Sans", mu)]
    img = S.text(img, items)
    return S.save(img, f"spectral_lines_{style}_{fmt}_{tag}.png")


def normalized_plate(tag, fmt="NVFP4", width=1600, n_rows=160, row_h=12, style="night"):
    """Same data in units of the block's own scale: the 15 lines become fixed vertical lines, and each row
    (blocks binned by scale quantile) shows how the weights spread over them. Large-scale (outlier) blocks
    collapse towards 0: most of their weights use only the levels 0 and +-0.5."""
    d = S.load(tag)
    W = d["W"].astype(np.float64)
    bs = 16 if fmt == "NVFP4" else 32
    H, Wd = W.shape
    B = W[:, : Wd // bs * bs].reshape(H, -1, bs)
    sc = d[f"{fmt}_scale"][:, : Wd // bs]
    y = (B / sc[..., None]).reshape(-1, bs)
    s = sc.ravel()
    order = np.argsort(s)
    chunks = np.array_split(order, n_rows)
    nb = 800
    edges = np.linspace(-7, 7, nb + 1)
    side, top = 150, 220
    img = S.canvas(top + n_rows * row_h + 200, width + 2 * side, S.NIGHT)
    cm = S.cmap("cmc.lajolla_r") if False else S.cmap("inferno")
    items = [(side, 40, "Spectral lines, in block units", 56, "Bold", (0.93, 0.91, 0.86)),
             (side, 115, f"{d['name']} {fmt}: blocks sorted by scale, {n_rows} quantile rows (top = smallest scale)", 22, "Sans", (0.55, 0.55, 0.6)),
             (side, 147, "x = w / block scale, so the FP4 levels are fixed lines. MXFP4: block max always lands in [4, 8) (floor rule); above 6 it is clipped.", 22, "Sans", (0.55, 0.55, 0.6)),
             (side, 179, "Fine vertical comb (MXFP4 only): w is a bf16 number and X a power of two, so w/X keeps the bf16 mantissa grid.", 22, "Sans", (0.55, 0.55, 0.6))]
    for i, ch in enumerate(chunks):
        h, _ = np.histogram(y[ch].ravel(), edges)
        t = np.log1p(h) / np.log1p(h.max())
        rowc = S.to_rgb(np.repeat(t, width // nb), cm)
        img[top + i * row_h: top + (i + 1) * row_h - 1, side:side + width] = rowc[None]
    for L in LV:
        x = side + int((L + 7) / 14 * width)
        img[top - 12:top, x:x + 2] = (0.6, 0.85, 1.0)
        img[top + n_rows * row_h: top + n_rows * row_h + 12, x:x + 2] = (0.6, 0.85, 1.0)
        items.append((x, top + n_rows * row_h + 16, f"{L:g}", 16, "Mono", (0.6, 0.85, 1.0), "ma"))
    items += [(side, top + n_rows * row_h + 60, "blue ticks = E2M1 levels. Colour = log count per row (inferno, declared). Emission beyond +-6 exists only for NVFP4 scale rounding / MXFP4 clipping.", 22, "Sans", (0.55, 0.55, 0.6)),
              (side, top + n_rows * row_h + 100, S.STACK, 20, "Sans", (0.55, 0.55, 0.6))]
    img = S.text(img, items)
    return S.save(img, f"spectral_lines_blockunits_{fmt}_{tag}.png")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["L26_q_proj", "L27_gate_proj"]
    for t in tags:
        for st in ("plate", "night", "line"):
            plate(t, "NVFP4", st)
        plate(t, "MXFP4", "plate", row_h=120)
        normalized_plate(t, "NVFP4")
        normalized_plate(t, "MXFP4")
