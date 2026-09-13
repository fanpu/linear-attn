"""Bitplanes: the 16 stored bits of a bf16 weight matrix as 1-bit images (4x4, MSB top-left), plus the
bitplanes of the FP8 / NVFP4 / MXFP4 codes.  Captions come from cache/bitplanes.npz (compute_bitplanes.py).
   python render_bitplanes.py
"""
import os

import numpy as np

import styles as S
from compute_bitplanes import BF16_LABELS, plane_stats

CROPS = {  # declared crop choices (square 1024^2 so the 4x4 grid is square)
    "L26_q_proj": (0, 0, 1024),
    "embed_rare": (0, 0, 1024),
    "L00_q_proj": (0, 0, 1024),       # rows 0:1024 of layer-0 q_proj: 8 heads x 128 dims, one high-magnitude head
    "embed_head": (0, 0, 1024),       # token rows 0:1024, includes the near-duplicate rows 177-187
    "L27_gate_proj": (0, 0, 1024),
}


def bf16_planes(raw):
    return [((raw >> (15 - k)) & 1).astype(np.uint8) for k in range(16)]


def describe(st):
    if st["H"] < 0.02:
        return f"constant ({'all 1' if st['p1'] > 0.5 else 'all 0'}, p1={st['p1']:.3f})"
    return f"H {st['H']:.3f}  row-stripe x{st['od_row']:.1f}  col x{st['od_col']:.1f}"


def grid(planes, labels, style, tag, title, sub, cols=4, gut=56, lab_h=90, side=90, top=220, bot=200, fname=None,
         field=None, notes=None):
    h, w = planes[0].shape
    rows = (len(planes) + cols - 1) // cols
    Wd = side * 2 + cols * w + (cols - 1) * gut
    Ht = top + rows * (h + lab_h) + (rows - 1) * gut + bot
    if style == "riso":
        bg, fg, mu = S.PAPER, S.RISO_BLACK, (0.40, 0.38, 0.35)
    elif style == "duotone":
        bg, fg, mu = S.PAPER, S.RISO_BLACK, (0.40, 0.38, 0.35)
    else:
        bg, fg, mu = S.NIGHT, (0.92, 0.90, 0.85), (0.55, 0.55, 0.6)
    img = S.canvas(Ht, Wd, bg)
    items = [(side, 50, title, 54, "Bold", fg if style != "duotone" else S.RISO_BLUE), (side, 125, sub, 24, "Sans", mu)]
    for i, b in enumerate(planes):
        r, c = divmod(i, cols)
        y, x = top + r * (h + lab_h + gut), side + c * (w + gut)
        if style == "riso":
            tile = S.ink(b.astype(float), S.RISO_BLACK)
        elif style == "duotone":
            col = S.RISO_BLUE if field[i] == "exp" else S.RISO_PINK
            tile = S.ink(b.astype(float), col)
        else:
            tile = np.where(b[..., None] == 1, np.array([0.96, 0.93, 0.84]), S.NIGHT)
        S.paste(img, tile, y, x)
        fr = np.array(mu) * 0.8 + np.array(bg) * 0.2          # declared hairline frame so blank planes read as tiles
        S.hline(img, y - 3, x - 3, x + w + 3, fr, 1); S.hline(img, y + h + 2, x - 3, x + w + 3, fr, 1)
        S.vline(img, x - 3, y - 3, y + h + 3, fr, 1); S.vline(img, x + w + 2, y - 3, y + h + 3, fr, 1)
        st = plane_stats(b)
        fs = max(17, w // 40)
        items.append((x, y + h + 12, labels[i], int(fs * 1.4), "Mono", fg))
        items.append((x, y + h + 14 + int(fs * 1.6), describe(st), fs, "Mono", mu))
    if notes:
        for j, n in enumerate(notes):
            items.append((side, Ht - bot + 40 + j * 32, n, 21, "Sans", mu))
    items.append((side, Ht - 50, S.STACK, 18, "Sans", mu))
    img = S.text(img, items)
    return S.save(img, fname, palette=64)


NOTES_BF16 = [
    "bf16 = 1 sign bit, 8 exponent bits (bias 127), 7 mantissa bits.  Ink = bit set.  One pixel per weight, rows = output units.",
    "H = binary entropy of the plane.  row-stripe xN = variance of row means / variance expected for i.i.d. bits (1.0 = no stripes).",
    "Measured on this crop; a random permutation of each plane gives x1.0 +- 0.1.  The sign plane is NOT structured here; the structure lives in exp b3..b1.",
]


def bf16_sheets(tag):
    d = S.load(tag)
    y0, x0, n = CROPS[tag]
    raw = d["raw"][y0:y0 + n, x0:x0 + n]
    planes = bf16_planes(raw)
    field = ["sign"] + ["exp"] * 8 + ["mant"] * 7
    name = str(d["name"])
    sub = f"{name}  rows {y0}:{y0 + n}, cols {x0}:{x0 + n}   16 bitplanes of the stored bfloat16 codes, MSB top-left"
    grid(planes, BF16_LABELS, "riso", tag, "BITPLANES", sub, fname=f"bitplanes_riso_{tag}.png", notes=NOTES_BF16)
    grid(planes, BF16_LABELS, "night", tag, "Bitplanes", sub, fname=f"bitplanes_night_{tag}.png", notes=NOTES_BF16)
    grid(planes, BF16_LABELS, "duotone", tag, "BITPLANES / two drums", sub + "   blue = exponent, pink = sign + mantissa",
         fname=f"bitplanes_duotone_{tag}.png", field=field, notes=NOTES_BF16)
    # zoom: 128x128 of each plane at 4x
    zy, zx = (CROPS_ZOOM.get(tag, (0, 0)))
    zp = [S.up(p[zy:zy + 160, zx:zx + 160], 4) for p in planes]
    grid(zp, BF16_LABELS, "riso", tag, "BITPLANES, magnified x4",
         f"{name}  rows {y0 + zy}:{y0 + zy + 160}, cols {x0 + zx}:{x0 + zx + 160}; each 4x4 px square is one bit of one weight",
         fname=f"bitplanes_zoom_riso_{tag}.png", notes=NOTES_BF16[:2])


CROPS_ZOOM = {"embed_rare": (0, 0), "L26_q_proj": (0, 0), "embed_head": (100, 0), "L00_q_proj": (540, 96), "L27_gate_proj": (0, 0)}


def code_sheet(tag, n=512):
    """FP8 byte planes, NVFP4/MXFP4 nibble planes, and their scale-byte planes (stretched x16 / x32 to line up)."""
    d = S.load(tag)
    name = str(d["name"])
    sets = []
    f8 = d["FP8_byte"][:n, :n]
    sets.append(("FP8 E4M3, per-tensor scale", [((f8 >> (7 - k)) & 1) for k in range(8)],
                 ["sign", "exp b3", "exp b2", "exp b1", "exp b0", "mant b2", "mant b1", "mant b0"]))
    nv = d["NVFP4_nea_nib"][:n, :n]
    nvs = np.repeat(d["NVFP4_scale_code"][:n, : n // 16], 16, axis=1)
    sets.append(("NVFP4 element nibble + E4M3 block-scale byte (each scale bit repeated across its 16-block)",
                 [((nv >> (3 - k)) & 1) for k in range(4)] + [((nvs >> (7 - k)) & 1) for k in range(8)][:4],
                 ["sign", "exp b1", "exp b0", "mant", "scale b7", "scale b6", "scale b5", "scale b4"]))
    sets.append(("", [((nvs >> (7 - k)) & 1) for k in range(4, 8)] + [None] * 4, ["scale b3", "scale b2", "scale b1", "scale b0"] + [""] * 4))
    mx = d["MXFP4_nea_nib"][:n, :n]
    mxs = np.repeat(d["MXFP4_scale_code"][:n, : n // 32], 32, axis=1)
    sets.append(("MXFP4 element nibble + E8M0 shared-exponent byte (repeated across its 32-block)",
                 [((mx >> (3 - k)) & 1) for k in range(4)] + [((mxs >> (7 - k)) & 1) for k in range(8)][:4],
                 ["sign", "exp b1", "exp b0", "mant", "E8M0 b7", "E8M0 b6", "E8M0 b5", "E8M0 b4"]))
    sets.append(("", [((mxs >> (7 - k)) & 1) for k in range(4, 8)] + [None] * 4, ["E8M0 b3", "E8M0 b2", "E8M0 b1", "E8M0 b0"] + [""] * 4))
    cols, gut, lab, side, top = 8, 36, 70, 70, 200
    rowgap = 60
    Wd = side * 2 + cols * n + (cols - 1) * gut
    Ht = top + len(sets) * (n + lab + rowgap) + 160
    img = S.canvas(Ht, Wd, S.PAPER)
    items = [(side, 40, "BITPLANES OF THE CODES", 54, "Bold", S.RISO_BLUE),
             (side, 115, f"{name} rows 0:{n}, cols 0:{n}: what an FP8 / FP4 kernel actually reads. Ink = bit set.", 24, "Sans", (0.4, 0.38, 0.35))]
    for r, (title, planes, labels) in enumerate(sets):
        y = top + r * (n + lab + rowgap)
        if title:
            items.append((side, y - 44, title, 26, "Bold", S.RISO_BLACK))
        for c, (p, lb) in enumerate(zip(planes, labels)):
            if p is None:
                continue
            x = side + c * (n + gut)
            colr = S.RISO_PINK if ("scale" in lb or "E8M0" in lb) else S.RISO_BLACK
            S.paste(img, S.ink(p.astype(float), colr), y, x)
            st = plane_stats(p)
            items += [(x, y + n + 8, lb, 22, "Mono", S.RISO_BLACK), (x, y + n + 36, describe(st)[:44], 19, "Mono", (0.4, 0.38, 0.35))]
    items.append((side, Ht - 90, "Pink = block-scale planes. Scale planes are stretched to the element grid (declared); their stripes are rows of blocks.", 21, "Sans", (0.4, 0.38, 0.35)))
    items.append((side, Ht - 50, S.STACK, 18, "Sans", (0.4, 0.38, 0.35)))
    img = S.text(img, items)
    return S.save(img, f"bitplanes_codes_riso_{tag}.png", palette=64)




# ---------------------------------------------------------------------------------------------------------
# Hero composition that follows the measurement: the planes that carry structure large, all 16 as an index.
HERO = {  # tag: (row0, row1, why)  -- chosen by compute_bitplane_scan.py ranking (see README)
    "L26_q_proj": (0, 1024, "top-ranked linear layer by exponent-plane stripe structure"),
    "embed_rare": (0, 1024, "top-ranked window overall: token ids 147456-148480, late (rare) BPE merges"),
    "embed_head": (0, 1024, "token ids 0-1023; rows 177-187 are the byte tokens 0xF5-0xFF, which never occur in UTF-8"),
}
BIG = [0, 5, 6, 7]   # sign, exp b3, exp b2, exp b1


def hero_strip(tag, sort=False, style="riso"):
    d = S.load(tag)
    r0, r1, why = HERO[tag]
    raw = d["raw"][r0:r1, :1024]
    W = d["W"][r0:r1, :1024].astype(np.float64)
    if sort:   # declared re-ordering: rows by descending RMS, columns by descending RMS
        ri = np.argsort(-np.sqrt((W ** 2).mean(1)))
        ci = np.argsort(-np.sqrt((W ** 2).mean(0)))
        raw = raw[ri][:, ci]
    planes = bf16_planes(raw)
    stats = [plane_stats(p) for p in planes]
    h, w = raw.shape
    sm = 256
    gsm = 10
    Wd_inner = 16 * sm + 15 * gsm
    gbig = (Wd_inner - 4 * w) // 3
    side, top = 110, 260
    chart_h = 330
    H = top + h + 150 + sm + 160 + chart_h + 260
    Wd = Wd_inner + 2 * side
    if style == "riso":
        bg, fg, mu, inkc = S.PAPER, S.RISO_BLACK, np.array([0.40, 0.38, 0.35]), S.RISO_BLACK
        on = lambda b: S.ink(b.astype(float), inkc)
        acc = S.RISO_PINK
    else:
        bg, fg, mu = S.NIGHT, np.array([0.93, 0.91, 0.86]), np.array([0.55, 0.55, 0.6])
        on = lambda b: np.where(b[..., None] == 1, np.array([0.96, 0.93, 0.84]), S.NIGHT)
        acc = np.array([1.0, 0.55, 0.25])
    img = S.canvas(H, Wd, bg)
    name = str(d["name"])
    rows_abs = (int(d["rows"][0]) + r0, int(d["rows"][0]) + r1)
    items = [(side, 60, "BITPLANES" if style == "riso" else "Bitplanes", 72, "Bold", fg),
             (side, 160, f"{name}  rows {rows_abs[0]}:{rows_abs[1]}, cols 0:{w}" + ("   rows & columns SORTED by RMS (declared)" if sort else "")
              + f"   ({why})", 28, "Sans", mu)]
    for j, k in enumerate(BIG):
        x = side + j * (w + gbig)
        S.paste(img, on(planes[k]), top, x)
        st = stats[k]
        items += [(x, top + h + 18, BF16_LABELS[k], 40, "Mono", fg),
                  (x, top + h + 70, describe(st), 26, "Mono", mu)]
    y2 = top + h + 150
    items.append((side, y2 - 8, "all 16 planes, MSB -> LSB (256x256 corner of each, 1:1)", 26, "Bold", fg))
    y2 += 36
    for k in range(16):
        x = side + k * (sm + gsm)
        S.paste(img, on(planes[k][:sm, :sm]), y2, x)
        fr = mu * 0.7 + bg * 0.3
        S.hline(img, y2 - 2, x - 2, x + sm + 2, fr); S.hline(img, y2 + sm + 1, x - 2, x + sm + 2, fr)
        S.vline(img, x - 2, y2 - 2, y2 + sm + 2, fr); S.vline(img, x + sm + 1, y2 - 2, y2 + sm + 2, fr)
        items.append((x + sm // 2, y2 + sm + 12, BF16_LABELS[k], 22, "Mono", fg, "ma"))
    # chart: entropy bar (fg) and log10 stripe overdispersion (accent), one pair per plane
    yc = y2 + sm + 120
    base = yc + chart_h
    odmax = 3.0   # log10(1000)
    for k in range(16):
        x = side + k * (sm + gsm)
        st = stats[k]
        hb = int(st["H"] * (chart_h - 40))
        img[base - hb:base, x + 40:x + 110] = fg
        od = np.log10(max(st["od_row"], st["od_col"], 1.0))
        ho = int(min(od / odmax, 1) * (chart_h - 40))
        img[base - ho:base, x + 146:x + 216] = acc
        items.append((x + 75, base + 10, f"{st['H']:.2f}", 20, "Mono", fg, "ma"))
        items.append((x + 181, base + 10, f"x{max(st['od_row'], st['od_col']):.0f}", 20, "Mono", acc, "ma"))
    S.hline(img, base, side, side + Wd_inner, mu)
    items += [(side, yc - 36, "dark bar: entropy H (bits/pixel, full = 1).   pink bar: stripe strength = max(row, column) overdispersion, log scale, "
               "full = x1000; x1 means indistinguishable from i.i.d. bits", 24, "Sans", mu)]
    notes = ["bf16 = 1 sign bit, 8 exponent bits (bias 127), 7 mantissa bits.  Ink = bit set; one pixel per weight; rows = output units, columns = input dims.",
             "Overdispersion = variance of row (column) means / binomial variance for i.i.d. bits; a random pixel permutation of every plane gives x1.0 +- 0.1.",
             S.STACK]
    for j, n in enumerate(notes):
        items.append((side, base + 70 + j * 40, n, 24, "Sans", mu))
    img = S.text(img, items)
    return S.save(img, f"bitplanes_hero_{style}_{tag}{'_sorted' if sort else ''}.png", palette=64)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["grids", "codes", "hero"]
    if "grids" in what:
        for t in CROPS:
            bf16_sheets(t)
    if "codes" in what:
        for t in ["L26_q_proj", "L27_gate_proj", "embed_rare"]:
            code_sheet(t)
    if "hero" in what:
        for t in HERO:
            for srt in (False, True):
                hero_strip(t, sort=srt, style="riso")
            hero_strip(t, style="night")
