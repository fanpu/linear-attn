"""Block Error renders: |Q(W) - W| per element, NVFP4 vs MXFP4 on the same matrix, several styles.
   python render_blockerr.py [tags...]
"""
import sys

import numpy as np

import styles as S

GAP = 24


def pick_crop(err_nv, h, w, bs=32):
    """Choose the block-aligned window (h x w) with the largest std of log mean |err| per 16-block:
    i.e. where the weave is most varied. Deterministic."""
    H, W = err_nv.shape
    B = np.log10(np.abs(err_nv[:, : W // 16 * 16]).reshape(H, -1, 16).mean(-1) + 1e-12)
    best, arg = -1, (0, 0)
    for y in range(0, H - h + 1, max(1, h // 4)):
        for x in range(0, W - w + 1, bs):
            s = B[y:y + h, x // 16:(x + w) // 16].std()
            if s > best:
                best, arg = s, (y, x)
    return arg


def header(img, title, sub, fg, y=18, x=28):
    return S.text(img, [(x, y, title, 34, "Bold", fg), (x, y + 46, sub, 18, "Sans", fg)])


def full_nocturne(tag, d, cm_name="magma"):
    en, em = d["NVFP4_nea_err"], d["MXFP4_nea_err"]
    ref = np.concatenate([np.abs(en).ravel(), np.abs(em).ravel()])
    nn, (lo, hi) = S.log_norm(en, 1.0, 99.9, ref=ref)
    nm, _ = S.log_norm(em, 1.0, 99.9, ref=ref)
    cm = S.cmap(cm_name)
    H, W = en.shape
    top, bot, side = 130, 150, 28
    img = S.canvas(H + top + bot, 2 * W + GAP + 2 * side, S.NIGHT)
    S.paste(img, S.to_rgb(nn, cm), top, side)
    S.paste(img, S.to_rgb(nm, cm), top, side + W + GAP)
    S.paste(img, S.colorbar(14, 520, cm), top + H + 58, side)
    fg = (0.9, 0.89, 0.86)
    mu = (0.55, 0.55, 0.6)
    name = str(d["name"])
    rows = d["rows"]
    rel_n = np.mean(en.astype(np.float64) ** 2) / np.mean(d["W"].astype(np.float64) ** 2)
    rel_m = np.mean(em.astype(np.float64) ** 2) / np.mean(d["W"].astype(np.float64) ** 2)
    crop = f" rows {rows[0]}:{rows[1]}" if "embed" in tag else ""
    img = S.text(img, [
        (side, 22, f"Block Error  |Q(W) - W|", 36, "Bold", fg),
        (side, 70, f"{name}{crop}  {H}x{W}, one pixel per weight, blocks run left to right along each row", 18, "Sans", mu),
        (side, top - 30, f"NVFP4  16-blocks, E4M3 scale x FP32 tensor scale   rel.MSE {rel_n:.4f}", 18, "Mono", fg),
        (side + W + GAP, top - 30, f"MXFP4  32-blocks, E8M0 power-of-two scale   rel.MSE {rel_m:.4f}", 18, "Mono", fg),
        (side, top + H + 22, f"log10 |error|, shared scale for both panels (black = exact / below 1st pct)", 16, "Sans", mu),
        (side, top + H + 78, f"{lo:.2f}", 15, "Mono", mu), (side + 520, top + H + 78, f"{hi:.2f}", 15, "Mono", mu, "ra"),
        (side, top + H + 110, "round-to-nearest, ties-to-even.  " + S.STACK, 15, "Sans", mu),
    ])
    return S.save(img, f"blockerr_nocturne_{tag}.png")


def crop_panels(tag, d, h=72, w=144, k=10, style="nocturne"):
    en, em = d["NVFP4_nea_err"], d["MXFP4_nea_err"]
    y, x = pick_crop(en, h, w)
    cn, cmx = en[y:y + h, x:x + w], em[y:y + h, x:x + w]
    ref = np.concatenate([np.abs(en).ravel(), np.abs(em).ravel()])
    nn, (lo, hi) = S.log_norm(cn, 1.0, 99.9, ref=ref)
    nm, _ = S.log_norm(cmx, 1.0, 99.9, ref=ref)
    Hh, Ww = h * k, w * k
    top, bot, side, gap = 150, 110, 40, 60
    if style == "nocturne":
        bg, fg, mu = S.NIGHT, (0.9, 0.89, 0.86), (0.55, 0.55, 0.6)
        pn, pm = S.to_rgb(S.up(nn, k), S.cmap("magma")), S.to_rgb(S.up(nm, k), S.cmap("magma"))
        tick = (0.45, 0.45, 0.5)
    elif style == "ledger":
        bg, fg, mu = S.PAPER, (0.09, 0.09, 0.1), (0.42, 0.40, 0.37)
        pn, pm = S.ink(S.up(nn, k), S.INK_DARK), S.ink(S.up(nm, k), S.INK_DARK)
        tick = (0.6, 0.57, 0.52)
    img = S.canvas(2 * Hh + gap + top + bot, Ww + 2 * side, bg)
    S.paste(img, pn, top, side)
    S.paste(img, pm, top + Hh + gap, side)
    # block-boundary ticks in the margin only (never drawn over data)
    for yy, bs in [(top, 16), (top + Hh + gap, 32)]:
        for c in range(0, w + 1):
            if (x + c) % bs == 0:
                S.vline(img, side + c * k - (1 if c == w else 0), yy - 12, yy - 2, tick, 2)
    img = S.text(img, [
        (side, 20, "Block Error, zoomed", 34, "Bold", fg),
        (side, 64, f"{d['name']}  rows {y + int(d['rows'][0])}:{y + h + int(d['rows'][0])}, cols {x}:{x + w};"
                   f" each square = one weight ({k}x{k} px)", 18, "Sans", mu),
        (side, top - 44, "NVFP4   (margin ticks = 16-element block edges)", 18, "Mono", fg),
        (side, top + Hh + gap - 44, "MXFP4   (margin ticks = 32-element block edges)", 18, "Mono", fg),
        (side, top + 2 * Hh + gap + 20, f"log10|error| from {lo:.2f} to {hi:.2f}, shared across panels; RTN ties-to-even", 16, "Sans", mu),
        (side, top + 2 * Hh + gap + 50, S.STACK, 15, "Sans", mu),
    ])
    return S.save(img, f"blockerr_zoom_{style}_{tag}.png")


def loom(tag, d, h=96, w=192, k=12):
    """Textile: each weight is a woven cell whose dye = its log |error| (data); thread shading, the
    plain-weave over/under alternation and the wider seam at block edges are declared ornament."""
    en, em = d["NVFP4_nea_err"], d["MXFP4_nea_err"]
    y, x = pick_crop(en, h, w)
    ref = np.concatenate([np.abs(en).ravel(), np.abs(em).ravel()])
    cm = S.madder()
    yy, xx = np.mgrid[0:k, 0:k]
    prof_h = 0.62 + 0.38 * np.sin(np.pi * (yy + 0.5) / k) ** 0.7     # weft on top: rounded across y
    prof_v = 0.62 + 0.38 * np.sin(np.pi * (xx + 0.5) / k) ** 0.7     # warp on top: rounded across x
    edge = np.ones((k, k))
    edge[0, :] *= 0.55
    edge[:, 0] *= 0.75

    def weave(e, bs):
        n, _ = S.log_norm(e[y:y + h, x:x + w], 1.0, 99.9, ref=ref)
        col = S.to_rgb(n, cm)                               # (h,w,3)
        big = S.up(col, k)
        ii, jj = np.mgrid[0:h, 0:w]
        over = ((ii + jj) % 2 == 0)
        shade = np.where(S.up(over, k), np.tile(prof_h, (h, w)), np.tile(prof_v, (h, w))) * np.tile(edge, (h, w))
        img = big * shade[..., None]
        for c in range(0, w):
            if (x + c) % bs == 0:
                img[:, c * k: c * k + 2] *= 0.25                  # seam at block edge
        return img

    pn, pm = weave(en, 16), weave(em, 32)
    Hh, Ww = h * k, w * k
    top, side, gap, bot = 170, 50, 70, 120
    bg = np.array([0.93, 0.90, 0.84])
    img = S.canvas(top + 2 * Hh + gap + bot, Ww + 2 * side, bg)
    S.paste(img, pn, top, side)
    S.paste(img, pm, top + Hh + gap, side)
    fg, mu = (0.2, 0.12, 0.1), (0.45, 0.38, 0.33)
    img = S.text(img, [
        (side, 26, "Loom", 44, "Serif", fg),
        (side, 86, f"{d['name']}  rows {y}:{y + h}, cols {x}:{x + w}.  Dye = log10 |Q(W)-W| per weight;"
                   f" dark seams = block edges", 19, "Serif", mu),
        (side, top - 40, "NVFP4 cloth, 16-thread repeat", 20, "SerifI", fg),
        (side, top + Hh + gap - 40, "MXFP4 cloth, 32-thread repeat", 20, "SerifI", fg),
        (side, top + 2 * Hh + gap + 24, "Weave shading and seams are ornament; colour is data (shared scale, RTN).", 17, "Serif", mu),
        (side, top + 2 * Hh + gap + 56, S.STACK, 15, "Serif", mu),
    ])
    return S.save(img, f"blockerr_loom_{tag}.png")


def riso(tag, d, h=384, w=768, k=3):
    """Two-spot risograph: NVFP4 error in blue, MXFP4 error in fluorescent pink, overprinted (multiply)
    with a declared 3 px misregistration; plus the two separations side by side."""
    en, em = d["NVFP4_nea_err"], d["MXFP4_nea_err"]
    H, W = en.shape
    h, w = min(h, H), min(w, W)
    y, x = pick_crop(en, h, w)
    ref = np.concatenate([np.abs(en).ravel(), np.abs(em).ravel()])
    dn, _ = S.lin_norm(en[y:y + h, x:x + w], 99.5, ref=ref)
    dm, _ = S.lin_norm(em[y:y + h, x:x + w], 99.5, ref=ref)
    dn, dm = S.up(dn, k) ** 0.8, S.up(dm, k) ** 0.8
    Hh, Ww = h * k, w * k
    side, top, gap = 60, 170, 50
    sep_w = (Ww - gap) // 2
    sep_h = Hh * sep_w // Ww
    img = S.canvas(top + Hh + gap + sep_h + 150, Ww + 2 * side, S.PAPER)
    over = S.ink_multi([(dn, S.RISO_BLUE), (dm, S.RISO_PINK)], shift=[(0, 0), (3, -2)])
    S.paste(img, over, top, side)
    from PIL import Image

    def small(dd, c):
        im = Image.fromarray((dd * 255).astype(np.uint8)).resize((sep_w, sep_h), Image.LANCZOS)
        return S.ink(np.asarray(im) / 255.0, c)
    S.paste(img, small(dn, S.RISO_BLUE), top + Hh + gap, side)
    S.paste(img, small(dm, S.RISO_PINK), top + Hh + gap, side + sep_w + gap)
    img *= S.grain(img.shape[:2], 0.04)
    fg, mu = S.RISO_BLACK, (0.4, 0.38, 0.36)
    img = S.text(img, [
        (side, 30, "BLOCK ERROR / two formats, one sheet", 40, "Bold", S.RISO_BLUE),
        (side, 90, f"{d['name']}  rows {y}:{y + h}  cols {x}:{x + w}   ink = |Q(W)-W| (linear, shared, 99.5 pct = full)", 19, "Sans", mu),
        (side, 120, "blue drum: NVFP4        pink drum: MXFP4        purple = both formats err there", 19, "Mono", fg),
        (side, top + Hh + gap + sep_h + 30, "Separations. 3 px misregistration and paper grain are declared ornament.", 17, "Sans", mu),
        (side, top + Hh + gap + sep_h + 62, S.STACK, 15, "Sans", mu),
    ])
    return S.save(img, f"blockerr_riso_{tag}.png")


def rounding_quad(tag, d, h=64, w=128, k=8):
    """RTN vs stochastic rounding: 2x2 zoomed panels, dark perceptual."""
    keys = [("NVFP4_nea_err", "NVFP4  round-to-nearest"), ("NVFP4_sto_err", "NVFP4  stochastic"),
            ("MXFP4_nea_err", "MXFP4  round-to-nearest"), ("MXFP4_sto_err", "MXFP4  stochastic")]
    y, x = pick_crop(d["NVFP4_nea_err"], h, w)
    ref = np.concatenate([np.abs(d[kk]).ravel()[::7] for kk, _ in keys])
    cm = S.cmap("cmc.lajolla_r") if False else S.cmap("inferno")
    Hh, Ww = h * k, w * k
    side, top, gap = 40, 140, 56
    img = S.canvas(top + 2 * Hh + 2 * gap + 80, 2 * Ww + gap + 2 * side, S.NIGHT)
    fg, mu = (0.9, 0.89, 0.86), (0.55, 0.55, 0.6)
    items = [(side, 20, "Rounding mode changes the grain", 34, "Bold", fg),
             (side, 66, f"{d['name']} rows {y}:{y + h} cols {x}:{x + w}. log10 |Q(W)-W|, shared scale."
                        f" Stochastic rounding is unbiased but has ~2x the MSE.", 18, "Sans", mu)]
    W64 = d["W"].astype(np.float64)
    for n, (kk, lab) in enumerate(keys):
        r, c = divmod(n, 2)
        nn, _ = S.log_norm(d[kk][y:y + h, x:x + w], 1.0, 99.9, ref=ref)
        py, px = top + r * (Hh + gap), side + c * (Ww + gap)
        S.paste(img, S.to_rgb(S.up(nn, k), cm), py, px)
        rel = np.mean(d[kk].astype(np.float64) ** 2) / np.mean(W64 ** 2)
        items.append((px, py - 30, f"{lab}   rel.MSE(whole matrix) {rel:.4f}", 17, "Mono", fg))
    items.append((side, top + 2 * Hh + 2 * gap - 20, S.STACK + "; stochastic seed 1234", 15, "Sans", mu))
    img = S.text(img, items)
    return S.save(img, f"blockerr_rounding_{tag}.png")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["L27_gate_proj", "L00_q_proj", "embed_head"]
    for t in tags:
        d = S.load(t)
        full_nocturne(t, d)
        crop_panels(t, d, style="nocturne")
        crop_panels(t, d, style="ledger")
        loom(t, d)
        riso(t, d)
        rounding_quad(t, d)
