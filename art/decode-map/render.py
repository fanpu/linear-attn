"""Render decode-map caches into the gallery styles. Pure CPU, reads cache/*.npz only.

Each sample of the (T, p) grid is drawn as a square tile of `tile` px (declared "tesserae"
rendering: the grid spacing is shown honestly, no interpolation). Cell boundaries are drawn
on tile edges, exactly where two neighbouring samples produce different text.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import analysis as A

sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
try:
    import palettes as PAL
except Exception:                                            # pragma: no cover
    PAL = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmcrameri.cm  # noqa: F401  (registers cmc.*)

GAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_MONO = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"

INK_PAPER = np.array([0.953, 0.937, 0.902])
SUMI = np.array([0.11, 0.11, 0.12])
# declared stained-glass palette: ruby, sapphire, emerald, amber, amethyst, teal, rose, cobalt
GLASS = ["#9b1b30", "#1f3f8f", "#1d7a4c", "#d99a1e", "#6a2c8c", "#18807f", "#c2446b", "#2a62b8"]


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def palette(name):
    if name == "glass":
        return np.array([hex2rgb(c) for c in GLASS])
    return np.array([hex2rgb(c) for c in PAL.SCHEMES[name].stops])


def tiles(img, s):
    """[H,W,...] -> [H*s, W*s, ...] with row 0 at the bottom of the plate (p increases upward)."""
    img = img[::-1]
    return np.repeat(np.repeat(img, s, 0), s, 1)


def edge_fields(tokens):
    """Birth step of every tile edge: first token index at which the two neighbouring outputs
    differ (L if they never differ). Returns (vert [H, W-1], horiz [H-1, W])."""
    L = tokens.shape[2]
    dv = tokens[:, :-1] != tokens[:, 1:]
    dh = tokens[:-1, :] != tokens[1:, :]
    bv = np.where(dv.any(-1), dv.argmax(-1), L)
    bh = np.where(dh.any(-1), dh.argmax(-1), L)
    return bv, bh


def paint_edges(canvas, vert, horiz, s, w, colour_fn):
    """Draw edges on a canvas of tile size s. vert/horiz: float arrays, NaN = no edge.
    colour_fn(values) -> (rgb [...,3], alpha [...]). Arrays are in data orientation (row 0 = bottom)."""
    H = vert.shape[0]
    W = horiz.shape[1]
    o0, o1 = w // 2, w - w // 2
    vflip, hflip = vert[::-1], horiz[::-1]
    # vertical edges between columns j and j+1 at x=(j+1)s
    rgb, al = colour_fn(vflip)
    rgb_t = np.repeat(rgb, s, 0)
    al_t = np.repeat(np.nan_to_num(al), s, 0)
    for off in range(-o0, o1):
        cols = np.arange(1, W) * s + off
        cur = canvas[:, cols]
        a = al_t[..., None]
        canvas[:, cols] = cur * (1 - a) + rgb_t * a
    rgb, al = colour_fn(hflip)
    rgb_t = np.repeat(rgb, s, 1)
    al_t = np.repeat(np.nan_to_num(al), s, 1)
    for off in range(-o0, o1):
        rows = np.arange(1, H) * s + off               # flipped: edge between flipped rows k-1,k
        cur = canvas[rows]
        a = al_t[..., None]
        canvas[rows] = cur * (1 - a) + rgb_t * a
    return canvas


def save(img, name, sub=""):
    path = os.path.join(GAL, sub, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, optimize=True)
    return path


def frame(img, title=None, caption=None, ground=(0.06, 0.06, 0.07), ink=(0.85, 0.85, 0.82),
          margin=None, axes=None, font_px=None):
    """Put a plate on a ground with a margin, optional title, caption and axis ticks.
    axes = (xs_range, ys_range, xlabel, ylabel)."""
    H, W = img.shape[:2]
    m = margin or max(40, W // 14)
    fp = font_px or max(14, W // 70)
    top = m + (int(fp * 2.2) if title else 0)
    bottom = m + (int(fp * 1.8 * (1 + (caption.count("\n") if caption else 0))) if caption else 0)
    canvas = np.ones((H + top + bottom, W + 2 * m, 3)) * np.array(ground)
    canvas[top:top + H, m:m + W] = img
    pil = Image.fromarray((canvas * 255).astype(np.uint8))
    dr = ImageDraw.Draw(pil)
    f = ImageFont.truetype(FONT, fp)
    fs = ImageFont.truetype(FONT, int(fp * 0.8))
    ink255 = tuple(int(c * 255) for c in ink)
    if title:
        dr.text((m, m // 2), title, font=ImageFont.truetype(FONT, int(fp * 1.3)), fill=ink255)
    if axes:
        (x0, x1), (y0, y1), xl, yl = axes
        for k in range(5):
            fx = k / 4
            xv = x0 + fx * (x1 - x0)
            X = m + fx * W
            dr.line([(X, top + H), (X, top + H + fp * 0.4)], fill=ink255, width=1)
            dr.text((X, top + H + fp * 0.5), f"{xv:g}", font=fs, fill=ink255, anchor="ma")
            yv = y0 + fx * (y1 - y0)
            Y = top + H - fx * H
            dr.line([(m - fp * 0.4, Y), (m, Y)], fill=ink255, width=1)
            dr.text((m - fp * 0.5, Y), f"{yv:g}", font=fs, fill=ink255, anchor="rm")
        dr.text((m + W, top + H + fp * 0.5), xl, font=fs, fill=ink255, anchor="ra")
        dr.text((m + 4, top - 4), yl, font=fs, fill=ink255, anchor="ld")
    if caption:
        dr.multiline_text((m, top + H + int(fp * 1.9)), caption, font=fs, fill=ink255, spacing=int(fp * 0.35))
    return np.asarray(pil) / 255.0


# ---------------------------------------------------------------------------- styles
def style_mosaic(d, s, pal="klimt_categorical", ground=(0.05, 0.05, 0.06), gamma=0.6, colours=None):
    """Categorical: proper colouring of the output-text cells (adjacent distinct outputs never share
    a colour); brightness = first-divergence position from the greedy output (late = bright)."""
    tk = d["tokens"]
    L = tk.shape[2]
    if colours is None:
        hs = A.prefix_hashes(tk)
        colours = A.stable_colours(hs, len(palette(pal)), [L])[0]
    P = palette(pal)
    fd = A.first_divergence(tk)
    v = 0.28 + 0.72 * (fd / L) ** gamma
    img = P[colours] * v[..., None]
    return tiles(img, s)


def style_glass(d, s, lead=None, pal="glass", colours=None, light=True):
    tk = d["tokens"]
    L = tk.shape[2]
    hs = A.prefix_hashes(tk)
    lab = A.labels_from_hash(hs[-1])
    if colours is None:
        colours = A.stable_colours(hs, len(palette(pal)), [L])[0]
    P = palette(pal)
    # per-cell brightness jitter from the cell's hash (declared: mimics uneven glass)
    jitter = ((hs[-1] % np.uint64(1000)).astype(float) / 1000.0)
    img = P[colours] * (0.78 + 0.35 * jitter[..., None])
    if light:                                               # soft backlight: brighter toward plate centre
        H, W = lab.shape
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot((yy - H / 2) / H, (xx - W / 2) / W)
        img = img * (1.1 - 0.5 * r[..., None])
    canvas = tiles(img, s)
    bv, bh = edge_fields(tk)
    lead = lead or max(2, s // 3)
    fn = lambda v: (np.zeros(v.shape + (3,)) + 0.03, np.where(v < L, 1.0, 0.0))
    return paint_edges(canvas, bv.astype(float), bh.astype(float), s, lead, fn)


def style_ink(d, s, w=None, paper=INK_PAPER, ink=SUMI, weight_by_age=True):
    """Single ink on paper: every cell boundary as a line along tile edges; boundaries born at
    an early token are fully dark, later-born ones lighter (declared: opacity = 1 - 0.75 t/L)."""
    tk = d["tokens"]
    L = tk.shape[2]
    H, W = tk.shape[:2]
    canvas = np.ones((H * s, W * s, 3)) * paper
    bv, bh = edge_fields(tk)
    w = w or max(1, s // 4)

    def fn(v):
        a = np.where(v < L, 1.0 - 0.75 * v / L if weight_by_age else 1.0, 0.0)
        return np.zeros(v.shape + (3,)) + ink, a
    return paint_edges(canvas, bv.astype(float), bh.astype(float), s, w, fn)


def style_age(d, s, cmap="cmc.batlow", ground=(0.02, 0.02, 0.03), w=None):
    """Boundary-age plate: each boundary coloured by the token index at which it was born."""
    tk = d["tokens"]
    L = tk.shape[2]
    H, W = tk.shape[:2]
    canvas = np.ones((H * s, W * s, 3)) * np.array(ground)
    bv, bh = edge_fields(tk)
    cm = plt.get_cmap(cmap)
    w = w or max(1, s // 3)

    def fn(v):
        return cm(np.clip(v / max(1, L - 1), 0, 1))[..., :3], np.where(v < L, 1.0, 0.0)
    return paint_edges(canvas, bv.astype(float), bh.astype(float), s, w, fn)


def style_continuous(field, s, cmap="magma", lo=None, hi=None):
    lo = np.nanpercentile(field, 0.5) if lo is None else lo
    hi = np.nanpercentile(field, 99.5) if hi is None else hi
    x = np.clip((field - lo) / (hi - lo + 1e-12), 0, 1)
    img = plt.get_cmap(cmap)(x)[..., :3]
    return tiles(img, s)


def style_split(signed, s, pairing="sd_spectral", near_boundary="small", lead_tokens=None, seam="dark"):
    img = PAL.render_split(signed, pairing, near_boundary=near_boundary, seam=seam)
    canvas = tiles(img, s)
    if lead_tokens is not None:                              # optional thin cell outlines
        L = lead_tokens.shape[2]
        bv, bh = edge_fields(lead_tokens)
        fn = lambda v: (np.zeros(v.shape + (3,)), np.where(v < L, 0.35, 0.0))
        canvas = paint_edges(canvas, bv.astype(float), bh.astype(float), s, 1, fn)
    return canvas


def coherence_split(d, thr=None):
    """Signed field for the Spectral split: mean model entropy along the generated text minus a
    threshold at the valley of its (bimodal) histogram. Negative = coherent text,
    positive = degenerate text. Returns (signed, thr)."""
    Hm = A.mean_entropy(d)
    if thr is None:
        x = np.log(Hm[np.isfinite(Hm)])
        hist, edges = np.histogram(x, 60)
        sm = np.convolve(hist, np.ones(5) / 5, "same")
        c = (edges[:-1] + edges[1:]) / 2
        # valley between the two largest modes
        pk = [i for i in range(1, 59) if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]]
        pk = sorted(pk, key=lambda i: -sm[i])[:2]
        if len(pk) == 2:
            a, b = sorted(pk)
            thr = float(np.exp(c[a + np.argmin(sm[a:b + 1])]))
        else:
            thr = float(np.median(Hm))
    return Hm - thr, thr


def style_riso(d, s, ink_a="#3255a4", ink_b="#ff48b0", paper="#f4efe3", shift=2, w=None, fill=None):
    """Two-ink riso: ink A = cell boundaries born within the first half of the horizon (full
    coverage) and later ones (40 %); ink B = contone fill, coverage = 1 - first-divergence/L
    (texts that leave greedy decoding early are inked heavily). Ink B misregistered by
    `shift` px (declared)."""
    tk = d["tokens"]
    L = tk.shape[2]
    H, W = tk.shape[:2]
    bv, bh = edge_fields(tk)
    w = w or max(1, s // 3)
    cov_a = np.zeros((H * s, W * s, 3))
    fn = lambda v: (np.ones(v.shape + (3,)), np.where(v < L, np.where(v < L / 2, 1.0, 0.4), 0.0))
    cov_a = paint_edges(cov_a, bv.astype(float), bh.astype(float), s, w, fn)[..., 0]
    fd = A.first_divergence(tk)
    cov_b = tiles(0.55 * (1 - fd / L) if fill is None else 0.08 + 0.62 * fill, s)
    cov_b = np.roll(cov_b, (shift, shift), (0, 1))
    return PAL.overprint([cov_b, cov_a], [ink_b, ink_a], paper=paper)


if __name__ == "__main__":
    path = sys.argv[1]
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    d = A.load(path)
    base = os.path.splitext(os.path.basename(path))[0]
    save(style_mosaic(d, s), f"{base}_mosaic.png", "test")
    save(style_glass(d, s), f"{base}_glass.png", "test")
    save(style_ink(d, s), f"{base}_ink.png", "test")
    save(style_age(d, s), f"{base}_age.png", "test")
    sig, thr = coherence_split(d)
    print("coherence threshold (nats)", thr)
    save(style_split(sig, s), f"{base}_spectral.png", "test")
    save(style_continuous(A.mean_entropy(d), s), f"{base}_entropy.png", "test")
    save(style_riso(d, s), f"{base}_riso.png", "test")
