"""Shared presentation vocabulary. Pure presentation: every number drawn comes from cache/*.npz.
All palettes, textures, grain and misregistration here are DECLARED aesthetic choices (see README)."""
import os

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
GALLERY = os.path.join(HERE, "gallery")
FONTDIR = "/home/fzeng/ml/research/art/.venv/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/"
STACK = "Qwen3-0.6B bf16 weights, formats computed exactly (CPU float64), torch 2.14.0+cu130, GB10 host, 2026-09-13"

PAPER = np.array([0.957, 0.945, 0.918])
RISO_BLUE = np.array([0.000, 0.471, 0.749])      # Riso "Blue"-like
RISO_PINK = np.array([1.000, 0.282, 0.690])      # Riso "Fluorescent Pink"-like
RISO_BLACK = np.array([0.10, 0.10, 0.12])
RISO_YELLOW = np.array([1.0, 0.91, 0.0])
INK_DARK = np.array([0.086, 0.086, 0.102])
NIGHT = np.array([0.027, 0.031, 0.047])


def font(size, kind="Sans"):
    name = {"Sans": "DejaVuSans.ttf", "Mono": "DejaVuSansMono.ttf", "Serif": "DejaVuSerif.ttf",
            "SerifI": "DejaVuSerif-Italic.ttf", "Bold": "DejaVuSans-Bold.ttf"}[kind]
    return ImageFont.truetype(FONTDIR + name, size)


def cmap(name):
    if name.startswith("cmc."):
        from cmcrameri import cm as ccm
        return getattr(ccm, name[4:])
    if name.startswith("cet."):
        import colorcet as cc
        return cc.cm[name[4:]]
    return colormaps[name]


def madder():
    """Declared textile dye ramp, monotone in lightness: undyed linen -> weld yellow -> madder red -> oak-gall."""
    return LinearSegmentedColormap.from_list("madder", ["#1b1320", "#4a1f2e", "#8c2f2f", "#c8612f", "#e2a85a", "#f3e2b8"])


def indigo():
    return LinearSegmentedColormap.from_list("indigo", ["#0d1326", "#1c2f5e", "#2f5a91", "#6d95bf", "#c2d4e2", "#f1eee4"])


def to_rgb(x, cm, lo=0.0, hi=1.0):
    x = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo + 1e-300), 0, 1)
    return cm(x)[..., :3]


def log_norm(a, lo_q=1.0, hi_q=99.7, ref=None, floor=None):
    """log10|a| mapped to [0,1] with percentiles of `ref` (default a). Exact zeros -> 0."""
    a = np.abs(a).astype(np.float64)
    r = np.abs(ref if ref is not None else a)
    r = r[r > 0]
    lo, hi = np.log10(np.percentile(r, lo_q)), np.log10(np.percentile(r, hi_q))
    if floor is not None:
        lo = floor
    out = np.zeros_like(a)
    nz = a > 0
    out[nz] = np.clip((np.log10(a[nz]) - lo) / (hi - lo), 0, 1)
    return out, (lo, hi)


def lin_norm(a, hi_q=99.7, ref=None):
    a = np.abs(a).astype(np.float64)
    hi = np.percentile(np.abs(ref if ref is not None else a), hi_q)
    return np.clip(a / hi, 0, 1), hi


def ink(density, color, paper=PAPER):
    """Multiply-blend one ink at density [0,1] onto paper."""
    d = np.clip(density, 0, 1)[..., None]
    return paper * (1 - d * (1 - color))


def ink_multi(layers, paper=PAPER, shift=None):
    """layers: list of (density, color). shift: per-layer (dy,dx) misregistration in pixels (declared)."""
    out = np.broadcast_to(paper, layers[0][0].shape + (3,)).copy()
    for k, (d, c) in enumerate(layers):
        if shift is not None and shift[k] != (0, 0):
            d = np.roll(d, shift[k], axis=(0, 1))
        out *= (1 - np.clip(d, 0, 1)[..., None] * (1 - c))
    return out


def grain(shape, amount=0.035, seed=0):
    """Declared paper grain (multiplicative, white noise, no periodicity)."""
    rng = np.random.default_rng(seed)
    return 1 - amount * rng.random(shape)[..., None]


def up(a, k):
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def save(rgb, name, meta=None):
    os.makedirs(GALLERY, exist_ok=True)
    arr = np.clip(np.asarray(rgb) * 255 + 0.5, 0, 255).astype(np.uint8) if np.asarray(rgb).dtype != np.uint8 else rgb
    im = Image.fromarray(arr)
    path = os.path.join(GALLERY, name)
    im.save(path, optimize=True)
    print("wrote", path, im.size)
    return path


def canvas(h, w, bg):
    return np.broadcast_to(np.asarray(bg, dtype=np.float64), (h, w, 3)).copy()


def paste(dst, src, y, x):
    h, w = src.shape[:2]
    dst[y:y + h, x:x + w] = src
    return dst


def text(img_arr, items):
    """items: list of (x, y, str, size, kind, rgb, anchor). Returns new float array."""
    im = Image.fromarray(np.clip(img_arr * 255 + .5, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for it in items:
        x, y, s, size, kind, col = it[:6]
        anchor = it[6] if len(it) > 6 else "la"
        dr.text((x, y), s, font=font(size, kind), fill=tuple(int(255 * c) for c in col), anchor=anchor)
    return np.asarray(im).astype(np.float64) / 255


def hline(a, y, x0, x1, col, t=1):
    a[y:y + t, x0:x1] = col


def vline(a, x, y0, y1, col, t=1):
    a[y0:y1, x:x + t] = col


def colorbar(h, w, cm, horizontal=True):
    t = np.linspace(0, 1, w if horizontal else h)
    bar = cm(t)[..., :3]
    if horizontal:
        return np.broadcast_to(bar[None], (h, w, 3)).copy()
    return np.broadcast_to(bar[::-1, None], (h, w, 3)).copy()


def load(tag):
    return np.load(os.path.join(CACHE, f"mat_{tag}.npz"))
