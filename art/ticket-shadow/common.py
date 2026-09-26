"""Shared drawing vocabulary (pure presentation; every value drawn comes from cache/).

Declared aesthetic choices:
  - specimen register: one ink on cream paper, madder red as the only second ink (used for
    negative / "fewer than expected" values), DejaVu Sans Mono captions;
  - a pixel is drawn as a round dot centred in its 28x28 cell, dot AREA proportional to the
    plotted value (halftone convention: ink coverage = value), so area, not radius, is the data.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CREAM = (245, 242, 234)
INK = (22, 22, 26)
MADDER = (178, 52, 42)
MUTED = (120, 116, 108)
RULE = (205, 199, 186)
NIGHT = (8, 8, 11)
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
CACHE = "/home/fzeng/ml/research/art/ticket-shadow/cache"
GALLERY = "/home/fzeng/ml/research/art/ticket-shadow/gallery"


def font(size, serif=False):
    return ImageFont.truetype(SERIF if serif else MONO, size)


def dots(values, cell, color=INK, neg_color=MADDER, bg=CREAM, max_frac=0.92, ss=3, vmax=None):
    """values (784,) -> RGB image (28*cell)^2. |value|/vmax sets dot area as a fraction of the
    largest disc that fits in the cell (max_frac of the cell width). Negative values use neg_color."""
    v = np.asarray(values, float).reshape(28, 28)
    vmax = np.abs(v).max() if vmax is None else vmax
    a = np.clip(np.abs(v) / (vmax + 1e-30), 0, 1)
    rmax = 0.5 * max_frac * cell * ss
    r = rmax * np.sqrt(a)
    S = 28 * cell * ss
    yy, xx = np.mgrid[0:cell * ss, 0:cell * ss] + 0.5
    c0 = cell * ss / 2
    d = np.sqrt((yy - c0) ** 2 + (xx - c0) ** 2)
    cov = np.zeros((S, S)); sign = np.zeros((S, S))
    for i in range(28):
        for j in range(28):
            if r[i, j] <= 0:
                continue
            blk = (d <= r[i, j]).astype(float)
            cov[i * cell * ss:(i + 1) * cell * ss, j * cell * ss:(j + 1) * cell * ss] = blk
            sign[i * cell * ss:(i + 1) * cell * ss, j * cell * ss:(j + 1) * cell * ss] = 1 if v[i, j] >= 0 else -1
    img = np.empty((S, S, 3))
    bg = np.array(bg, float); pc = np.array(color, float); nc = np.array(neg_color, float)
    for k in range(3):
        img[..., k] = bg[k] * (1 - cov) + np.where(sign < 0, nc[k], pc[k]) * cov
    img = img.reshape(28 * cell, ss, 28 * cell, ss, 3).mean((1, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def text(draw, xy, s, size, fill=INK, anchor="la", serif=False):
    draw.text(xy, s, font=font(size, serif), fill=fill, anchor=anchor)


def load_stats():
    import json
    return json.load(open(f"{CACHE}/stats.json"))


def load_analysis():
    return np.load(f"{CACHE}/analysis.npz")


def imp_idx(z, cond, mode="imp"):
    md = z[f"{cond}__modes"]
    return [m for m in range(len(md)) if md[m] == mode]
