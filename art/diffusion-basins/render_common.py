"""Shared rendering helpers: declared palettes, boundary shading, riso overprint, split maps, text plates."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P  # noqa: E402

from common import GALLERY, boundary_mask  # noqa: E402

PAPER = "#f4efe3"
NIGHT = "#0d0d12"
INK = "#1b1b24"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def hx(h):
    return P.hex2rgb(h)


# ----------------------------------------------------------------------------- palettes (declared)
def lch_to_rgb(L, C, h_deg):
    h = np.deg2rad(h_deg)
    lab = np.stack([np.full_like(h, L, dtype=float), C * np.cos(h), C * np.sin(h)], -1)
    return np.clip(P.lab_to_rgb(lab), 0, 1)


def ring_palette(k, phase=0.0):
    """modes on a ring: CIELAB hue follows mode angle (the class order is genuinely cyclic);
    lightness alternates 72 / 52 so neighbouring modes separate in lightness too."""
    h = (phase + 360.0 * np.arange(k) / k) % 360
    L = np.where(np.arange(k) % 2 == 0, 72.0, 52.0)
    return np.stack([lch_to_rgb(L[i], 48.0, np.array(h[i])) for i in range(k)])


# digits 0-9 on dark ground: dataviz reference dark categorical (8) + bronze + cerulean; validated with the
# dataviz validator (adjacent CVD dE 8.4, normal-vision dE 18.6; lightness-band/chroma warnings noted in README)
DIGITS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#8a6a3a", "#9085e9", "#e66767", "#008300", "#2aa7c4"]
TAB10 = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
TAB10_DARK = ["#5b8fd1", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#c38fd6", "#ff9da7", "#b88a6f", "#d9d0c8"]


def scatter_palette(k):
    """Tableau-10 (Stone 2016) extended by two glasbey picks; slot i follows mode index i."""
    extra = ["#2f8f6f", "#8e6bd1"]
    return np.stack([hx(c) for c in (TAB10 + extra)[:k]])


def grid_palette(n=5):
    """bivariate: hue by column (5 hues), CIELAB lightness by row"""
    hues = [25, 85, 150, 235, 300]
    Ls = np.linspace(40, 85, n)
    out = []
    for i in range(n):          # mode index = i*n + j with mu = (g[i], g[j])  -> i = x column, j = y row
        for j in range(n):
            out.append(lch_to_rgb(Ls[j], 42.0, np.array(hues[i])))
    return np.stack(out)


def layout_palette(layout):
    k = {"ring8": 8, "ring6": 6, "ring5": 5, "ring12": 12, "grid25": 25, "scatter12": 12, "tri3": 3, "pair2": 2}[layout]
    if layout.startswith("ring"):
        return ring_palette(k)
    if layout == "grid25":
        return grid_palette()
    return scatter_palette(k)


def memo_palette(digits):
    """40 memorised images: hue = digit class (Tableau-10), lightness = instance (4 steps)"""
    out = []
    count = {}
    for d in digits:
        i = count.get(d, 0)
        count[d] = i + 1
        lab = P.rgb_to_lab(hx(TAB10[d]))
        lab = lab.copy()
        lab[0] = [38, 55, 70, 84][i % 4]
        out.append(np.clip(P.lab_to_rgb(lab), 0, 1))
    return np.stack(out)


# ----------------------------------------------------------------------------- shading
def boundary_distance(lab):
    """Euclidean distance (px) from each pixel to the nearest pixel of a different label"""
    b = boundary_mask(lab)
    return ndimage.distance_transform_edt(~b)


def seam_shade(lab, d0=6.0, floor=0.25):
    d = boundary_distance(lab)
    return floor + (1 - floor) * (1 - np.exp(-d / d0))


def colorize(lab, pal, shade=None, bad=(0, 0, 0)):
    lab = np.asarray(lab)
    img = np.zeros(lab.shape + (3,))
    ok = lab < len(pal)
    img[ok] = pal[lab[ok]]
    img[~ok] = bad
    if shade is not None:
        img = img * shade[..., None]
    return img


def line_art(lab, ink=INK, paper=PAPER, width=1.0, tint=None, pal=None, tint_strength=0.18):
    """single-ink boundary drawing; optional faint basin tint"""
    d = boundary_distance(lab)
    cov = np.clip(width + 0.5 - d, 0, 1)  # anti-aliased width in px
    base = np.ones(lab.shape + (3,)) * hx(paper)
    if pal is not None:
        base = base * (1 - tint_strength) + tint_strength * colorize(lab, pal)
    return base * (1 - cov[..., None]) + cov[..., None] * hx(ink)


def halftone(coverage, cell=6, angle_deg=15.0, offset=(0.0, 0.0)):
    """amplitude-modulated round-dot screen, returns binary-ish (anti-aliased) coverage"""
    H, W = coverage.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    yy += offset[1]
    xx += offset[0]
    t = np.deg2rad(angle_deg)
    u = (xx * np.cos(t) + yy * np.sin(t)) / cell
    v = (-xx * np.sin(t) + yy * np.cos(t)) / cell
    fu, fv = u - np.floor(u) - 0.5, v - np.floor(v) - 0.5
    r = np.hypot(fu, fv)
    rad = np.sqrt(np.clip(coverage, 0, 1) / np.pi)
    return np.clip((rad - r) * cell + 0.5, 0, 1)


def riso(lab, levels, inks, paper=P.RISO_PAPER, cell=7, misreg=((0, 0), (3, -2), (-2, 3)), seam=True):
    """levels: (k, n_inks) coverage per class. Each ink is screened at its own angle and shifted
    (deliberate misregistration), then overprinted multiplicatively."""
    covs = []
    angles = [15, 75, 45]
    for j, ink in enumerate(inks):
        c = levels[np.minimum(lab, len(levels) - 1), j]
        c = ndimage.shift(c, misreg[j][::-1], order=0, mode="nearest")
        covs.append(halftone(c, cell, angles[j]))
    if seam:
        d = boundary_distance(lab)
        line = np.clip(1.6 - d, 0, 1)
        covs[-1] = np.maximum(covs[-1], ndimage.shift(line, misreg[len(inks) - 1][::-1], order=0))
    return P.overprint(covs, inks, paper)


def riso_levels(k, n_inks=3, seed=0):
    """distinct coverage combinations for k classes from {0, 0.35, 0.75} per ink (declared, seeded order)"""
    vals = [0.0, 0.35, 0.75]
    combos = [(a, b, c) for a in vals for b in vals for c in vals if (a, b, c) != (0, 0, 0)]
    rng = np.random.default_rng(seed)
    combos = [combos[i] for i in rng.permutation(len(combos))]
    return np.array(combos[:k])[:, :n_inks]


# ----------------------------------------------------------------------------- output
def save(img, name, sub=""):
    d = os.path.join(GALLERY, sub)
    os.makedirs(d, exist_ok=True)
    arr = np.clip(np.asarray(img) * 255 + 0.5, 0, 255).astype(np.uint8) if np.asarray(img).dtype != np.uint8 else img
    path = os.path.join(d, name)
    Image.fromarray(arr).save(path, optimize=True)
    return path


def flipud(a):
    """compute arrays have row 0 = smallest y; images have row 0 at the top"""
    return a[::-1]


def upscale_nearest(img, f):
    return np.repeat(np.repeat(img, f, axis=0), f, axis=1)


def caption_strip(img, lines, ground=PAPER, ink=INK, size=22, pad=18, mono=False):
    """append a caption strip below an image (float RGB)"""
    H, W = img.shape[:2]
    font = ImageFont.truetype(FONT_MONO if mono else FONT, size)
    h = pad * 2 + len(lines) * int(size * 1.35)
    strip = Image.new("RGB", (W, h), tuple(int(255 * c) for c in hx(ground)))
    dr = ImageDraw.Draw(strip)
    for i, t in enumerate(lines):
        dr.text((pad, pad + i * int(size * 1.35)), t, fill=tuple(int(255 * c) for c in hx(ink)), font=font)
    return np.concatenate([img, np.asarray(strip) / 255.0], 0)


def text_on(img, xy, text, size=20, fill=(1, 1, 1), mono=True):
    im = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    dr.text(xy, text, fill=tuple(int(255 * c) for c in fill), font=ImageFont.truetype(FONT_MONO if mono else FONT, size))
    return np.asarray(im) / 255.0


def pad_to(img, H, W, ground):
    out = np.ones((H, W, 3)) * hx(ground)
    h, w = img.shape[:2]
    y, x = (H - h) // 2, (W - w) // 2
    out[y:y + h, x:x + w] = img
    return out


def grid_images(imgs, ncol, gap, ground):
    h, w = imgs[0].shape[:2]
    nrow = int(np.ceil(len(imgs) / ncol))
    out = np.ones((nrow * h + (nrow + 1) * gap, ncol * w + (ncol + 1) * gap, 3)) * hx(ground)
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncol)
        out[gap + r * (h + gap):gap + r * (h + gap) + h, gap + c * (w + gap):gap + c * (w + gap) + w] = im
    return out


# ----------------------------------------------------------------------------- confidence shading (measured)
def margin_confidence(x0, mu):
    """sampler maps: commitment margin of the generated sample, m = 1 - d1/d2 in [0, 1], where d1, d2 are the
    distances from x0 to the nearest and second-nearest mixture mean. m ~ 1: sample sits on its mode;
    m ~ 0: the sample landed half-way between two modes (low-density bridge)."""
    d = np.sqrt(((x0[..., None, :] - mu[None, None]) ** 2).sum(-1))
    d.sort(-1)
    return 1 - d[..., 0] / np.maximum(d[..., 1], 1e-12)


def confidence_shade(conf, floor=0.10, gamma=0.6):
    return floor + (1 - floor) * np.clip(conf, 0, 1) ** gamma


def nu_confidence(nu):
    """iterated map: fast convergence = confident; rank-normalised log(nu), inf -> 0"""
    v = np.log(np.where(np.isfinite(nu), nu, np.nan))
    ok = np.isfinite(v)
    r = np.zeros(nu.shape)
    r[ok] = (np.argsort(np.argsort(v[ok])) + 0.5) / ok.sum()
    return np.where(ok, 1 - r, 0.0)
