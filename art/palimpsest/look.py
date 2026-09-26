"""Declared aesthetic choices, in one place.

normal light : ink density d (the network's output, clipped to [0,1]) laid on vellum as iron-gall
               sepia: rgb = vellum * (1 - d) + ink * d.
pseudocolour : the Archimedes Palimpsest imaging trick. Two exposures: one that shows only the
               over-text (here: page B itself, which the reader of the palimpsest can see), and one
               that shows everything the network holds (its output). The first drives the red
               channel, the second green and blue. Where both have ink: near-black. Where only the
               network has ink (the ghost of A): red. Where B has ink the network has not yet
               written: blue-green. The gain on (output - B), when used, is stated on every plate.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VELLUM = np.array([238, 226, 200], np.float32) / 255  # warm calf vellum
INK = np.array([52, 34, 22], np.float32) / 255        # iron-gall sepia
RED_DEPTH = 0.88                                      # how dark full density prints in pseudocolour

SERIF = "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf"
SERIF_I = "/usr/share/fonts/opentype/urw-base35/P052-Italic.otf"


def normal(d):
    d = np.clip(np.asarray(d, np.float32), 0, 1)[..., None]
    return VELLUM * (1 - d) + INK * d


def pseudo(d_net, d_over, gain=1.0):
    """d_net: the network's output; d_over: the over-text alone (page B)."""
    d_net = np.asarray(d_net, np.float32)
    d_over = np.asarray(d_over, np.float32)
    uv = np.clip(d_over + gain * (d_net - d_over), 0, 1)
    red = np.clip(d_over, 0, 1)
    T = np.stack([1 - RED_DEPTH * red, 1 - RED_DEPTH * uv, 1 - RED_DEPTH * uv], -1)
    return VELLUM * T


def ghost_ink(r, scale, color=(150, 30, 22)):
    """A signed residual printed as a single red ink on vellum: positive = ink, negative = bare."""
    c = np.array(color, np.float32) / 255
    d = np.clip(np.asarray(r, np.float32) / scale, 0, 1)[..., None]
    return VELLUM * (1 - d) + c * d


def to_img(rgb):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def up(rgb, k):
    """Nearest-neighbour enlargement: one network sample becomes k x k print pixels (no invented detail)."""
    return np.repeat(np.repeat(rgb, k, 0), k, 1)


def font(size, italic=False):
    return ImageFont.truetype(SERIF_I if italic else SERIF, size)


def sheet(w, h):
    return Image.new("RGB", (w, h), tuple(int(255 * v) for v in VELLUM))


def text(img, xy, s, size, italic=False, fill=None, anchor="la"):
    fill = fill or tuple(int(255 * v) for v in INK)
    ImageDraw.Draw(img).text(xy, s, font=font(size, italic), fill=fill, anchor=anchor)


MADDER = np.array([158, 38, 28], np.float32) / 255   # red ink for the part of the output page B lacks


def two_ink(f, over, red=MADDER, gain=1.0):
    """Split the output exactly into two inks: f = min(f, B) + max(f - B, 0).
    The first part (ink the network shares with page B) prints in sepia, the excess (ink page B
    does not have: the ghost) in madder red, with an optional stated gain. Inks mix
    multiplicatively (Beer-Lambert), so overlapping density darkens as it would on vellum."""
    f = np.clip(np.asarray(f, np.float32), 0, 1)
    over = np.clip(np.asarray(over, np.float32), 0, 1)
    d_s = np.minimum(f, over)[..., None]
    d_r = np.clip(gain * np.maximum(f - over, 0), 0, 1)[..., None]
    Ts = np.clip(INK / VELLUM, 1e-3, 1)
    Tr = np.clip(red / VELLUM, 1e-3, 1)
    return VELLUM * Ts ** d_s * Tr ** d_r


def normal_bl(d):
    """Normal light with the same Beer-Lambert ink law as two_ink (for side-by-side consistency)."""
    d = np.clip(np.asarray(d, np.float32), 0, 1)[..., None]
    return VELLUM * np.clip(INK / VELLUM, 1e-3, 1) ** d
