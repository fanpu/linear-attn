"""Shared rendering helpers for Dither: dB -> unit, power-domain resampling, colour idioms, 1-bit error diffusion.

All colour mappings here are declared aesthetic choices; the *data* mapped is always a dB power value.
"""
import ctypes
import json
import os
import subprocess
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
GAL = os.path.join(HERE, "gallery")
META = json.load(open(os.path.join(CACHE, "spec_meta.json"))) if os.path.exists(os.path.join(CACHE, "spec_meta.json")) else None
STACK = "numpy float64 exact math, CPU; rendered 2026-09-13"

PAPER = np.array([244, 239, 228]) / 255
INK = np.array([22, 22, 26]) / 255
RISO_BLUE = np.array([0, 120, 191]) / 255
RISO_PINK = np.array([255, 72, 176]) / 255
RISO_YELLOW = np.array([255, 232, 0]) / 255
RISO_TEAL = np.array([0, 131, 138]) / 255


def load_spec(name):
    return np.load(os.path.join(CACHE, f"spec_{name}.npy")).astype(np.float32)


def resample_power(S_db, out_w=None, out_h=None):
    """Area-average in linear power (energy-honest), S is frames x bins. Returns bins x frames image, low f at bottom."""
    P = 10 ** (S_db.astype(np.float64) / 10)
    n_t, n_f = P.shape
    if out_w and out_w < n_t:
        k = n_t // out_w
        P = P[:k * out_w].reshape(out_w, k, n_f).mean(1)
    if out_h and out_h < n_f:
        k = n_f // out_h
        P = P[:, :k * out_h].reshape(P.shape[0], out_h, k).mean(2)
    return (10 * np.log10(P + 1e-30)).T[::-1]


def unit(S_db, lo, hi, gamma=1.0):
    return np.clip((S_db - lo) / (hi - lo), 0, 1) ** gamma


def cmap_rgb(v, name="magma"):
    return colormaps[name](v)[..., :3]


def ink_on_paper(v, ink=INK, paper=PAPER, gamma=1.0):
    a = v[..., None] ** gamma
    return paper * (1 - a) + ink * a


def multiply_layers(layers, paper=PAPER):
    """layers: list of (coverage[0..1] HxW, ink rgb). Subtractive-ish multiply blend on paper."""
    out = np.ones(layers[0][0].shape + (3,)) * paper
    for cov, ink in layers:
        out *= 1 - cov[..., None] * (1 - ink)
    return out


def shift(a, dx, dy):
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def save_rgb(rgb, path):
    return save_png(rgb, path)


def _old_save_rgb(rgb, path):
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, optimize=True)


# ------------------------------------------------------------------ Floyd-Steinberg in C (pure python is too slow)
_C = r"""
void fs(double *a, unsigned char *out, int h, int w, int serpentine){
  for(int y=0;y<h;y++){
    int ltr = !serpentine || (y%2==0);
    for(int i=0;i<w;i++){
      int x = ltr ? i : w-1-i; int d = ltr ? 1 : -1;
      double v=a[y*w+x]; int q = v>=0.5; out[y*w+x]=q; double e=v-q;
      if(x+d>=0 && x+d<w) a[y*w+x+d]+=e*7/16.;
      if(y+1<h){ if(x-d>=0&&x-d<w) a[(y+1)*w+x-d]+=e*3/16.; a[(y+1)*w+x]+=e*5/16.; if(x+d>=0&&x+d<w) a[(y+1)*w+x+d]+=e*1/16.; }
    }
  }
}
"""
_lib = None


def _load():
    global _lib
    if _lib is None:
        src = os.path.join(CACHE, "fs.c")
        so = os.path.join(CACHE, "fs.so")
        if not os.path.exists(so):
            open(src, "w").write(_C)
            subprocess.run(["gcc", "-O3", "-shared", "-fPIC", src, "-o", so], check=True)
        _lib = ctypes.CDLL(so)
    return _lib


def floyd_steinberg(v, serpentine=True):
    a = np.ascontiguousarray(v, dtype=np.float64).copy()
    out = np.zeros(a.shape, dtype=np.uint8)
    _load().fs(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
               a.shape[0], a.shape[1], int(serpentine))
    return out


def freq_axis_labels(ax, height_px, fmax=24000, ticks=(0, 6000, 12000, 18000, 24000), color="0.6", size=7):
    ax.set_yticks([height_px * (1 - f / fmax) for f in ticks])
    ax.set_yticklabels([f"{f // 1000} kHz" for f in ticks], color=color, fontsize=size)


# ------------------------------------------------------------------ PIL composition helpers
from PIL import ImageDraw, ImageFont

FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"


def canvas(w, h, color):
    return Image.new("RGB", (w, h), tuple(int(c * 255) for c in color))


def paste(cv, rgb, x, y):
    cv.paste(Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)), (x, y))


def text(cv, xy, s, size=24, color=(0.8, 0.8, 0.8), font=FONT_SANS, anchor="la"):
    d = ImageDraw.Draw(cv)
    d.text(xy, s, font=ImageFont.truetype(font, size), fill=tuple(int(c * 255) for c in color), anchor=anchor)


def line(cv, xy, color, width=1):
    ImageDraw.Draw(cv).line(xy, fill=tuple(int(c * 255) for c in color), width=width)


def save_png(img, path, max_mb=19.0):
    """Lossless RGB PNG if it fits under max_mb, else a 256-colour palette PNG (colormaps are 256-entry LUTs anyway;
    only antialiased text/blends are approximated). Declared in the README."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    img.save(path, optimize=True)
    if os.path.getsize(path) > max_mb * 1e6:
        img.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
        print(f"  palette PNG: {os.path.basename(path)} {os.path.getsize(path) / 1e6:.1f} MB")
        while os.path.getsize(path) > max_mb * 1e6:   # still too big for commit.sh's 20 MB cap: downscale 0.85x (declared)
            img = img.resize((int(img.width * 0.85), int(img.height * 0.85)), Image.LANCZOS)
            img.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
            print(f"  downscaled to {img.width}x{img.height}: {os.path.getsize(path) / 1e6:.1f} MB")


def logfreq_power(S_db, out_h, fmin=100.0, fmax=24000.0, fs=48000, nfft=4096, out_w=None):
    """Re-grid the frequency axis logarithmically: each output row averages the linear power of the STFT bins whose
    centres fall in its [f_lo, f_hi) band (or takes the nearest bin when the band is narrower than a bin).
    Returns image rows (high f at top) x columns."""
    P = 10 ** (S_db.astype(np.float64) / 10)
    if out_w and out_w < P.shape[0]:
        k = P.shape[0] // out_w
        P = P[:k * out_w].reshape(out_w, k, -1).mean(1)
    edges = np.geomspace(fmin, fmax, out_h + 1)
    df = fs / nfft
    out = np.empty((out_h, P.shape[0]))
    cs = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)], axis=1)
    for r in range(out_h):
        a = int(np.floor(edges[r] / df + 0.5))
        b = int(np.floor(edges[r + 1] / df + 0.5))
        if b <= a:
            out[r] = P[:, min(a, P.shape[1] - 1)]
        else:
            out[r] = (cs[:, b] - cs[:, a]) / (b - a)
    return (10 * np.log10(out + 1e-30))[::-1]
