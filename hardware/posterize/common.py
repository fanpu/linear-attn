"""Shared exact math + rendering helpers for *Posterize*.

Number formats (declared conventions, identical to hardware/dither):
  int-b   symmetric abs-max: levels k * s, k in [-(2^(b-1)-1), 2^(b-1)-1], s = peak/(2^(b-1)-1); round-half-even; clipped.
          int2 -> 3 levels {-1,0,1}, int3 -> 7, int4 -> 15.
  FP4 E2M1 (15 values: 0, .5, 1, 1.5, 2, 3, 4, 6), FP8 E4M3FN (253 values, max 448): all bit patterns enumerated,
          round-to-nearest, ties-to-even mantissa, saturating. Signal peak is mapped to the format max.
Halftones: Floyd-Steinberg (serpentine, C), ordered Bayer (recursive), blue noise (void-and-cluster, Ulichney 1993).
"""
import ctypes
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
GAL = os.path.join(HERE, "gallery")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(GAL, exist_ok=True)
STACK = "numpy float64 exact math, CPU; rendered 2026-09-13"

# ------------------------------------------------------------------ formats

def fp_grid(exp_bits, man_bits, bias, fn_nan_top=False):
    vals, even = [], []
    for e in range(2 ** exp_bits):
        for m in range(2 ** man_bits):
            if fn_nan_top and e == 2 ** exp_bits - 1 and m == 2 ** man_bits - 1:
                continue
            v = 2.0 ** (1 - bias) * (m / 2 ** man_bits) if e == 0 else 2.0 ** (e - bias) * (1 + m / 2 ** man_bits)
            vals.append(v)
            even.append(m % 2 == 0)
    vals = np.array(vals)
    o = np.argsort(vals)
    pos, ev = vals[o], np.array(even)[o]
    return np.concatenate([-pos[:0:-1], pos]), np.concatenate([ev[:0:-1], ev])


FP4 = fp_grid(2, 1, 1)
FP8 = fp_grid(4, 3, 7, fn_nan_top=True)


def round_to_grid(x, grid, even):
    x = np.asarray(x, dtype=np.float64)
    xc = np.clip(x, grid[0], grid[-1])
    hi = np.clip(np.searchsorted(grid, xc, side="left"), 1, len(grid) - 1)
    lo = hi - 1
    dlo = xc - grid[lo]
    dhi = grid[hi] - xc
    return np.where((dhi < dlo) | ((dhi == dlo) & even[hi]), grid[hi], grid[lo])


def q_fp(x, fmt, peak=1.0):
    g, e = {"fp4": FP4, "fp8": FP8}[fmt]
    s = g[-1] / peak
    return round_to_grid(x * s, g, e) / s


def q_int(x, bits, peak=1.0):
    L = 2 ** (bits - 1) - 1
    s = peak / L
    return np.clip(np.round(x / s), -L, L) * s


def quantize(x, fmt, peak=1.0):
    if fmt.startswith("int"):
        return q_int(x, int(fmt[3:]), peak)
    return q_fp(x, fmt, peak)


FORMATS = ["int2", "int3", "int4", "fp4", "fp8"]
LEVELS = {"int2": 3, "int3": 7, "int4": 15, "fp4": 15, "fp8": 253}

# ------------------------------------------------------------------ halftoning

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


def floyd_steinberg(v, serpentine=True):
    global _lib
    if _lib is None:
        so = os.path.join(CACHE, "fs.so")
        if not os.path.exists(so):
            open(os.path.join(CACHE, "fs.c"), "w").write(_C)
            subprocess.run(["gcc", "-O3", "-shared", "-fPIC", os.path.join(CACHE, "fs.c"), "-o", so], check=True)
        _lib = ctypes.CDLL(so)
    a = np.ascontiguousarray(v, dtype=np.float64).copy()
    out = np.zeros(a.shape, dtype=np.uint8)
    _lib.fs(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            a.shape[0], a.shape[1], int(serpentine))
    return out


def bayer(n):
    """Recursive Bayer index matrix of size 2^n: M_1 = [[0,2],[3,1]], M_2k = [[4M, 4M+2], [4M+3, 4M+1]]."""
    M = np.array([[0]], dtype=np.int64)
    for _ in range(n):
        M = np.block([[4 * M, 4 * M + 2], [4 * M + 3, 4 * M + 1]])
    return M


def bayer_closed_form(n):
    """Closed form: interleave the bits of (x XOR y) and y, most significant first, bit-reversed per level."""
    s = 2 ** n
    y, x = np.mgrid[0:s, 0:s]
    M = np.zeros((s, s), dtype=np.int64)
    for b in range(n):
        xb = (x >> b) & 1
        yb = (y >> b) & 1
        # level b (finest = b 0) contributes the most significant 2 bits
        M |= (((xb ^ yb) << 1) | yb) << (2 * (n - 1 - b))
    return M


def void_and_cluster(n=128, sigma=1.5, seed=0):
    """Ulichney's void-and-cluster blue-noise threshold matrix (toroidal Gaussian energy), ranks 0..n^2-1."""
    rng = np.random.default_rng(seed)
    N = n * n
    yy, xx = np.mgrid[0:n, 0:n]
    dy = np.minimum(yy, n - yy); dx = np.minimum(xx, n - xx)
    K = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
    Kf = np.fft.rfft2(K)

    def energy(B):
        return np.fft.irfft2(np.fft.rfft2(B) * Kf, s=B.shape)

    # 1) initial binary pattern: ~10% random, relaxed by moving tightest cluster -> largest void
    B = np.zeros((n, n))
    B.flat[rng.choice(N, N // 10, replace=False)] = 1
    E = energy(B)
    Kroll = lambda i: np.roll(np.roll(K, i // n, 0), i % n, 1)
    while True:
        c = np.argmax(np.where(B == 1, E, -np.inf))
        B.flat[c] = 0; E -= Kroll(c)
        v = np.argmin(np.where(B == 0, E, np.inf))
        if v == c:
            B.flat[c] = 1; E += Kroll(c)
            break
        B.flat[v] = 1; E += Kroll(v)
    proto = B.copy()
    ones = int(proto.sum())
    R = np.zeros(N, dtype=np.int64)
    # 2) phase 1: remove tightest clusters, rank ones-1 .. 0
    B = proto.copy(); E = energy(B)
    for r in range(ones - 1, -1, -1):
        c = np.argmax(np.where(B == 1, E, -np.inf))
        B.flat[c] = 0; E -= Kroll(c); R[c] = r
    # 3) phase 2/3: fill largest voids, rank ones .. N-1
    B = proto.copy(); E = energy(B)
    for r in range(ones, N):
        v = np.argmin(np.where(B == 0, E, np.inf))
        B.flat[v] = 1; E += Kroll(v); R[v] = r
    return R.reshape(n, n)


def ordered(v, T):
    """1-bit ordered dither of v in [0,1] with rank matrix T (tiled). Pixel on iff v > (rank + 0.5)/size."""
    n = T.shape[0]
    h, w = v.shape
    th = (np.tile(T, (h // n + 1, w // n + 1))[:h, :w] + 0.5) / T.size
    return (v > th).astype(np.uint8)


# ------------------------------------------------------------------ rendering helpers

PAPER = np.array([244, 239, 228]) / 255
INK = np.array([22, 22, 26]) / 255
RISO_BLUE = np.array([0, 120, 191]) / 255
RISO_PINK = np.array([255, 72, 176]) / 255
RISO_TEAL = np.array([0, 131, 138]) / 255
RISO_YELLOW = np.array([255, 232, 0]) / 255
FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"


def cmap_rgb(v, name):
    return colormaps[name](v)[..., :3]


def diverging(e, lim, name):
    return cmap_rgb(np.clip(0.5 + 0.5 * e / lim, 0, 1), name)


def ink_on_paper(v, ink=INK, paper=PAPER, gamma=1.0):
    a = np.clip(v, 0, 1)[..., None] ** gamma
    return paper * (1 - a) + ink * a


def multiply_layers(layers, paper=PAPER):
    out = np.ones(layers[0][0].shape + (3,)) * paper
    for cov, ink in layers:
        out *= 1 - cov[..., None] * (1 - ink)
    return out


def shift(a, dx, dy):
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def canvas(w, h, color):
    return Image.new("RGB", (w, h), tuple(int(c * 255) for c in color))


def to_img(rgb):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def paste(cv, rgb, x, y):
    cv.paste(to_img(rgb) if not isinstance(rgb, Image.Image) else rgb, (x, y))


def text(cv, xy, s, size=24, color=(0.8, 0.8, 0.8), font=FONT_SANS, anchor="la"):
    ImageDraw.Draw(cv).text(xy, s, font=ImageFont.truetype(font, size), fill=tuple(int(c * 255) for c in color), anchor=anchor)


def line(cv, xy, color, width=1):
    ImageDraw.Draw(cv).line(xy, fill=tuple(int(c * 255) for c in color), width=width)


def circle(cv, cx, cy, r, color, width=1):
    ImageDraw.Draw(cv).ellipse([cx - r, cy - r, cx + r, cy + r], outline=tuple(int(c * 255) for c in color), width=width)


def save_png(img, path, max_mb=19.0):
    if not isinstance(img, Image.Image):
        img = to_img(img)
    img.save(path, optimize=True)
    if os.path.getsize(path) > max_mb * 1e6:
        img.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
        while os.path.getsize(path) > max_mb * 1e6:
            img = img.resize((int(img.width * 0.85), int(img.height * 0.85)), Image.LANCZOS)
            img.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
        print(f"  size-capped: {os.path.basename(path)} {img.width}x{img.height} {os.path.getsize(path) / 1e6:.1f} MB")


def box_down(v, f):
    h, w = v.shape[:2]
    return v[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f, *v.shape[2:]).mean((1, 3))
