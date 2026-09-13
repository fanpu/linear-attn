"""Shared rendering helpers: density rasters, tone maps, paper textures, text."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/fzeng/ml/research/art/gd-bifurcation"
CACHE = ROOT + "/cache"
GAL = ROOT + "/gallery"

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/dejavu/"]


def font(size, kind="mono"):
    names = {"mono": "DejaVuSansMono.ttf", "sans": "DejaVuSans.ttf", "serif": "DejaVuSerif.ttf",
             "serif-it": "DejaVuSerif-Italic.ttf", "sans-bold": "DejaVuSans-Bold.ttf",
             "mono-bold": "DejaVuSansMono-Bold.ttf", "serif-bold": "DejaVuSerif-Bold.ttf"}
    for d in FONT_DIRS:
        try:
            return ImageFont.truetype(d + names[kind], size)
        except OSError:
            continue
    return ImageFont.load_default()


def density(etas, vals, lo, hi, ylo, yhi, W, H):
    """Count of visited iterates per pixel.  etas: (N,), vals: (N,R).  Row 0 = top (yhi)."""
    N, R = vals.shape
    cx = np.floor((etas - lo) / (hi - lo) * W).astype(np.int64)
    cx = np.repeat(cx, R)
    v = vals.reshape(-1).astype(np.float64)
    cy = np.floor((yhi - v) / (yhi - ylo) * H)
    ok = np.isfinite(cy) & (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    idx = cx[ok] * H + cy[ok].astype(np.int64)
    c = np.bincount(idx, minlength=W * H).reshape(W, H).T
    return c.astype(np.float64)


def column_norm(c, etas, lo, hi, W, R):
    """Divide by the number of iterates that landed in each pixel column (so columns are comparable)."""
    per = np.bincount(np.clip(np.floor((etas - lo) / (hi - lo) * W).astype(int), 0, W - 1), minlength=W) * R
    return c / np.maximum(per, 1)[None, :]


def tone_log(c, sat=None, floor=0.0):
    """log tone map to [0,1]: log(1 + c/c0) / log(1 + sat/c0) with c0 the median nonzero density."""
    nz = c[c > 0]
    if nz.size == 0:
        return np.zeros_like(c)
    c0 = np.percentile(nz, 20)
    sat = sat if sat is not None else np.percentile(nz, 99.7)
    t = np.log1p(c / c0) / np.log1p(sat / c0)
    return np.clip(t, 0, 1)


def cmap_lut(name, n=1024):
    import matplotlib
    if name.startswith("cc:"):
        import colorcet as cc
        m = cc.cm[name[3:]]
    elif name.startswith("cmc:"):
        import cmcrameri.cm as cmc
        m = getattr(cmc, name[4:])
    else:
        m = matplotlib.colormaps[name]
    return (m(np.linspace(0, 1, n))[:, :3] * 255).astype(np.float64)


def apply_lut(t, lut):
    i = np.clip((t * (len(lut) - 1)).astype(int), 0, len(lut) - 1)
    return lut[i]


def paper(H, W, base=(244, 239, 228), grain=6.0, seed=0):
    rng = np.random.default_rng(seed)
    p = np.ones((H, W, 3)) * np.array(base, float)
    n = rng.normal(0, grain, (H // 4 + 1, W // 4 + 1))
    n = np.kron(n, np.ones((4, 4)))[:H, :W]
    fine = rng.normal(0, grain * 0.5, (H, W))
    return np.clip(p + (n + fine)[..., None], 0, 255)


def ink_multiply(paper_rgb, alpha, ink_rgb):
    """Subtractive (multiply) ink laydown: alpha in [0,1] is ink coverage."""
    ink = np.array(ink_rgb, float) / 255.0
    a = alpha[..., None]
    return paper_rgb * (1 - a + a * ink)


def save(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path, optimize=True)
    print("wrote", path, arr.shape)


def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def lyap_raster(etas, lyap, lo, hi, W, H, vmin=-1.0, vmax=0.8):
    """Per-column max/min of the Lyapunov exponent -> filled trace mask (H,W) for lambda>0 and <0 parts."""
    cx = np.clip(np.floor((etas - lo) / (hi - lo) * W).astype(int), 0, W - 1)
    lam = np.full(W, np.nan)
    lmin = np.full(W, np.nan)
    ok = np.isfinite(lyap)
    for arr, fn in [(lam, np.fmax), (lmin, np.fmin)]:
        tmp = np.full(W, np.nan)
        fn.at(tmp, cx[ok], lyap[ok])
        arr[:] = tmp
    # rows: value -> y
    def row(v):
        return (vmax - np.clip(v, vmin, vmax)) / (vmax - vmin) * (H - 1)
    y0 = row(0.0)
    yy = np.arange(H)[:, None]
    top = row(lam)[None, :]
    bot = row(lmin)[None, :]
    pos = (yy >= np.minimum(top, y0)) & (yy <= y0) & (lam[None, :] > 0)
    neg = (yy <= np.maximum(bot, y0)) & (yy >= y0) & (lmin[None, :] < 0)
    # thin trace between min and max (for chaotic columns the mean sits inside)
    return pos, neg, y0, lam, lmin
