"""Shared rendering: every style maps cached measurements (class labels, status, tconv) to RGB.

Declared aesthetic choices are named here once so captions can cite them.
"""
import json, sys
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0, '/home/fzeng/ml/research/art/color-research')
import palettes as P
from fractal import boundary

PAPER = np.array(P.hex2rgb('#f3eee2'))
INK = np.array(P.hex2rgb('#1c1b1a'))
DIV_DARK = np.array(P.hex2rgb('#141217'))     # diverged, dark styles
NC_GREY = np.array(P.hex2rgb('#6b6763'))      # not converged by T

# depth-3 factorisation: +++ gold, the three S3-rotated classes as three jewel tones (Klimt set)
F3_COLORS = {0: '#eacc62', 1: '#c93f55', 2: '#469d76', 3: '#3c4b99'}
F3_CANON = {0: '#eacc62', 1: '#8a5a9e'}      # +++ vs "one positive" (mod permutation)

# XOR 2-2-1: hue family = canonical solution; within family 8 raw variants (unit order x signs)
XOR_FAMILY = {  # canonical code -> (base, 8 raw shades)
    (1, 7): ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#7a1f3d', '#e7298a', '#b35806', '#fee08b'],
    (2, 4): ['#5e4fa2', '#3288bd', '#66c2a5', '#abdda4', '#2d2a6e', '#35978f', '#1b7837', '#c7eae5'],
}
XOR_CANON = {(1, 7): '#e0583a', (2, 4): '#3a78b5'}


def load(path):
    d = dict(np.load(path, allow_pickle=False))
    d['meta'] = json.loads(str(d['meta']))
    return d


def flipud(a):
    return a[::-1]


def time_q(tconv, mask, lo=None, hi=None, pct=(0.5, 99.5)):
    """log convergence time -> [0,1] (0 fast). Range from percentiles unless given (zoom films fix it)."""
    lt = np.log10(np.maximum(tconv, 1.0))
    if lo is None:
        lo, hi = np.percentile(lt[mask], pct) if mask.any() else (0, 1)
    return np.clip((lt - lo) / max(hi - lo, 1e-9), 0, 1), (lo, hi)


def class_rgb(labels, colors):
    rgb = np.zeros(labels.shape + (3,))
    for k, c in colors.items():
        rgb[labels == k] = P.hex2rgb(c)
    return rgb


def xor_raw_colors(labels):
    """raw XOR code (c1*16+c2) -> shade inside its canonical family."""
    cols = {}
    for code in np.unique(labels):
        if code < 0:
            continue
        c1, c2 = divmod(int(code), 16)
        m1, m2 = min(c1, 15 - c1), min(c2, 15 - c2)
        fam = tuple(sorted((m1, m2)))
        if fam not in XOR_FAMILY:
            cols[code] = '#bbbbbb'; continue
        order = int(m1 > m2)                      # which unit carries the smaller canonical code
        s1, s2 = int(c1 != m1), int(c2 != m2)     # sign flips of unit 1 / unit 2
        cols[code] = XOR_FAMILY[fam][order * 4 + s1 * 2 + s2]
    return cols


def xor_canon_colors(labels):
    cols = {}
    for code in np.unique(labels):
        if code < 0:
            continue
        fam = divmod(int(code), 16)
        cols[code] = XOR_CANON.get(fam, '#bbbbbb')
    return cols


# ------------------------------------------------------------------ styles
def style_newton(labels, status, tconv, colors, tq=None, vmin=0.18, div_rgb=DIV_DARK, nc_rgb=NC_GREY):
    """Categorical hue = solution identity; brightness = 1 - 0.82 * normalised log10 convergence time."""
    conv = status == 0
    q, rng = time_q(tconv, conv) if tq is None else time_q(tconv, conv, *tq)
    rgb = class_rgb(labels, colors)
    v = 1 - (1 - vmin) * q
    rgb = rgb * v[..., None]
    # diverged: flat neutral with a faint escape-time glow (slow escape = lighter), declared
    dq, _ = time_q(tconv, status == 2)
    rgb[status == 2] = div_rgb + (dq[status == 2, None]) * 0.10
    rgb[status == 1] = nc_rgb
    return np.clip(rgb, 0, 1), rng


def style_ink(labels, status, weight_between=2, weight_edge=1, paper=PAPER, ink=INK, ss=1):
    """Single ink on paper: basin-to-basin boundaries heavy, converge/diverge edge light."""
    conv = status == 0
    lab_c = np.where(conv, labels, -99)
    b_all = boundary(np.where(conv, labels, -2))
    # between two converged classes: boundary of labels restricted to converged pixels
    L = np.where(conv, labels, -99)
    between = np.zeros_like(b_all)
    for axis in (0, 1):
        a = np.take(L, range(L.shape[axis] - 1), axis=axis); b = np.take(L, range(1, L.shape[axis]), axis=axis)
        d = (a != b) & (a != -99) & (b != -99)
        pad = [(0, 0), (0, 0)]; pad[axis] = (0, 1); between |= np.pad(d, pad)
        pad[axis] = (1, 0); between |= np.pad(d, pad)
    edge = b_all & ~between
    if weight_between > 1:
        between = ndimage.binary_dilation(between, iterations=weight_between - 1)
    if weight_edge > 1:
        edge = ndimage.binary_dilation(edge, iterations=weight_edge - 1)
    cov = np.maximum(between * 1.0, edge * 0.55)
    return paper * (1 - cov[..., None]) + ink * cov[..., None]


def screen(cov, rng, kind='stochastic'):
    return (rng.random(cov.shape) < cov).astype(float)


def style_riso(labels, status, tconv, class_inks, inks, seed=0, misreg=((0, 0), (3, -2), (-2, 3)),
               density=(0.95, 0.35), paper=P.RISO_PAPER):
    """Three spot inks with stochastic screens. class_inks[k] = tuple of ink indices for class k.
    Screen density ~ convergence speed (fast = dense); diverged = bare paper. Per-drum offsets
    (misregistration, pixels) are a declared aesthetic choice."""
    rng = np.random.default_rng(seed)
    conv = status == 0
    q, _ = time_q(tconv, conv)
    dens = density[0] + (density[1] - density[0]) * q
    covs = []
    for i in range(len(inks)):
        m = np.zeros(labels.shape, bool)
        for k, idx in class_inks.items():
            if i in idx:
                m |= conv & (labels == k)
        c = screen(np.where(m, dens, 0.0), rng)
        dy, dx = misreg[i]
        covs.append(np.roll(c, (dy, dx), axis=(0, 1)) * 0.92)
    return P.overprint(covs, inks, paper=paper)


def style_dark_time(status, tconv, cmap='cmc.lajolla_r', div_rgb=np.array([0.02, 0.02, 0.03])):
    """Dark ground, perceptually uniform map of log10 convergence time; diverged = near-black."""
    import matplotlib
    conv = status == 0
    q, rng = time_q(tconv, conv)
    cm = matplotlib.colormaps[cmap]
    rgb = cm(1 - q)[..., :3]
    rgb[~conv] = div_rgb
    rgb[status == 1] = NC_GREY * 0.5
    return rgb, rng


def style_spectral(status, tconv):
    """Sohl-Dickstein split: converged ranked by convergence time (purple at boundary -> yellow),
    diverged ranked by escape time (deep red at boundary -> yellow). Not-converged sits on the seam."""
    x = np.where(status == 0, -np.maximum(tconv, 1e-3), np.maximum(tconv, 1e-3))
    x = np.where(status == 1, np.nan, x)
    return P.render_split(x, 'sd_spectral', near_boundary='large', nan_color='#3b0f2e')


def save(rgb, path, flip=True):
    a = np.clip(np.asarray(rgb), 0, 1)
    if flip:
        a = a[::-1]
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(path, optimize=False)
