"""Style renderers: measure field M (R x R, his signed convergence measure) -> uint8 RGB.
Row 0 of M is the bottom of the plot (eta1 smallest); every renderer returns images with
row 0 at the TOP (i.e. flipped), ready for PIL.

Measured quantity in every style: sign(M) = converged / diverged, |M| = sum of normalised
losses (converged) or sum of inverse normalised losses (diverged) - small |M| = fast.
Aesthetic choices are listed per function."""
import numpy as np
import colorcet as cc
import matplotlib as mpl
from matplotlib.colors import LightSource
from scipy import ndimage

from common_render import cdf_img, edges, speed01


def _flip(a):
    return a[::-1].copy()


def _to_u8(rgb):
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def upscale(img, f):
    if f == 1:
        return img
    return np.repeat(np.repeat(img, f, axis=0), f, axis=1)


# ---------------------------------------------------------------- 0. his Spectral (primary)
def spectral(M, M_ref=None):
    """Faithful reproduction of Sohl-Dickstein's colab colouring: cdf_img (rank-normalise
    negatives = converged into [-1,-0.25] and positives = diverged into [0.25,1], keep the
    sign, then negate) shown with matplotlib 'Spectral', vmin=-1, vmax=1, nearest
    interpolation. Converged: slowest (next to the boundary) = deep purple (+1), fastest =
    pale yellow-green (+0.25). Diverged: slowest = deep red (-1), fastest = pale orange
    (-0.25). The two dark ends meet at the boundary. M_ref: distribution used for the rank
    normalisation (his reference_scale; default = the image itself)."""
    y = cdf_img(M, M_ref)
    cm = mpl.colormaps['Spectral']
    return _to_u8(cm((_flip(y) + 1) / 2)[..., :3])


# ---------------------------------------------------------------- 1. dark continuous
def dark_magma(M, M_ref=None, cmap='magma'):
    """His colouring (cdf restretch, sign kept, gap of +/-0.25 around zero) shown through
    magma instead of Spectral: slowest diverging = black, slowest converging = near
    white, the fast interiors sit in the middle of the map. Aesthetic: colormap choice."""
    y = cdf_img(M, M_ref)
    cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    return _to_u8(cm((_flip(y) + 1) / 2)[..., :3])


def dark_fire_ice(M, M_ref=None):
    """Two sequential maps meeting in black at the boundary: converged = colorcet fire
    (brightness = convergence speed rank), diverged = colorcet kbc reversed-dark
    (brightness = divergence speed rank). Slow training near the boundary goes dark, so
    the fractal edge reads as black filaments. Aesthetic: map pair, gamma 0.8."""
    s, conv = speed01(M)
    s = _flip(s) ** 0.8
    conv = _flip(conv)
    fire = mpl.colors.ListedColormap(cc.fire)
    ice = mpl.colors.ListedColormap(cc.kbc)
    out = np.where(conv[..., None], fire(0.08 + 0.9 * s)[..., :3], ice(0.05 + 0.9 * s)[..., :3])
    return _to_u8(out)


# ---------------------------------------------------------------- 2. two-ink riso
PAPER = np.array([0.96, 0.94, 0.89])
INK_A = np.array([1.00, 0.28, 0.53])   # riso fluorescent pink (converged)
INK_B = np.array([0.00, 0.47, 0.75])   # riso blue (diverged)


def _screen(shape, angle_deg, period):
    """AM halftone screen threshold in [0,1] at an angle (round dots)."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    a = np.deg2rad(angle_deg)
    u = (xx * np.cos(a) + yy * np.sin(a)) / period
    v = (-xx * np.sin(a) + yy * np.cos(a)) / period
    t = (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v)) / 4 + 0.5   # dot profile
    return t


def riso_two_ink(M, scale=3, period=5.0, misreg=(2, -1), seed=0,
                 ink_a=INK_A, ink_b=INK_B, slow_dense=True):
    """Converged pixels printed in ink A, diverged in ink B, each through its own
    rotated halftone screen (A 15 deg, B 75 deg). Screen density = speed rank within
    phase; slow (near-boundary) = dense ink when slow_dense. Ink B layer offset by
    `misreg` px (deliberate misregistration), multiplicative overprint, paper grain.
    Measured: phase + speed rank. Aesthetic: inks, screen period/angles, misregistration,
    grain."""
    s, conv = speed01(M)
    s = _flip(s); conv = _flip(conv)
    s = upscale(s, scale); conv = upscale(conv, scale)
    dens = (1 - s) if slow_dense else s
    dens = 0.12 + 0.85 * dens ** 1.3
    H, W = s.shape
    ta = _screen((H, W), 15, period)
    tb = _screen((H, W), 75, period)
    cov_a = (conv & (dens > ta)).astype(np.float64)
    cov_b = ((~conv) & (dens > tb)).astype(np.float64)
    cov_b = np.roll(cov_b, misreg, axis=(0, 1))
    rng = np.random.default_rng(seed)
    grain = ndimage.gaussian_filter(rng.standard_normal((H, W)), 1.0)
    cov_a = np.clip(cov_a * (0.9 + 0.08 * grain), 0, 1)
    cov_b = np.clip(cov_b * (0.9 + 0.08 * np.roll(grain, 7, 0)), 0, 1)
    img = PAPER[None, None] * (1 - cov_a[..., None] * (1 - ink_a)) * (1 - cov_b[..., None] * (1 - ink_b))
    img *= (1 + 0.015 * grain[..., None])
    return _to_u8(img)


# ---------------------------------------------------------------- 3. single-ink line
INK = np.array([0.10, 0.10, 0.14])
PAPER_W = np.array([0.985, 0.98, 0.965])


def line_boundary(M, scale=2, weight=1.0, ink=INK, paper=PAPER_W):
    """Only the converge/diverge boundary (his sign-change edge set, on the pixel grid),
    drawn as one ink on paper white. Pixel edge set upscaled, lightly anti-aliased by a
    Gaussian of `weight` px. Measured: boundary location only. Aesthetic: ink, weight."""
    E = np.zeros(M.shape, bool)
    e = edges(M)
    E[:-1, :-1] |= e
    E = _flip(E)
    E = upscale(E.astype(np.float64), scale)
    if weight > 0:
        E = ndimage.gaussian_filter(E, weight * 0.6)
        E = np.clip(E * 2.2, 0, 1)
    img = paper[None, None] * (1 - E[..., None]) + ink[None, None] * E[..., None]
    return _to_u8(img)


# ---------------------------------------------------------------- 4. hillshade relief
def relief_height(M):
    """Signed log speed: +log10(sum v) on converged side, -log10(sum 1/v) on the
    diverged side. Slow convergence = high plateau, slow divergence = deep trench;
    the trainability boundary is the cliff between them."""
    a = np.log10(np.maximum(np.abs(M), 1e-12))
    return np.where(M < 0, a, -a)


def hillshade(M, azdeg=315, altdeg=35, vert_exag=None, tint='copper', paper=None):
    """Engraved-plate relief of relief_height(M) under raking light. Measured: height.
    Aesthetic: light direction, vertical exaggeration, tint ramp."""
    z = _flip(relief_height(M))
    R = M.shape[0]
    if vert_exag is None:
        vert_exag = R / 200.0
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    shade = ls.hillshade(z, vert_exag=vert_exag, dx=1, dy=1)
    zn = (z - np.percentile(z, 1)) / (np.percentile(z, 99) - np.percentile(z, 1) + 1e-12)
    zn = np.clip(zn, 0, 1)
    if tint == 'copper':
        base = mpl.colormaps['cmc.oslo'](0.25 + 0.6 * zn)[..., :3] if False else None
        dark = np.array([0.16, 0.11, 0.08]); light = np.array([0.95, 0.90, 0.80])
        col = dark + (light - dark) * (0.35 + 0.65 * shade[..., None]) * (0.75 + 0.25 * zn[..., None])
    else:  # grey engraving
        col = np.repeat((0.08 + 0.9 * shade)[..., None], 3, -1)
    return _to_u8(col)
