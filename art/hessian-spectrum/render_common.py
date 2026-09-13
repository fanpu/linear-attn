"""Rendering helpers shared by all render_*.py scripts (no computation of spectra here)."""
import os, glob, json
import numpy as np
from common import CACHE, HERE

GAL = os.path.join(HERE, 'gallery')
os.makedirs(GAL, exist_ok=True)
SERIES = [2, 3, 4, 5, 7, 10]
TAU = 1e-4          # symlog linear threshold (eigenvalue units)


def set_tau(t):
    global TAU
    TAU = t


def slog(lam, tau=None):
    tau = TAU if tau is None else tau
    """Symmetric log: sign(l) * log10(1 + |l|/tau). Negative eigenvalues map to the left of 0."""
    lam = np.asarray(lam, dtype=np.float64)
    return np.sign(lam) * np.log10(1 + np.abs(lam) / tau)


def islog(s, tau=None):
    tau = TAU if tau is None else tau
    return np.sign(s) * tau * (10 ** np.abs(s) - 1)


def n_structural(d, thr=0.5):
    """Leading eigenvectors whose squared projection onto span{d_c} (C class-mean gradient directions) exceeds thr."""
    ov = np.asarray(d['ov_dc'])
    k = 0
    while k < len(ov) and ov[k] > thr:
        k += 1
    return k


def load_exact(C, run='s_mlps', step=None):
    d = os.path.join(CACHE, 'exact', f'{run}_C{C}')
    fs = sorted(glob.glob(os.path.join(d, 'step_*.npz')))
    f = fs[-1] if step is None else os.path.join(d, f'step_{step:06d}.npz')
    return dict(np.load(f))


def load_lanczos(C, run='s_mlp', step=None):
    d = os.path.join(CACHE, 'spectra', f'{run}_C{C}')
    fs = sorted(glob.glob(os.path.join(d, 'step_*.npz')))
    f = fs[-1] if step is None else os.path.join(d, f'step_{step:06d}.npz')
    return dict(np.load(f))


def exposure_profile(lam, x0, x1, W, sigma_px=1.2, d0=0.7, weights=None):
    """Photographic-plate model along the dispersion axis.

    Every eigenvalue is a Gaussian line of unit peak (sigma_px pixels) at its symlog position; the plate
    darkens as 1 - exp(-density/d0).  A lone eigenvalue reaches ~0.76; the bulk saturates.
    Returns (exposure in [0,1], raw density) arrays of length W.
    """
    s = slog(lam)
    px = (s - x0) / (x1 - x0) * (W - 1)
    dens = np.zeros(W)
    w = np.ones_like(px) if weights is None else weights
    r = int(np.ceil(4 * sigma_px))
    base = np.floor(px).astype(int)
    for off in range(-r, r + 2):
        idx = base + off
        ok = (idx >= 0) & (idx < W)
        np.add.at(dens, idx[ok], w[ok] * np.exp(-0.5 * ((idx[ok] - px[ok]) / sigma_px) ** 2))
    return 1 - np.exp(-dens / d0), dens


def log_density(lam, x0, x1, W, sigma_px=6.0):
    """Smoothed count per pixel column (for microdensitometer tracings). Kernel width declared by sigma_px."""
    s = slog(lam)
    h, _ = np.histogram(s, bins=W, range=(x0, x1))
    k = np.arange(-int(4 * sigma_px), int(4 * sigma_px) + 1)
    g = np.exp(-0.5 * (k / sigma_px) ** 2); g /= g.sum()
    return np.convolve(h.astype(float), g, mode='same')


def wavelength_rgb(wl):
    """Approximate visible colour of a wavelength in nm (Bruton 1996), vectorised, gamma 0.8. Aesthetic only."""
    wl = np.asarray(wl, dtype=float)
    r = np.zeros_like(wl); g = np.zeros_like(wl); b = np.zeros_like(wl)
    m = (wl >= 380) & (wl < 440); r[m] = -(wl[m] - 440) / 60; b[m] = 1
    m = (wl >= 440) & (wl < 490); g[m] = (wl[m] - 440) / 50; b[m] = 1
    m = (wl >= 490) & (wl < 510); g[m] = 1; b[m] = -(wl[m] - 510) / 20
    m = (wl >= 510) & (wl < 580); r[m] = (wl[m] - 510) / 70; g[m] = 1
    m = (wl >= 580) & (wl < 645); r[m] = 1; g[m] = -(wl[m] - 645) / 65
    m = (wl >= 645) & (wl <= 780); r[m] = 1
    f = np.ones_like(wl)
    m = (wl >= 380) & (wl < 420); f[m] = 0.3 + 0.7 * (wl[m] - 380) / 40
    m = (wl > 700) & (wl <= 780); f[m] = 0.3 + 0.7 * (780 - wl[m]) / 80
    return np.clip(np.stack([r, g, b], -1) * f[..., None], 0, 1) ** 0.8


def hexrgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.


def save_png(rgb, path):
    from PIL import Image
    a = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path, optimize=True)
    print('wrote', path, a.shape)
