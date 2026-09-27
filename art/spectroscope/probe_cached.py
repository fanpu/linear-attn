"""Spectroscope probe: render cached weight-spectrum runs as light (CPU only, throwaway).

Each weight matrix at each checkpoint gives eigenvalues lam_i of W^T W / N, sorted descending,
and the same for a copy with its entries shuffled (lam_shuf, the Marchenko-Pastur null: same
entries, no structure). The ratio r_i = lam_i / lam_shuf_i is 1 for every i when the matrix is
indistinguishable from its shuffled self, and sum(lam) == sum(lam_shuf) always, so training can
only move energy between modes.

Declared choices:
  * mode rank i -> wavelength. 'log' maps u = log(i)/log(n) (top mode at u=0); 'lin' maps i/n.
  * top mode sits at the SHORT (violet) end: the most energetic mode is the most energetic photon.
  * spectral power at a wavelength = r at that rank, raised to gamma (gamma=1 is faithful).
  * integrate against CIE 1931 2-degree CMFs (Wyman-Sloan-Shirley 2013 analytic fit), equal-energy
    white point, so r == 1 everywhere renders exactly white.
  * colour carries the spectrum's SHAPE only: out-of-gamut chromaticities are desaturated toward
    white (add white until no channel is negative), then shown at max displayable brightness.
"""
import sys
import numpy as np
from PIL import Image

WL = np.linspace(400, 700, 301)


def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cmf(wl):
    x = 1.056 * _g(wl, 599.8, 37.9, 31.0) + 0.362 * _g(wl, 442.0, 16.0, 26.7) - 0.065 * _g(wl, 501.1, 20.4, 26.2)
    y = 0.821 * _g(wl, 568.8, 46.9, 40.5) + 0.286 * _g(wl, 530.9, 16.3, 31.1)
    z = 1.217 * _g(wl, 437.0, 11.8, 36.0) + 0.681 * _g(wl, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], 1)  # (W, 3)


CMF = cmf(WL)
XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def spectra_to_rgb(P):
    """P: (..., W) spectral power on WL. Returns sRGB in [0,1], luminance-normalised."""
    xyz = P @ CMF
    white = np.ones(len(WL)) @ CMF
    xyz = xyz / white[1]                       # equal-energy white has Y = 1 per unit power
    xyz = xyz / np.maximum(xyz[..., 1:2], 1e-9)  # luminance-normalise: shape only
    wxyz = white / white[1]
    # chromatic adaptation E -> D65 by von Kries scaling in XYZ (declared, crude): make E map to D65 white
    d65 = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz * (d65 / wxyz)
    rgb = xyz @ XYZ2RGB.T
    # out-of-gamut chromaticities: add white until no channel is negative (desaturate toward white)
    rgb = rgb - np.minimum(rgb.min(-1, keepdims=True), 0)
    # then show each chromaticity at the brightest displayable level (max channel = 1)
    rgb = rgb / np.maximum(rgb.max(-1, keepdims=True), 1e-9)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def ratio_spectrum(lam, lam_shuf, mapping='log', gamma=1.0):
    """lam, lam_shuf: (T, n). Returns (T, W) spectral power."""
    a = -np.sort(-lam, 1)
    b = -np.sort(-lam_shuf, 1)
    if mapping == 'energy':
        # wavelength position = null's cumulative energy fraction by rank; power = dF_trained/dF_null
        Fa = np.concatenate([np.zeros((len(a), 1)), np.cumsum(a, 1) / a.sum(1, keepdims=True)], 1)
        Fb = np.concatenate([np.zeros((len(b), 1)), np.cumsum(b, 1) / b.sum(1, keepdims=True)], 1)
        edges = np.linspace(0, 1, len(WL) + 1)
        P = np.empty((len(a), len(WL)))
        for t in range(len(a)):
            Fe = np.interp(edges, Fb[t], Fa[t])
            P[t] = np.diff(Fe) * len(WL)
        return np.maximum(P, 0) ** gamma
    r = (a / np.maximum(b, 1e-30)) ** gamma
    n = r.shape[1]
    i = np.arange(1, n + 1)
    u = np.log(i) / np.log(n) if mapping == 'log' else (i - 1) / (n - 1)
    # piecewise-constant: wavelength bin at position v in [0,1] takes the mode whose u is nearest below
    v = (WL - WL[0]) / (WL[-1] - WL[0])
    idx = np.clip(np.searchsorted(u, v, side='right') - 1, 0, n - 1)
    return r[:, idx]


if __name__ == '__main__':
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, '..', 'weight-spectrum', 'cache')
    runs = [f'mlp_bs{b}_s{s}' for b in (16, 128, 256, 1024) for s in (0, 1)]
    layers = ['FC1', 'FC2', 'FC3']
    out = os.path.join(here, 'gallery')
    os.makedirs(out, exist_ok=True)
    for mapping in ('energy', 'log'):
        for gamma in (1.0, 2.0):
            rows = []
            for run in runs:
                d = np.load(os.path.join(cache, run + '.npz'))
                for L in layers:
                    rgb = spectra_to_rgb(ratio_spectrum(d[f'{L}/lam'], d[f'{L}/lam_shuf'], mapping, gamma))
                    rows.append(np.repeat(rgb[None], 12, 0))
                rows.append(np.full((6, rgb.shape[0], 3), 0.1))
            W = max(r.shape[1] for r in rows)
            rows = [np.concatenate([r, np.full((r.shape[0], W - r.shape[1], 3), 0.1)], 1) for r in rows]
            img = np.concatenate(rows, 0)
            img = np.repeat(img, 4, 1)
            Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(out, f'probe_{mapping}_g{gamma:g}.png'))
    # the colour of a flat spectrum and of single-mode spikes at each position, as a key
    key = []
    for pos in np.linspace(0, 1, 60):
        P = np.ones(len(WL)) * 0.3
        P[int(pos * (len(WL) - 1))::1][:30] = 5.0
        key.append(P)
    rgb = spectra_to_rgb(np.array(key))
    Image.fromarray((np.repeat(np.repeat(rgb[None], 40, 0), 8, 1) * 255).astype(np.uint8)).save(os.path.join(out, 'probe_key.png'))
    print('flat ->', spectra_to_rgb(np.ones((1, len(WL)))))
