"""Spectroscope colour pipeline (module form of probe_cached.py, energy mapping only).

A matrix's eigenvalues of W^T W / N, sorted descending, are compared with a null spectrum (the mean
of two entry-shuffled copies, or the Marchenko-Pastur quantiles at the measured variance).
  * wavelength position = the null's cumulative energy fraction by rank (top modes at 400 nm);
  * spectral power      = dF_trained / dF_null, so a spectrum identical to the null is flat;
  * integrate against CIE 1931 2-degree CMFs (Wyman-Sloan-Shirley 2013 fit), equal-energy white
    -> exactly white; out-of-gamut chromaticities desaturate toward white; every cell is shown at
    maximum displayable brightness (colour carries the spectrum's SHAPE, not its energy).
"""
import numpy as np

WL = np.linspace(400, 700, 301)
NW = len(WL)


def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cmf(wl):
    x = 1.056 * _g(wl, 599.8, 37.9, 31.0) + 0.362 * _g(wl, 442.0, 16.0, 26.7) - 0.065 * _g(wl, 501.1, 20.4, 26.2)
    y = 0.821 * _g(wl, 568.8, 46.9, 40.5) + 0.286 * _g(wl, 530.9, 16.3, 31.1)
    z = 1.217 * _g(wl, 437.0, 11.8, 36.0) + 0.681 * _g(wl, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], 1)


CMF = cmf(WL)
WHITE_XYZ = np.ones(NW) @ CMF
XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
D65 = np.array([0.95047, 1.0, 1.08883])


def spectra_to_xyz(P):
    return (P @ CMF) / WHITE_XYZ[1]


def xy(P):
    X = spectra_to_xyz(P)
    return X[..., :2] / np.maximum(X.sum(-1, keepdims=True), 1e-12)


def spectra_to_rgb(P):
    """P (..., NW) -> sRGB in [0,1] at max brightness (identical to probe_cached.spectra_to_rgb)."""
    xyz = spectra_to_xyz(P)
    xyz = xyz / np.maximum(xyz[..., 1:2], 1e-9)
    xyz = xyz * (D65 / (WHITE_XYZ / WHITE_XYZ[1]))
    rgb = xyz @ XYZ2RGB.T
    rgb = rgb - np.minimum(rgb.min(-1, keepdims=True), 0)
    rgb = rgb / np.maximum(rgb.max(-1, keepdims=True), 1e-9)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)


def energy_power(lam, null, gamma=1.0):
    """lam, null: (T, n) eigenvalues (any order). Returns (T, NW) power = dF_lam/dF_null on the null's
    cumulative-energy axis, mean 1. gamma != 1 is a declared exaggeration of the SHAPE: P**gamma."""
    a = -np.sort(-np.asarray(lam, float), -1)
    b = -np.sort(-np.asarray(null, float), -1)
    z = np.zeros(a.shape[:-1] + (1,))
    Fa = np.concatenate([z, np.cumsum(a, -1) / np.maximum(a.sum(-1, keepdims=True), 1e-30)], -1)
    Fb = np.concatenate([z, np.cumsum(b, -1) / np.maximum(b.sum(-1, keepdims=True), 1e-30)], -1)
    edges = np.linspace(0, 1, NW + 1)
    P = np.empty(a.shape[:-1] + (NW,))
    for idx in np.ndindex(a.shape[:-1]):
        fb = Fb[idx]
        # guard against repeated null values (zero eigenvalues) for interp monotonicity
        fb = np.maximum.accumulate(fb + np.arange(len(fb)) * 1e-15)
        P[idx] = np.diff(np.interp(edges, fb, Fa[idx])) * NW
    P = np.maximum(P, 0)
    if gamma != 1.0:
        P = P ** gamma
        P = P / np.maximum(P.mean(-1, keepdims=True), 1e-30)
    return P


def centroid_nm(P):
    """Power-weighted mean wavelength: 550 nm for a flat (white) spectrum; <550 = blue-shift."""
    return (P * WL).sum(-1) / np.maximum(P.sum(-1), 1e-30)


# dominant wavelength: xy of monochromatic lights, for hue labelling
_LOCUS = CMF[:, :2] / CMF.sum(1, keepdims=True)
_WXY = WHITE_XYZ[:2] / WHITE_XYZ.sum()


def dominant_wavelength(xys):
    """Dominant wavelength (nm) of chromaticities relative to equal-energy white; negative values are
    complementary wavelengths (purples). NaN if within 1e-4 of white."""
    xys = np.atleast_2d(xys)
    v = xys - _WXY
    ang = np.arctan2(v[:, 1], v[:, 0])
    la = np.arctan2(*(_LOCUS - _WXY)[:, ::-1].T)
    out = np.full(len(xys), np.nan)
    for k, a in enumerate(ang):
        if np.hypot(*v[k]) < 1e-4:
            continue
        d = np.angle(np.exp(1j * (la - a)))
        dc = np.angle(np.exp(1j * (la - a - np.pi)))
        if np.abs(d).min() < np.abs(dc).min():
            out[k] = WL[np.abs(d).argmin()]
        else:
            out[k] = -WL[np.abs(dc).argmin()]
    return out


def chroma(xys):
    return np.hypot(*(np.atleast_2d(xys) - _WXY).T)


def mp_null(n, m, grid=20000):
    """Marchenko-Pastur quantile eigenvalues (descending, mean 1) for W^T W / m, W is m x n, n <= m."""
    q = n / m
    lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    # substitute x = lo + (hi-lo) sin^2(t) to tame the edge singularities
    t = np.linspace(0, np.pi / 2, grid)
    x = lo + (hi - lo) * np.sin(t) ** 2
    dx = (hi - lo) * 2 * np.sin(t) * np.cos(t)
    dens = np.sqrt(np.maximum((hi - x) * (x - lo), 0)) / (2 * np.pi * q * np.maximum(x, 1e-30))
    f = dens * dx
    cdf = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])
    cdf /= cdf[-1]
    u = (np.arange(n) + 0.5) / n
    ev = np.interp(u, cdf, x)
    ev = ev / ev.mean()
    return ev[::-1]
