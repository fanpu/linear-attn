"""Orthographic globe sampling of cached GP draws (spectra + seed => exact band-limited sample)."""
import numpy as np
from common import *

SEED_GLOBE = 11
_sp = None


def spec(name, lmax):
    global _sp
    if _sp is None:
        _sp = np.load("cache/spectra.npz")
    names = list(_sp["names"])
    return _sp["C"][names.index(name)][: lmax + 1]


def rotation(lon, lat, roll=0.0):
    """Rotation taking view-space (x right, y up, z towards viewer) to world."""
    lon, lat = np.radians(lon), np.radians(lat)
    # viewer direction (sub-observer point)
    c = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    up0 = np.array([0, 0, 1.0])
    e1 = np.cross(up0, c); e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    r = np.radians(roll)
    e1, e2 = np.cos(r) * e1 + np.sin(r) * e2, -np.sin(r) * e1 + np.cos(r) * e2
    return np.stack([e1, e2, c], 1)   # columns


def globe_points(n, R):
    """n x n image, unit disk mapped to pixels; returns mask, world dirs (masked), z (view)."""
    s = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(s, -s)
    r2 = X**2 + Y**2
    mask = r2 < 1
    Z = np.sqrt(np.clip(1 - r2, 0, 1))
    V = np.stack([X, Y, Z], -1)[mask]
    W = V @ R.T
    return mask, W, Z


def globe_field(alm, lmax, n, R, nthreads=6):
    mask, W, Z = globe_points(n, R)
    th, ph = vec_to_thetaphi(W)
    vals = synth(alm, lmax, th, ph, nthreads=nthreads)
    f = np.full((n, n), np.nan)
    f[mask] = vals
    return f, mask, Z


def make_alm(name, lmax, seed=SEED_GLOBE, z=None):
    if z is None:
        z = white_alm(lmax, seed)
    return alm_from_white(z, spec(name, lmax), lmax)


def sphere_median(alm, lmax, nthreads=6):
    """Median over the sphere from an equal-area Fibonacci point set."""
    m = 400000
    i = np.arange(m) + 0.5
    th = np.arccos(1 - 2 * i / m)
    ph = (np.pi * (1 + 5**0.5) * i) % (2 * np.pi)
    return float(np.median(synth(alm, lmax, th, ph, nthreads=nthreads)))
