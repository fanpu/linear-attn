"""Slice plate: the existing game-chaos Poincare plate (eps = 0.5, H = 2.8) beside the same crossings drawn where
they live in the stereographic chart, on the translucent section membrane g = 0 (dg/dt > 0).

The dots on the right are the game-chaos kam_eps0.50 section crossings (395 orbits), mapped through the chart; the
left panel is game-chaos/gallery/poincare_ink_eps0.50.png unchanged (downscaled).

python render_plates.py [--size N]
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import chart as C
import render_lib as L
from render_lib import r3d

GC = os.path.join(L.HERE, '..', 'game-chaos')


def section_dots():
    k = np.load(os.path.join(GC, 'cache', 'kam_eps0.50.npz'))
    sec, ns = k['sec'], k['nsec']
    P = np.concatenate([sec[o, :ns[o]] for o in range(len(ns))]).astype(np.float64)
    x = P[:, :3] / P[:, :3].sum(1, keepdims=True); y = P[:, 3:] / P[:, 3:].sum(1, keepdims=True)   # float32 -> renormalise
    ch = C.Chart(np.array(json.load(open(os.path.join(L.CACHE, 'pole.json')))['pole']), 2.8)
    return ch.forward_strategies(x, y), x, y


KEY = 'cmc.batlow'


def key_colours(x, y):
    """Declared colour key: position along the plate's long diagonal, t = rank of (x_R + y_P), through cmc.batlow."""
    import cmcrameri  # noqa: F401
    import matplotlib
    v = x[:, 0] + y[:, 1]
    t = np.argsort(np.argsort(v)) / (len(v) - 1)
    return matplotlib.colormaps[KEY](0.05 + 0.9 * t)[:, :3]


def ink(W, CW, gain):
    """Declared ink idiom (as game-chaos plates): coverage = 1 - exp(-gain * hits); colour = mean key colour."""
    cov = 1 - torch.exp(-gain * W)[..., None]
    col = CW / W.clamp_min(1e-12)[..., None]
    return L.T(L.PAPER) * (1 - cov) + col * cov


def flat_view(size, x, y, cols):
    """The plate's own axes: x_R horizontal, y_P vertical, same bounds rule as game-chaos render_poincare.bounds."""
    pts = np.stack([x[:, 0], y[:, 1]], 1)
    lo = pts.min(0) - 0.03; hi = pts.max(0) + 0.03; c = 0.5 * (lo + hi); w = (hi - lo).max()
    lo, hi = c - w / 2, c + w / 2
    ij = np.floor((pts - lo) / (hi - lo) * size).astype(int).clip(0, size - 1)
    flat = (size - 1 - ij[:, 1]) * size + ij[:, 0]
    W = np.bincount(flat, minlength=size * size).astype(np.float32)
    CW = np.stack([np.bincount(flat, cols[:, k], minlength=size * size) for k in range(3)], 1).astype(np.float32)
    return ink(L.T(W.reshape(size, size)), L.T(CW.reshape(size, size, 3)), 0.9 * (size / 3000) ** 2)


def membrane_view(size, dots, cols):
    dz = np.load(os.path.join(L.CACHE, 'density_eps05.npz'))   # frame on the fog box used by the sea renders
    lo_b, hi_b = dz['lo'], dz['hi']
    core = np.all((dots >= lo_b) & (dots <= hi_b), 1)
    D = dots[core]
    c = D.mean(0); w, v = np.linalg.eigh(np.cov((D - c).T))
    nrm = v[:, 0] if v[2, 0] >= 0 else -v[:, 0]                    # view along the sheet's least-variance direction
    e = nrm + 0.35 * v[:, 1]; e /= np.linalg.norm(e)               # declared: slight oblique so the curvature reads
    up = v[:, 2] - (v[:, 2] @ e) * e; up /= np.linalg.norm(up)
    f = -e; r = np.cross(f, up)
    pr, pu = (D - c) @ r, (D - c) @ up
    ext = 1.04 * max(np.ptp(pr), np.ptp(pu))
    tgt = c + 0.5 * (pr.max() + pr.min()) * r + 0.5 * (pu.max() + pu.min()) * up
    cam = r3d.Camera(tuple(tgt + 40 * e), tuple(tgt), up=tuple(up), width=size, height=size, ortho_height=ext)
    with L.Timer(f'plate membrane {size}'):
        P = L.T(dots); Cc = L.T(cols)
        W = r3d.splat_additive(P, cam, sigma_px=0.5)
        CW = r3d.splat_additive(P, cam, sigma_px=0.5, color=Cc)
        img = ink(W, CW, 3.0 * 0.9 * (size / 3000) ** 2)             # declared: 3x the flat panel's ink gain
    return img, dict(n_dots=int(len(dots)), view=e.tolist())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=768)
    a = ap.parse_args()
    torch.set_num_threads(4)
    dots, x, y = section_dots()
    cols = key_colours(x, y)
    S = a.size
    right, info = membrane_view(S, dots, cols)
    mid = flat_view(S, x, y, cols)
    left = Image.open(os.path.join(GC, 'gallery', 'poincare_ink_eps0.50.png')).convert('RGB').resize((S, S), Image.LANCZOS)
    left = torch.tensor(np.asarray(left) / 255.0, dtype=torch.float32)
    gap = torch.ones(S, S // 24, 3) * L.T(L.PAPER)
    sfx = "_" + str(S) if S < 2000 else ""
    r3d.save_png(os.path.join(L.GAL, f'plate_section_flat_vs_membrane{sfx}.png'), torch.cat([left, gap, mid, gap, right], 1))
    print(info)
