"""Nested tori at eps = 0, H = 2.8: 12 orbits in the stereographic chart, rendered three ways.

  plaster   splat_spheres tubes, cutaway by a clip plane through the core, tinted by torus index,
            AO from a voxel occupancy of the tubes + one raking light (form only)
  glow      additive Gaussian hairlines on a dark ground, colour = torus index (no occlusion)
  plotter   hidden-line SVG (tube depth buffer), one pen colour per torus, same cutaway as plaster

python render_tori.py [plaster|glow|plotter|all] [--size N] [--tw T]
"""
import argparse
import os

import matplotlib
import numpy as np
import torch

import render_lib as L
from render_lib import r3d

AZ, EL = -62.0, 24.0


def load(tw):
    d = np.load(os.path.join(L.CACHE, 'stereo_eps0_fine.npz'))
    n = int(round(tw / 0.01)) + 1
    return [d['X'][i, :n] for i in range(d['X'].shape[0])]


def palette(n):
    # Declared: torus index 0 (seeded nearest the core periodic orbit) ... 11 (nearest the section fold),
    # cmcrameri-like 'viridis' trimmed at the bright end so the outermost stays visible on paper.
    return matplotlib.colormaps['viridis'](np.linspace(0.0, 0.9, n))[:, :3]


def scene(polys=None):
    return L.tori_frame()


def camera(polys, keep_masks, size, tilt=52.0, spin=0.0, margin=1.08):
    """Orthographic camera at `tilt` degrees from the torus axis (declared view), framed on the kept points."""
    c, a, b = scene(polys)
    b2 = np.cross(a, b)
    s = np.radians(spin); bb = np.cos(s) * b + np.sin(s) * b2
    e = np.cos(np.radians(tilt)) * a + np.sin(np.radians(tilt)) * bb
    f = -e; r = np.cross(f, a); r /= np.linalg.norm(r); u = np.cross(r, f)
    pts = np.concatenate([P[k][::20] for P, k in zip(polys, keep_masks)])
    pr, pu = (pts - c) @ r, (pts - c) @ u
    cx, cy = 0.5 * (pr.max() + pr.min()), 0.5 * (pu.max() + pu.min())
    ext = margin * max(pr.max() - pr.min(), pu.max() - pu.min())
    tgt = c + cx * r + cy * u
    return r3d.Camera(tuple(tgt + 40 * e), tuple(tgt), up=tuple(u), width=size, height=size, ortho_height=ext), e, bb


def wedge_keep(P, c, a, bb, half_deg=50.0):
    """Declared cutaway: remove the wedge of half-angle `half_deg` about the torus axis a, centred on the
    camera-side direction bb (identical for all tori), exposing the nested cross-sections on its two faces."""
    q = P - c; q = q - np.outer(q @ a, a)
    ang = np.degrees(np.arctan2(q @ np.cross(a, bb), q @ bb))
    return np.abs(ang) > half_deg


def dense(P, spacing):
    p, _ = r3d.sample_polyline(L.T(P), spacing)
    return p.cpu().numpy()


def plaster(size, tw, R):
    polys = [dense(P, 0.5 * R) for P in load(tw)]
    c, a, b = scene(polys)
    keeps = [wedge_keep(P, c, a, b) for P in polys]
    cam, e, bb = camera(polys, keeps, size)
    with L.Timer(f'tori plaster {size} tw={tw}'):
        pts = L.T(np.concatenate([P[k] for P, k in zip(polys, keeps)]))
        idx = L.T(np.concatenate([np.full(k.sum(), i) for i, k in enumerate(keeps)]))
        hit = r3d.splat_spheres(pts, R, cam, attrs=idx[:, None])
        m = hit['mask']
        pos = L.surface_pos(cam, hit['depth'])[m]; nrm = hit['normal'][m]
        lo = pts.min(0).values.cpu().numpy() - 3 * R; hi = pts.max(0).values.cpu().numpy() + 3 * R
        occ, hvox = L.occupancy_occluder(pts, 0.7 * R, lo, hi, 400)
        ao = r3d.ambient_occlusion(pos, nrm, occ, n_rays=32, radius=0.15, bias=1.2 * hvox)
        light = tuple(0.8 * a + 0.9 * np.cross(a, bb) + 0.5 * e)          # one raking light, declared
        sh = r3d.hard_shadow(pos, nrm, light, occ, bias=1.2 * hvox)
        lum = (0.5 + 0.5 * r3d.lambert(nrm, light, ambient=0.0) * (0.5 + 0.5 * sh)) * (0.45 + 0.55 * ao)
        pal = L.T(palette(len(polys)))
        tint = 0.55 * torch.ones(3, device=L.dev()) + 0.45 * pal[hit['attr'][m][:, 0].long()]
        img = torch.ones(size, size, 3, device=L.dev()) * L.T(L.PAPER * 0.9)
        img[m] = (tint * lum[:, None]).clamp(0, 1) * 1.08
    r3d.save_png(os.path.join(L.GAL, f'tori_plaster{"_" + str(size) if size < 2000 else ""}.png'), L.to_img(img))


def glow(size, tw, R):
    polys = [dense(P, 0.5 * R) for P in load(tw)]
    c, a, b = scene(polys)
    keeps = [wedge_keep(P, c, a, b) for P in polys]
    cam, e, bb = camera(polys, keeps, size)          # same view and framing as the plaster cutaway
    pal = palette(len(polys))
    with L.Timer(f'tori glow {size} tw={tw}'):
        acc = torch.zeros(size, size, 3, device=L.dev())
        spacing = 0.5 * R
        for i, (P, k) in enumerate(zip(polys, keeps)):
            g = r3d.splat_additive(L.T(P[k]), cam, sigma_px=0.55 * max(1.0, size / 1024),
                                   weight=spacing / cam.pixel_scale())
            col = matplotlib.colormaps['plasma'](0.15 + 0.8 * i / (len(polys) - 1))[:3]   # declared: index -> plasma
            acc += g[..., None] * L.T(col)
        img = r3d.glow_tonemap(acc, 0.22 * max(1.0, size / 1024) ** 1.2) + L.T(L.NIGHT)   # declared exposure
    r3d.save_png(os.path.join(L.GAL, f'tori_glow{"_" + str(size) if size < 2000 else ""}.png'), L.to_img(img))


def plotter(size, tw, R):
    polys = [dense(P, 0.5 * R) for P in load(tw)]
    c, a, b = scene(polys)
    keeps = [wedge_keep(P, c, a, b) for P in polys]
    cam, e, bb = camera(polys, keeps, size)
    pal = palette(len(polys))
    with L.Timer(f'tori plotter {size} tw={tw}'):
        pts = L.T(np.concatenate([P[k] for P, k in zip(polys, keeps)]))
        depth = r3d.splat_spheres(pts, R, cam)['depth']
        groups = []
        for i, (P, k) in enumerate(zip(polys, keeps)):
            edges = np.flatnonzero(np.diff(np.r_[0, k.astype(int), 0]))     # split strokes at the wedge
            runs = []
            for a0, b0 in zip(edges[::2], edges[1::2]):
                if b0 - a0 >= 2:
                    runs += r3d.visible_runs(L.T(P[a0:b0]), cam, depth, eps=3 * R)
            groups.append((L.hexcol(pal[i] * 0.42), 1.1 * size / 2400, [np.asarray(r) for r in runs]))
        name = os.path.join(L.GAL, f'tori_plotter{"_" + str(size) if size < 2000 else ""}.svg')
        L.write_svg_multi(name, groups, size, size, background=L.hexcol(L.PAPER))
    return name


def slice_plate(size):
    """Exact meridional slice (compute_slice.py): every crossing of the plane spanned by the doughnut axis a and b,
    t <= 1e5, as ink dots coloured by torus index (same viridis as the plaster tint). Declared ink coverage."""
    d = np.load(os.path.join(L.CACHE, 'slice_eps0.npz'))
    u, v, k = d['u'], d['v'], d['torus']
    pal = palette(12)
    lo = np.array([u.min(), v.min()]); hi = np.array([u.max(), v.max()])
    ctr = 0.5 * (lo + hi); w = 1.06 * (hi - lo).max(); lo, hi = ctr - w / 2, ctr + w / 2
    with L.Timer(f'tori slice {size}'):
        ij = np.floor((np.stack([u, v], 1) - lo) / (hi - lo) * size).astype(int).clip(0, size - 1)
        flat = (size - 1 - ij[:, 1]) * size + ij[:, 0]
        W = np.bincount(flat, minlength=size * size).astype(np.float32)
        CW = np.stack([np.bincount(flat, pal[k][:, j] * 0.8, minlength=size * size) for j in range(3)], 1)
        from scipy.ndimage import gaussian_filter
        sig = max(0.7, size / 2400 * 1.4)                                   # declared dot footprint (pixels)
        W = gaussian_filter(W.reshape(size, size), sig).ravel() * 2 * np.pi * sig ** 2
        CW = np.stack([gaussian_filter(CW[:, j].reshape(size, size), sig).ravel() * 2 * np.pi * sig ** 2 for j in range(3)], 1)
        cov = (1 - np.exp(-0.9 * W))[:, None]
        col = CW / np.maximum(W, 1e-9)[:, None]
        img = (L.PAPER[None] * (1 - cov) + col * cov).reshape(size, size, 3)
    r3d.save_png(os.path.join(L.GAL, f'tori_slice{"_" + str(size) if size < 2000 else ""}.png'), torch.tensor(img))
    print('slice extent (chart units)', w, 'crossings', len(u))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='*', default=['all'])
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--tw', type=float, default=1000.0)
    ap.add_argument('--R', type=float, default=0.01)
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(L.GAL, exist_ok=True)
    if 'plaster' in a.what or 'all' in a.what:
        plaster(a.size, a.tw, a.R)
    if 'glow' in a.what or 'all' in a.what:
        glow(a.size, a.tw, a.R)
    if 'plotter' in a.what or 'all' in a.what:
        plotter(a.size, a.tw, a.R)
    if 'slice' in a.what or 'all' in a.what:
        slice_plate(a.size)
