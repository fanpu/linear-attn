"""Shared helpers for Invariant Tori renders (reads cache/ only). Uses the shared renderer art/_shared/r3d.

Chart (declared in every caption): energy level set H = 2.8 of the two-player RPS replicator dynamics,
radial projection to S^3 in zero-mean logit coordinates, stereographic projection from the pole in
cache/pole.json.  World (x, y, z) = chart (X0, X1, X2), z up.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/home/fzeng/ml/research/art/_shared')
import r3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
GAL = os.path.join(HERE, 'gallery')
PAPER = np.array([0.953, 0.937, 0.902])
INK = np.array([0.12, 0.11, 0.10])
NIGHT = np.array([0.02, 0.02, 0.03])
COPPER = np.array([0.80, 0.47, 0.28])
DT = torch.float32


def dev():
    """Device for splatting and everything except volume marching: CPU (r3d splats crash on CUDA until the
    Camera.project fix lands, controller note 2026-09-15)."""
    return 'cpu'


def vdev():
    """Device for render_volume (set R3D_VOL_DEVICE=cuda inside a gpu1.sh job)."""
    return os.environ.get('R3D_VOL_DEVICE', 'cpu')


def volume(data, lo, hi, cam, tf, step, depth=None, clip=()):
    d = data.to(vdev())
    rgb, a = r3d.render_volume(d, lo, hi, cam, tf, step, depth=None if depth is None else depth.to(vdev()),
                               clip=clip, device=vdev())
    return rgb.to(dev()), a.to(dev())


def T(a, dtype=DT):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=dev())


def pole():
    return json.load(open(os.path.join(CACHE, 'pole.json')))


def tube_points(polylines, spacing, clip=(), attr=None):
    """Resample polylines (list of (m,3) numpy) at `spacing`; drop samples removed by clip planes.
    Returns centres (N,3) torch and per-point attribute (N,) (the polyline index unless attr given)."""
    pts, ids = [], []
    for i, P in enumerate(polylines):
        p, _ = r3d.sample_polyline(T(P), spacing)
        if clip:
            p = p[r3d.clip_keep(p, clip)]
        pts.append(p); ids.append(torch.full((len(p),), float(i if attr is None else attr[i]), dtype=DT, device=dev()))
    return torch.cat(pts), torch.cat(ids)


def surface_pos(cam, depth):
    o, d = cam.rays(device=dev(), dtype=DT)
    t = cam.depth_to_t(depth, d)
    return o + t[..., None] * d


def occupancy_occluder(centres, radius, lo, hi, n):
    """Voxel occluder made from tube sample centres (cells within `radius` of a centre, approximated by
    splatting each centre and its 6 axis neighbours at radius).  Used for AO / shadows only (form, not data)."""
    lo_t, hi_t = T(lo), T(hi)
    h = (hi_t - lo_t) / (n - 1)
    solid = torch.zeros((n, n, n), dtype=torch.bool, device=dev())
    offs = [(0, 0, 0)] + [tuple(radius * np.eye(3)[k] * s) for k in range(3) for s in (-1, 1)]
    for off in offs:
        q = torch.round((centres + T(off) - lo_t) / h).long()
        ok = ((q >= 0) & (q < n)).all(1)
        q = q[ok]
        solid[q[:, 2], q[:, 1], q[:, 0]] = True
    return r3d.voxel_occluder(solid, tuple(lo), tuple(hi)), float(h.max())


def to_img(x):
    return x.detach().float().cpu()


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time(); return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t
        print(f'[{self.name}] {self.dt:.1f}s', flush=True)
        with open(os.path.join(HERE, 'logs', 'render_times.log'), 'a') as fh:
            fh.write(f'{time.strftime("%H:%M:%S")} {self.name} {self.dt:.1f}s volume_device={vdev()}\n')


def write_svg_multi(path, groups, width, height, background=None):
    """groups: list of (stroke_hex, stroke_width, [polylines (M,2)])."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    if background:
        out.append(f'<rect width="{width}" height="{height}" fill="{background}"/>')
    for stroke, sw, lines in groups:
        out.append(f'<g fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">')
        for pl in lines:
            if len(pl) < 2:
                continue
            out.append('<polyline points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in pl) + '"/>')
        out.append('</g>')
    out.append('</svg>')
    with open(path, 'w') as fh:
        fh.write('\n'.join(out))


def hexcol(rgb):
    r, g, b = (int(round(255 * float(c))) for c in rgb[:3])
    return f'#{r:02x}{g:02x}{b:02x}'
