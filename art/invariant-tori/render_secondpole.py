"""Second-pole comparison plate (spec §5 pitfall: stereographic scale depends on the pole).

The same orbits drawn through two charts that differ only in the pole: the current pole (cache/pole.json, min angle
23.6 deg) and the first pole of M1 (cache/pole_first.json, which sat inside the corner island).  Top row: the 12
eps = 0 tori (t <= 1000); bottom row: the 8 regular eps = 0.5 orbits (t <= 600).  Glow hairlines, same declared
colours and exposure as tori_glow; each panel is framed on its own points (0.5-99.5 percentile box), so apparent
size is not comparable across columns; the per-panel chart-scale range is printed.
python render_secondpole.py [--size 1600]
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import chart as C
import render_lib as L
from render_lib import r3d


def panel(Xs, cols, size, title):
    allp = np.concatenate([X[::20] for X in Xs])
    lo, hi = np.percentile(allp, 0.5, axis=0), np.percentile(allp, 99.5, axis=0)
    c = 0.5 * (lo + hi)
    w, v = np.linalg.eigh(np.cov((allp - allp.mean(0)).T))
    e = np.cos(np.radians(52)) * v[:, 0] + np.sin(np.radians(52)) * v[:, 2]
    up = v[:, 1] - (v[:, 1] @ e) * e; up /= np.linalg.norm(up)
    f = -e; r = np.cross(f, up)
    pr, pu = (allp - c) @ r, (allp - c) @ up
    ext = 1.08 * max(np.percentile(pr, 99.5) - np.percentile(pr, 0.5), np.percentile(pu, 99.5) - np.percentile(pu, 0.5))
    cam = r3d.Camera(tuple(c + 40 * e), tuple(c), up=tuple(up), width=size, height=size, ortho_height=ext)
    acc = torch.zeros(size, size, 3, device=L.dev())
    for X, col in zip(Xs, cols):
        g = r3d.splat_additive(L.T(X), cam, sigma_px=0.55 * size / 1024, weight=0.005 / cam.pixel_scale())
        acc += g[..., None] * L.T(col)
    img = L.to_img(r3d.glow_tonemap(acc, 0.22 * (size / 1024) ** 1.2) + L.T(L.NIGHT)).numpy()
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size // 40)
    for j, ln in enumerate(title):
        dr.text((size // 40, size // 40 + j * size // 30), ln, fill=(225, 222, 214), font=font)
    return np.asarray(im)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--size', type=int, default=1600); a = ap.parse_args()
    torch.set_num_threads(4)
    L0 = np.load(os.path.join(L.CACHE, 'orbits_eps0_fine.npz'))['L'][:, :100001]
    L5 = np.load(os.path.join(L.CACHE, 'orbits_eps05_fine.npz'))['L'][:, :60001]
    kam = np.load(os.path.join(L.CACHE, 'orbits_eps05_fine.npz'))['kam_index']
    c0 = [np.array(matplotlib.colormaps['plasma'](0.15 + 0.8 * i / 11)[:3]) for i in range(12)]
    c5 = [np.array(matplotlib.colormaps['viridis'](0.1 + 0.85 * i / 7)[:3]) for i in range(8)]
    rows = [[], []]; stats = {}
    with L.Timer(f'second pole plate {a.size}'):
        for name in ('pole.json', 'pole_first.json'):
            p = np.array(json.load(open(os.path.join(L.CACHE, name)))['pole'])
            ch = C.Chart(p, 2.8)
            for row, (Ls, cols, what) in enumerate(((L0, c0, 'eps = 0 tori'), (L5, c5, 'eps = 0.5 regular orbits'))):
                Xs = [ch.forward_logits(Lk) for Lk in Ls]
                q = C.to_sphere(C.logits_to_u(Ls.reshape(-1, 4)))
                ang = float(np.degrees(np.arccos(np.clip(q @ p, -1, 1))).min())
                sc = np.concatenate([C.stereo_scale(X) for X in Xs])
                stats[f'{name}:{what}'] = dict(min_angle_deg=ang, scale_min=float(sc.min()), scale_max=float(sc.max()))
                label = 'current pole' if name == 'pole.json' else 'first (M1) pole'
                rows[row].append(panel(Xs, cols, a.size, [f'{what}, {label}',
                                                          f'min angle to pole {ang:.1f} deg, chart scale {sc.min():.2f}-{sc.max():.1f}']))
    gap = np.full((a.size, a.size // 40, 3), 8, np.uint8)
    top = np.concatenate([rows[0][0], gap, rows[0][1]], 1); bot = np.concatenate([rows[1][0], gap, rows[1][1]], 1)
    hgap = np.full((a.size // 40, top.shape[1], 3), 8, np.uint8)
    Image.fromarray(np.concatenate([top, hgap, bot], 0)).save(os.path.join(L.GAL, 'plate_second_pole.png'))
    json.dump(stats, open(os.path.join(L.CACHE, 'second_pole_stats.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1), 'kam', kam.tolist())
