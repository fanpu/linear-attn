"""Films for Invariant Tori (M3).

sweep      the eps sweep: 26 measured keyframes eps = 0, 0.02, ..., 0.5 (same 12 starts, t <= 1000 of each orbit,
           cache/stereo_sweep_fine.npz), glow hairlines, one camera.  Declared tween: between keyframes the two
           keyframe renders are cross-faded (opacity only) over TWEEN frames while the camera spins slowly about
           the torus axis; orbits are never interpolated.  Colour: regular orbits (lambda <= 5e-3 at T = 1e4) in
           plasma(torus index) as tori_glow; chaotic orbits (lambda > 5e-3) in pale cyan (declared).
turntable  sea and islands cutaway, 360 degrees about the vertical axis; the clip plane turns with the camera
           (always removing the near half, declared), fog/sheet/tubes as sea_islands_cutaway.

python render_film.py [sweep|turntable] [--size 1080]
"""
import argparse
import os

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import render_lib as L
from render_lib import r3d

HOLD, TWEEN, FPS = 10, 6, 24
CHAOS_RGB = np.array([0.55, 0.85, 1.0])


def label(img, lines, size):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', max(14, size // 42))
    except OSError:
        font = ImageFont.load_default()
    y = size // 40
    for ln in lines:
        dr.text((size // 40, y), ln, fill=(225, 222, 214), font=font); y += int(size / 32)
    return im


def sweep(size, test=False):
    d = np.load(os.path.join(L.CACHE, 'stereo_sweep_fine.npz')); X, eps = d['X'], d['eps']
    lam = np.load(os.path.join(L.CACHE, 'stereo_sweep.npz'))['lyap']
    c, a, b = L.tori_frame()
    allp = X[:, :, ::50].reshape(-1, 3)
    lo_p, hi_p = np.percentile(allp, 0.5, axis=0), np.percentile(allp, 99.5, axis=0)
    ext = 0.85 * np.linalg.norm(hi_p - lo_p)
    out = os.path.join(L.CACHE, 'frames_sweep' + ('_test' if test else '')); os.makedirs(out, exist_ok=True)
    nfr = len(eps) * HOLD + (len(eps) - 1) * TWEEN
    plan = []                                   # (frame, k0, k1, s)
    f = 0
    for k in range(len(eps)):
        for _ in range(HOLD):
            plan.append((f, k, k, 0.0)); f += 1
        if k < len(eps) - 1:
            for j in range(TWEEN):
                plan.append((f, k, k + 1, (j + 1) / (TWEEN + 1))); f += 1

    def cam_at(fr):
        spin = 60.0 * fr / (nfr - 1)                                       # declared camera tween: 60 deg total
        b2 = np.cross(a, b); s = np.radians(spin); bb = np.cos(s) * b + np.sin(s) * b2
        e = np.cos(np.radians(52)) * a + np.sin(np.radians(52)) * bb
        f_ = -e; r = np.cross(f_, a); r /= np.linalg.norm(r); u = np.cross(r, f_)
        mid = 0.5 * (lo_p + hi_p)
        return r3d.Camera(tuple(mid + 40 * e), tuple(mid), up=tuple(u), width=size, height=size, ortho_height=ext)

    def keyframe(k, cam):
        acc = torch.zeros(size, size, 3, device=L.dev())
        for i in range(X.shape[1]):
            P = L.T(X[k, i])
            col = CHAOS_RGB if lam[k, i] > 5e-3 else np.array(matplotlib.colormaps['plasma'](0.15 + 0.8 * i / 11)[:3])
            g = r3d.splat_additive(P, cam, sigma_px=0.55 * size / 1024, weight=0.01 * 0.5 / cam.pixel_scale())
            acc += g[..., None] * L.T(col)
        return acc

    with L.Timer(f'film sweep {size} {nfr} frames'):
        for fr, k0, k1, s in plan:
            path = os.path.join(out, f'{fr:05d}.png')
            if os.path.exists(path):
                continue
            cam = cam_at(fr)
            acc = keyframe(k0, cam) if s == 0 else (1 - s) * keyframe(k0, cam) + s * keyframe(k1, cam)
            img = r3d.glow_tonemap(acc, 0.22 * (size / 1024) ** 1.2) + L.T(L.NIGHT)
            k = k0 if s < 0.5 else k1
            lines = [f'eps = {eps[k]:.2f}    H = 2.8', f'chaotic (lambda > 5e-3, T = 1e4): {int((lam[k] > 5e-3).sum())} of 12',
                     'first to go chaotic: the innermost torus (eps = 0.04)' if eps[k] >= 0.035 else 'all 12 orbits on invariant tori']
            label(L.to_img(img).numpy(), lines, size).save(path)
    if test:
        return
    r3d.write_film(os.path.join(out, '%05d.png'), os.path.join(L.GAL, 'film_eps_sweep.mp4'), fps=FPS,
                   gif=os.path.join(L.GAL, 'film_eps_sweep.gif'), gif_width=480)


def turntable(size, n=240):
    import render_sea as S
    out = os.path.join(L.CACHE, 'frames_turntable'); os.makedirs(out, exist_ok=True)
    with L.Timer(f'film turntable {size} {n} frames'):
        for fr in range(n):
            path = os.path.join(out, f'{fr:05d}.png')
            if os.path.exists(path):
                continue
            az = S.AZ + 360.0 * fr / n
            img = S.render(size, 'raw256', 0.008, True, az=az, save=False, **S.CUT)
            r3d.save_png(path, L.to_img(img))
    r3d.write_film(os.path.join(out, '%05d.png'), os.path.join(L.GAL, 'film_turntable_sea.mp4'), fps=FPS,
                   gif=os.path.join(L.GAL, 'film_turntable_sea.gif'), gif_width=420)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='*', default=['sweep'])
    ap.add_argument('--size', type=int, default=1080)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--hold', type=int, default=HOLD)
    ap.add_argument('--tween', type=int, default=TWEEN)
    a = ap.parse_args()
    HOLD, TWEEN = a.hold, a.tween
    torch.set_num_threads(4)
    if 'sweep' in a.what:
        sweep(a.size, a.test)
    if 'turntable' in a.what:
        turntable(a.size)
