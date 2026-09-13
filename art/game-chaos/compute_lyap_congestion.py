"""Lyapunov parameter planes of MWU in the two-agent, two-link congestion game.
  u' = u - s (sigma(u) - y*)      (symmetric start: both agents at the same mixed strategy)
s = eta (a+b) effective step size, y* = equilibrium load of link 1.
Largest Lyapunov exponent per pixel, float64, exact derivative 1 - s sigma'(u).

Usage:
  python compute_lyap_congestion.py plates     # hero + zoom plates  -> cache/cong_*.npz
  python compute_lyap_congestion.py multistab  # attractor coexistence map -> cache/cong_multistab.npz
  python compute_lyap_congestion.py video      # zoom-video frames -> cache/cong_zoom_frames.npy
  python compute_lyap_congestion.py rescheck   # 1x/2x/4x resolution check on one zoom window
"""
import sys
import time

import numpy as np
import torch

from gamelib import DT, gpu_setup, lyap_cong1d, dev

# (name, s_lo, s_hi, y_lo, y_hi, R_s, R_y)
PLATES = [
    ('full', 1.0, 100.0, 0.0, 1.0, 2400, 2400),
    ('z1_crossing', 18.0, 32.0, 0.36, 0.44, 2000, 2000),
    ('z2_hooks', 40.0, 70.0, 0.66, 0.76, 2000, 2000),
    ('z3_shrimp', 29.6, 30.6, 0.3875, 0.3945, 2000, 2000),
    ('z4_shrimp', 23.2, 24.4, 0.4145, 0.4215, 2000, 2000),
    ('z5_shrimp', 24.07, 24.15, 0.4170, 0.4173, 2000, 2000),
    ('z6_chain', 23.92, 23.95, 0.41612, 0.41624, 2000, 2000),
]


def grid(s_lo, s_hi, y_lo, y_hi, Rs, Ry):
    s = torch.linspace(s_lo, s_hi, Rs, dtype=DT)
    y = torch.linspace(y_lo, y_hi, Ry + 2, dtype=DT)[1:-1] if y_lo == 0.0 else torch.linspace(y_lo, y_hi, Ry, dtype=DT)
    Y, S = torch.meshgrid(y, s, indexing='ij')
    return S, Y, s.numpy(), y.numpy()


def plates():
    for name, *win in PLATES:
        t = time.time()
        S, Y, s, y = grid(*win)
        L = lyap_cong1d(S, Y, T0=3000, T1=5000)
        np.savez_compressed(f'cache/cong_{name}.npz', L=L.astype(np.float32), s=s, y=y, T0=3000, T1=5000, u0=0.1)
        print(f'{name}: {time.time()-t:.0f}s  frac(lambda>0)={np.mean(L>0):.3f}  max={L.max():.3f}', flush=True)


@torch.no_grad()
def multistab(Rs=1600, Ry=1600, n0=12, T0=3000, T1=64):
    """For each (s,y*) start from n0 initial logits spread over [-8,8]; count distinct attractors
    (distinct sorted-orbit signatures), and store the Lyapunov exponent spread across starts."""
    S, Y, s, y = grid(1.0, 100.0, 0.0, 1.0, Rs, Ry)
    u0s = np.linspace(-8, 8, n0)
    Ls = []
    sigs = []
    for u0 in u0s:
        L = lyap_cong1d(S, Y, T0=T0, T1=3000, u0=float(u0), chunk=1 << 21)
        Ls.append(L.astype(np.float32))
    Ls = np.stack(Ls)
    spread = Ls.max(0) - Ls.min(0)
    np.savez_compressed('cache/cong_multistab.npz', Ls=Ls, spread=spread.astype(np.float32), s=s, y=y, u0s=u0s)
    print('multistab: frac pixels with lambda spread > 0.05:', np.mean(spread > 0.05))


def zoom_path(nf, start=(1.0, 100.0, 0.0, 1.0), end_center=(24.103, 0.41722), end_w=(0.036, 0.00014)):
    """Per-axis geometric zoom from the full plane into a small shrimp; centre follows the
    same geometric schedule so the target stays in frame."""
    s0c = 0.5 * (start[0] + start[1]); y0c = 0.5 * (start[2] + start[3])
    ws0 = start[1] - start[0]; wy0 = start[3] - start[2]
    out = []
    for i in range(nf):
        f = i / (nf - 1)
        ws = ws0 * (end_w[0] / ws0) ** f; wy = wy0 * (end_w[1] / wy0) ** f
        # centre moves so that the relative position of the target inside the frame decays with the zoom
        a = (ws - end_w[0]) / (ws0 - end_w[0])
        sc = end_center[0] + (s0c - end_center[0]) * a
        yc = end_center[1] + (y0c - end_center[1]) * ((wy - end_w[1]) / (wy0 - end_w[1]))
        out.append((sc - ws / 2, sc + ws / 2, max(yc - wy / 2, 1e-6), min(yc + wy / 2, 1 - 1e-6)))
    return out


def video(nf=360, R=900):
    path = zoom_path(nf)
    frames = np.lib.format.open_memmap('cache/cong_zoom_frames.npy', mode='w+', dtype=np.float32, shape=(nf, R, R))
    t = time.time()
    for i, (a, b, c, d) in enumerate(path):
        S, Y, _, _ = grid(a, b, c, d, R, R)
        frames[i] = lyap_cong1d(S, Y, T0=1000, T1=1500, chunk=R * R)
        if i % 20 == 0:
            print(f'frame {i}/{nf} window {a:.6f}-{b:.6f} x {c:.7f}-{d:.7f}  {time.time()-t:.0f}s', flush=True)
    frames.flush()
    np.save('cache/cong_zoom_path.npy', np.array(path))


def rescheck():
    """Same window at 1x, 2x, 4x resolution (and 2x iterations) for the verification section."""
    win = (23.2, 24.4, 0.4145, 0.4215)
    out = {}
    for R in (500, 1000, 2000):
        S, Y, s, y = grid(*win, R, R)
        out[f'L{R}'] = lyap_cong1d(S, Y, T0=3000, T1=5000).astype(np.float32)
    S, Y, s, y = grid(*win, 500, 500)
    out['L500_long'] = lyap_cong1d(S, Y, T0=6000, T1=20000).astype(np.float32)
    np.savez_compressed('cache/cong_rescheck.npz', **out)
    a, b = out['L500'], out['L500_long']
    print('rescheck: sign agreement 5k vs 20k iterations', np.mean((a > 0) == (b > 0)),
          ' corr', np.corrcoef(a.ravel(), b.ravel())[0, 1])


if __name__ == '__main__':
    gpu_setup()
    for w in sys.argv[1:]:
        globals()[w]()
