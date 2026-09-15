"""Emission-absorption volume rendering. The transfer function is a declared aesthetic choice (spec §0.6)."""
import math

import numpy as np
import torch

from ._hash import hash01
from .camera import ray_box
from .grid import clip_keep, sample_grid


class TransferFunction:
    def __init__(self, cmap, vmin, vmax, opacity, density=1.0, n=1024):
        import matplotlib
        cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        self.lut = torch.tensor(cm(np.linspace(0, 1, n))[:, :3], dtype=torch.float32)
        self.vmin, self.vmax, self.opacity, self.density, self.n = vmin, vmax, opacity, density, n

    def __call__(self, v):
        x = ((v - self.vmin) / (self.vmax - self.vmin)).clamp(0, 1)
        f = x * (self.n - 1)
        i0 = f.floor().long().clamp(max=self.n - 2)
        w = (f - i0)[..., None]
        lut = self.lut.to(v.device, v.dtype)
        return lut[i0] * (1 - w) + lut[i0 + 1] * w, self.density * self.opacity(x)


def lut_tf(rgb, sigma):
    def tf(v):
        i = v.round().long().clamp(0, rgb.shape[0] - 1)
        return rgb.to(v.device, v.dtype)[i], sigma.to(v.device, v.dtype)[i]
    return tf


def march_volume(data, lo, hi, o, d, tf, step, *, tmax=None, clip=(), mode="linear", jitter=True, seed=0,
                 chunk_samples=2 ** 22):
    """Emission-absorption along rays. Segment boundaries sit at tn + (k - u) * step clamped to [tn, tfar], so the
    segment lengths sum exactly to the path. With `jitter` each ray gets its own deterministic u in [0, 1) keyed on
    (ray index, seed), which turns step-aligned wood-grain rings into fine noise; jitter=False uses u = 0."""
    N = o.shape[0]
    rgb = torch.zeros(N, 3, dtype=o.dtype, device=o.device)
    alpha = torch.zeros(N, dtype=o.dtype, device=o.device)
    tn, tfar = ray_box(o, d, lo, hi)
    if tmax is not None:
        tfar = torch.minimum(tfar, tmax)
    idx = (tfar > tn).nonzero().squeeze(1)
    if idx.numel() == 0:
        return rgb, alpha
    nmax = max(1, int(math.ceil(((tfar - tn)[idx].max().item()) / step))) + (1 if jitter else 0)
    per = max(1, chunk_samples // nmax)
    k = torch.arange(nmax, dtype=o.dtype, device=o.device)
    for s in range(0, idx.numel(), per):
        ii = idx[s:s + per]
        t0, t1 = tn[ii, None], tfar[ii, None]
        seg0 = t0 + k * step                                   # unjittered segment starts (R, S)
        if jitter:
            seg0 = seg0 - hash01(ii, seed, 0, o.dtype)[:, None] * step
        start = torch.maximum(seg0, t0)
        ds = torch.minimum(t1 - start, step - (start - seg0)).clamp_min(0)   # exact lengths; 0 past the exit
        ts = torch.minimum(start + 0.5 * ds, t1)
        pts = o[ii, None, :] + ts[..., None] * d[ii, None, :]
        c, sig = tf(sample_grid(data, lo, hi, pts, mode))
        if clip:
            sig = sig * clip_keep(pts, clip)
        a = 1 - torch.exp(-sig * ds)
        T = torch.cumprod(torch.cat([torch.ones_like(a[:, :1]), 1 - a[:, :-1]], 1), 1)
        rgb[ii] = ((T * a)[..., None] * c).sum(1)
        alpha[ii] = 1 - T[:, -1] * (1 - a[:, -1])
    return rgb, alpha


def render_volume(data, lo, hi, cam, tf, step, *, depth=None, clip=(), mode="linear", jitter=True, seed=0, device="cpu"):
    o, d = cam.rays(device=device, dtype=data.dtype if data.is_floating_point() else torch.float32)
    H, W = o.shape[:2]
    tmax = None if depth is None else cam.depth_to_t(depth.to(d), d).reshape(-1)
    rgb, a = march_volume(data, lo, hi, o.reshape(-1, 3), d.reshape(-1, 3), tf, step, tmax=tmax, clip=clip, mode=mode,
                          jitter=jitter, seed=seed)
    return rgb.reshape(H, W, 3), a.reshape(H, W)


def over(front_rgb, front_alpha, back_rgb):
    back = torch.as_tensor(back_rgb, dtype=front_rgb.dtype, device=front_rgb.device)
    return front_rgb + (1 - front_alpha)[..., None] * back
