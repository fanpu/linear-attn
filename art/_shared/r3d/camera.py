"""Cameras and rays. World is right-handed with z up; images are (H, W, ...) with row 0 at the top.
Depth buffers everywhere in r3d hold view-axis depth z = (p - eye) . forward."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass
class Camera:
    eye: tuple
    target: tuple
    up: tuple = (0.0, 0.0, 1.0)
    width: int = 512
    height: int = 512
    fov_deg: float | None = None      # vertical field of view; None = orthographic
    ortho_height: float = 2.0         # world height of the view (orthographic only)

    def basis(self):
        f = np.asarray(self.target, float) - np.asarray(self.eye, float)
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(self.up, float))
        if np.linalg.norm(r) < 1e-9:
            raise ValueError("camera up is parallel to the view direction")
        r /= np.linalg.norm(r)
        return f, r, np.cross(r, f)

    def _screen(self):
        W, H = self.width, self.height
        xs = ((torch.arange(W, dtype=torch.float64) + 0.5) / W * 2 - 1) * (W / H)
        ys = 1 - (torch.arange(H, dtype=torch.float64) + 0.5) / H * 2
        return torch.meshgrid(ys, xs, indexing="ij")          # Y (H,W), X (H,W)

    def rays(self, device="cpu", dtype=torch.float32):
        f, r, u = (torch.tensor(v, dtype=torch.float64) for v in self.basis())
        e = torch.tensor(self.eye, dtype=torch.float64)
        Y, X = self._screen()
        H, W = Y.shape
        if self.fov_deg is None:
            s = self.ortho_height / 2
            o = e + X[..., None] * s * r + Y[..., None] * s * u
            d = f.expand(H, W, 3).clone()
        else:
            s = math.tan(math.radians(self.fov_deg) / 2)
            d = f + X[..., None] * s * r + Y[..., None] * s * u
            d = d / d.norm(dim=-1, keepdim=True)
            o = e.expand(H, W, 3).clone()
        return o.to(device, dtype), d.to(device, dtype)

    def project(self, pts):
        pts = torch.as_tensor(pts, dtype=torch.float64)
        f, r, u = (torch.tensor(v, dtype=torch.float64) for v in self.basis())
        rel = pts - torch.tensor(self.eye, dtype=torch.float64)
        z, x, y = rel @ f, rel @ r, rel @ u
        if self.fov_deg is None:
            s = self.ortho_height / 2
            X, Y = x / s, y / s
        else:
            s = math.tan(math.radians(self.fov_deg) / 2)
            X, Y = x / (z * s), y / (z * s)
        col = (X / (self.width / self.height) + 1) / 2 * self.width
        row = (1 - Y) / 2 * self.height
        return torch.stack([col, row], -1), z

    def _fdot(self, d):
        f = torch.tensor(self.basis()[0], dtype=d.dtype, device=d.device)
        return (d * f).sum(-1)

    def t_to_depth(self, t, d):
        return t * self._fdot(d)

    def depth_to_t(self, depth, d):
        return depth / self._fdot(d)

    def pixel_scale(self, depth=None):
        if self.fov_deg is None:
            return self.ortho_height / self.height
        return 2 * depth * math.tan(math.radians(self.fov_deg) / 2) / self.height


def orbit(target, radius, az_deg, el_deg, **cam_kw):
    a, e = math.radians(az_deg), math.radians(el_deg)
    t = np.asarray(target, float)
    eye = t + radius * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return Camera(eye=tuple(eye), target=tuple(t), **cam_kw)


def turntable(n_frames, target, radius, el_deg, az0_deg=0.0, **cam_kw):
    return [orbit(target, radius, az0_deg + 360.0 * i / n_frames, el_deg, **cam_kw) for i in range(n_frames)]


def stereo_pair(cam, separation):
    """Parallel-axis stereo: both eye and target shift sideways by +-separation/2."""
    _, r, _ = cam.basis()
    h = 0.5 * separation * r
    shift = lambda s: replace(cam, eye=tuple(np.asarray(cam.eye) + s * h), target=tuple(np.asarray(cam.target) + s * h))
    return shift(-1.0), shift(1.0)


def ray_box(o, d, lo, hi):
    lo = torch.as_tensor(lo, dtype=o.dtype, device=o.device)
    hi = torch.as_tensor(hi, dtype=o.dtype, device=o.device)
    safe = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    t0, t1 = (lo - o) / safe, (hi - o) / safe
    tnear = torch.minimum(t0, t1).amax(-1).clamp_min(0)
    tfar = torch.maximum(t0, t1).amin(-1)
    return tnear, tfar
