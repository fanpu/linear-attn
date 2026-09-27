"""Generic lit-solid heightfield render for wild-landscapes, built on dense-landscape's renderer v2
(solid.py: bicubic prep, horizon AO, soft shadows, Blinn-Phong/Schlick, ACES; hfk.py: Triton march).

render(F, albedo_srgb, ...) -> uint8 image.
  F        (R, R) field in data units (e.g. log10 loss), row = beta (y), col = alpha (x), drawn on
           the world square [-1, 1]^2 whatever its data extent (declared per scene).
  K        world units per data unit (height exaggeration, declared per scene), Z0 = base level.
  up       bicubic upsampling factor (1 = bilinear sheet through the samples, nothing invented).
"""
import math, os, sys
import numpy as np
import torch
sys.path.insert(0, "/home/fzeng/ml/research/art/dense-landscape")
sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import r3d
import hf, solid


def _uv_safe(Hm, x, y):
    """hf._uv clamps to R - 1 - 1e-4, which float32 rounds to R - 1 once R > ~2049 (so idx + 1 runs
    off the end).  Same bilinear coordinates, with the cell index clamped to R - 2 instead."""
    R = Hm.shape[0]
    u = ((x - hf.LO) / (hf.HI - hf.LO) * (R - 1)).clamp(0, R - 1)
    v = ((y - hf.LO) / (hf.HI - hf.LO) * (R - 1)).clamp(0, R - 1)
    i0 = u.floor().long().clamp(0, R - 2); j0 = v.floor().long().clamp(0, R - 2)
    return i0, j0, u - i0, v - j0


hf._uv = _uv_safe


def srgb2lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lut(name, n=1024):
    import palettes as P
    return np.asarray(P.get(name)(np.linspace(0, 1, n)))[:, :3]


def cmap(t, name):
    L = lut(name)
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, np.linspace(0, 1, len(L)), L[:, c]) for c in range(3)], -1)


def make_cam(orbit, W, H):
    tx, ty, tz, rad, az, el, fov = (float(v) for v in orbit.split(",")) if isinstance(orbit, str) else orbit
    return r3d.orbit((tx, ty, tz), rad, az_deg=az, el_deg=el, width=W, height=H, fov_deg=fov)


def prepare(F, K, Z0, up, device):
    Hm, GX, GY = solid.prep(np.asarray(F, np.float64), up, K, Z0, device)
    return Hm, GX, GY


class Scene:
    """Precomputes height, normals, AO and albedo once; shade(cam, key) many times (films)."""

    def __init__(self, F, albedo_srgb, K, Z0=None, up=1, device="cuda", ao_cells=None):
        F = np.asarray(F, np.float64)
        self.Z0 = float(np.nanmin(F)) if Z0 is None else Z0
        self.up = up
        self.Hm, self.GX, self.GY = prepare(F, K, self.Z0, up, device)
        assert torch.isfinite(self.Hm).all() and torch.isfinite(self.GX).all(), "non-finite height field"
        n = self.Hm.shape[0]
        alb = srgb2lin(np.clip(albedo_srgb, 0, 1))
        if alb.shape[0] != n:
            alb = solid.up_colour(alb, n)
        self.alb = torch.tensor(alb, dtype=torch.float32, device=device)
        self.ao = solid.ao_coarse(self.Hm, up) if ao_cells is None else solid.ao_coarse(self.Hm, up, cells=ao_cells)
        self.device = device
        self._sh = {}

    def shadow(self, key, soft):
        k = (tuple(round(v, 5) for v in key), soft)
        if k not in self._sh:
            if len(self._sh) > 2:
                self._sh.clear()
            self._sh[k] = solid.shadow_fast(self.Hm, key, softness=soft, step_texels=0.75)
        return self._sh[k]

    def render(self, orbit, width=1920, aspect=16 / 9, ss=2, key=(200, 30), fill=(20, 35), key_I=3.2, fill_I=0.55,
               ka=0.55, soft=18.0, fog=0.25, ks=0.35, shin=40.0, exposure=1.2, bg=(0, 0, 0), under_gain=0.15,
               amb_rgb=(0.32, 0.36, 0.5), key_rgb=(1.0, 0.93, 0.82), fill_rgb=(0.55, 0.65, 1.0), cam=None, G=None):
        H = int(round(width / aspect))
        cam = cam or make_cam(orbit, width, H)
        kd = solid.light_dir(*key); fd = solid.light_dir(*fill) if fill else None
        sh = self.shadow(kd, soft)
        if G is None:
            G = solid.gbuffer(self.Hm, cam, ss=ss, step_texels=0.6, device=self.device)
            ok = torch.isfinite(G["p"]).all(-1)
            self.bad_rays = int((G["m"] & ~ok).sum())
            G["m"] = G["m"] & ok
            G["p"] = torch.where(ok[:, None], G["p"], torch.zeros_like(G["p"]))
        img = solid.shade(G, self.Hm, self.GX, self.GY, self.alb, kd, shadow=sh, ao=self.ao, fill=fd, key_I=key_I,
                          fill_I=fill_I, ka=ka, fog=fog, ks=ks, shininess=shin, bg=bg, under_gain=under_gain,
                          amb_rgb=amb_rgb, key_rgb=key_rgb, fill_rgb=fill_rgb)
        return solid.tonemap(img, exposure), G


def save(arr, path, q=None):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im = Image.fromarray(arr)
    if path.endswith(".jpg"):
        im.save(path, quality=q or 92, subsampling=0)
    else:
        im.save(path, compress_level=3)
