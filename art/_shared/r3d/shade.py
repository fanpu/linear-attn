"""Form lighting. Brightness from these functions shows shape only; data lives in colour and position (spec §0.3)."""
import math

import torch

from ._hash import hash01


def _unit(v, like):
    v = torch.as_tensor(v, dtype=like.dtype, device=like.device)
    return v / v.norm()


def lambert(normal, light_dir, ambient=0.2):
    L = _unit(light_dir, normal)
    return ambient + (1 - ambient) * (normal * L).sum(-1).clamp_min(0)


def hemisphere_dirs(n_rays, seed=0, dtype=torch.float64, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    u0 = torch.rand(1, generator=g, dtype=torch.float64).item()
    i = torch.arange(n_rays, dtype=torch.float64) + 0.5
    r = torch.sqrt(i / n_rays)
    phi = 2 * math.pi * (i * (math.sqrt(5) - 1) / 2 + u0)
    d = torch.stack([r * torch.cos(phi), r * torch.sin(phi), torch.sqrt((1 - r * r).clamp_min(0))], 1)
    return d.to(device, dtype)


def _frame(n):
    """Branchless orthonormal basis (Duff et al. 2017): continuous except across n_z = 0 (hairy ball)."""
    sgn = torch.where(n[:, 2] >= 0, 1.0, -1.0).to(n.dtype)
    a = -1.0 / (sgn + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    t = torch.stack([1 + sgn * n[:, 0] ** 2 * a, sgn * b, -sgn * n[:, 0]], 1)
    return t, torch.stack([b, sgn + n[:, 1] ** 2 * a, -n[:, 1]], 1)


def ambient_occlusion(pos, normal, occluded, n_rays=16, radius=1.0, seed=0, bias=1e-4):
    """Fraction of cosine-weighted hemisphere rays that escape within `radius` (1 = open).

    Each point gets its own Cranley-Patterson shift of the stratified pattern and its own rotation about the normal,
    keyed on (point index, seed), so the error is per-point noise rather than bands shared by neighbours."""
    N, dt, dev = pos.shape[0], pos.dtype, pos.device
    idx = torch.arange(N, device=dev)
    shift = hash01(idx, seed, 0, dt)[:, None]                                   # (N,1) radial stratum shift
    spin = hash01(idx, seed, 1, dt)[:, None]                                    # (N,1) rotation about the normal
    i = torch.arange(n_rays, dtype=dt, device=dev)[None]
    u = torch.frac((i + 0.5) / n_rays + shift)
    r = torch.sqrt(u)
    phi = 2 * math.pi * torch.frac(i * (math.sqrt(5) - 1) / 2 + spin)
    lx, ly, lz = r * torch.cos(phi), r * torch.sin(phi), torch.sqrt((1 - u).clamp_min(0))
    t, b = _frame(normal)
    w = lx[..., None] * t[:, None] + ly[..., None] * b[:, None] + lz[..., None] * normal[:, None]
    o = (pos + bias * normal)[:, None, :].expand(N, n_rays, 3).reshape(-1, 3)
    tmax = torch.full((N * n_rays,), float(radius), dtype=dt, device=dev)
    hit = occluded(o, w.reshape(-1, 3), tmax).reshape(N, n_rays)
    return 1 - hit.to(dt).mean(1)


def hard_shadow(pos, normal, light_dir, occluded, bias=1e-4, tmax=1e9):
    L = _unit(light_dir, pos)
    facing = (normal * L).sum(-1) > 0
    o = pos + bias * normal
    hit = occluded(o, L.expand_as(pos).contiguous(), torch.full((pos.shape[0],), float(tmax), dtype=pos.dtype, device=pos.device))
    return (facing & ~hit).to(pos.dtype)
