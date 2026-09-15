"""Form lighting. Brightness from these functions shows shape only; data lives in colour and position (spec §0.3)."""
import math

import torch


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
    ex = torch.tensor([1.0, 0, 0], dtype=n.dtype, device=n.device)
    ey = torch.tensor([0, 1.0, 0], dtype=n.dtype, device=n.device)
    a = torch.where((n[:, :1].abs() < 0.9), ex, ey)
    t = torch.linalg.cross(n, a)
    t = t / t.norm(dim=-1, keepdim=True)
    return t, torch.linalg.cross(n, t)


def ambient_occlusion(pos, normal, occluded, n_rays=16, radius=1.0, seed=0, bias=1e-4):
    N = pos.shape[0]
    loc = hemisphere_dirs(n_rays, seed, pos.dtype, pos.device)
    t, b = _frame(normal)
    w = loc[None, :, 0:1] * t[:, None] + loc[None, :, 1:2] * b[:, None] + loc[None, :, 2:3] * normal[:, None]
    o = (pos + bias * normal)[:, None, :].expand(N, n_rays, 3).reshape(-1, 3)
    tmax = torch.full((N * n_rays,), float(radius), dtype=pos.dtype, device=pos.device)
    hit = occluded(o, w.reshape(-1, 3), tmax).reshape(N, n_rays)
    return 1 - hit.to(pos.dtype).mean(1)


def hard_shadow(pos, normal, light_dir, occluded, bias=1e-4, tmax=1e9):
    L = _unit(light_dir, pos)
    facing = (normal * L).sum(-1) > 0
    o = pos + bias * normal
    hit = occluded(o, L.expand_as(pos).contiguous(), torch.full((pos.shape[0],), float(tmax), dtype=pos.dtype, device=pos.device))
    return (facing & ~hit).to(pos.dtype)
