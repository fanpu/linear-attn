"""First crossing of field == level along rays, with bisection refinement on the trilinear field.
Features thinner than `step` can be missed; plates of fractal data should use voxels.py instead (spec §0.5)."""
import math

import torch

from .camera import ray_box
from .grid import clip_keep, sample_grid


def _solid(field, lo, hi, pts, level, clip):
    s = sample_grid(field, lo, hi, pts, padding="border") >= level
    return s & clip_keep(pts, clip) if clip else s


def march_iso(field, lo, hi, o, d, level, step, *, refine=12, clip=(), tmax=None, chunk_samples=2 ** 22):
    N = o.shape[0]
    out = torch.full((N,), math.inf, dtype=o.dtype, device=o.device)
    tn, tf = ray_box(o, d, lo, hi)
    if tmax is not None:
        tf = torch.minimum(tf, tmax)
    idx = (tf > tn).nonzero().squeeze(1)
    if idx.numel() == 0:
        return out
    nmax = int(math.ceil((tf - tn)[idx].max().item() / step)) + 1
    per = max(1, chunk_samples // nmax)
    k = torch.arange(nmax, dtype=o.dtype, device=o.device)
    for s in range(0, idx.numel(), per):
        ii = idx[s:s + per]
        ts = torch.minimum(tn[ii, None] + k * step, tf[ii, None])
        pts = o[ii, None, :] + ts[..., None] * d[ii, None, :]
        solid = _solid(field, lo, hi, pts, level, clip)
        has = solid.any(1)
        first = solid.to(torch.int8).argmax(1)
        r = torch.arange(ii.numel(), device=o.device)
        t_hi = ts[r, first]
        t_lo = torch.where(first > 0, ts[r, (first - 1).clamp_min(0)], t_hi)
        for _ in range(refine):
            mid = 0.5 * (t_lo + t_hi)
            sm = _solid(field, lo, hi, o[ii] + mid[:, None] * d[ii], level, clip)
            t_hi = torch.where(sm, mid, t_hi)
            t_lo = torch.where(sm, t_lo, mid)
        out[ii] = torch.where(has, t_hi, torch.full_like(t_hi, math.inf))
    return out


def iso_normals(field, lo, hi, pts, level, *, clip=(), tol=None):
    lo_t = torch.as_tensor(lo, dtype=pts.dtype, device=pts.device)
    hi_t = torch.as_tensor(hi, dtype=pts.dtype, device=pts.device)
    shape = torch.tensor(field.shape[::-1], dtype=pts.dtype, device=pts.device)   # (W, H, D) = x, y, z counts
    h = (hi_t - lo_t) / (shape - 1)
    tol = float(h.min()) * 0.25 if tol is None else tol
    g = torch.zeros_like(pts)
    for ax in range(3):
        e = torch.zeros(3, dtype=pts.dtype, device=pts.device); e[ax] = h[ax]
        g[:, ax] = (sample_grid(field, lo, hi, pts + e, padding="border") -
                    sample_grid(field, lo, hi, pts - e, padding="border")) / (2 * h[ax])
    n = -g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    for ax in range(3):                                          # box faces
        for side, bound in ((-1.0, lo_t[ax]), (1.0, hi_t[ax])):
            on = (pts[:, ax] - bound).abs() < tol
            face = torch.zeros(3, dtype=pts.dtype, device=pts.device); face[ax] = side
            n = torch.where(on[:, None], face, n)
    for point, normal in clip:                                   # clip faces win over box faces
        p = torch.as_tensor(point, dtype=pts.dtype, device=pts.device)
        c = torch.as_tensor(normal, dtype=pts.dtype, device=pts.device)
        c = c / c.norm()
        on = ((pts - p) * c).sum(-1).abs() < tol
        n = torch.where(on[:, None], c, n)
    return n


def render_iso(field, lo, hi, cam, level, step=None, *, clip=(), device="cpu"):
    if step is None:
        h = [(b - a) / (n - 1) for a, b, n in zip(lo, hi, field.shape[::-1])]
        step = 0.5 * min(h)
    o, d = cam.rays(device=device, dtype=field.dtype)
    H, W = o.shape[:2]
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    t = march_iso(field, lo, hi, o, d, level, step, clip=clip)
    mask = torch.isfinite(t)
    pos = o + torch.where(mask, t, torch.zeros_like(t))[:, None] * d
    normal = torch.zeros_like(pos)
    if mask.any():
        normal[mask] = iso_normals(field, lo, hi, pos[mask], level, clip=clip)
    depth = torch.where(mask, cam.t_to_depth(t, d), torch.full_like(t, math.inf))
    return dict(depth=depth.reshape(H, W), pos=pos.reshape(H, W, 3), normal=normal.reshape(H, W, 3), mask=mask.reshape(H, W))


def iso_occluder(field, lo, hi, level, step, clip=()):
    def occluded(o, d, tmax):
        return torch.isfinite(march_iso(field, lo, hi, o, d, level, step, clip=clip, tmax=tmax))
    return occluded
