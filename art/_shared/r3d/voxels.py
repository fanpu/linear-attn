"""Exact voxel traversal (Amanatides & Woo 1987), vectorised over rays. No interpolation: every face is a real cell face."""
import math

import torch

from .camera import ray_box
from .grid import clip_keep


def _geometry(solid, lo, hi, dtype, device):
    lo_t = torch.as_tensor(lo, dtype=dtype, device=device)
    hi_t = torch.as_tensor(hi, dtype=dtype, device=device)
    n = torch.tensor(solid.shape[::-1], device=device)                 # (W, H, D) = counts along x, y, z
    h = (hi_t - lo_t) / (n - 1).to(dtype)
    return lo_t, hi_t, n, h


def _apply_clip(solid, lo_t, h, clip):
    if not clip:
        return solid
    D, H, W = solid.shape
    z, y, x = (torch.arange(m, dtype=h.dtype, device=h.device) for m in (D, H, W))
    Z, Y, X = torch.meshgrid(lo_t[2] + z * h[2], lo_t[1] + y * h[1], lo_t[0] + x * h[0], indexing="ij")
    return solid & clip_keep(torch.stack([X, Y, Z], -1), clip)


def march_voxels(solid, lo, hi, o, d, *, tmax=None, clip=(), chunk=2 ** 20):
    dt, dev = o.dtype, o.device
    solid = solid.to(dev)
    lo_t, hi_t, n, h = _geometry(solid, lo, hi, dt, dev)
    solid = _apply_clip(solid, lo_t, h, clip)
    blo, bhi = lo_t - h / 2, hi_t + h / 2
    N = o.shape[0]
    t_out = torch.full((N,), math.inf, dtype=dt, device=dev)
    n_out = torch.zeros(N, 3, dtype=dt, device=dev)
    c_out = torch.full((N, 3), -1, dtype=torch.long, device=dev)
    tn, tf = ray_box(o, d, blo, bhi)
    if tmax is not None:
        tf = torch.minimum(tf, tmax)
    idx = (tf > tn).nonzero().squeeze(1)
    for s in range(0, idx.numel(), chunk):
        ii = idx[s:s + chunk]
        oo, dd, tfar = o[ii], d[ii], tf[ii]
        t = tn[ii].clone()
        R = ii.numel()
        r_all = torch.arange(R, device=dev)
        safe = torch.where(dd == 0, torch.ones_like(dd), dd)
        ta, tb = (blo - oo) / safe, (bhi - oo) / safe
        entry = torch.where(dd == 0, torch.full_like(dd, -math.inf), torch.minimum(ta, tb))
        ax = entry.argmax(1)                                           # axis of the entry face
        sgn = -torch.sign(dd[r_all, ax])
        p = oo + t[:, None] * dd
        cell = torch.floor((p - blo) / h).long()
        cell = torch.minimum(torch.maximum(cell, torch.zeros_like(cell)), (n - 1).expand_as(cell))
        stp = torch.sign(dd).long()
        tdelta = torch.where(dd == 0, torch.full_like(dd, math.inf), h / dd.abs())
        nb = blo + (cell + (stp > 0).long()).to(dt) * h
        tmx = torch.where(dd == 0, torch.full_like(dd, math.inf), (nb - oo) / safe)
        active = r_all
        for _ in range(int(n.sum().item()) + 3):
            if active.numel() == 0:
                break
            c = cell[active]
            hit = solid[c[:, 2], c[:, 1], c[:, 0]]
            if hit.any():
                a = active[hit]
                t_out[ii[a]] = t[a]
                nv = torch.zeros(a.numel(), 3, dtype=dt, device=dev)
                nv[torch.arange(a.numel(), device=dev), ax[a]] = sgn[a]
                n_out[ii[a]] = nv
                c_out[ii[a]] = cell[a]
            rem = active[~hit]
            if rem.numel() == 0:
                break
            rr = torch.arange(rem.numel(), device=dev)
            axr = tmx[rem].argmin(1)
            t[rem] = tmx[rem, axr]
            cell[rem, axr] = cell[rem, axr] + stp[rem, axr]
            tmx[rem, axr] = tmx[rem, axr] + tdelta[rem, axr]
            ax[rem] = axr
            sgn[rem] = -stp[rem, axr].to(dt)
            cr = cell[rem]
            ok = (cr >= 0).all(1) & (cr < n).all(1) & (t[rem] <= tfar[rem])
            active = rem[ok]
    return t_out, n_out, c_out


def render_voxels(labels, lo, hi, cam, *, solid=None, clip=(), device="cpu"):
    labels = labels.to(device)
    solid = (labels > 0) if solid is None else solid.to(device)
    o, d = cam.rays(device=device, dtype=torch.float64)
    H, W = o.shape[:2]
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    t, nrm, cell = march_voxels(solid, lo, hi, o, d, clip=clip)
    mask = torch.isfinite(t)
    pos = o + torch.where(mask, t, torch.zeros_like(t))[:, None] * d
    label = torch.full((o.shape[0],), -1, dtype=labels.dtype, device=device)
    if mask.any():
        c = cell[mask]
        label[mask] = labels[c[:, 2], c[:, 1], c[:, 0]]
    depth = torch.where(mask, cam.t_to_depth(t, d), torch.full_like(t, math.inf))
    return dict(depth=depth.reshape(H, W), pos=pos.reshape(H, W, 3), normal=nrm.reshape(H, W, 3),
                cell=cell.reshape(H, W, 3), label=label.reshape(H, W), mask=mask.reshape(H, W))


def voxel_occluder(solid, lo, hi, clip=()):
    def occluded(o, d, tmax):
        return torch.isfinite(march_voxels(solid, lo, hi, o, d, tmax=tmax, clip=clip)[0])
    return occluded
