"""Curves in space: tubes as z-buffered sphere impostors, hairlines as additive Gaussian splats, and hidden-line SVG."""
import math

import numpy as np
import torch


def sample_polyline(P, spacing):
    seg = P[1:] - P[:-1]
    L = seg.norm(dim=1)
    k = torch.ceil(L / spacing).clamp_min(1).long()
    sid = torch.repeat_interleave(torch.arange(len(seg), device=P.device), k)
    start = torch.cumsum(k, 0) - k
    frac = (torch.arange(int(k.sum()), device=P.device) - start[sid]).to(P.dtype) / k[sid].to(P.dtype)
    pts = torch.cat([P[sid] + frac[:, None] * seg[sid], P[-1:]], 0)
    s = torch.cat([sid.to(P.dtype) + frac, torch.tensor([float(len(seg))], dtype=P.dtype, device=P.device)])
    return pts, s


def _offsets(K, device):
    o = torch.arange(-K, K + 1, device=device)
    oy, ox = torch.meshgrid(o, o, indexing="ij")
    return ox.reshape(-1), oy.reshape(-1)


def splat_spheres(centers, radius, cam, *, attrs=None, chunk=2 ** 22, max_radius_px=256):
    """Z-buffered sphere impostors. Samples are grouped by splat radius into power-of-two kernel buckets, so each
    chunk allocates only its own kernel. Samples behind the eye or with discs fully off-screen are skipped. A visible
    sample whose splat radius exceeds `max_radius_px` (None = no cap) raises ValueError instead of allocating a huge
    kernel: a perspective camera very near a curve. Move the camera away or shrink the radius."""
    dt, dev = centers.dtype, centers.device
    H, W = cam.height, cam.width
    M = centers.shape[0]
    R = torch.as_tensor(radius, dtype=dt, device=dev).expand(M)
    pix, z = cam.project(centers, device=dev)
    pix, z = pix.to(dev, dt), z.to(dev, dt)
    ps = torch.full_like(z, cam.pixel_scale()) if cam.fov_deg is None else cam.pixel_scale(z)
    Rpx = R / ps
    f, r, u = (torch.tensor(v, dtype=dt, device=dev) for v in cam.basis())
    zbuf = torch.full((H * W,), math.inf, dtype=dt, device=dev)
    nbuf = torch.zeros(H * W, 3, dtype=dt, device=dev)
    abuf = None if attrs is None else torch.zeros(H * W, attrs.shape[1], dtype=dt, device=dev)
    pc0, pr0 = pix[:, 0], pix[:, 1]
    live = (z > 0) & (pc0 + Rpx >= 0) & (pc0 - Rpx < W) & (pr0 + Rpx >= 0) & (pr0 - Rpx < H)
    if max_radius_px is not None:
        big = live & ~(Rpx <= max_radius_px)
        if big.any():
            raise ValueError(f"splat_spheres: {int(big.sum())} sample(s) need a splat radius of up to "
                             f"{Rpx[big].max().item():.4g} px, above max_radius_px={max_radius_px}: move the camera "
                             f"away from the curve or shrink the radius")
    idx = live.nonzero().squeeze(1)
    K_all = torch.ceil(Rpx[idx]).clamp_min(1)
    lvl = torch.ceil(torch.log2(K_all)).long()
    lvl = lvl + (2.0 ** lvl < K_all).long()                            # guard log2 rounding: 2**lvl >= ceil(Rpx)
    for L in lvl.unique().tolist():
        bidx = idx[lvl == L]
        ox, oy = _offsets(2 ** L, dev)
        per = max(1, chunk // ox.numel())
        for s in range(0, bidx.numel(), per):
            ii = bidx[s:s + per]
            col, row = pix[ii, 0:1], pix[ii, 1:2]
            pc = torch.floor(col) + ox + 0.5
            pr = torch.floor(row) + oy + 0.5
            rp = Rpx[ii, None]
            dx, dy = (pc - col) / rp, (pr - row) / rp
            rho2 = dx * dx + dy * dy
            inside = (rho2 <= 1) & (pc >= 0) & (pc < W) & (pr >= 0) & (pr < H) & (z[ii, None] > 0)
            if not inside.any():
                continue
            nz = torch.sqrt((1 - rho2).clamp_min(0))
            depth = (z[ii, None] - R[ii, None] * nz)[inside]
            flat = (pr.long() * W + pc.long())[inside]
            nrm = (dx[..., None] * r - dy[..., None] * u - nz[..., None] * f)[inside]
            cmin = torch.full((H * W,), math.inf, dtype=dt, device=dev).scatter_reduce(0, flat, depth, "amin")
            win = (depth == cmin[flat]) & (depth < zbuf[flat])
            fw = flat[win]
            zbuf[fw] = depth[win]
            nbuf[fw] = nrm[win]
            if abuf is not None:
                sid = ii[:, None].expand_as(inside)[inside]
                abuf[fw] = attrs[sid[win]].to(dt)
    mask = torch.isfinite(zbuf)
    return dict(depth=zbuf.reshape(H, W), normal=nbuf.reshape(H, W, 3),
                attr=None if abuf is None else abuf.reshape(H, W, -1), mask=mask.reshape(H, W))


def splat_additive(points, cam, *, weight=None, color=None, sigma_px=0.7, depth=None, eps=0.0, chunk=2 ** 22):
    dt, dev = points.dtype, points.device
    H, W = cam.height, cam.width
    M = points.shape[0]
    pix, z = cam.project(points, device=dev)
    pix, z = pix.to(dev, dt), z.to(dev, dt)
    w = torch.ones(M, dtype=dt, device=dev) if weight is None else torch.as_tensor(weight, dtype=dt, device=dev).expand(M)
    C = 1 if color is None else 3
    acc = torch.zeros(H * W, C, dtype=dt, device=dev)
    K = int(math.ceil(3 * sigma_px))
    ox, oy = _offsets(K, dev)
    per = max(1, chunk // ox.numel())
    norm = 1.0 / (2 * math.pi * sigma_px ** 2)
    for s in range(0, M, per):
        sl = slice(s, s + per)
        col, row = pix[sl, 0:1], pix[sl, 1:2]
        pc = torch.floor(col) + ox + 0.5
        pr = torch.floor(row) + oy + 0.5
        g = norm * torch.exp(-((pc - col) ** 2 + (pr - row) ** 2) / (2 * sigma_px ** 2)) * w[sl, None]
        ok = (pc >= 0) & (pc < W) & (pr >= 0) & (pr < H) & (z[sl, None] > 0)
        flat = (pr.long().clamp(0, H - 1) * W + pc.long().clamp(0, W - 1))
        if depth is not None:
            ok &= z[sl, None] <= depth.reshape(-1).to(dt)[flat] + eps
        val = g[ok][:, None]
        if color is not None:
            val = val * color[sl].to(dt)[:, None, :].expand(-1, ox.numel(), 3)[ok]
        acc.index_add_(0, flat[ok], val)
    out = acc.reshape(H, W, C)
    return out[..., 0] if color is None else out


def visible_runs(P, cam, depth, eps):
    pix, z = cam.project(P, device=P.device)
    pix, z = pix.cpu(), z.cpu()                                        # the run walk below is host-side
    col, row = pix[:, 0], pix[:, 1]
    H, W = depth.shape
    inb = (col >= 0) & (col < W) & (row >= 0) & (row < H) & (z > 0)
    ci, ri = col.long().clamp(0, W - 1), row.long().clamp(0, H - 1)
    vis = inb & (z <= depth.double().cpu()[ri, ci] + eps)
    runs, cur = [], []
    xy = pix.numpy()
    for i, v in enumerate(vis.tolist()):
        if v:
            cur.append(xy[i])
        elif cur:
            runs.append(np.array(cur)); cur = []
    if cur:
        runs.append(np.array(cur))
    return [r for r in runs if len(r) >= 2]


def write_svg(path, polylines, width, height, stroke="#000000", stroke_width=1.0, background=None):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    if background:
        out.append(f'<rect width="{width}" height="{height}" fill="{background}"/>')
    for pl in polylines:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in pl)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')
    out.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(out))
