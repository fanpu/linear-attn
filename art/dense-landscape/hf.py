"""hf: a small torch heightfield renderer for dense loss landscapes.

Built alongside art/_shared/r3d (uses r3d.Camera for rays / projection, unchanged).
The surface is the BILINEAR interpolant of an R x R height grid over x, y in [-1, 1]
(x = alpha = column, y = beta = row), z = height in world units.  Every measured grid point
is one vertex; nothing finer than the grid is invented.

  march(Hm, o, d)              primary rays -> t, hit mask, wall mask (first crossing, bisection-refined)
  shadow_map(Hm, L)            per-texel soft shadow toward a directional light (march on the grid)
  ao_map(Hm)                   per-texel horizon-based ambient occlusion
  shade(...)                   Blinn-Phong + Schlick gloss, AO ambient, distance fog to black
  xray(Hm, cam, ...)           additive hairlines along every grid row and column (no depth test)
Light, shadow, AO, gloss and fog show form only; they are declared rendering choices.
"""
import math
import numpy as np
import torch

LO, HI = -1.0, 1.0


def _uv(Hm, x, y):
    R = Hm.shape[0]
    u = ((x - LO) / (HI - LO) * (R - 1)).clamp(0, R - 1 - 1e-4)
    v = ((y - LO) / (HI - LO) * (R - 1)).clamp(0, R - 1 - 1e-4)
    i0 = u.floor().long(); j0 = v.floor().long()
    return i0, j0, u - i0, v - j0


def bilin(T, x, y, grad=False):
    """T: (R,R) or (R,R,C). Returns value (and d/dx, d/dy in world units for 2-D T)."""
    R = T.shape[0]
    i0, j0, fu, fv = _uv(T, x, y)
    flat = T.reshape(R * R, -1)
    idx = j0 * R + i0
    t00, t10, t01, t11 = flat[idx], flat[idx + 1], flat[idx + R], flat[idx + R + 1]
    fu_, fv_ = fu[:, None], fv[:, None]
    val = (t00 * (1 - fu_) + t10 * fu_) * (1 - fv_) + (t01 * (1 - fu_) + t11 * fu_) * fv_
    if T.dim() == 2:
        val = val[:, 0]
    if not grad:
        return val
    s = (R - 1) / (HI - LO)
    dx = ((t10 - t00) * (1 - fv_) + (t11 - t01) * fv_)[:, 0] * s
    dy = ((t01 - t00) * (1 - fu_) + (t11 - t10) * fu_)[:, 0] * s
    return val, dx, dy


def ray_box(o, d, lo, hi):
    inv = 1.0 / torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    lo = torch.tensor(lo, device=o.device, dtype=o.dtype); hi = torch.tensor(hi, device=o.device, dtype=o.dtype)
    t0 = (lo - o) * inv; t1 = (hi - o) * inv
    tn = torch.minimum(t0, t1).amax(-1); tf = torch.maximum(t0, t1).amin(-1)
    return tn.clamp_min(0), tf


@torch.no_grad()
def march(Hm, o, d, step_texels=0.5, refine=10, chunk=1 << 21, zfloor=None, sheet=False):
    """First intersection of rays with the bilinear heightfield solid {z <= H(x,y)} (and
    its vertical side walls down to zfloor).  Returns t (inf = miss), wall (bool)."""
    R = Hm.shape[0]
    zmin = float(Hm.min()) if zfloor is None else zfloor
    zmax = float(Hm.max()) + 1e-6
    texel = (HI - LO) / (R - 1)
    N = o.shape[0]
    t_out = torch.full((N,), float("inf"), device=o.device)
    wall_out = torch.zeros(N, dtype=torch.bool, device=o.device)
    for c0 in range(0, N, chunk):
        oc, dc = o[c0:c0 + chunk], d[c0:c0 + chunk]
        tn, tf = ray_box(oc, dc, (LO, LO, zmin), (HI, HI, zmax))
        ok = tn < tf
        idx = ok.nonzero()[:, 0]
        ro, rd, t, tfar = oc[idx], dc[idx], tn[idx] + 1e-6, tf[idx]
        dxy = rd[:, :2].norm(dim=-1).clamp_min(1e-6)
        dt = step_texels * texel / dxy
        # entry below the surface: solid mode -> side wall; sheet mode -> look for the underside
        p = ro + t[:, None] * rd
        below = p[:, 2] < bilin(Hm, p[:, 0], p[:, 1])
        tres = torch.full_like(t, float("inf")); wall = torch.zeros_like(below)
        if sheet:
            wall = below.clone()            # here: 'wall' flags an underside hit
            act = torch.arange(t.numel(), device=o.device)
            sgn = torch.where(below, -1.0, 1.0)
        else:
            tres[below] = t[below]; wall[below] = True
            act = (~below).nonzero()[:, 0]
            sgn = torch.ones_like(t)
        tprev = t.clone()
        k = 0
        while act.numel() > 0:
            tprev[act] = t[act]
            t[act] = t[act] + dt[act]
            pa = ro[act] + t[act, None] * rd[act]
            f = (pa[:, 2] - bilin(Hm, pa[:, 0], pa[:, 1])) * sgn[act]
            hit = f < 0
            gone = t[act] > tfar[act]
            if hit.any():
                h_i = act[hit]
                a_, b_ = tprev[h_i], t[h_i]
                for _ in range(refine):
                    m = 0.5 * (a_ + b_)
                    pm = ro[h_i] + m[:, None] * rd[h_i]
                    crossed = (pm[:, 2] - bilin(Hm, pm[:, 0], pm[:, 1])) * sgn[h_i] < 0
                    b_ = torch.where(crossed, m, b_); a_ = torch.where(crossed, a_, m)
                tres[h_i] = b_
            act = act[~(hit | gone)]
            k += 1
        tt = torch.full((oc.shape[0],), float("inf"), device=o.device); tt[idx] = tres
        ww = torch.zeros(oc.shape[0], dtype=torch.bool, device=o.device); ww[idx] = wall
        t_out[c0:c0 + chunk] = tt; wall_out[c0:c0 + chunk] = ww
    return t_out, wall_out


@torch.no_grad()
def shadow_map(Hm, L, softness=24.0, step_texels=0.75, chunk=1 << 20):
    """Soft shadow factor per texel (1 = lit) toward directional light L (pointing TO the light),
    by marching from each vertex along L over the bilinear surface. Penumbra ~ min clearance/dist."""
    R = Hm.shape[0]
    L = torch.tensor(L, dtype=torch.float32, device=Hm.device); L = L / L.norm()
    texel = (HI - LO) / (R - 1)
    ax = torch.linspace(LO, HI, R, device=Hm.device)
    Y, X = torch.meshgrid(ax, ax, indexing="ij")
    X, Y, Z = X.flatten(), Y.flatten(), Hm.flatten()
    zmax = float(Hm.max())
    out = torch.ones(R * R, device=Hm.device)
    dxy = float(L[:2].norm()); dt = step_texels * texel / max(dxy, 1e-6)
    for c0 in range(0, R * R, chunk):
        x0, y0, z0 = X[c0:c0 + chunk], Y[c0:c0 + chunk], Z[c0:c0 + chunk] + 1e-5
        s = torch.ones_like(x0)
        act = torch.arange(x0.numel(), device=Hm.device)
        t = torch.zeros_like(x0)
        while act.numel() > 0:
            t[act] += dt
            px = x0[act] + t[act] * L[0]; py = y0[act] + t[act] * L[1]; pz = z0[act] + t[act] * L[2]
            inside = (px >= LO) & (px <= HI) & (py >= LO) & (py <= HI) & (pz <= zmax)
            clear = pz - bilin(Hm, px, py)
            s[act] = torch.minimum(s[act], (softness * clear / t[act]).clamp(0, 1))
            act = act[inside & (s[act] > 0)]
        out[c0:c0 + chunk] = s
    return out.view(R, R)


@torch.no_grad()
def ao_map(Hm, n_dirs=16, radii_texels=(1, 2, 3, 5, 8, 12, 18, 27, 40, 60), strength=1.0):
    """Horizon-based AO per texel: 1 - mean over directions of sin(max horizon elevation)."""
    R = Hm.shape[0]
    texel = (HI - LO) / (R - 1)
    occ = torch.zeros_like(Hm)
    Hp = Hm[None, None]
    for k in range(n_dirs):
        ang = 2 * math.pi * (k + 0.5) / n_dirs
        best = torch.zeros_like(Hm)
        for r in radii_texels:
            sx, sy = r * math.cos(ang), r * math.sin(ang)
            # shift by (sx, sy) texels with bilinear sampling via grid_sample
            ax = torch.linspace(-1, 1, R, device=Hm.device)
            gy, gx = torch.meshgrid(ax, ax, indexing="ij")
            grid = torch.stack([gx + 2 * sx / (R - 1), gy + 2 * sy / (R - 1)], -1)[None]
            Hs = torch.nn.functional.grid_sample(Hp, grid, mode="bilinear", padding_mode="border",
                                                 align_corners=True)[0, 0]
            elev = (Hs - Hm) / (r * texel)
            best = torch.maximum(best, elev)
        occ += torch.sin(torch.atan(best))
    return (1 - strength * occ / n_dirs).clamp(0, 1)


def normals_at(Hm, x, y):
    _, gx, gy = bilin(Hm, x, y, grad=True)
    n = torch.stack([-gx, -gy, torch.ones_like(gx)], -1)
    return n / n.norm(dim=-1, keepdim=True)


def wall_normal(p):
    ax = p[:, :2].abs()
    nx = torch.where(ax[:, 0] >= ax[:, 1], torch.sign(p[:, 0]), torch.zeros_like(p[:, 0]))
    ny = torch.where(ax[:, 0] < ax[:, 1], torch.sign(p[:, 1]), torch.zeros_like(p[:, 1]))
    return torch.stack([nx, ny, torch.zeros_like(nx)], -1)


@torch.no_grad()
def render(Hm, albedo, cam, L, *, ss=2, shadow=None, ao=None, light_rgb=(1.0, 0.96, 0.9),
           amb_rgb=(0.35, 0.4, 0.5), kd=1.0, ka=0.35, ks=0.55, shininess=60.0, f0=0.04,
           fog=0.0, fog_ref=None, wall_rgb=(0.05, 0.05, 0.06), bg=(0.0, 0.0, 0.0),
           step_texels=0.5, device="cuda", row_chunk=256, sheet=True, under_gain=0.18, rim=None,
           window=None):
    """Render with ss x ss supersampling (box-filtered). albedo: (R,R,3) texture (data colour).
    Returns (H, W, 3) linear-ish RGB in [0, inf)."""
    import r3d
    W, H = cam.width, cam.height
    big = r3d.Camera(cam.eye, cam.target, cam.up, W * ss, H * ss, cam.fov_deg, cam.ortho_height)
    o_all, d_all = big.rays(device, torch.float32)
    if window is not None:           # (x0, y0, w, h) in output pixels: render a crop of the same camera
        x0, y0, ww, wh = window
        o_all = o_all[y0 * ss:(y0 + wh) * ss, x0 * ss:(x0 + ww) * ss]
        d_all = d_all[y0 * ss:(y0 + wh) * ss, x0 * ss:(x0 + ww) * ss]
        W, H = ww, wh
    Lv = torch.tensor(L, dtype=torch.float32, device=device); Lv = Lv / Lv.norm()
    lrgb = torch.tensor(light_rgb, device=device); argb = torch.tensor(amb_rgb, device=device)
    img = torch.zeros(H * ss, W * ss, 3, device=device)
    eye = torch.tensor(cam.eye, dtype=torch.float32, device=device)
    if fog_ref is None:               # distance from the eye to the nearest corner of the plane
        cx = torch.tensor([[x, y, float(Hm.mean())] for x in (LO, HI) for y in (LO, HI)], device=device)
        fog_ref = float((cx - eye).norm(dim=-1).min())
    for r0 in range(0, H * ss, row_chunk):
        o = o_all[r0:r0 + row_chunk].reshape(-1, 3); d = d_all[r0:r0 + row_chunk].reshape(-1, 3)
        t, wall = march(Hm, o, d, step_texels=step_texels, sheet=sheet)
        m = torch.isfinite(t)
        col = torch.tensor(bg, device=device).expand(o.shape[0], 3).clone()
        if m.any():
            p = o[m] + t[m, None] * d[m]
            n = normals_at(Hm, p[:, 0], p[:, 1])
            wm = wall[m]
            alb = bilin(albedo, p[:, 0], p[:, 1])
            if wm.any():
                if sheet:                   # underside of the sheet: flipped normal, dimmed albedo
                    n[wm] = -n[wm]; alb[wm] = alb[wm] * under_gain
                else:
                    n[wm] = wall_normal(p[wm]); alb[wm] = torch.tensor(wall_rgb, device=device)
            sh = bilin(shadow, p[:, 0], p[:, 1]) if shadow is not None else torch.ones(p.shape[0], device=device)
            a_o = bilin(ao, p[:, 0], p[:, 1]) if ao is not None else torch.ones(p.shape[0], device=device)
            if not sheet:
                sh = torch.where(wm, torch.ones_like(sh), sh); a_o = torch.where(wm, torch.ones_like(a_o) * 0.6, a_o)
            ndl = (n @ Lv).clamp_min(0)
            v = -d[m]
            hv = Lv[None] + v; hv = hv / hv.norm(dim=-1, keepdim=True)
            ndh = (n * hv).sum(-1).clamp_min(0)
            vdh = (v * hv).sum(-1).clamp(0, 1)
            fres = f0 + (1 - f0) * (1 - vdh) ** 5
            spec = ks * fres * ndh ** shininess * (shininess + 8) / 8 * ndl
            sky = 0.55 + 0.45 * n[:, 2]
            c = alb * (kd * ndl[:, None] * sh[:, None] * lrgb + ka * a_o[:, None] * sky[:, None] * argb) \
                + (spec * sh)[:, None] * lrgb
            if rim is not None:             # second, unshadowed light (declared): diffuse + gloss
                R2 = torch.tensor(rim[:3], dtype=torch.float32, device=device); R2 = R2 / R2.norm()
                n2 = (n @ R2).clamp_min(0)
                h2 = R2[None] + v; h2 = h2 / h2.norm(dim=-1, keepdim=True)
                s2 = ks * (f0 + (1 - f0) * (1 - (v * h2).sum(-1).clamp(0, 1)) ** 5) * \
                    (n * h2).sum(-1).clamp_min(0) ** shininess * (shininess + 8) / 8 * n2
                c = c + rim[3] * (alb * n2[:, None] * 0.5 + s2[:, None]) * lrgb
            if fog > 0:
                dist = (p - eye).norm(dim=-1)
                ref = fog_ref                 # one global reference distance (per-chunk min made seams)
                c = c * torch.exp(-fog * (dist - ref).clamp_min(0))[:, None]
            col[m] = c
        img[r0:r0 + row_chunk] = col.view(-1, W * ss, 3)
    return img.view(H, ss, W, ss, 3).mean((1, 3))


@torch.no_grad()
def xray(Hm, cam, colour=None, densify=4, every=1, device="cuda", orients=(0, 1)):
    """Additive hairlines along every `every`-th grid row and column, no depth test.
    Weight per sample = its projected segment length in px, so a line's brightness is set by
    how many lines cross a pixel, not by sampling density. colour: (R,R,3) or None (white).
    Returns (H, W, 3) accumulated radiance (tonemap outside)."""
    R = Hm.shape[0]
    W, H = cam.width, cam.height
    acc = torch.zeros(H * W, 3, device=device, dtype=torch.float32)
    ax = torch.linspace(LO, HI, R, device=device)
    fine = torch.linspace(LO, HI, (R - 1) * densify + 1, device=device)
    for orient in orients:
        for j0 in range(0, R, 64 * every):
            js = torch.arange(j0, min(j0 + 64 * every, R), every, device=device)
            if orient == 0:   # rows: y fixed, x varies
                Y, X = torch.meshgrid(ax[js], fine, indexing="ij")
            else:
                X, Y = torch.meshgrid(ax[js], fine, indexing="ij")
            x, y = X.flatten(), Y.flatten()
            z = bilin(Hm, x, y)
            pts = torch.stack([x, y, z], -1).double()
            pix, depth = cam.project(pts, device=device)
            pix = pix.float().view(len(js), -1, 2)
            seg = torch.zeros(pix.shape[:2], device=device)
            dl = (pix[:, 1:] - pix[:, :-1]).norm(dim=-1)
            seg[:, 1:] += 0.5 * dl; seg[:, :-1] += 0.5 * dl
            wgt = seg.flatten().clamp(max=4.0) * (depth.float() > 0)
            c = bilin(colour, x, y) if colour is not None else torch.ones(x.numel(), 3, device=device)
            px, py = pix[..., 0].flatten() - 0.5, pix[..., 1].flatten() - 0.5
            i0 = px.floor(); j0_ = py.floor(); fx = px - i0; fy = py - j0_
            for di, dj, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                ii = (i0 + di).long(); jj = (j0_ + dj).long()
                ok = (ii >= 0) & (ii < W) & (jj >= 0) & (jj < H)
                acc.index_add_(0, (jj * W + ii)[ok], (c * (wgt * ww)[:, None])[ok])
    return acc.view(H, W, 3)
