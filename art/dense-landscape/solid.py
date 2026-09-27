"""solid.py: the 'dense solid' renderer (v2) for the fur, films and stills.

Why a second renderer.  hf.py draws the bilinear interpolant of the 1024^2 grid: one grid cell is
~5 px at 5120 px, so every needle is a hairline and the frame reads as scattered strokes.  Here:

  * GRAIN (declared): the surface is built from a COARSER sampling of the same measurement
    (e.g. every 4th point of B = its 256^2 subgrid; the boiling film measures 256^2 directly).
    The analysis in README showed that the needle size is set by the grid spacing, so the grid
    is an honest, declared choice of grain, not a property of the network.
  * SMOOTH INTERPOLANT (declared): an interpolating bicubic spline through every coarse sample
    (scipy RectBivariateSpline, kx = ky = 3), evaluated on an `up`-times finer grid together with
    its analytic derivatives (normal map).  Each sample becomes a smooth cone several cells wide;
    nothing between samples is measured.
  * LIGHT (declared, form only): a strong low key light with per-texel soft shadows marched on the
    fine grid, a cool unshadowed fill from the opposite side, horizon ambient occlusion (radii in
    coarse cells) that also darkens the direct light a little (cavity), Blinn-Phong/Schlick gloss,
    mild depth fog, filmic tonemap.

API: prep(F, up, K, Z0) -> Hm, GX, GY (torch);  shadow/AO from hf;  render(...) -> linear RGB;
     tonemap(img) -> sRGB uint8.  G-buffer variant for fixed cameras (light sweep).
"""
import math, os, sys
import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline
sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import hf
from dl_common import grid_axis

LO, HI = -1.0, 1.0


def prep(F, up, K, Z0, device):
    """F: (R,R) log10-loss on the coarse grid (row = beta = y, col = alpha = x).
    Returns Hm (height, world units), GX, GY (dz/dx, dz/dy, world units) on the fine grid."""
    R = F.shape[0]
    n = (R - 1) * up + 1
    ax, fx = grid_axis(R), grid_axis(n)
    if up == 1:
        Hn = F
        gy, gx = np.gradient(F, ax, ax)
    else:
        s = RectBivariateSpline(ax, ax, F, kx=3, ky=3)
        Hn = s(fx, fx)
        gy = s(fx, fx, dx=1)          # d/d(beta) = d/dy
        gx = s(fx, fx, dy=1)          # d/d(alpha) = d/dx
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    return t(K * (Hn - Z0)), t(K * gx), t(K * gy)


def up_colour(rgb, n):
    """bicubic upsample of a colour texture to n x n (clipped)."""
    R = rgb.shape[0]
    if R == n:
        return rgb
    ax, fx = grid_axis(R), grid_axis(n)
    return np.stack([RectBivariateSpline(ax, ax, rgb[..., c], kx=3, ky=3)(fx, fx) for c in range(3)], -1).clip(0, 1)


def ao_coarse(Hm, up, n_dirs=12, cells=(0.35, 0.7, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5, 9.0, 13.0, 20.0, 30.0)):
    """horizon AO with radii given in COARSE cells (so the crevices between cones darken)."""
    return hf.ao_map(Hm, n_dirs=n_dirs, radii_texels=tuple(max(0.5, c * up) for c in cells))


def max_blocks(Hm, b):
    """block-max pyramid level: max of Hm over each b x b block of texels, then dilated over the
    3 x 3 block neighbourhood (a conservative bound for any segment shorter than one block)."""
    R = Hm.shape[0]
    nb = (R - 1) // b + 1
    pad = nb * b + 1 - R
    Hp = torch.nn.functional.pad(Hm[None, None], (0, pad, 0, pad), mode="replicate")
    # block i covers texels [i*b, i*b + b] (inclusive: the bilinear cell edges)
    Bm = torch.nn.functional.max_pool2d(Hp, kernel_size=b + 1, stride=b)[0, 0]
    Bm = torch.nn.functional.max_pool2d(Bm[None, None], 3, 1, 1)[0, 0]
    return Bm


@torch.no_grad()
def march_fast(Hm, o, d, Bm, b, step_texels=0.6, refine=10):
    """First hit of rays with the bilinear heightfield solid {z <= H}.  Pass 1 skips empty space in
    whole blocks using the dilated block maxima Bm (conservative: a segment is skipped only if both
    its ends are above the max over the 3x3 blocks around its start); pass 2 is hf.march's fixed
    fine step + bisection from there.  Returns t (inf = miss), wall (entered below the surface)."""
    R = Hm.shape[0]
    texel = (HI - LO) / (R - 1)
    nb = Bm.shape[0]
    tn, tf = hf.ray_box(o, d, (LO, LO, float(Hm.min())), (HI, HI, float(Hm.max()) + 1e-6))
    N = o.shape[0]
    t_out = torch.full((N,), float("inf"), device=o.device)
    wall_out = torch.zeros(N, dtype=torch.bool, device=o.device)
    idx = (tn < tf).nonzero()[:, 0]
    if idx.numel() == 0:
        return t_out, wall_out
    ro, rd, t, tfar = o[idx], d[idx], tn[idx] + 1e-6, tf[idx]
    dxy = rd[:, :2].norm(dim=-1).clamp_min(1e-6)
    p = ro + t[:, None] * rd
    below = p[:, 2] < hf.bilin(Hm, p[:, 0], p[:, 1])
    t_out_l = torch.full_like(t, float("inf")); t_out_l[below] = t[below]
    wall = below.clone()
    # ---- pass 1: block skipping
    dtc = 0.9 * b * texel / dxy
    act = (~below).nonzero()[:, 0]
    while act.numel() > 0:
        ta = t[act]; pa = ro[act] + ta[:, None] * rd[act]
        tb = ta + dtc[act]; zb = ro[act, 2] + tb * rd[act, 2]
        bi = ((pa[:, 0] - LO) / texel / b).floor().long().clamp(0, nb - 1)
        bj = ((pa[:, 1] - LO) / texel / b).floor().long().clamp(0, nb - 1)
        bound = Bm[bj, bi]
        ok = (torch.minimum(pa[:, 2], zb) >= bound)
        adv = act[ok]
        t[adv] = tb[ok]
        out = t[adv] > tfar[adv]
        act = adv[~out]
    # ---- pass 2: fine march from there
    dt = step_texels * texel / dxy
    act = ((~below) & (t <= tfar)).nonzero()[:, 0]
    tprev = t.clone()
    while act.numel() > 0:
        tprev[act] = t[act]
        t[act] = t[act] + dt[act]
        pa = ro[act] + t[act, None] * rd[act]
        hit = pa[:, 2] < hf.bilin(Hm, pa[:, 0], pa[:, 1])
        gone = t[act] > tfar[act]
        if hit.any():
            h_i = act[hit]
            a_, b_ = tprev[h_i], t[h_i]
            for _ in range(refine):
                m = 0.5 * (a_ + b_)
                pm = ro[h_i] + m[:, None] * rd[h_i]
                crossed = pm[:, 2] < hf.bilin(Hm, pm[:, 0], pm[:, 1])
                b_ = torch.where(crossed, m, b_); a_ = torch.where(crossed, a_, m)
            t_out_l[h_i] = b_
        act = act[~(hit | gone)]
    t_out[idx] = t_out_l; wall_out[idx] = wall
    return t_out, wall_out


@torch.no_grad()
def shadow_fast(Hm, L, softness=18.0, step_texels=0.75, block=16, chunk=1 << 21):
    """Same soft shadow as hf.shadow_map (s = min over the ray of softness*clearance/t), with exact
    block skipping: a whole-block step is taken only when both ends of the segment clear the 3x3
    block max by enough that the penumbra term is already saturated (softness*clear/t >= 1)."""
    if Hm.is_cuda and not os.environ.get("NO_TRITON"):
        import hfk
        return hfk.shadow(Hm, L, softness=softness, step_texels=step_texels, block=block)
    R = Hm.shape[0]
    Lv = torch.tensor(_norm(L), dtype=torch.float32, device=Hm.device)
    texel = (HI - LO) / (R - 1)
    Bm = max_blocks(Hm, block); nb = Bm.shape[0]
    ax = torch.linspace(LO, HI, R, device=Hm.device)
    Y, X = torch.meshgrid(ax, ax, indexing="ij")
    X, Y, Z = X.flatten(), Y.flatten(), Hm.flatten()
    zmax = float(Hm.max())
    out = torch.ones(R * R, device=Hm.device)
    dxy = float(Lv[:2].norm())
    dt = step_texels * texel / max(dxy, 1e-6); dtc = 0.9 * block * texel / max(dxy, 1e-6)
    for c0 in range(0, R * R, chunk):
        x0, y0, z0 = X[c0:c0 + chunk], Y[c0:c0 + chunk], Z[c0:c0 + chunk] + 1e-5
        s = torch.ones_like(x0); t = torch.zeros_like(x0)
        act = torch.arange(x0.numel(), device=Hm.device)
        while act.numel() > 0:
            ta = t[act]
            px = x0[act] + ta * Lv[0]; py = y0[act] + ta * Lv[1]; pz = z0[act] + ta * Lv[2]
            bi = ((px - LO) / texel / block).floor().long().clamp(0, nb - 1)
            bj = ((py - LO) / texel / block).floor().long().clamp(0, nb - 1)
            clear_b = pz - Bm[bj, bi]                        # pz is the lower end (rays rise)
            big = (ta > 0) & (softness * clear_b >= (ta + dtc))
            tn = ta + torch.where(big, torch.full_like(ta, dtc), torch.full_like(ta, dt))
            t[act] = tn
            px = x0[act] + tn * Lv[0]; py = y0[act] + tn * Lv[1]; pz = z0[act] + tn * Lv[2]
            inside = (px >= LO) & (px <= HI) & (py >= LO) & (py <= HI) & (pz <= zmax)
            clear = pz - hf.bilin(Hm, px.clamp(LO, HI), py.clamp(LO, HI))
            sv = torch.where(inside, (softness * clear / tn).clamp(0, 1), torch.ones_like(tn))
            s[act] = torch.minimum(s[act], sv)
            act = act[inside & (s[act] > 0)]
        out[c0:c0 + chunk] = s
    return out.view(R, R)


def _norm(v):
    v = torch.as_tensor(v, dtype=torch.float32); return (v / v.norm()).tolist()


@torch.no_grad()
def gbuffer(Hm, cam, ss=2, step_texels=0.6, device="cuda", row_chunk=512, window=None, block=16, fast=True):
    """primary hits for a camera: returns dict(p (N,3), m (N,) hit mask, under (N,) underside flag,
    d (N,3), shape (H*ss, W*ss))."""
    import r3d
    W, H = cam.width, cam.height
    big = r3d.Camera(cam.eye, cam.target, cam.up, W * ss, H * ss, cam.fov_deg, cam.ortho_height)
    o_all, d_all = big.rays(device, torch.float32)
    if window is not None:
        x0, y0, ww, wh = window
        o_all = o_all[y0 * ss:(y0 + wh) * ss, x0 * ss:(x0 + ww) * ss]
        d_all = d_all[y0 * ss:(y0 + wh) * ss, x0 * ss:(x0 + ww) * ss]
    Hs, Ws = o_all.shape[:2]
    ts, us = [], []
    Bm = max_blocks(Hm, block).contiguous() if fast else None
    if fast and Hm.is_cuda and not os.environ.get("NO_TRITON"):
        row_chunk = Hs
    for r0 in range(0, Hs, row_chunk):
        o = o_all[r0:r0 + row_chunk].reshape(-1, 3); d = d_all[r0:r0 + row_chunk].reshape(-1, 3)
        if fast and Hm.is_cuda and not os.environ.get("NO_TRITON"):
            import hfk
            t, under = hfk.march(Hm, o, d, Bm=Bm, block=block, step_texels=step_texels)
        elif fast:
            t, under = march_fast(Hm, o, d, Bm, block, step_texels=step_texels)
        else:
            t, under = hf.march(Hm, o, d, step_texels=step_texels, sheet=True)
        ts.append(t); us.append(under)
    t = torch.cat(ts); under = torch.cat(us)
    o = o_all.reshape(-1, 3); d = d_all.reshape(-1, 3)
    m = torch.isfinite(t)
    p = o + torch.where(m, t, torch.zeros_like(t))[:, None] * d
    return dict(p=p, m=m, under=under, d=d, o=o, shape=(Hs, Ws), ss=ss)


@torch.no_grad()
def shade(G, Hm, GX, GY, albedo, key, *, shadow=None, ao=None, fill=None, key_rgb=(1.0, 0.93, 0.82),
          fill_rgb=(0.55, 0.65, 1.0), amb_rgb=(0.32, 0.36, 0.5), key_I=3.2, fill_I=0.55, ka=0.55,
          ao_pow=1.6, cavity=0.6, ks=0.35, shininess=40.0, f0=0.04, fog=0.25, fog_ref=None,
          under_gain=0.15, bg=(0.0, 0.0, 0.0), chunk=1 << 22):
    """Shade a G-buffer. Returns (H, W, 3) linear RGB (box-filtered over ss)."""
    dev = Hm.device
    p, m, d = G["p"], G["m"], G["d"]
    N = p.shape[0]
    out = torch.tensor(bg, device=dev, dtype=torch.float32).expand(N, 3).clone()
    Lk = torch.tensor(_norm(key), device=dev)
    Lf = torch.tensor(_norm(fill), device=dev) if fill is not None else None
    krgb = torch.tensor(key_rgb, device=dev); frgb = torch.tensor(fill_rgb, device=dev)
    argb = torch.tensor(amb_rgb, device=dev)
    eye = G["o"][0]
    if fog_ref is None:
        pm = p[m]
        fog_ref = float((pm - eye).norm(dim=-1).min()) if pm.numel() else 0.0
    idx_all = m.nonzero()[:, 0]
    for c0 in range(0, idx_all.numel(), chunk):
        idx = idx_all[c0:c0 + chunk]
        pp = p[idx]; dd = d[idx]; x, y = pp[:, 0], pp[:, 1]
        gx = hf.bilin(GX, x, y); gy = hf.bilin(GY, x, y)
        n = torch.stack([-gx, -gy, torch.ones_like(gx)], -1); n = n / n.norm(dim=-1, keepdim=True)
        alb = hf.bilin(albedo, x, y)
        un = G["under"][idx]
        n = torch.where(un[:, None], -n, n); alb = torch.where(un[:, None], alb * under_gain, alb)
        sh = hf.bilin(shadow, x, y) if shadow is not None else torch.ones_like(x)
        a_o = hf.bilin(ao, x, y) if ao is not None else torch.ones_like(x)
        v = -dd
        ndl = (n @ Lk).clamp_min(0)
        hv = Lk[None] + v; hv = hv / hv.norm(dim=-1, keepdim=True)
        ndh = (n * hv).sum(-1).clamp_min(0)
        vdh = (v * hv).sum(-1).clamp(0, 1)
        fres = f0 + (1 - f0) * (1 - vdh) ** 5
        spec = ks * fres * ndh ** shininess * (shininess + 8) / 8 * ndl
        cav = (1 - cavity) + cavity * a_o                    # crevices also get less direct light
        c = key_I * (alb * ndl[:, None] + spec[:, None]) * (sh * cav)[:, None] * krgb
        sky = 0.5 + 0.5 * n[:, 2]
        c = c + ka * alb * (a_o ** ao_pow * sky)[:, None] * argb
        if Lf is not None:
            nf = (n @ Lf).clamp_min(0)
            c = c + fill_I * alb * (nf * a_o)[:, None] * frgb
        if fog > 0:
            dist = (pp - eye).norm(dim=-1)
            c = c * torch.exp(-fog * (dist - fog_ref).clamp_min(0))[:, None]
        out[idx] = c
    Hs, Ws = G["shape"]; ss = G["ss"]
    return out.view(Hs // ss, ss, Ws // ss, ss, 3).mean((1, 3))


def tonemap(img, exposure=1.0):
    """ACES-fitted filmic curve (Narkowicz 2015) then sRGB; declared."""
    x = img * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    y = ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0, 1)
    y = torch.where(y <= 0.0031308, 12.92 * y, 1.055 * y.clamp_min(1e-8) ** (1 / 2.4) - 0.055)
    return (y.clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()


def light_dir(az_deg, el_deg):
    """direction TO the light: azimuth measured in the x-y plane from +x, elevation above horizon."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
