"""hfk.py: Triton kernels for the heightfield renderer (same algorithms as solid.march_fast and
solid.shadow_fast, one GPU thread per ray instead of a Python loop over torch ops).

The surface is the bilinear interpolant of Hm (R x R) over [-1, 1]^2 (x = column, y = row).
  march(Hm, Bm, block, o, d)  first hit of each ray (block-max empty-space skipping, fixed fine
                              step, 10 bisection steps) -> t (inf = miss), wall flag
  shadow(Hm, Bm, block, L)    soft shadow per texel, s = min over the ray of softness*clear/t
"""
import torch
import triton
import triton.language as tl

LO, HI = -1.0, 1.0


@triton.jit
def _bilin(H, x, y, R, texel, mask):
    u = (x - (-1.0)) / texel
    v = (y - (-1.0)) / texel
    u = tl.minimum(tl.maximum(u, 0.0), R - 1 - 1e-4)
    v = tl.minimum(tl.maximum(v, 0.0), R - 1 - 1e-4)
    i0 = u.to(tl.int32)
    j0 = v.to(tl.int32)
    fu = u - i0.to(tl.float32)
    fv = v - j0.to(tl.float32)
    base = j0 * R + i0
    h00 = tl.load(H + base, mask=mask, other=0.0)
    h10 = tl.load(H + base + 1, mask=mask, other=0.0)
    h01 = tl.load(H + base + R, mask=mask, other=0.0)
    h11 = tl.load(H + base + R + 1, mask=mask, other=0.0)
    return (h00 * (1 - fu) + h10 * fu) * (1 - fv) + (h01 * (1 - fu) + h11 * fu) * fv


@triton.jit
def _bound(Bm, x, y, nb, texel, block, mask):
    bi = ((x - (-1.0)) / texel / block).to(tl.int32)
    bj = ((y - (-1.0)) / texel / block).to(tl.int32)
    bi = tl.minimum(tl.maximum(bi, 0), nb - 1)
    bj = tl.minimum(tl.maximum(bj, 0), nb - 1)
    return tl.load(Bm + bj * nb + bi, mask=mask, other=0.0)


@triton.jit
def _march_kernel(O, D, T0, T1, H, Bm, TOUT, WALL, N, R, nb, texel, block, step, REFINE: tl.constexpr,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    ox = tl.load(O + offs * 3 + 0, mask=m, other=0.0)
    oy = tl.load(O + offs * 3 + 1, mask=m, other=0.0)
    oz = tl.load(O + offs * 3 + 2, mask=m, other=0.0)
    dx = tl.load(D + offs * 3 + 0, mask=m, other=0.0)
    dy = tl.load(D + offs * 3 + 1, mask=m, other=0.0)
    dz = tl.load(D + offs * 3 + 2, mask=m, other=1.0)
    t = tl.load(T0 + offs, mask=m, other=0.0) + 1e-6
    tfar = tl.load(T1 + offs, mask=m, other=-1.0)
    ok = m & (t < tfar)
    dxy = tl.maximum(tl.sqrt(dx * dx + dy * dy), 1e-6)
    h = _bilin(H, ox + t * dx, oy + t * dy, R, texel, ok)
    below = ok & ((oz + t * dz) < h)
    tout = tl.where(below, t, float("inf"))
    # pass 1: block skipping
    dtc = 0.9 * block * texel / dxy
    act = ok & (~below)
    while tl.max(act.to(tl.int32), axis=0) > 0:
        px = ox + t * dx
        py = oy + t * dy
        pz = oz + t * dz
        zb = oz + (t + dtc) * dz
        bnd = _bound(Bm, px, py, nb, texel, block, act)
        adv = act & (tl.minimum(pz, zb) >= bnd)
        t = tl.where(adv, t + dtc, t)
        act = adv & (t <= tfar)
    # pass 2: fine march
    dt = step * texel / dxy
    act = ok & (~below) & (t <= tfar)
    ta = t
    tb = t
    hit = tl.zeros((BLOCK,), dtype=tl.int1)
    while tl.max(act.to(tl.int32), axis=0) > 0:
        tn = t + dt
        px = ox + tn * dx
        py = oy + tn * dy
        pz = oz + tn * dz
        hh = _bilin(H, px, py, R, texel, act)
        crossed = act & (pz < hh)
        ta = tl.where(crossed, t, ta)
        tb = tl.where(crossed, tn, tb)
        hit = hit | crossed
        t = tl.where(act, tn, t)
        act = act & (~crossed) & (t <= tfar)
    # bisection
    for _ in range(REFINE):
        mid = 0.5 * (ta + tb)
        hh = _bilin(H, ox + mid * dx, oy + mid * dy, R, texel, hit)
        c = (oz + mid * dz) < hh
        tb = tl.where(c, mid, tb)
        ta = tl.where(c, ta, mid)
    tout = tl.where(hit, tb, tout)
    tl.store(TOUT + offs, tout, mask=m)
    tl.store(WALL + offs, below.to(tl.int8), mask=m)


def max_blocks(Hm, b):
    R = Hm.shape[0]
    nb = (R - 1) // b + 1
    pad = nb * b + 1 - R
    Hp = torch.nn.functional.pad(Hm[None, None], (0, pad, 0, pad), mode="replicate")
    Bm = torch.nn.functional.max_pool2d(Hp, kernel_size=b + 1, stride=b)[0, 0]
    return torch.nn.functional.max_pool2d(Bm[None, None], 3, 1, 1)[0, 0].contiguous()


def ray_box(o, d, lo, hi):
    inv = 1.0 / torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    lo = torch.tensor(lo, device=o.device, dtype=o.dtype); hi = torch.tensor(hi, device=o.device, dtype=o.dtype)
    t0 = (lo - o) * inv; t1 = (hi - o) * inv
    return torch.minimum(t0, t1).amax(-1).clamp_min(0), torch.maximum(t0, t1).amin(-1)


@torch.no_grad()
def march(Hm, o, d, Bm=None, block=16, step_texels=0.6, refine=10):
    Hm = Hm.contiguous().float()
    R = Hm.shape[0]
    if Bm is None:
        Bm = max_blocks(Hm, block)
    texel = (HI - LO) / (R - 1)
    o = o.contiguous().float(); d = d.contiguous().float()
    tn, tf = ray_box(o, d, (LO, LO, float(Hm.min())), (HI, HI, float(Hm.max()) + 1e-6))
    N = o.shape[0]
    tout = torch.empty(N, device=o.device); wall = torch.empty(N, device=o.device, dtype=torch.int8)
    BLOCK = 128
    _march_kernel[(triton.cdiv(N, BLOCK),)](o, d, tn.contiguous(), tf.contiguous(), Hm, Bm, tout, wall, N, R,
                                            Bm.shape[0], texel, float(block), step_texels, REFINE=refine,
                                            BLOCK=BLOCK)
    return tout, wall.bool()


@triton.jit
def _shadow_kernel(H, Bm, S, R, nb, texel, block, lx, ly, lz, zmax, soft, dt, dtc, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < R * R
    j = offs // R
    i = offs % R
    x0 = -1.0 + i.to(tl.float32) * texel
    y0 = -1.0 + j.to(tl.float32) * texel
    z0 = tl.load(H + offs, mask=m, other=0.0) + 1e-5
    s = tl.where(m, 1.0, 0.0)
    t = x0 * 0.0
    act = m
    while tl.max(act.to(tl.int32), axis=0) > 0:
        px = x0 + t * lx
        py = y0 + t * ly
        pz = z0 + t * lz
        bnd = _bound(Bm, px, py, nb, texel, block, act)
        big = (t > 0) & (soft * (pz - bnd) >= (t + dtc))
        tn = tl.where(big, t + dtc, t + dt)
        px = x0 + tn * lx
        py = y0 + tn * ly
        pz = z0 + tn * lz
        inside = (px >= -1.0) & (px <= 1.0) & (py >= -1.0) & (py <= 1.0) & (pz <= zmax)
        hh = _bilin(H, px, py, R, texel, act & inside)
        sv = tl.minimum(tl.maximum(soft * (pz - hh) / tn, 0.0), 1.0)
        s = tl.where(act & inside, tl.minimum(s, sv), s)
        t = tl.where(act, tn, t)
        act = act & inside & (s > 0)
    tl.store(S + offs, s, mask=m)


@torch.no_grad()
def shadow(Hm, L, softness=18.0, step_texels=0.75, block=16, Bm=None):
    Hm = Hm.contiguous().float()
    R = Hm.shape[0]
    if Bm is None:
        Bm = max_blocks(Hm, block)
    L = torch.tensor(L, dtype=torch.float64); L = (L / L.norm()).tolist()
    texel = (HI - LO) / (R - 1)
    dxy = max((L[0] ** 2 + L[1] ** 2) ** 0.5, 1e-6)
    S = torch.empty(R * R, device=Hm.device)
    BLOCK = 256
    _shadow_kernel[(triton.cdiv(R * R, BLOCK),)](Hm, Bm, S, R, Bm.shape[0], texel, float(block), L[0], L[1], L[2],
                                                 float(Hm.max()), softness, step_texels * texel / dxy,
                                                 0.9 * block * texel / dxy, BLOCK=BLOCK)
    return S.view(R, R)
