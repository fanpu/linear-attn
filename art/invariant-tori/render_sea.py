"""Sea and islands at eps = 0.5, H = 2.8, in the stereographic chart.

  fog       visit density of the chaotic orbit (256^3 counts, T = 6e6), emission-absorption volume
  sheet     Poincare section g = x_P - x_R + y_P - y_R = 0 (upward, dg/dt > 0), a translucent volume sheet
            whose opacity is a Gaussian of the first-order distance |g|/|grad g| (computed exactly on the grid)
  tubes     8 regular orbits (lambda <= 5e-3) as copper splat_spheres tubes, cropped to the density box
  dots      section crossings of the chaotic and regular orbits, small spheres on the sheet
Fog and sheet are one multi-channel volume marched once, composited over the opaque tubes/dots via depth.

python render_sea.py [hero|cutaway|stereo|all] [--size N] [--fog raw256|blur128] [--step S]
"""
import argparse
import os

import matplotlib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import render_lib as L
from render_lib import r3d

AZ, EL = 38.0, 22.0
SHEET_RGB = np.array([0.96, 0.90, 0.74])     # declared: ivory sheet
CHAOS_DOT = np.array([0.86, 0.18, 0.16])     # declared: chaotic crossings red (as the game-chaos plates)
FOG_CMAP = 'cmc.oslo'


def load_volume(fog):
    import cmcrameri  # noqa: F401  (registers cmc.* colormaps)
    dz = np.load(os.path.join(L.CACHE, 'density_eps05.npz'))
    lo, hi = dz['lo'], dz['hi']
    C = dz['counts'].astype(np.float32)                                    # (X0, X1, X2)
    fz = np.load(os.path.join(L.CACHE, 'fields256.npz'))
    sd, up = fz['sdist'], (fz['gdot_eps05'] > 0).astype(np.float32)
    if fog == 'blur128':                                                  # declared: 2x2x2 sum, Gaussian sigma 1 voxel
        C = C.reshape(128, 2, 128, 2, 128, 2).sum((1, 3, 5))
        C = gaussian_filter(C, 1.0, mode='constant')
        sd = sd.reshape(128, 2, 128, 2, 128, 2).mean((1, 3, 5)); up = up.reshape(128, 2, 128, 2, 128, 2).mean((1, 3, 5))
    R = C.shape[0]
    wv = 1.5                                                                # declared sheet half-width, in voxels
    hh = ((hi - lo) / R).max()
    sheet = np.exp(-(sd / (wv * hh)) ** 2) * up                             # 0 outside the box (zero padding is exact)
    v = np.log1p(C); vmax = np.quantile(v[C > 0], 0.999)
    fogv = np.clip(v / vmax, 0, 1)
    h = (hi - lo) / R
    lo_c, hi_c = lo + h / 2, hi - h / 2                                      # r3d: lo/hi are corner voxel centres
    data = np.stack([fogv, sheet]).transpose(0, 3, 2, 1)                  # (C, X2, X1, X0) = (C, z, y, x)
    return L.T(np.ascontiguousarray(data)), tuple(lo_c), tuple(hi_c), float(h.max()), dict(vmax=float(vmax), R=R)


class FogSheetTF:
    """Declared transfer function.
    fog:   x = log1p(count) / q99.9(log1p(count)); colour cmc.oslo(0.15 + 0.85 x); extinction dens_f * x^2
    sheet: extinction dens_s * exp(-(sdist / 1.5 voxels)^2) * [dg/dt > 0], evaluated on the grid, trilinear; ivory."""

    def __init__(self, dens_f, dens_s):
        import cmcrameri  # noqa: F401
        self.lut = L.T(matplotlib.colormaps[FOG_CMAP](np.linspace(0.15, 1.0, 1024))[:, :3])
        self.df, self.ds = dens_f, dens_s
        self.sheet = L.T(SHEET_RGB)

    def __call__(self, v):
        x = v[..., 0].clamp(0, 1)
        cf = self.lut.to(v.device, v.dtype)[(x * 1023).long()]
        sf = self.df * x * x
        ss = self.ds * v[..., 1].clamp(0, 1)
        tot = sf + ss
        c = (sf[..., None] * cf + ss[..., None] * self.sheet.to(v.device, v.dtype)) / tot.clamp_min(1e-12)[..., None]
        return c, tot


def box_camera(lo, hi, size, az=AZ, el=EL, eye_shift=None):
    c = 0.5 * (np.asarray(lo) + np.asarray(hi))
    ext = 0.80 * np.linalg.norm(np.asarray(hi) - np.asarray(lo))
    cam = r3d.orbit(tuple(c), 40.0, az_deg=az, el_deg=el, width=size, height=size, ortho_height=ext)
    return cam, c


def opaque_layer(cam, lo, hi, size, clip, tw, R, dotR):
    s = np.load(os.path.join(L.CACHE, 'stereo_eps05_fine.npz'))['X']
    n = int(round(tw / 0.01)) + 1
    lo_, hi_ = np.asarray(lo), np.asarray(hi)
    pts = []
    for P in s[:, :n]:
        p, _ = r3d.sample_polyline(L.T(P), 0.5 * R)
        pts.append(p)
    pts = torch.cat(pts)
    keep = ((pts >= L.T(lo_)) & (pts <= L.T(hi_))).all(1)                  # declared: tubes cropped to the fog box
    if clip:
        keep &= r3d.clip_keep(pts, clip)
    pts = pts[keep]
    e = np.load(os.path.join(L.CACHE, 'stereo_eps05.npz'))
    dots = np.concatenate([e['chaotic_sec_X'], e['regular_sec_X']])
    kind = np.r_[np.zeros(len(e['chaotic_sec_X'])), np.ones(len(e['regular_sec_X']))]
    dk = np.all((dots >= lo_) & (dots <= hi_), 1)
    dots, kind = L.T(dots[dk]), L.T(kind[dk])
    if clip:
        kk = r3d.clip_keep(dots, clip); dots, kind = dots[kk], kind[kk]
    allc = torch.cat([pts, dots])
    rad = torch.cat([torch.full((len(pts),), R, device=L.dev()), torch.full((len(dots),), dotR, device=L.dev())])
    attr = torch.cat([torch.full((len(pts),), 2.0, device=L.dev()), kind])[:, None]
    hit = r3d.splat_spheres(allc, rad, cam, attrs=attr)
    m = hit['mask']
    light = (np.cos(np.radians(AZ + 60)), np.sin(np.radians(AZ + 60)), 0.8)
    lam = r3d.lambert(hit['normal'][m], light, ambient=0.25)
    a = hit['attr'][m][:, 0]
    base = torch.where((a == 2)[:, None], L.T(L.COPPER), torch.where((a == 0)[:, None], L.T(CHAOS_DOT), L.T(L.COPPER * 0.7)))
    img = torch.zeros(size, size, 3, device=L.dev()) + L.T(L.NIGHT)
    img[m] = base * lam[:, None]
    return img, hit['depth'], dict(n_tube=int(len(pts)), n_dots=int(len(dots)))


def render(size, fog, step, clip_mode, tw=150.0, R=0.009, name=None, az=AZ, el=EL, clip_az=None, dens_f=12.0,
           dens_s=3.0, save=True):
    data, lo, hi, h, info = load_volume(fog)
    cam, c = box_camera(lo, hi, size, az=az, el=el)
    clip = ()
    if clip_mode:
        # Declared cutaway: vertical plane through the box centre, normal toward the (central) camera, near half removed.
        a = np.radians(az if clip_az is None else clip_az)
        n = np.array([np.cos(a), np.sin(a), 0.0])
        clip = [(tuple(c), tuple(n))]
    tf = FogSheetTF(dens_f, dens_s)
    with L.Timer(f'sea {name} {size} fog={fog} step={step} az={az}'):
        back, depth, cnt = opaque_layer(cam, lo, hi, size, clip, tw, R, 0.6 * R)
        rgb, a_ = L.volume(data, lo, hi, cam, tf, step, depth=depth, clip=clip)
        img = r3d.over(rgb, a_, back)
    if save:
        out = os.path.join(L.GAL, f'{name}{"_" + str(size) if size < 2000 else ""}.png')
        r3d.save_png(out, L.to_img(img))
        print(out, info, cnt, flush=True)
    return img


HERO = dict(dens_f=12.0)        # exterior: translucent sea, tubes visible through it
CUT = dict(dens_f=80.0)         # cutaway: dense fog, so the cut face reads as a slice with island holes


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='*', default=['all'])
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--fog', default='raw256')
    ap.add_argument('--step', type=float, default=0.006)
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(L.GAL, exist_ok=True)
    if 'hero' in a.what or 'all' in a.what:
        render(a.size, a.fog, a.step, False, name='sea_islands_hero', **HERO)
    if 'cutaway' in a.what or 'all' in a.what:
        render(a.size, a.fog, a.step, True, name='sea_islands_cutaway', **CUT)
    if 'stereo' in a.what or 'all' in a.what:
        # Declared: rotation stereo (orthographic cameras have no translation parallax), eyes at az -/+ 2.5 deg,
        # identical clip plane for both eyes.
        Lh = render(a.size, a.fog, a.step, True, az=AZ - 2.5, clip_az=AZ, save=False, **CUT)
        Rh = render(a.size, a.fog, a.step, True, az=AZ + 2.5, clip_az=AZ, save=False, **CUT)
        sfx = "_" + str(a.size) if a.size < 2000 else ""
        gap = torch.zeros(a.size, a.size // 24, 3) + L.T(L.NIGHT)
        r3d.save_png(os.path.join(L.GAL, f'sea_stereo_crosseye{sfx}.png'), torch.cat([Rh, gap, Lh], 1))   # cross-eye: right eye left
        lum = torch.tensor([0.299, 0.587, 0.114])
        ana = torch.stack([(Lh * lum).sum(-1), (Rh * lum).sum(-1), (Rh * lum).sum(-1)], -1)          # red = left eye
        r3d.save_png(os.path.join(L.GAL, f'sea_anaglyph_redcyan{sfx}.png'), ana)
