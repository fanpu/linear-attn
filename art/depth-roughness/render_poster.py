"""Whole-sphere Hammer (equal-area) poster of one draw. usage: render_poster.py <kernel> <style>
styles: topo (9 quantile contours, sepia on paper), gold (median level set on quiet dark relief)"""
import sys, numpy as np
from scipy import ndimage
from render_common import *
from common import *
from globe import make_alm, sphere_median
import cmcrameri.cm as cmc

kernel, style = sys.argv[1], sys.argv[2]
LMAX = 4096
Wp, SS = 6000, 2
Hp = Wp // 2
n_w, n_h = Wp * SS, Hp * SS
x = (np.arange(n_w) + 0.5) / n_w * 4 * np.sqrt(2) - 2 * np.sqrt(2)     # Hammer x in [-2sqrt2, 2sqrt2]
y = -((np.arange(n_h) + 0.5) / n_h * 2 * np.sqrt(2) - np.sqrt(2))
X, Y = np.meshgrid(x, y)
inside = (X**2 / 8 + Y**2 / 2) < 1
zz = np.sqrt(np.clip(1 - (X / 4) ** 2 - (Y / 2) ** 2, 0, 1))
lon = 2 * np.arctan2(zz * X, 2 * (2 * zz**2 - 1))
lat = np.arcsin(np.clip(zz * Y, -1, 1))
th = (np.pi / 2 - lat)[inside]; ph = lon[inside] % (2 * np.pi)
alm = make_alm(kernel, LMAX)
vals = synth(alm, LMAX, th, ph, nthreads=8)
c = sphere_median(alm, LMAX, nthreads=8)
f = np.full((n_h, n_w), c); f[inside] = vals
inner = ndimage.binary_erosion(inside, iterations=2)
def lines(lv, w=1):
    e = boundary(f, lv) & inner
    if w > 1:
        e = ndimage.binary_dilation(e, structure=np.ones((w, w), bool))
    return downsample(e.astype(np.float32), SS)
md = downsample(inside.astype(np.float32), SS)
edge = downsample((inside & ~ndimage.binary_erosion(inside, iterations=3)).astype(np.float32), SS)
if style == "topo":
    qs = np.quantile(vals[::7], np.linspace(0.1, 0.9, 9))
    cov = np.zeros((Hp, Wp), np.float32)
    for i, q in enumerate(qs):
        cov = np.maximum(cov, lines(q, 3 if i == 4 else 1) * (1.0 if i == 4 else 0.8))
    rgb = mix(PAPER, np.array([0.42, 0.2, 0.09]), np.clip(np.maximum(cov * 1.8, edge), 0, 1))
    rgb *= grain((Hp, Wp), 9, 0.01)[..., None]
else:
    fd = downsample(f.astype(np.float32), SS)
    lo, hi = np.percentile(vals[::7], [1, 99])
    t = np.clip((fd - lo) / (hi - lo), 0, 1)
    rgb = mix(DARK, cmc.oslo(0.08 + 0.32 * t)[..., :3], md)
    rgb = mix(rgb, np.array([0.55, 0.55, 0.6]), edge * 0.6)
    rgb = mix(rgb, np.array([1.0, 0.80, 0.38]), np.clip(lines(c) * 2.2, 0, 1))
to_img(rgb).save(f"gallery/poster_hammer_{kernel}_{style}.png")
print("ok")
