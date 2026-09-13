"""1-bit halftones: Floyd-Steinberg (serpentine) vs ordered Bayer (8x8 and 256x256) vs blue noise (void-and-cluster
128x128, sigma 1.5) vs white-noise threshold (the null model), on
  zone     (1 + cos(k r^2))/2, 2048^2, R_N = 1448 (well sampled inside the inscribed circle, aliased in the corners)
  basin    luminance of art/gd-bifurcation/gallery/basin_wide_dark.png (read-only; 4096^2 RGB -> 2x box -> 2048^2,
           Rec.709 luma of the 8-bit sRGB values, no gamma decode: declared)
  ramp     horizontal 0..1 ramp, 2048 x 256
  flat_g   constant grays g in {1/8, 1/4, 1/3, 1/2}, 1024^2, for spectra
Radially averaged power spectra of the flat-gray halftones are saved with low-frequency energy fractions.
Run: OMP_NUM_THREADS=4 python compute_halftones.py
"""
import json
import numpy as np
from PIL import Image
import common as c

rng = np.random.default_rng(7)
bn = c.void_and_cluster(128, sigma=1.5, seed=0)
np.save("cache/bluenoise128.npy", bn)
B8, B256 = c.bayer(3), c.bayer(8)
white = rng.permutation(128 * 128).reshape(128, 128)  # a random rank matrix: white-noise thresholds


def all_methods(v):
    return {"fs": c.floyd_steinberg(v), "bayer8": c.ordered(v, B8), "bayer256": c.ordered(v, B256),
            "blue": c.ordered(v, bn), "white": c.ordered(v, white)}


targets = {}
n = 2048
x = np.arange(n) - n / 2 + 0.5
r2 = x[None] ** 2 + x[:, None] ** 2
k = np.pi / (2 * 1448.0)
targets["zone"] = (1 + np.cos(k * r2)) / 2
im = np.asarray(Image.open("/home/fzeng/ml/research/art/gd-bifurcation/gallery/basin_wide_dark.png").convert("RGB"), dtype=np.float64) / 255
im = c.box_down(im, 2)
targets["basin"] = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]
targets["ramp"] = np.tile(np.linspace(0, 1, 2048), (256, 1))
out = {}
for name, v in targets.items():
    out[f"{name}_input"] = v.astype(np.float32)
    for m, b in all_methods(v).items():
        out[f"{name}_{m}"] = np.packbits(b, axis=None)
        out[f"{name}_{m}_shape"] = np.array(b.shape)
    print(name, flush=True)

stats = {"mean_tone_error": {}}
spectra = {}
for g in (1 / 8, 1 / 4, 1 / 3, 1 / 2):
    v = np.full((1024, 1024), g)
    for m, b in all_methods(v).items():
        bb = b.astype(np.float64)
        P = np.abs(np.fft.fftshift(np.fft.fft2(bb - bb.mean()))) ** 2
        yy, xx = np.mgrid[-512:512, -512:512] / 1024
        rad = np.sqrt(xx ** 2 + yy ** 2)
        bins = np.clip((rad / 0.5 * 128).astype(int), 0, 180)
        prof = np.bincount(bins.ravel(), P.ravel(), 181) / np.maximum(np.bincount(bins.ravel(), minlength=181), 1)
        spectra[f"{g:.4f}_{m}"] = prof.astype(np.float32)
        spectra[f"{g:.4f}_{m}_2d"] = np.log10(P + 1e-6).astype(np.float16)
        lf = float(P[rad < 0.1].sum() / P.sum())
        stats.setdefault(f"lowfreq_energy_frac_r<0.1cyc/px_gray{g:.4f}", {})[m] = round(lf, 5)
        stats["mean_tone_error"][f"{g:.4f}_{m}"] = round(float(bb.mean() - g), 5)
stats["lowfreq_area_frac_(white-noise expectation)"] = float(np.mean(np.sqrt(((np.mgrid[-512:512, -512:512] / 1024) ** 2).sum(0)) < 0.1))
np.savez("cache/halftones.npz", **out)
np.savez("cache/halftone_spectra.npz", **spectra)
json.dump(stats, open("cache/halftone_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
