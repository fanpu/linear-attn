"""The recursive Bayer ordered-dither matrix: construction, exact self-similarity checks, 1-bit ramps, spectra.

M_1 = [[0, 2], [3, 1]],   M_2k = [[4 M_k, 4 M_k + 2], [4 M_k + 3, 4 M_k + 1]]      (sizes 2, 4, ..., 256)
Equivalent closed form (checked for all 8 orders): M(x, y) = sum_b d_b 4^(n-1-b),  d_b = 2 (x_b XOR y_b) + y_b,
so the FINEST coordinate bits set the MOST significant digits: neighbouring pixels get far-apart thresholds.
Self-similarity (exact, checked): the on-set {M_n < c 4^(n-m)} depends only on (x mod 2^m, y mod 2^m) and equals the
tiled on-set {M_m < c}: gray level c/4^m is a 2^m-periodic tiling of the order-m pattern.
Outputs: cache/bayer.npz, cache/bayer_stats.json.  Run: python compute_bayer.py
"""
import json
import numpy as np
import common as c

st = {}
mats = {n: c.bayer(n) for n in range(1, 9)}
st["closed_form_equals_recursion_orders_1_8"] = all(np.array_equal(mats[n], c.bayer_closed_form(n)) for n in range(1, 9))
st["is_permutation"] = all(np.array_equal(np.sort(mats[n].ravel()), np.arange(4 ** n)) for n in range(1, 9))

# exact self-similarity of level sets
n = 8
M = mats[n]
ok = True
checked = 0
for m in range(1, n + 1):
    Mm = mats[m]
    for cc in range(0, 4 ** m + 1):
        A = M < cc * 4 ** (n - m)
        B = np.tile(Mm < cc, (2 ** (n - m), 2 ** (n - m)))
        ok &= np.array_equal(A, B)
        checked += 1
st["level_sets_c_over_4^m_equal_tiled_order_m_pattern"] = bool(ok)
st["level_sets_checked"] = checked

# minimum toroidal distance between the first j pixels switched on (dispersion), vs order
def min_dist(Mn, j):
    s = Mn.shape[0]
    ys, xs = np.nonzero(Mn < j)
    if len(xs) < 2:
        return None
    dx = np.abs(xs[:, None] - xs[None]); dx = np.minimum(dx, s - dx)
    dy = np.abs(ys[:, None] - ys[None]); dy = np.minimum(dy, s - dy)
    d = np.sqrt(dx ** 2 + dy ** 2); np.fill_diagonal(d, np.inf)
    return float(d.min())
st["min_dist_first_4^k_on_pixels_order8"] = {k: min_dist(mats[8], 4 ** k) for k in range(1, 6)}

# 1-bit ramps: horizontal gray ramp 0..1 dithered with each order (and FS, blue noise for later comparison)
W, H = 2048, 160
ramp = np.tile(np.linspace(0, 1, W), (H, 1))
ramps = {f"bayer{2 ** k}": c.ordered(ramp, mats[k]) for k in range(1, 9)}

# 8x8 specimen: all 65 patterns
spec8 = np.array([(mats[3] < t).astype(np.uint8) for t in range(65)])
# bit planes of M_8 (16 bits)
planes = np.array([(M >> b) & 1 for b in range(16)], dtype=np.uint8)

# spectra of Bayer-256 halftones of constant grays (periodic tile -> exact line spectrum)
grays = [1 / 2, 1 / 4, 1 / 3, 0.1]
spectra = {}
for g in grays:
    B = (M < g * 65536).astype(np.float64)
    F = np.abs(np.fft.fftshift(np.fft.fft2(B - B.mean()))) ** 2
    spectra[f"{g:.4f}"] = F.astype(np.float32)
    nz = int((F > 1e-9 * F.max()).sum())
    st[f"gray_{g:.4f}_nonzero_spectral_lines"] = nz
    # all lines at multiples of 256/2^m ... report the finest period present
np.savez("cache/bayer.npz", **{f"M{2 ** k}": mats[k] for k in range(1, 9)}, spec8=spec8, planes=planes,
         **{f"ramp_{k}": v for k, v in ramps.items()}, **{f"spec_{k}": v for k, v in spectra.items()})
json.dump(st, open("cache/bayer_stats.json", "w"), indent=1)
print(json.dumps(st, indent=1))
