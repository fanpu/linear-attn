"""Deeper golden-ratio zoom panels with a pushed resolution floor.
Levels 3, 4, 5 (x125, x625, x3125; half-widths 2.4e-4, 4.8e-5, 9.6e-6 in rho) around rho = 1/phi, 2048 inputs per
panel, N = 2^20 samples per input after 4096 warm-up (C modulator, checked bit-exact against dsp.sigma_delta),
Blackman-Harris. New floor: 1/N ~ 9.5e-7 in rho.
Frequency axis: at these depths a full-band column would max-pool ~256 bins, each full of high-k lines, which paints
vertical stripes (first attempt). So the deep panels keep only the band f in [F0 - 0.01, F0 + 0.01] with
F0 = 1 - 1/phi = fold(rho*) (the k = 1 ray passes through the centre), max-pooled from ~20 970 bins to 2048 columns.
Saves cache/sdzoom_deep.npz"""
import ctypes
import os
import subprocess
import numpy as np
from scipy.signal import get_window
import dsp

if not os.path.exists("cache/sd.so"):
    raise SystemExit("run compute_extras.py B2 first (builds cache/sd.so)")
lib = ctypes.CDLL(os.path.abspath("cache/sd.so"))
ut = np.full(6000, 0.2360679)
yc = np.empty(6000, np.int8)
lib.sd1(ut.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), yc.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)), ctypes.c_long(6000))
assert np.array_equal(yc, dsp.sigma_delta(ut[None], order=1)[0])

PHI = (1 + 5 ** 0.5) / 2
N, WARM, ROWS, COLS = 2 ** 20, 4096, 2048, 2048
w = get_window("blackmanharris", N, fftbins=True)
norm = (w.sum() / 2) ** 2
F0, FHW = 1 - 1 / PHI, 0.01
POOL = int(np.floor(2 * FHW * N / COLS))
B0 = int(round((F0 - FHW) * N))
out = {}
for lev in (3, 4, 5):
    hw = 0.03 / 5 ** lev
    rho = np.linspace(1 / PHI + hw, 1 / PHI - hw, ROWS)
    S = np.empty((ROWS, COLS), np.float16)
    u = np.empty(N + WARM)
    y = np.empty(N + WARM, np.int8)
    for r in range(ROWS):
        u[:] = 2 * rho[r] - 1
        lib.sd1(u.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)), ctypes.c_long(N + WARM))
        yy = y[WARM:].astype(np.float64)
        P = np.abs(np.fft.rfft((yy - yy.mean()) * w)) ** 2 / norm
        P = P[B0:B0 + COLS * POOL].reshape(COLS, POOL).max(1)
        S[r] = 10 * np.log10(P + 1e-20)
        if r % 512 == 0:
            print(lev, r, flush=True)
    out[f"gold_{lev}"] = S
    out[f"gold_{lev}_rho"] = rho
np.savez("cache/sdzoom_deep.npz", N=N, f_lo=B0 / N, f_hi=(B0 + COLS * POOL) / N, **out)
print("done")
