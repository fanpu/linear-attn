"""Zooms into the first-order sigma-delta idle-tone fan along the DC-input axis.

Rotation number rho = (1+u)/2. Rays fold(k rho) cross at rational rho = p/q (at frequencies j/q folded).
  static: two centres, rho = 2/3 (rational) and rho = 1/phi = 0.6180... (golden, the "most irrational" number,
          whose best rational approximants are Fibonacci ratios F_n/F_{n+1}), 5 zoom levels each (x5 per level);
          plus the second-order modulator at the rational centre.
  movie:  continuous exponential zoom toward rho = 1/phi, 240 frames, half-width 0.38 -> 5e-5 in rho.
Frequency axis is never zoomed (0 .. fs/2). Outputs cache/sdzoom_static.npz, cache/sdzoom_movie.npy
Run: OMP_NUM_THREADS=4 python compute_sdzoom.py [static] [movie | movie_c]   (movie_c needs cache/sd.so from compute_extras.py B2; used for the final movie)
"""
import sys
import numpy as np
from scipy.signal import get_window
import dsp

which = sys.argv[1:] or ["static", "movie"]
PHI = (1 + 5 ** 0.5) / 2


def fan(rho_lo, rho_hi, rows, N, pool, order=1, warm=2048):
    rho = np.linspace(rho_hi, rho_lo, rows)           # row 0 = top = highest rho
    u = 2 * rho - 1
    y = dsp.sigma_delta(np.broadcast_to(u[:, None], (rows, N + warm)), order=order)[:, warm:].astype(np.float64)
    w = get_window("blackmanharris", N, fftbins=True)
    P = np.abs(np.fft.rfft((y - y.mean(1, keepdims=True)) * w, axis=1)) ** 2 / (w.sum() / 2) ** 2
    P = P[:, :(P.shape[1] // pool) * pool].reshape(rows, -1, pool).max(2)
    return (10 * np.log10(P + 1e-20)).astype(np.float16), rho


if "static" in which:
    out = {}
    centres = {"r23": 2 / 3, "gold": 1 / PHI}
    hws = [0.03 / 5 ** i for i in range(5)]           # half-widths in rho: 0.03 ... 4.8e-5
    for name, c in centres.items():
        for i, hw in enumerate(hws):
            S, rho = fan(c - hw, c + hw, 2048, 2 ** 14, 4)
            out[f"{name}_{i}"] = S
            out[f"{name}_{i}_rho"] = rho
            print(name, i, hw)
    for i, hw in enumerate(hws[:3]):
        S, rho = fan(2 / 3 - hw, 2 / 3 + hw, 2048, 2 ** 14, 4, order=2)
        out[f"r23o2_{i}"] = S
        out[f"r23o2_{i}_rho"] = rho
        print("order2", i)
    np.savez("cache/sdzoom_static.npz", hws=np.array(hws), **out)

if "movie_c" in which:
    # same zoom, C modulator and N = 2^17 samples per input (floor ~7.6e-6 in rho instead of 1.2e-4)
    import ctypes, os
    lib = ctypes.CDLL(os.path.abspath("cache/sd.so"))
    F, N, WARM, ROWS, COLS = 240, 2 ** 17, 4096, 1080, 1024
    c = 1 / PHI
    hw = 0.38 * (5e-5 / 0.38) ** (np.arange(F) / (F - 1))
    w = get_window("blackmanharris", N, fftbins=True)
    norm = (w.sum() / 2) ** 2
    frames = np.empty((F, ROWS, COLS), np.float16)
    u = np.empty(N + WARM); y = np.empty(N + WARM, np.int8)
    for f in range(F):
        rho = np.linspace(c + hw[f], c - hw[f], ROWS)
        for r in range(ROWS):
            u[:] = 2 * rho[r] - 1
            lib.sd1(u.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)), ctypes.c_long(N + WARM))
            yy = y[WARM:].astype(np.float64)
            P = np.abs(np.fft.rfft((yy - yy.mean()) * w)) ** 2 / norm
            frames[f, r] = 10 * np.log10(P[: COLS * (len(P) // COLS)].reshape(COLS, -1).max(1) + 1e-20)
        if f % 10 == 0:
            print("frame", f, hw[f], flush=True)
    np.save("cache/sdzoom_movie.npy", frames)
    np.save("cache/sdzoom_movie_hw.npy", hw)

if "movie" in which:
    F = 240
    c = 1 / PHI
    hw = 0.38 * (5e-5 / 0.38) ** (np.arange(F) / (F - 1))
    frames = np.empty((F, 1080, 1024), np.float16)
    for f in range(F):
        frames[f], _ = fan(c - hw[f], c + hw[f], 1080, 2 ** 13, 4)
        if f % 20 == 0:
            print("frame", f, hw[f], flush=True)
    np.save("cache/sdzoom_movie.npy", frames)
    np.save("cache/sdzoom_movie_hw.npy", hw)
