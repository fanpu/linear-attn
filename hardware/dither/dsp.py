"""Shared exact-math DSP for *Dither*: quantizers, number-format grids, dither, chirp, STFT, sigma-delta.

Conventions (declared, used everywhere in this project):
  * Signals are float64 in "peak units": the test signal has peak amplitude A (default 1).
  * int-b quantizer is ML-style *symmetric* abs-max: step  Delta = A / (2^(b-1) - 1),
    so the signal spans the 2^b - 1 codes  -(2^(b-1)-1) .. +(2^(b-1)-1).
    The grid is NOT clipped (dither can push a sample one or two codes past the peak code;
    a real b-bit converter would clip there). This keeps harmonics from clipping out of the picture.
  * FP formats: every bit pattern is enumerated and decoded; values are rounded to the nearest
    representable value, ties to even mantissa, saturating at +-max (E4M3FN has no infinity).
    The signal is scaled so its peak equals the format's max representable value.
"""
import numpy as np
from scipy.signal import get_window

FS = 48_000

# ---------------------------------------------------------------- number formats

def fp_grid(exp_bits, man_bits, bias, fn_nan_top=False):
    """Return (sorted non-negative values, even-mantissa flag) for a sign/exp/mantissa minifloat.
    fn_nan_top: E4M3FN convention: top exponent is finite, only the all-ones pattern is NaN."""
    vals, even = [], []
    for e in range(2 ** exp_bits):
        for m in range(2 ** man_bits):
            if fn_nan_top and e == 2 ** exp_bits - 1 and m == 2 ** man_bits - 1:
                continue  # NaN
            if e == 0:
                v = 2.0 ** (1 - bias) * (m / 2 ** man_bits)
            else:
                v = 2.0 ** (e - bias) * (1 + m / 2 ** man_bits)
            vals.append(v)
            even.append(m % 2 == 0)
    vals = np.array(vals)
    order = np.argsort(vals)
    return vals[order], np.array(even)[order]


FP4_E2M1 = fp_grid(2, 1, 1)               # 0, .5, 1, 1.5, 2, 3, 4, 6
FP8_E4M3 = fp_grid(4, 3, 7, fn_nan_top=True)  # max 448, min subnormal 2^-9


def signed_grid(pos_vals, pos_even):
    g = np.concatenate([-pos_vals[:0:-1], pos_vals])
    ev = np.concatenate([pos_even[:0:-1], pos_even])
    return g, ev


def round_to_grid(x, grid, even):
    """Nearest value of sorted `grid`, ties to the even-mantissa neighbour, saturating."""
    x = np.asarray(x, dtype=np.float64)
    xc = np.clip(x, grid[0], grid[-1])
    hi = np.clip(np.searchsorted(grid, xc, side="left"), 1, len(grid) - 1)
    lo = hi - 1
    dlo = xc - grid[lo]
    dhi = grid[hi] - xc
    pick_hi = (dhi < dlo) | ((dhi == dlo) & even[hi])
    return np.where(pick_hi, grid[hi], grid[lo])


def make_fp_quantizer(fmt):
    pos, ev = {"fp4": FP4_E2M1, "fp8": FP8_E4M3}[fmt]
    g, e = signed_grid(pos, ev)
    vmax = pos[-1]

    def q(x, peak=1.0):
        """Quantize signal whose nominal peak is `peak`, mapping peak -> format max."""
        s = vmax / peak
        return round_to_grid(x * s, g, e) / s

    q.vmax = vmax
    q.grid = g
    return q


def int_step(bits, peak=1.0):
    return peak / (2 ** (bits - 1) - 1)


def quant_int(x, bits, peak=1.0, dither=None, rng=None, subtractive=False):
    """Uniform mid-tread quantizer with step int_step(bits). dither in {None,'rpdf','tpdf'} (LSB units)."""
    d = int_step(bits, peak)
    u = x / d
    nu = make_dither(dither, u.shape, rng)
    y = np.round(u + nu)
    if subtractive:
        y = y - nu
    return y * d


def make_dither(kind, shape, rng):
    if kind is None:
        return 0.0
    rng = rng or np.random.default_rng(0)
    if kind == "rpdf":
        return rng.random(shape) - 0.5
    if kind == "tpdf":
        return (rng.random(shape) - 0.5) + (rng.random(shape) - 0.5)
    raise ValueError(kind)


# ---------------------------------------------------------------- signals

def log_chirp(T, f0, f1, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(int(T * fs)) / fs
    k = np.log(f1 / f0) / T
    ph = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    return amp * np.sin(ph + phase), t


def inst_freq(t, T, f0, f1):
    return f0 * (f1 / f0) ** (t / T)


def fold(f, fs=FS):
    """Alias frequency f (Hz) into [0, fs/2]."""
    r = np.mod(f, fs)
    return np.minimum(r, fs - r)


# ---------------------------------------------------------------- analysis

def stft_db(x, nfft=4096, hop=256, window="blackmanharris", fs=FS, ref=None):
    """Power spectrogram in dB, frames x bins (float32). ref: power normalisation (default: full-scale sine)."""
    w = get_window(window, nfft, fftbins=True)
    n = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    out = np.empty((n, nfft // 2 + 1), dtype=np.float32)
    # full-scale sine of amplitude 1 -> peak bin power (sum(w)/2)^2
    ref = ref or (w.sum() / 2) ** 2
    for s in range(0, n, 512):
        fr = x[idx[s:s + 512]] * w
        P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        out[s:s + 512] = 10 * np.log10(P / ref + 1e-30)
    times = (np.arange(n) * hop + nfft / 2) / fs
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    return out, times, freqs


# ---------------------------------------------------------------- sigma-delta

def sigma_delta(u, order=2):
    """1-bit sigma-delta modulator in error-feedback form, vectorised over leading axes of u ([..., T]).
    w[n] = u[n] + sum_k c_k e[n-k],  y[n] = sign(w[n]) (+-1),  e[n] = y[n] - w[n]
    => Y = U + (1 - z^-1)^order E   (noise transfer function (1-z^-1)^order).
    order 1 is the classic first-order modulator (idle tones); order 2 is stable for |u| <~ 0.6."""
    c = {1: [-1.0], 2: [-2.0, 1.0], 3: [-3.0, 3.0, -1.0]}[order]
    u = np.asarray(u, dtype=np.float64)
    shp = u.shape[:-1]
    T = u.shape[-1]
    y = np.empty(u.shape, dtype=np.int8)
    es = [np.zeros(shp) for _ in c]
    for n in range(T):
        w = u[..., n].copy()
        for ck, ek in zip(c, es):
            w += ck * ek
        yn = np.where(w >= 0, 1.0, -1.0)
        e = yn - w
        es = [e] + es[:-1]
        y[..., n] = yn
    return y
