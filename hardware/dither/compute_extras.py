"""Extra pieces (chosen from the brainstorm in the README):

(A) Harmonic-amplitude maps: |b_k(A)| of the quantization error of a sine vs amplitude A, for uniform (int) and
    floating-point (FP8 E4M3, FP4 E2M1) grids. Exact up to FFT aliasing of the sampled period (M = 2^16 points per
    period; contamination of k <= 301 from k' ~ 2^16 is below ~ -45 dB re the 1/k envelope).
(B) 1-bit sigma-delta modulators (error-feedback form, NTF (1 - z^-1)^L):
    B1 idle-tone maps: spectrum of the 1-bit output vs DC input u (order 1 and 2), and a zoom (order 1).
    B2 a log chirp at DSD64 rate (fs = 64 x 48 kHz = 3.072 MHz), order 2, as a spectrogram.
Outputs in cache/. Run: OMP_NUM_THREADS=4 python compute_extras.py [A] [B1] [B2]
"""
import ctypes
import os
import subprocess
import sys
import numpy as np
from scipy.signal import get_window
import dsp

which = sys.argv[1:] or ["A", "B1", "B2"]
os.makedirs("cache", exist_ok=True)

# ------------------------------------------------------------------ (A) harmonic amplitude maps
if "A" in which:
    M = 2 ** 16
    K = 301
    th = 2 * np.pi * (np.arange(M) + 0.5) / M
    s = np.sin(th)

    def harm(qfun, amps, rel):
        out = np.empty((len(amps), K + 1), dtype=np.float32)
        for i0 in range(0, len(amps), 32):
            a = amps[i0:i0 + 32, None]
            x = a * s[None]
            e = qfun(x) - x
            c = np.abs(np.fft.rfft(e, axis=1)[:, :K + 1]) / (M / 2)
            if rel:
                c = c / a
            out[i0:i0 + 32] = c
        return out

    lin = np.linspace(0.005, 24.0, 2400)                         # LSB units, uniform grid step 1
    hint = harm(np.round, lin, rel=False)
    fp8 = dsp.make_fp_quantizer("fp8")
    fp4 = dsp.make_fp_quantizer("fp4")
    log8 = 448.0 * 2.0 ** np.linspace(-12, 0, 2400)              # FP8 format units, 12 octaves below max
    h8 = harm(lambda x: fp8(x, peak=448.0), log8, rel=True)
    log4 = 6.0 * 2.0 ** np.linspace(-6, 0, 2400)                 # FP4: 6 octaves below max
    h4 = harm(lambda x: fp4(x, peak=6.0), log4, rel=True)
    logi = 127.0 * 2.0 ** np.linspace(-12, 0, 2400)              # int8-like uniform grid on the same log axis
    hi8 = harm(np.round, logi, rel=True)
    np.savez("cache/harmonic_maps.npz", lin_amp=lin, int_lin=hint, fp8_amp=log8, fp8=h8, fp4_amp=log4, fp4=h4,
             int8log_amp=logi, int8log=hi8)
    # octave check on the FP8 map: row i vs row i + 200 (exactly one octave apart on this grid)
    d = np.abs(20 * np.log10(h8[200:1000] + 1e-12) - 20 * np.log10(h8[400:1200] + 1e-12))
    print("A done; FP8 relative-harmonic map, one-octave shift, median |diff| dB over k<=301, 8-4 octaves below max:",
          float(np.median(d)))

# ------------------------------------------------------------------ (B1) idle-tone maps
if "B1" in which:
    def dc_map(us, order, N=2 ** 14, warm=4096, pool=2):
        y = dsp.sigma_delta(np.broadcast_to(us[:, None], (len(us), N + warm)), order=order)[:, warm:].astype(np.float64)
        w = get_window("blackmanharris", N, fftbins=True)
        P = np.abs(np.fft.rfft((y - y.mean(1, keepdims=True)) * w, axis=1)) ** 2 / (w.sum() / 2) ** 2
        P = P[:, :(P.shape[1] // pool) * pool].reshape(len(us), -1, pool).max(2)  # max-pool: keeps thin lines
        return (10 * np.log10(P + 1e-20)).astype(np.float16)

    rng = np.random.default_rng(3)
    u1 = np.linspace(-0.999, 0.999, 4096) + 1e-7
    m1 = dc_map(u1, 1)
    print("B1 order1")
    u2 = np.linspace(-0.6, 0.6, 4096) + 1e-7
    m2 = dc_map(u2, 2)
    print("B1 order2")
    uz = np.linspace(0.30, 0.42, 4096) + 1e-7
    mz = dc_map(uz, 1)
    print("B1 zoom")
    np.savez("cache/sd_dc_maps.npz", u1=u1, m1=m1, u2=u2, m2=m2, uz=uz, mz=mz, pool=2, pool_mode="max", N=2 ** 14, window="blackmanharris")

# ------------------------------------------------------------------ (B2) DSD64 chirp
if "B2" in which:
    C = r"""
    void sd2(const double *u, signed char *y, long n){
      double e1=0,e2=0;
      for(long i=0;i<n;i++){ double w=u[i]-2*e1+e2; double q = w>=0?1.0:-1.0; double e=q-w; e2=e1; e1=e; y[i]=(signed char)q; }
    }
    void sd1(const double *u, signed char *y, long n){
      double e1=0;
      for(long i=0;i<n;i++){ double w=u[i]-e1; double q = w>=0?1.0:-1.0; e1=q-w; y[i]=(signed char)q; }
    }"""
    open("cache/sd.c", "w").write(C)
    subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "cache/sd.c", "-o", "cache/sd.so"], check=True)
    lib = ctypes.CDLL(os.path.abspath("cache/sd.so"))
    # check the C modulator against the numpy reference
    ut = 0.4 * np.sin(np.arange(5000) * 0.01)
    yc = np.empty(5000, np.int8)
    lib.sd2(ut.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), yc.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)), ctypes.c_long(5000))
    assert np.array_equal(yc, dsp.sigma_delta(ut[None], order=2)[0]), "C and numpy sigma-delta disagree"
    FS64 = 64 * 48000
    T = 8.0
    t = np.arange(int(T * FS64)) / FS64
    k = np.log(1000.0) / T
    u = 0.5 * np.sin(2 * np.pi * 20.0 * (np.exp(k * t) - 1) / k)  # 20 Hz -> 20 kHz
    res = {}
    for order, fn in ((1, lib.sd1), (2, lib.sd2)):
        y = np.empty(len(u), np.int8)
        fn(u.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)), ctypes.c_long(len(u)))
        nfft, hop = 2 ** 16, 2 ** 14
        w = get_window("blackmanharris", nfft, fftbins=True)
        nfr = 1 + (len(y) - nfft) // hop
        freqs = np.fft.rfftfreq(nfft, 1 / FS64)
        edges = np.geomspace(20.0, FS64 / 2, 1201)
        idx = np.searchsorted(freqs, edges)
        S = np.empty((nfr, 1200), np.float32)
        for i in range(nfr):
            P = np.abs(np.fft.rfft(y[i * hop:i * hop + nfft] * w)) ** 2 / (w.sum() / 2) ** 2
            cs = np.concatenate([[0], np.cumsum(P)])
            a, b = idx[:-1], np.maximum(idx[1:], idx[:-1] + 1)
            S[i] = 10 * np.log10((cs[b] - cs[a]) / (b - a) + 1e-20)
        res[f"order{order}"] = S
        # in-band (<20 kHz) vs out-of-band noise, excluding the chirp region is not needed for a coarse figure
        print("B2 order", order, S.shape)
        if order == 2:
            # 48 kHz decimated audio: brick-wall FFT lowpass of the 1-bit stream, block-wise
            blk = FS64  # 1 s blocks
            out = []
            for s0 in range(0, len(y), blk):
                seg = y[s0:s0 + blk].astype(np.float64)
                F = np.fft.rfft(seg)
                F[int(20000 * len(seg) / FS64):] = 0
                out.append(np.fft.irfft(F, len(seg))[::64])
            np.save("cache/sd2_chirp_48k.npy", np.concatenate(out).astype(np.float32))
    np.savez("cache/sd_chirp_spec.npz", edges=edges, hop_s=hop / FS64, nfft=nfft, fs=FS64, **res)
    print("B2 done")
