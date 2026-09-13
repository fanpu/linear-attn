"""Compute every chirp spectrogram used by the Dither renders, plus lossless FLAC audio.

Signal: log chirp 20 Hz -> 24 kHz (= Nyquist), 30 s, fs = 48 kHz, peak amplitude 1 (float64).
STFT: 4096-sample Blackman-Harris window, hop 256, power in dB re a full-scale sine's peak bin.
Output: cache/spec_<name>.npy (float16, frames x 2049 bins), cache/spec_meta.json, gallery/audio/*.flac

Run: OMP_NUM_THREADS=4 python compute_spectrograms.py
"""
import json
import os
import subprocess
import numpy as np
from scipy.io import wavfile
import dsp

T, F0, F1 = 30.0, 20.0, 24000.0
NFFT, HOP = 4096, 256
os.makedirs("cache", exist_ok=True)
os.makedirs("gallery/audio", exist_ok=True)

x, t = dsp.log_chirp(T, F0, F1)
fp4, fp8 = dsp.make_fp_quantizer("fp4"), dsp.make_fp_quantizer("fp8")
meta = {}


def save(name, y, desc, audio=False, audio_kind="int", bits=None):
    S, times, freqs = dsp.stft_db(y, nfft=NFFT, hop=HOP)
    np.save(f"cache/spec_{name}.npy", S.astype(np.float16))
    e = y - x
    meta[name] = dict(desc=desc, sinad_dB=float(10 * np.log10(np.mean(x ** 2) / np.mean(e ** 2))),
                      err_rms=float(np.sqrt(np.mean(e ** 2))))
    if audio:
        write_flac(name, y, audio_kind, bits)
    print(name, S.shape, f"SINAD {meta[name]['sinad_dB']:.2f} dB")


def write_flac(name, y, kind, bits):
    """Lossless: int-b codes are written exactly into 16-bit samples; FP values exactly into 24-bit samples.
    Level: -6 dB (codes are shifted/scaled by powers of two only, so the stored values stay exact)."""
    path = f"gallery/audio/{name}.flac"
    if kind == "int":
        codes = np.round(y / dsp.int_step(bits)).astype(np.int64)
        shift = 16 - bits - 1  # peak code 2^(b-1)-1  ->  <= 2^14 : -6 dB, room for dither excursions
        pcm = (codes << shift).astype(np.int16)
        tmp = f"/tmp/dither_{name}.wav"
        wavfile.write(tmp, dsp.FS, pcm)
    else:
        # FP: values in format units (y * vmax); 24-bit scale 2^20 (FP4, max 6) or 2^13 (FP8, max 448 -> -6 dB)
        q = fp4 if kind == "fp4" else fp8
        sc = 2 ** 20 if kind == "fp4" else 2 ** 13
        v = y * q.vmax * sc
        assert np.max(np.abs(v - np.round(v))) < 1e-6, "FP value not representable in 24-bit"
        v = np.round(v)
        pcm = (v.astype(np.int32) << 8)  # 24-bit left-justified in 32-bit container
        tmp = f"/tmp/dither_{name}.wav"
        wavfile.write(tmp, dsp.FS, pcm)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp] +
                   (["-sample_fmt", "s32", "-bits_per_raw_sample", "24"] if kind != "int" else []) +
                   ["-compression_level", "8", path], check=True)
    os.remove(tmp)


rng = np.random.default_rng(2026)

# 1) reference and bit-depth series (undithered)
save("float64", x, "unquantized float64 chirp (window sidelobe floor only)")
for b in (2, 3, 4, 5, 6, 8, 10, 12):
    save(f"int{b}", dsp.quant_int(x, b), f"int{b} undithered", audio=b in (2, 3, 4, 6, 8, 12), bits=b)

# 2) dither variants at 3 and 4 bits
for b in (3, 4):
    save(f"int{b}_rpdf", dsp.quant_int(x, b, dither="rpdf", rng=rng), f"int{b} RPDF (+-1/2 LSB) nonsubtractive",
         audio=b == 4, bits=b)
    save(f"int{b}_tpdf", dsp.quant_int(x, b, dither="tpdf", rng=rng), f"int{b} TPDF (+-1 LSB) nonsubtractive",
         audio=b == 4, bits=b)
    save(f"int{b}_sub", dsp.quant_int(x, b, dither="rpdf", rng=rng, subtractive=True),
         f"int{b} RPDF subtractive (dither removed after quantizing)")

# 3) floating-point grids at three amplitudes, with uniform grids of equal bit count for comparison
for adb in (0, -12, -24):
    a = 10 ** (adb / 20)
    xa = a * x
    tag = f"{-adb:02d}dB"
    # quantizer grid fixed to peak 1 (as a real tensor scale would be); signal is a dB below it
    for name, q in [("fp4", fp4), ("fp8", fp8)]:
        y = q(xa, peak=1.0) / a
        save(f"{name}_{tag}", y, f"{name} grid, chirp at {adb} dB re format max", audio=adb == 0, audio_kind=name)
    for b in (4, 8):
        y = dsp.quant_int(xa, b) / a
        save(f"int{b}_{tag}", y, f"int{b} grid, chirp at {adb} dB re full scale")

json.dump(dict(T=T, F0=F0, F1=F1, NFFT=NFFT, HOP=HOP, FS=dsp.FS, window="blackmanharris",
               db_ref="full-scale sine peak bin", variants=meta), open("cache/spec_meta.json", "w"), indent=1)

# 4) decaying tone trio (noise modulation demo), 6 bits, audio only + short-time error RMS
td = np.arange(int(6 * dsp.FS)) / dsp.FS
xd = np.sin(2 * np.pi * 220 * td) * np.exp(-td / 0.8)
rms = {}
for kind in (None, "rpdf", "tpdf"):
    y = dsp.quant_int(xd, 6, dither=kind, rng=rng)
    nm = f"decay_int6_{kind or 'undithered'}"
    write_flac(nm, y, "int", 6)
    e = (y - xd) / dsp.int_step(6)
    w = 480  # 10 ms
    n = len(e) // w
    rms[nm] = np.sqrt(np.mean(e[:n * w].reshape(n, w) ** 2, axis=1))
np.savez("cache/decay_rms.npz", t=(np.arange(n) + 0.5) * w / dsp.FS, env=np.exp(-(np.arange(n) + .5) * w / dsp.FS / 0.8) / dsp.int_step(6), **rms)
print("done")
