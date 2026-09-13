"""Scrolling spectrogram video with its own audio: a 3-bit chirp undithered, then the same chirp with TPDF dither.

The audio track IS the signal that is analysed (computed here, seed fixed), so picture and sound are the same data.
Timeline: 0-30 s undithered | 0.75 s silence | 30.75-60.75 s TPDF.  1920x1080, 30 fps, 192 px per second:
one STFT column (hop 250 samples) per pixel. Playhead at x = 1440; the not-yet-heard future to its right is dimmed.
Audio in the MP4 is AAC 192 kb/s (lossy, for compatibility); the lossless FLAC is saved next to it.
Run: OMP_NUM_THREADS=4 python render_video.py
"""
import os
import subprocess
import numpy as np
from scipy.io import wavfile
from PIL import Image
import dsp
import render_common as rc

FS = dsp.FS
PXS = 192
HOP = FS // PXS            # 250
NFFT = 4096
W, H = 1920, 1080
SPEC_T, SPEC_H = 110, 820  # spectrogram band
PLAY = 1440
FPS = 30
BITS = 3

x, _ = dsp.log_chirp(30.0, 20.0, 24000.0)
rng = np.random.default_rng(99)
gap = np.zeros(int(0.75 * FS))
y = np.concatenate([dsp.quant_int(x, BITS), gap, dsp.quant_int(x, BITS, dither="tpdf", rng=rng)])
codes = np.round(y / dsp.int_step(BITS)).astype(np.int64)
pcm = (codes << (16 - BITS - 1)).astype(np.int16)        # exact codes, -6 dB headroom
os.makedirs(f"{rc.GAL}/audio", exist_ok=True)
wav = "/tmp/dither_video.wav"
wavfile.write(wav, FS, pcm)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-compression_level", "8",
                f"{rc.GAL}/audio/video_int3_undithered_then_tpdf.flac"], check=True)

# analysis of exactly that signal (pad so column i is centred on sample i*HOP)
yp = np.concatenate([np.zeros(NFFT // 2), y, np.zeros(NFFT)])
S, _, _ = dsp.stft_db(yp, nfft=NFFT, hop=HOP)
S = S[: len(y) // HOP]
P = 10 ** (S.astype(np.float64) / 10)
k = P.shape[1] // SPEC_H
P = P[:, : k * SPEC_H].reshape(P.shape[0], SPEC_H, k).mean(2)   # 2049 bins -> 900 rows, power-mean
D = (10 * np.log10(P + 1e-30)).T[::-1]
lo = np.percentile(D[:, : 30 * PXS], 65)
strip = (rc.cmap_rgb(rc.unit(D, lo, lo + 50, 0.9), "magma") * 255).astype(np.uint8)   # H x T x 3
print("strip", strip.shape, "black point", lo)

# static overlay (labels) rendered once
bg = Image.new("RGB", (W, H), (4, 3, 7))
rc.text(bg, (60, 30), "DITHER  -  hear the lattice, then hear it melt", 46, (0.88, 0.85, 0.8), rc.FONT_SERIF)
rc.text(bg, (60, H - 52), f"3-bit (7-level) log chirp 20 Hz - 24 kHz; left 30 s undithered, then TPDF dither.  STFT 4096 "
        f"Blackman-Harris, 1 px = {1000 / PXS:.2f} ms, 0-24 kHz linear.  numpy float64, 2026-09-13", 22, (0.7, 0.68, 0.64), rc.FONT_MONO)
base = np.array(bg)
for f_k in (0, 6, 12, 18, 24):
    pass
labels = {}
for key, txt in (("u", "UNDITHERED  -  error = harmonics, folded at Nyquist"), ("t", "TPDF DITHER  -  error = steady noise, no lattice")):
    im = Image.new("RGB", (1100, 60), (4, 3, 7))
    rc.text(im, (0, 8), txt, 32, (1.0, 0.8, 0.5), rc.FONT_SANS)
    labels[key] = np.array(im)

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", f"{rc.GAL}/video_scroll_int3.mp4"]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
T = strip.shape[1]
nframes = int(len(y) / FS * FPS)
for n in range(nframes):
    t = n / FPS
    c = int(round(t * PXS))
    fr = base.copy()
    a, b = c - PLAY, c + (W - PLAY)          # strip columns visible in [0, W)
    sa, sb = max(a, 0), min(b, T)
    if sb > sa:
        fr[SPEC_T:SPEC_T + SPEC_H, sa - a:sb - a] = strip[:, sa:sb]
    fr[SPEC_T:SPEC_T + SPEC_H, PLAY + 1:] = (fr[SPEC_T:SPEC_T + SPEC_H, PLAY + 1:] * 0.35).astype(np.uint8)
    fr[SPEC_T:SPEC_T + SPEC_H, PLAY - 1:PLAY + 1] = 230
    fr[H - 125:H - 65, 60:60 + 1100] = labels["u"] if t < 30.4 else labels["t"]
    proc.stdin.write(fr.tobytes())
proc.stdin.close()
proc.wait()
os.remove(wav)
print("video", os.path.getsize(f"{rc.GAL}/video_scroll_int3.mp4") / 1e6, "MB")
