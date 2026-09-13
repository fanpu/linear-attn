"""Key plate: the measured 3-bit spectrogram with the predicted folded harmonic curves fold(k f(t)) drawn on top.
f(t) = 20 Hz * 1200^(t/30 s); fold(f) = min(f mod fs, fs - f mod fs). Only odd k (the error of a symmetric
mid-tread quantizer on a sine has half-wave symmetry, so even harmonics vanish: measured odd/even power ratio 53.5 dB).
Styles: paper (survey sheet, red key) | dark."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import render_common as rc
import dsp

T0, T1 = 16.0, 26.0
KS = (1, 3, 5, 7, 9, 11, 13)


def render(st):
    S = rc.load_spec("int3")
    meta = rc.META
    hop, nfft = meta["HOP"], meta["NFFT"]
    times = (np.arange(S.shape[0]) * hop + nfft / 2) / dsp.FS
    sel = (times >= T0) & (times <= T1)
    img = rc.resample_power(S[sel])
    lo = np.percentile(img, 50)
    v = rc.unit(img, lo, lo + 55, 0.9)
    if st == "paper":
        bg, fg, key = tuple(rc.PAPER), tuple(rc.INK), (0.78, 0.1, 0.1)
        rgb = rc.ink_on_paper(v, gamma=0.8)
    else:
        bg, fg, key = (0.02, 0.018, 0.03), (0.86, 0.83, 0.78), (0.35, 0.95, 0.85)
        rgb = rc.cmap_rgb(v, "magma")
    fig = plt.figure(figsize=(28, 16), dpi=110, facecolor=bg)
    ax = fig.add_axes([0.05, 0.08, 0.93, 0.78])
    ax.imshow(rgb, aspect="auto", extent=[T0, T1, 0, 24], interpolation="antialiased")
    t = np.linspace(T0, T1, 4000)
    f = dsp.inst_freq(t, 30.0, 20.0, 24000.0)
    for k in KS:
        fk = dsp.fold(k * f) / 1000
        ax.plot(t, fk, color=key, lw=1.3, ls=(0, (6, 5)), alpha=0.95)
        # label at first point where the curve is well inside the frame
        i = np.argmin(np.abs(t - (T0 + 0.35 + 0.8 * KS.index(k))))
        ax.text(t[i], fk[i] + 0.35, f"k={k}", color=key, fontsize=15, family="DejaVu Sans Mono",
                bbox=dict(facecolor=bg, edgecolor="none", pad=1.5, alpha=0.85))
    ax.axhline(24, color=key, lw=2)
    ax.text(T1 - 0.05, 23.3, "Nyquist 24 kHz: harmonics reflect down", color=key, fontsize=16, ha="right", va="top",
            bbox=dict(facecolor=bg, edgecolor="none", alpha=0.85))
    ax.text(T1 - 0.05, 0.4, "0 Hz: and reflect up again", color=key, fontsize=16, ha="right",
            bbox=dict(facecolor=bg, edgecolor="none", alpha=0.85))
    ax.set_xlim(T0, T1); ax.set_ylim(0, 24)
    ax.tick_params(colors=fg, labelsize=14)
    for s in ax.spines.values():
        s.set_color(fg)
    ax.set_xlabel("time [s]", color=fg, fontsize=16); ax.set_ylabel("frequency [kHz]", color=fg, fontsize=16)
    fig.text(0.05, 0.965, "KEY  -  every line in the lattice is a harmonic k of the chirp, folded by sampling", fontsize=34,
             color=fg, family="DejaVu Serif", va="top")
    fig.text(0.05, 0.922, "measured: 3-bit undithered spectrogram, 16-26 s.   drawn: fold(k f(t)) for odd k <= 13, "
             "f(t) = 20 Hz x 1200^(t/30 s).   Unlabelled lines are higher odd k (the strongest harmonics of a 7-level "
             "quantizer are near k ~ 2 pi A/Delta ~ 19).\nNote the dashed k=5 with NO line under it: at A = 3 LSB the 5th harmonic's Bessel sum "
             "b_5 = sum_n 2(-1)^n J_5(2 pi n A)/(pi n) nearly vanishes (-48 dB, vs -18 to -27 dB for k = 1..13). See the amplitude map.", fontsize=15, color=fg, va="top")
    out = f"{rc.GAL}/key_int3_{st}.png"
    fig.savefig(out, facecolor=bg); plt.close(fig)
    rc.save_png(Image.open(out).convert("RGB"), out)
    print(out)


for s in (sys.argv[1:] or ["paper", "dark"]):
    render(s)
