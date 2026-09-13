"""Render the extra pieces from compute_extras.py in three styles each (dark | paper | riso).

  harmonics_int     |b_k(A)| of round(A sin) - A sin vs amplitude A (0-24 LSB, linear) and odd harmonic k (1-301)
  harmonics_fp8     relative |b_k|/A for FP8 E4M3 over 12 octaves of amplitude (log axis): exact octave repeat
  harmonics_fp4     same for FP4 E2M1 over 6 octaves
  harmonics_int8log uniform 255-level grid on the same log axis (the non-repeating control)
  sd_idle_order1 / sd_idle_order2 / sd_idle_zoom    1-bit sigma-delta output spectrum vs DC input
  sd_dsd64_order1 / order2                         spectrogram of a chirp through a 1-bit modulator at 3.072 MHz
"""
import sys
import numpy as np
from PIL import Image
import render_common as rc

H = np.load(f"{rc.CACHE}/harmonic_maps.npz")
SD = np.load(f"{rc.CACHE}/sd_dc_maps.npz")
CH = np.load(f"{rc.CACHE}/sd_chirp_spec.npz")


def colorize(v, st, cmap="magma"):
    if st == "dark":
        return rc.cmap_rgb(v, cmap)
    if st == "paper":
        return rc.ink_on_paper(v, gamma=1.0)
    blue = rc.floyd_steinberg(v)
    pink = rc.floyd_steinberg(np.clip((v - 0.55) / 0.35, 0, 1))
    return rc.multiply_layers([(blue, rc.RISO_BLUE), (rc.shift(pink, 2, -2), rc.RISO_PINK)])


def plate(v, st, title, sub, xlab, ylab, out, xt=(), yt=()):
    """v: image in [0,1], row 0 = top. xt/yt: lists of (fraction, label)."""
    h, w = v.shape
    import textwrap
    L, R, T, B = 190, 60, 250, 170
    lines = textwrap.wrap(sub, width=max(60, int((w + 100) / 13.6)))
    dark = st == "dark"
    bg = (0.02, 0.018, 0.03) if dark else rc.PAPER
    fg = (0.86, 0.83, 0.78) if dark else (rc.INK if st == "paper" else rc.RISO_BLUE * 0.75)
    cv = rc.canvas(L + w + R, T + h + B, bg)
    rc.paste(cv, colorize(v, st), L, T)
    rc.text(cv, (L, 40), title, 58, fg, rc.FONT_SERIF)
    for i, ln in enumerate(lines[:3]):
        rc.text(cv, (L, 112 + 32 * i), ln, 24, fg, rc.FONT_SANS)
    for f, s in xt:
        x = L + int(f * (w - 1))
        rc.line(cv, [(x, T + h), (x, T + h + 14)], fg, 2)
        rc.text(cv, (x, T + h + 22), s, 24, fg, rc.FONT_MONO, anchor="ma")
    for f, s in yt:
        y = T + int(f * (h - 1))
        rc.line(cv, [(L - 14, y), (L, y)], fg, 2)
        rc.text(cv, (L - 22, y), s, 24, fg, rc.FONT_MONO, anchor="rm")
    rc.text(cv, (L + w // 2, T + h + 70), xlab, 30, fg, rc.FONT_SANS, anchor="ma")
    rc.text(cv, (L - 22, T - 18), ylab, 26, fg, rc.FONT_SANS, anchor="rb")
    rc.text(cv, (L + w, T + h + 125), rc.STACK, 20, fg, rc.FONT_MONO, anchor="ra")
    rc.save_png(cv, f"{rc.GAL}/{out}_{st}.png")
    print(out, st)


def harm_img(key, lo, hi, width, band=8, gamma=1.0):
    a = H[key][:, 1::2]                                  # odd k: 1, 3, ..., 301
    dB = 20 * np.log10(a + 1e-12)
    v = rc.unit(dB, lo, hi, gamma).T[::-1]               # rows = k (high at top), cols = amplitude
    v = np.repeat(v, band, axis=0)                       # each harmonic drawn as a band of `band` px (declared)
    if v.shape[1] != width:
        v = np.array(Image.fromarray((v * 65535).astype(np.uint16)).resize((width, v.shape[0]), Image.BOX)) / 65535
    return v


def kticks(nrows_k=151):
    return [(1 - (k - 1) / 2 / (nrows_k - 1), f"k={k}") for k in (1, 51, 101, 151, 201, 251, 301)]


def render_harmonics(st):
    v = harm_img("int_lin", -62, -12, 2400)
    plate(v, st, "Harmonics of rounding, uniform grid",
          "|b_k(A)| of the error round(A sin t) - A sin t, odd k only (even k vanish). b_k = sum_n 2(-1)^n J_k(2 pi n A)/(pi n): "
          "rays k ~ 2 pi A and Bessel interference.  colour [-62, -12] dB re 1 LSB",
          "sine amplitude A  [LSB]", "harmonic", "harmonics_int",
          xt=[(a / 24, f"{a}") for a in range(0, 25, 4)], yt=kticks())
    octs = 12
    v = harm_img("fp8", -80, -38, 2400)
    plate(v, st, "Harmonics of rounding, FP8 E4M3: one octave, twelve times",
          "relative |b_k(A)|/A of the FP8 rounding error vs log2 amplitude. Q(2x) = 2Q(x) outside the subnormal range, so every "
          "octave is the same column pattern (measured: median 3e-5 dB).  colour [-80, -38] dB",
          "sine amplitude  [log2, re FP8 max 448]", "harmonic", "harmonics_fp8",
          xt=[(o / octs, f"2^{o - octs}") for o in range(0, octs + 1, 2)], yt=kticks())
    v = harm_img("fp4", -70, -20, 2400)
    plate(v, st, "Harmonics of rounding, FP4 E2M1: runs out of octaves",
          "relative |b_k(A)|/A for the 15-value FP4 grid over 6 octaves below its max (6). Only 3 normal octaves exist; below "
          "amplitude 0.25 everything rounds to 0 (black).  colour [-70, -20] dB",
          "sine amplitude  [log2, re FP4 max 6]", "harmonic", "harmonics_fp4",
          xt=[(o / 6, f"2^{o - 6}") for o in range(0, 7)], yt=kticks())
    v = harm_img("int8log", -85, -35, 2400)
    plate(v, st, "Control: uniform 255-level grid on the same log axis",
          "relative |b_k(A)|/A for int8 (peak 127 LSB) over 12 octaves: no repetition; the lattice coarsens as the "
          "signal spans fewer levels, then vanishes below 0.5 LSB.  colour [-85, -35] dB",
          "sine amplitude  [log2, re 127 LSB]", "harmonic", "harmonics_int8log",
          xt=[(o / 12, f"2^{o - 12}") for o in range(0, 13, 2)], yt=kticks())


def render_sd(st):
    N = int(SD["N"])
    for key, ukey, title, sub, lo, hi, g in [
        ("m1", "u1", "Idle tones of a 1-bit first-order sigma-delta",
         "output spectrum vs DC input u. The output is a rotation sequence with density rho = (1+u)/2, so its lines sit at "
         "fold(k rho): straight rays meeting at rationals (a Farey fan). Noise shaping darkens low frequencies.", -50, -22, 0.55),
        ("m2", "u2", "Idle tones of a 1-bit second-order sigma-delta",
         "NTF (1 - z^-1)^2: the low band is swept clean (black wedge, 40 dB/decade), tones survive as rays and "
         "the noise is no longer a pure rotation. |u| <= 0.6 (stable range).", -60, -10, 0.8),
        ("mz", "uz", "Zoom: first-order idle tones, u in [0.30, 0.42]",
         "the same fan, 16x finer in u: rays re-converge at every rational rho = p/q in the window, "
         "a number-theoretic (not fractal-dimension) self-similarity.", -50, -22, 0.55)]:
        a = SD[key].astype(np.float32)
        v = rc.unit(a, lo, hi, g)[::-1]                  # rows: u high at top
        v = np.array(Image.fromarray((v * 65535).astype(np.uint16)).resize((3072, 3072), Image.BOX)) / 65535
        u = SD[ukey]
        plate(v, st, title, sub + f"  N = 2^14 samples/row, Blackman-Harris, max-pooled 2x in f, [{lo}, {hi}] dB",
              "frequency  [fraction of sample rate]", "DC in", f"sd_idle_{ {'m1': 'order1', 'm2': 'order2', 'mz': 'zoom'}[key]}",
              xt=[(f / 0.5, f"{f:.2f}") for f in (0, 0.1, 0.2, 0.3, 0.4, 0.5)],
              yt=[(1 - (x - u[0]) / (u[-1] - u[0]), f"{x:+.2f}") for x in np.linspace(u[0], u[-1], 7)])


def render_dsd(st):
    edges = CH["edges"]
    for order, lo in ((1, -110), (2, -125)):
        keep = edges[:-1] >= 400.0      # below ~400 Hz the log rows are narrower than one 47 Hz FFT bin (staircase): cropped
        a = CH[f"order{order}"][:, keep]
        e0 = edges[:-1][keep][0]
        v = rc.unit(a, lo, -5, 0.9).T[::-1]
        v = np.repeat(v, 2, axis=0)
        v = np.array(Image.fromarray((v * 65535).astype(np.uint16)).resize((2994, 2400), Image.BOX)) / 65535
        fr = lambda f: 1 - np.log(f / e0) / np.log(edges[-1] / e0)
        plate(v, st, f"DSD64: a chirp through a 1-bit order-{order} sigma-delta at 3.072 MHz",
              f"log chirp 20 Hz - 20 kHz, amplitude 0.5, 8 s. The audio band (below 20 kHz) is pushed clean; quantization noise "
              f"rises {20 * order} dB/decade toward 1.5 MHz. Tones and harmonics ride on the noise. [{lo}, -5] dB",
              "time [s]", "freq", f"sd_dsd64_order{order}",
              xt=[(t / 8, f"{t}") for t in range(0, 9)],
              yt=[(fr(f), lab) for f, lab in ((400, "400 Hz"), (2000, "2 k"), (20000, "20 k"), (200000, "200 k"), (1536000, "1.5 M"))])


if __name__ == "__main__":
    which = sys.argv[1:] or ["harm", "sd", "dsd"]
    styles = [s for s in which if s in ("dark", "paper", "riso")] or ["dark", "paper", "riso"]
    for st in styles:
        if "harm" in which:
            render_harmonics(st)
        if "sd" in which:
            render_sd(st)
        if "dsd" in which:
            render_dsd(st)
