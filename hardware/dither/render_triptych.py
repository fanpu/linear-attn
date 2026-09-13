"""Undithered vs RPDF vs TPDF (and optionally subtractive): same chirp, same bit depth, same render settings.

Row 1: spectrogram (shared absolute dB scale across columns, black point declared).
Row 2: conditional moments of the total error vs where the input sits inside an LSB -- exact theory (lines, from
       verify.py) with the values measured on the 30 s chirp (dots, compute_moments.py).
Row 3: noise modulation on a decaying 220 Hz tone at 6 bits: short-time (10 ms) RMS error.
Styles: dark | paper | riso.   Usage: python render_triptych.py [bits] [style...] [--quad]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import render_common as rc

args = [a for a in sys.argv[1:] if not a.startswith("--")]
QUAD = "--quad" in sys.argv
BITS = int(args[0]) if args and args[0].isdigit() else 3
STYLES = [a for a in args if not a.isdigit()] or ["dark", "paper", "riso"]

TH = np.load(f"{rc.CACHE}/moments.npz")
MEAS = np.load(f"{rc.CACHE}/chirp_moments.npz")
DEC = np.load(f"{rc.CACHE}/decay_rms.npz")
M = rc.META["variants"]

COLS = [("none", f"int{BITS}", "undithered", "none", "decay_int6_undithered"),
        ("rpdf", f"int{BITS}_rpdf", "RPDF dither, 1 LSB p-p (+-1/2)", "rpdf", "decay_int6_rpdf"),
        ("tpdf", f"int{BITS}_tpdf", "TPDF dither, 2 LSB p-p (+-1)", "tpdf", "decay_int6_tpdf")]
if QUAD:
    COLS.append(("sub", f"int{BITS}_sub", "subtractive RPDF", "subtractive_rpdf", None))

VERDICT = {"none": "error = deterministic sawtooth of the input:\nmean and power both depend on x  ->  harmonics",
           "rpdf": "mean error independent of x (no distortion)\npower still depends on x  ->  noise modulation",
           "tpdf": "mean AND power independent of x\n(higher moments are not; power is 3x larger)",
           "sub": "dither subtracted after quantizing: error is\nuniform, independent of x, power 1/12 LSB^2"}


def style_params(style):
    if style == "dark":
        return dict(bg=(0.02, 0.018, 0.03), fg=(0.86, 0.83, 0.78), c1=(1.0, 0.72, 0.35), c2=(0.55, 0.75, 1.0), grid=(0.25, 0.25, 0.3))
    if style == "paper":
        return dict(bg=tuple(rc.PAPER), fg=tuple(rc.INK), c1=(0.1, 0.1, 0.1), c2=(0.55, 0.52, 0.48), grid=(0.8, 0.78, 0.72))
    return dict(bg=tuple(rc.PAPER), fg=tuple(rc.RISO_BLUE * 0.8), c1=tuple(rc.RISO_PINK), c2=tuple(rc.RISO_BLUE), grid=(0.82, 0.86, 0.9))


def spec_image(name, style, lo, hi, w=1100, h=1025):
    S = rc.resample_power(rc.load_spec(name), out_w=w, out_h=h)
    v = rc.unit(S, lo, hi, 0.9)
    if style == "dark":
        return rc.cmap_rgb(v, "magma")
    if style == "paper":
        return rc.ink_on_paper(v, gamma=0.8)
    hb = rc.floyd_steinberg(v)
    return rc.multiply_layers([(hb, rc.RISO_BLUE)])


def render(style):
    P = style_params(style)
    n = len(COLS)
    fig = plt.figure(figsize=(9.6 * n, 21), dpi=110, facecolor=P["bg"])
    gs = GridSpec(4, n, height_ratios=[0.2, 1.0, 0.55, 0.42], hspace=0.28, wspace=0.12,
                  left=0.045, right=0.985, top=0.925, bottom=0.05, figure=fig)
    ref = rc.resample_power(rc.load_spec(f"int{BITS}"), out_w=1100, out_h=1025)
    lo = float(np.percentile(ref, 65))
    hi = lo + 55
    fig.text(0.045, 0.99, f"DITHER II  -  one {BITS}-bit chirp, three ways of rounding it", fontsize=38, color=P["fg"],
             family="DejaVu Serif", va="top")
    fig.text(0.045, 0.962, f"Same 30 s log chirp (20 Hz - 24 kHz), same {2 ** BITS - 1}-level grid, same colour scale "
             f"[{lo:.0f}, {hi:.0f}] dB re full-scale sine (black point = 65th pct of the undithered plate, declared). "
             "Theory: Lipshitz, Wannamaker & Vanderkooy, JAES 40(5) 1992.", fontsize=15, color=P["fg"], va="top")
    for j, (key, spec, title, thkey, dkey) in enumerate(COLS):
        axh = fig.add_subplot(gs[0, j]); axh.axis("off")
        axh.text(0, 0.95, title, fontsize=26, color=P["fg"], family="DejaVu Serif", va="top", transform=axh.transAxes)
        axh.text(0, 0.5, f"SINAD {M[spec]['sinad_dB']:.1f} dB\n" + VERDICT[key], fontsize=14,
                 color=P["fg"], va="top", transform=axh.transAxes, family="DejaVu Sans Mono", linespacing=1.4)
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(spec_image(spec, style, lo, hi), aspect="auto", extent=[0, 30, 0, 24], interpolation="antialiased")
        ax.set_facecolor(P["bg"])
        for s in ax.spines.values():
            s.set_color(P["grid"])
        ax.tick_params(colors=P["fg"], labelsize=12)
        ax.set_xlabel("time [s]  (instantaneous input frequency rises exponentially)", color=P["fg"], fontsize=13)
        if j == 0:
            ax.set_ylabel("frequency [kHz]", color=P["fg"], fontsize=14)

        # moments
        axm = fig.add_subplot(gs[2, j], facecolor=P["bg"])
        xs = TH["xs"]
        th = TH[thkey]
        mk = {"none": "none", "rpdf": "rpdf", "tpdf": "tpdf", "sub": "sub"}[key]
        c = MEAS["centers"]
        axm.axhline(0, color=P["grid"], lw=1)
        axm.axhline(1 / 12, color=P["grid"], lw=1, ls=":")
        axm.plot(xs, th[0], color=P["c2"], lw=2.5, label="mean error  E[e | x]")
        axm.plot(xs, th[1], color=P["c1"], lw=2.5, label="error power  E[e^2 | x]")
        axm.plot(c, MEAS[f"b{BITS}_{mk}_m1"], "o", ms=5, color=P["c2"], mfc="none", mew=1.3)
        axm.plot(c, MEAS[f"b{BITS}_{mk}_m2"], "o", ms=5, color=P["c1"], mfc="none", mew=1.3)
        axm.set_ylim(-0.55, 0.55); axm.set_xlim(-0.5, 0.5)
        axm.set_xlabel("input position inside one LSB,  x - nearest level  [LSB]", color=P["fg"], fontsize=13)
        if j == 0:
            axm.set_ylabel("[LSB], [LSB^2]", color=P["fg"], fontsize=13)
            axm.legend(loc="lower left", fontsize=12, frameon=False, labelcolor=P["fg"])
            axm.text(0.48, 1 / 12 + 0.02, "1/12", color=P["fg"], fontsize=11, ha="right")
        axm.text(0.01, 0.97, "lines: exact (integration over the dither pdf)\ndots: measured on the chirp above",
                 transform=axm.transAxes, fontsize=10.5, color=P["fg"], va="top", family="DejaVu Sans Mono")
        axm.tick_params(colors=P["fg"], labelsize=11)
        for s in axm.spines.values():
            s.set_color(P["grid"])

        # noise modulation on a decaying tone
        axd = fig.add_subplot(gs[3, j], facecolor=P["bg"])
        t = DEC["t"]
        axd.plot(t, DEC["env"], color=P["grid"], lw=1.5)
        if dkey is not None:
            axd.plot(t, DEC[dkey], color=P["c1"], lw=1.6)
            axd.set_title("", color=P["fg"])
        else:
            axd.axhline(np.sqrt(1 / 12), color=P["c1"], lw=1.6)
            axd.text(3, np.sqrt(1 / 12) * 1.6, "(not simulated; exact value sqrt(1/12))", color=P["fg"], fontsize=10.5, ha="center")
        axd.set_yscale("log"); axd.set_ylim(0.02, 40); axd.set_xlim(0, 6)
        axd.set_xlabel("time [s]:  220 Hz tone decaying (grey = its amplitude), 6-bit, RMS error in 10 ms windows",
                       color=P["fg"], fontsize=12)
        if j == 0:
            axd.set_ylabel("RMS error [LSB]", color=P["fg"], fontsize=13)
        axd.tick_params(colors=P["fg"], labelsize=11)
        for s in axd.spines.values():
            s.set_color(P["grid"])
    fig.text(0.985, 0.012, "STFT 4096 Blackman-Harris, hop 256, power-averaged into pixels  |  " + rc.STACK,
             fontsize=11, color=P["fg"], ha="right")
    out = f"{rc.GAL}/triptych_int{BITS}_{style}{'_quad' if QUAD else ''}.png"
    fig.savefig(out, facecolor=P["bg"])
    plt.close(fig)
    rc.save_png(__import__("PIL.Image", fromlist=["Image"]).open(out).convert("RGB"), out)
    print(out)


for s in STYLES:
    render(s)
