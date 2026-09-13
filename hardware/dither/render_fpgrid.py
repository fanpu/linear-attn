"""The same chirp stored on uniform (int4, int8) and floating-point (FP4 E2M1, FP8 E4M3FN) grids, at 0/-12/-24 dB.

Spectrograms are normalised to the chirp's own level (output divided by the amplitude), so the chirp line is always
0 dB and the lattice shows the error *relative to the signal*. One absolute colour scale for all 12 plates.
Bottom strip: SINAD vs level (compute_fpcurves.py). Styles: dark | paper | riso.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import render_common as rc
import dsp

ROWS = [("int4", "int4  (15 uniform levels)"), ("fp4", "FP4 E2M1  (15 levels: 0, .5, 1, 1.5, 2, 3, 4, 6)"),
        ("int8", "int8  (255 uniform levels)"), ("fp8", "FP8 E4M3FN  (253 levels, max 448)")]
COLS = [("00dB", "chirp at 0 dB (peak = grid max)"), ("12dB", "-12 dB"), ("24dB", "-24 dB")]
M = rc.META["variants"]
CUR = np.load(f"{rc.CACHE}/fp_sinad_curves.npz")
LO, HI = -78.0, -8.0


def style(st):
    if st == "dark":
        return dict(bg=(0.02, 0.018, 0.03), fg=(0.86, 0.83, 0.78), cols={"int4": "#8fb8ff", "int8": "#4f78c9", "fp4": "#ffb35c", "fp8": "#ff6f4a"}, grid=(0.25, 0.25, 0.3))
    if st == "paper":
        return dict(bg=tuple(rc.PAPER), fg=tuple(rc.INK), cols={"int4": "0.55", "int8": "0.75", "fp4": "0.05", "fp8": "0.3"}, grid=(0.8, 0.78, 0.72))
    return dict(bg=tuple(rc.PAPER), fg=tuple(rc.RISO_BLUE * 0.8), cols={"int4": rc.RISO_BLUE, "int8": rc.RISO_BLUE * 0.6 + 0.4, "fp4": rc.RISO_PINK, "fp8": rc.RISO_PINK * 0.6 + 0.4 * rc.RISO_BLUE}, grid=(0.82, 0.86, 0.9))


def img(name, st):
    S = rc.resample_power(rc.load_spec(name), out_w=1122, out_h=683)
    v = rc.unit(S, LO, HI, 0.9)
    if st == "dark":
        return rc.cmap_rgb(v, "magma")
    if st == "paper":
        return rc.ink_on_paper(v, gamma=0.8)
    return rc.multiply_layers([(rc.floyd_steinberg(v), rc.RISO_BLUE), (rc.shift(rc.floyd_steinberg(np.clip((v - .45) / .4, 0, 1)), 2, -2), rc.RISO_PINK)])


def render(st):
    P = style(st)
    fig = plt.figure(figsize=(30, 30), dpi=100, facecolor=P["bg"])
    gs = GridSpec(5, 3, height_ratios=[1, 1, 1, 1, 1.05], hspace=0.2, wspace=0.06, left=0.06, right=0.985, top=0.9, bottom=0.04)
    fig.text(0.06, 0.985, "DITHER III  -  a chirp on floating-point grids", fontsize=44, color=P["fg"], family="DejaVu Serif", va="top")
    fig.text(0.06, 0.955, "Uniform grids lose resolution as the signal gets quieter; floating-point grids are logarithmic, so the "
             "error lattice keeps its shape (FP8) or runs out of levels (FP4).\nEach plate is divided by its amplitude: the chirp line "
             f"is always 0 dB, colour = [{LO:.0f}, {HI:.0f}] dB re the signal, one scale for all plates. Round-to-nearest, ties-to-even, "
             "saturating; grid verified bit-exact against torch.float8_e4m3fn.", fontsize=16, color=P["fg"], va="top")
    for i, (fmt, rlab) in enumerate(ROWS):
        for j, (tag, clab) in enumerate(COLS):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img(f"{fmt}_{tag}", st), aspect="auto", extent=[0, 30, 0, 24], interpolation="antialiased")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(P["grid"])
            sin = M[f"{fmt}_{tag}"]["sinad_dB"]
            ax.set_title(f"SINAD {sin:.1f} dB" if sin > 0.05 else "SINAD 0 dB: every sample rounds to 0 (silence)",
                         color=P["fg"], fontsize=15, loc="right", family="DejaVu Sans Mono")
            if j == 0:
                ax.set_ylabel(rlab, color=P["fg"], fontsize=17, family="DejaVu Serif")
            if i == 0:
                ax.set_title(clab, color=P["fg"], fontsize=22, loc="left", family="DejaVu Serif")
    ax = fig.add_subplot(gs[4, :], facecolor=P["bg"])
    for k, lab in (("int4", "int4"), ("fp4", "FP4 E2M1"), ("int8", "int8"), ("fp8", "FP8 E4M3")):
        y = np.where(np.isfinite(CUR[k]), CUR[k], np.nan)
        ax.plot(CUR["amp_db"], y, color=P["cols"][k], lw=3.2 if k.startswith("fp") else 2.2, label=lab)
    for d in (0, -12, -24):
        ax.axvline(d, color=P["grid"], lw=1, ls=":")
    for o in range(0, 14):
        ax.axvline(-6.0206 * o, color=P["grid"], lw=0.6, alpha=0.5)
    ax.set_xlim(-84, 0.5); ax.set_ylim(-1, 55)
    ax.set_xlabel("sine level [dB re grid max]   (thin rules: octaves, factor 2 in amplitude)", color=P["fg"], fontsize=17)
    ax.set_ylabel("SINAD [dB]", color=P["fg"], fontsize=17)
    ax.tick_params(colors=P["fg"], labelsize=14)
    for s in ax.spines.values():
        s.set_color(P["grid"])
    ax.legend(loc="upper right", fontsize=17, frameon=False, labelcolor=P["fg"])
    ax.text(-83, 50, "997.3 Hz sine, 2^15 samples, 0.25 dB steps.  FP8 ripple repeats every octave (exact: Q(x/2) = Q(x)/2 "
            "outside the subnormal range);\nFP4 has only 3 normal octaves, so it collapses after ~-20 dB.", color=P["fg"], fontsize=15, va="top")
    fig.text(0.985, 0.012, "STFT 4096 Blackman-Harris, hop 256  |  " + rc.STACK, fontsize=12, color=P["fg"], ha="right")
    out = f"{rc.GAL}/fpgrid_{st}.png"
    fig.savefig(out, facecolor=P["bg"])
    plt.close(fig)
    rc.save_png(Image.open(out).convert("RGB"), out)
    print(out)


for s in (sys.argv[1:] or ["dark", "paper", "riso"]):
    render(s)
