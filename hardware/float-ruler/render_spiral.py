"""Nautilus: every representable positive value v on a log-spiral, one turn per octave.

    angle  = 2*pi * log2(v)            (mod 2*pi: the position of v inside its octave)
    radius = R_in + (R_out - R_in) * (log2 v - log2 vmin) / (log2 vmax - log2 vmin)

Binary floats: v(e+1, m) = 2 v(e, m) has the same angle, so the ticks of successive turns line up
radially into exactly 2^M spokes.  Uniform integers do not (n, m share an angle only if n/m is a power of 2).
Subnormals break the spokes near the centre.
Exact: angles, radii.  Tick length = ruler rank of the mantissa field (as on the specimen sheet).
Aesthetic: colours, tick opacity, sheet layout.

    python render_spiral.py sheet [engraved observatory]
    python render_spiral.py hero  [observatory engraved]
    python render_spiral.py grow            # MP4 + GIF: the float16 spiral grows one turn per octave
"""
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from formats import BF16, FP4_E2M1, FP8_E4M3, FP8_E5M2, FP16, positive_finite
import colorcet  # noqa: F401
from plate import CACHE, GALLERY, MONO, SERIF, STACK, STYLES, trailing_zeros


CYC = plt.get_cmap("cet_CET_C6")  # cyclic: mantissa position = angle on the spiral, genuinely cyclic


def spiral_data(fmt, nbits=None, window=None):
    if fmt is None:
        v = np.arange(1, 1 << (nbits - 1), dtype=float)
        lev = np.zeros(len(v))
        return v, np.full(len(v), 0.55), np.zeros(len(v), bool), np.zeros(len(v))
    d = positive_finite(fmt)
    lev = fmt.M - trailing_zeros(d["m"], fmt.M)
    h = (0.25 + 0.75 * 0.62 ** lev) if fmt.M < 7 else (0.06 + 0.94 * 0.55 ** lev)
    v, sub, frac = d["value"], d["cls"] == 1, d["m"] / (1 << fmt.M)
    if window is not None:
        w = (np.log2(v) >= window[0]) & (np.log2(v) <= window[1])
        return v[w], h[w], sub[w], frac[w]
    return v, h, sub, frac


def spokes(v):
    """Number of distinct angles (exact: fractional part of log2 v via frexp mantissa)."""
    mant, _ = np.frexp(v)
    return len(np.unique(mant))


def draw_spiral(ax, v, h, sub, frac, style, R_in=0.12, R_out=1.0, lw=0.6, alpha=1.0, cmap=None, show_path=True,
                vmax_reveal=None):
    st = STYLES[style]
    L = np.log2(v)
    lo, hi = L.min(), L.max()
    span = max(hi - lo, 1e-9)
    turns = span
    r = R_in + (R_out - R_in) * (L - lo) / span
    th = 2 * np.pi * L
    ring = (R_out - R_in) / max(turns, 1.0)
    half = 0.5 * ring * h * 0.92
    sel = np.ones(len(v), bool) if vmax_reveal is None else (L <= vmax_reveal)
    c, s = np.cos(th), np.sin(th)
    segs = np.stack([np.stack([(r - half) * c, (r - half) * s], 1), np.stack([(r + half) * c, (r + half) * s], 1)], 1)
    if show_path:
        tt = np.linspace(lo, hi if vmax_reveal is None else min(hi, vmax_reveal), 4000)
        rr = R_in + (R_out - R_in) * (tt - lo) / span
        ax.plot(rr * np.cos(2 * np.pi * tt), rr * np.sin(2 * np.pi * tt), color=st["faint"], lw=0.35, alpha=0.6)
    # dense formats: opacity follows the tick's ruler rank (h), so the coarse spokes read through the fine ones
    a_i = alpha * (0.3 + 0.7 * np.clip((h - 0.06) / 0.94, 0.0, 1.0) ** 0.6) if len(v) > 1000 else np.full(len(v), alpha)
    if cmap is not None:
        cols = cmap(frac)
        cols[:, 3] = a_i
    else:
        cols = np.array([plt.matplotlib.colors.to_rgba(st["ink"], 1.0)] * len(v))
        cols[:, 3] = a_i
    cols[sub] = plt.matplotlib.colors.to_rgba(st["sub"], min(1.0, alpha * 1.5))
    ax.add_collection(LineCollection(segs[sel], colors=cols[sel], linewidths=lw, capstyle="butt"))
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_aspect("equal")
    ax.axis("off")
    return r, th


SHEET = [
    ("int4", None, 4, 3.2, 1.0),
    ("int8", None, 8, 1.4, 1.0),
    ("FP4 E2M1", FP4_E2M1, None, 3.2, 1.0),
    ("FP8 E5M2", FP8_E5M2, None, 2.0, 1.0),
    ("FP8 E4M3FN", FP8_E4M3, None, 1.6, 1.0),
    ("float16", FP16, None, 0.5, 1.0),
    ("bfloat16", BF16, None, 0.7, 1.0),
]


def sheet(style):
    st = STYLES[style]
    fig = plt.figure(figsize=(26, 15), dpi=300, facecolor=st["bg"])
    cmap = CYC if style == "observatory" else None
    boxes = [(0.02, 0.50), (0.27, 0.50), (0.52, 0.50), (0.77, 0.50), (0.02, 0.04), (0.27, 0.04), (0.52, 0.04)]
    for (name, fmt, nb, lw, alpha), (x0, y0) in zip(SHEET, boxes):
        ax = fig.add_axes([x0 + 0.01, y0 + 0.02, 0.21, 0.37])
        ax.set_facecolor(st["bg"])
        v, h, sub, frac = spiral_data(fmt, nb, window=(-24, 16) if fmt is BF16 else None)
        draw_spiral(ax, v, h, sub, frac, style, lw=lw, alpha=alpha, cmap=cmap)
        L = np.log2(v)
        ns = spokes(v[~sub])
        fig.text(x0 + 0.01, y0 + 0.415, name, family=SERIF, fontsize=26, color=st["ink"])
        fig.text(x0 + 0.01, y0 + 0.40, f"{len(v)} positive values · {L.max() - L.min():.1f} turns · "
                 f"{ns} distinct angles{' (normals)' if sub.any() else ''}"
                 + ("\nwindow 2^-24 … 2^16 of the full 2^-133 … 2^128" if fmt is BF16 else ""),
                 family=MONO, fontsize=10, color=st["ink"], va="top")
    ax = fig.add_axes([0.79, 0.06, 0.19, 0.36])
    ax.axis("off")
    txt = ("NAUTILUS\n\n"
           "Each tick is one representable\npositive value v.\n\n"
           "angle  = 2π · log2 v\nradius grows one ring per octave\n\n"
           "Floats: v(e+1, m) = 2·v(e, m),\nso every octave lands on the same\nangles and the ticks stack into\n"
           "exactly 2^M straight spokes.\n\nIntegers are uniform: n and m share\nan angle only if n/m is a power of 2,\nso int8 scatters over 64 angles\nand never forms a comb.\n\n"
           "Subnormals (second colour) are\nevenly spaced, so near the centre\nthe spokes dissolve.")
    ax.text(0, 1, txt, family=MONO, fontsize=13, color=st["ink"], va="top", linespacing=1.35)
    fig.text(0.98, 0.012, STACK + ("  ·  colour = mantissa position = angle (cyclic colorcet C6)" if cmap else ""), family=MONO,
             fontsize=10, color=st["ink"], ha="right")
    p = GALLERY / f"nautilus_sheet_{style}.png"
    fig.savefig(p, facecolor=st["bg"])
    plt.close(fig)
    print("wrote", p)


def hero(style):
    st = STYLES[style]
    fig = plt.figure(figsize=(18, 18), dpi=300, facecolor=st["bg"])
    ax = fig.add_axes([0.05, 0.02, 0.9, 0.9])
    ax.set_facecolor(st["bg"])
    v, h, sub, frac = spiral_data(FP16)
    lev = np.log(np.clip((h - 0.06) / 0.94, 1e-12, 1)) / np.log(0.55)
    h = 0.22 + 0.78 * 0.6 ** lev          # hero: longer fine ticks (declared; rank order preserved)
    cmap = CYC if style == "observatory" else None
    draw_spiral(ax, v, h, sub, frac, style, R_in=0.05, lw=0.75, alpha=1.0, cmap=cmap, show_path=False)
    fig.text(0.03, 0.965, "float16", family=SERIF, fontsize=40, color=st["ink"])
    fig.text(0.03, 0.948, f"{len(v)} positive values · one turn per octave · {spokes(v[~sub])} spokes · subnormals at the centre",
             family=MONO, fontsize=13, color=st["ink"])
    fig.text(0.97, 0.03, STACK, family=MONO, fontsize=10, color=st["ink"], ha="right")
    p = GALLERY / f"nautilus_float16_{style}.png"
    fig.savefig(p, facecolor=st["bg"])
    plt.close(fig)
    print("wrote", p)


def grow():
    style = "observatory"
    st = STYLES[style]
    outdir = CACHE / "nautilus_grow"
    outdir.mkdir(exist_ok=True)
    v, h, sub, frac = spiral_data(FP8_E4M3)
    v2, h2, sub2, frac2 = spiral_data(FP16)
    L1, L2 = np.log2(v), np.log2(v2)
    frames = 480
    cmap = CYC
    for i in range(frames):
        q = i / (frames - 1)
        fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=st["bg"])
        a1 = fig.add_axes([0.0, 0.05, 0.5, 0.82])
        a2 = fig.add_axes([0.5, 0.05, 0.5, 0.82])
        for a in (a1, a2):
            a.set_facecolor(st["bg"])
        k1 = L1.min() + q * (L1.max() - L1.min())
        k2 = L2.min() + q * (L2.max() - L2.min())
        draw_spiral(a1, v, h, sub, frac, style, lw=2.2, cmap=cmap, vmax_reveal=k1)
        draw_spiral(a2, v2, h2, sub2, frac2, style, lw=0.35, alpha=0.6, cmap=cmap, vmax_reveal=k2)
        fig.text(0.03, 0.93, "Nautilus: one turn per octave", family=SERIF, fontsize=28, color=st["ink"])
        fig.text(0.25, 0.88, f"FP8 E4M3FN  ·  up to 2^{k1:.2f}", family=MONO, fontsize=14, color=st["ink"], ha="center")
        fig.text(0.75, 0.88, f"float16  ·  up to 2^{k2:.2f}", family=MONO, fontsize=14, color=st["ink"], ha="center")
        fig.savefig(outdir / f"{i:05d}.png", facecolor=st["bg"])
        plt.close(fig)
    mp4 = GALLERY / "nautilus_grow.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(outdir / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)], check=True)
    gif = GALLERY / "nautilus_grow.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(outdir / "%05d.png"),
                    "-vf", "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=sierra2_4a",
                    str(gif)], check=True)
    print("wrote", mp4, gif)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    if what == "grow":
        grow()
    else:
        for s in sys.argv[2:] or ["engraved", "observatory"]:
            (sheet if what == "sheet" else hero)(s)
