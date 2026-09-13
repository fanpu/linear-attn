"""Type-specimen sheet: one ruler per format on a shared log2 axis (+ a linear-axis band).

Exact: tick positions (every positive finite value of each format).
Tick height (mode 'ruler'): ruler rank of the mantissa field m — octave marks (m=0) tallest,
  m=2^(M-1) next, then quarters ... (height = 0.12 + 0.88 * 0.7^(M - trailing_zeros(m))).
Tick height (mode 'sawtooth'): the significand 1 + m/2^M, so each octave is one ramp.
Aesthetic: colours, fonts, ink-accumulation gain, paper grain, riso misregistration.

    python render_specimen.py [engraved observatory riso]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from formats import BF16, FP4_E2M1, FP8_E4M3, FP8_E5M2, FP16, positive_finite
from plate import MONO, SERIF, STACK, STYLES, save, trailing_zeros
from raster import InkLayer, composite, multiply

W, H, DPI = 7200, 4560, 300
X0, X1 = 1350, 7000
LO, HI = -25.0, 18.0
ROWS = [
    ("int4", None, 4, "1·3  two's complement", "uniform: 7 positive values"),
    ("int8", None, 8, "1·7  two's complement", "uniform: 127 positive values"),
    ("FP4 E2M1", FP4_E2M1, 4, "1·2·1  bias 1", "15 values, max 6, no inf/NaN"),
    ("FP8 E5M2", FP8_E5M2, 8, "1·5·2  bias 15", "max 57344, ±inf, 6 NaN codes"),
    ("FP8 E4M3FN", FP8_E4M3, 8, "1·4·3  bias 7", "max 448, no inf, 2 NaN codes"),
    ("float16", FP16, 16, "1·5·10  bias 15", "max 65504, 1024 values/octave"),
    ("bfloat16", BF16, 16, "1·8·7  bias 127", "max 3.39e38, 128 values/octave"),
]
ROW_Y0, ROW_PITCH, TICK_H = 640, 318, 250
LIN_ROWS = [("int4", None), ("FP4 E2M1", FP4_E2M1), ("FP8 E4M3FN", FP8_E4M3), ("float16", FP16), ("bfloat16", BF16)]
LIN_Y0, LIN_PITCH, LIN_H = 3260, 190, 150


def lx(v):
    return X0 + (np.log2(v) - LO) / (HI - LO) * (X1 - X0)


def linx(v):
    return X0 + v / 8.0 * (X1 - X0)


def heights(fmt, nb, mode):
    if fmt is None:
        v = np.arange(1, 1 << (nb - 1), dtype=np.float64)
        tz = trailing_zeros(v.astype(np.int64), nb - 1)
        return v, 0.12 + 0.88 * 0.7 ** (nb - 2 - np.minimum(tz, nb - 2)), np.zeros(len(v), bool), np.zeros(len(v))
    d = positive_finite(fmt)
    v, m, cls = d["value"], d["m"], d["cls"]
    frac = m / (1 << fmt.M)
    if mode == "sawtooth":
        h = 0.12 + 0.88 * frac
    else:
        h = 0.12 + 0.88 * 0.7 ** (fmt.M - trailing_zeros(m, fmt.M))
    return v, h, cls == 1, frac


def build_layers(style, mode):
    st = STYLES[style]
    rgb = style == "observatory"
    main = InkLayer(H, W, rgb=rgb)
    sub = InkLayer(H, W)
    rules = InkLayer(H, W)
    cmap = plt.get_cmap("cmc.batlow")
    for i, (name, fmt, nb, _, _) in enumerate(ROWS):
        yb = ROW_Y0 + (i + 1) * ROW_PITCH - 40
        v, h, is_sub, frac = heights(fmt, nb, mode)
        x = lx(v)
        win = (x >= X0) & (x <= X1)
        dense = fmt is not None and fmt.M >= 7
        w = 0.45 if dense else 3.0
        s = win & ~is_sub
        cols = cmap(0.2 + 0.8 * frac[s]) if rgb else None
        main.ticks(x[s], yb, TICK_H * h[s], w_px=w, colors=cols)
        s = win & is_sub
        if s.any():
            sub.ticks(x[s], yb, TICK_H * h[s], w_px=w)
        rules.hline(yb + 1, X0, X1, 1.2, 0.35)
    # log axis
    ya = ROW_Y0 + len(ROWS) * ROW_PITCH - 10
    for k in range(int(LO), int(HI) + 1):
        L = 34 if k % 5 == 0 else 16
        rules.ticks(np.array([lx(2.0 ** k)]), ya + L, np.array([L]), w_px=2.5 if k % 5 == 0 else 1.5)
    rules.hline(ya, X0, X1, 2.0, 1.0)
    # linear band
    for j, (name, fmt) in enumerate(LIN_ROWS):
        yb = LIN_Y0 + (j + 1) * LIN_PITCH - 20
        if fmt is None:
            v = np.arange(0, 8, dtype=float)
            h = np.ones_like(v) * 0.9
            is_sub = np.zeros(len(v), bool)
            frac = np.zeros(len(v))
        else:
            v, h, is_sub, frac = heights(fmt, 0, mode)
            v = np.concatenate([[0.0], v])
            h = np.concatenate([[1.0], h])
            is_sub = np.concatenate([[False], is_sub])
            frac = np.concatenate([[0.0], frac])
        sel = v <= 8
        dense = fmt is not None and fmt.M >= 7
        w = 0.45 if dense else 3.0
        s = sel & ~is_sub
        main.ticks(linx(v[s]), yb, LIN_H * h[s], w_px=w, colors=cmap(0.2 + 0.8 * frac[s]) if rgb else None)
        s = sel & is_sub
        if s.any():
            sub.ticks(linx(v[s]), yb, LIN_H * h[s], w_px=w)
        rules.hline(yb + 1, X0, X1, 1.2, 0.35)
    ya2 = LIN_Y0 + len(LIN_ROWS) * LIN_PITCH
    for k in range(9):
        rules.ticks(np.array([linx(k)]), ya2 + 30, np.array([30]), w_px=2.5)
    rules.hline(ya2, X0, X1, 2.0, 1.0)
    return main, sub, rules


def text_figure(style, mode, bgimg=None, color="black"):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    if bgimg is not None:
        fig.figimage((bgimg * 255).astype(np.uint8), 0, 0)
    else:
        fig.patch.set_facecolor("white")

    def T(xpx, ypx, s, **kw):
        fig.text(xpx / W, 1 - ypx / H, s, color=color, **kw)

    T(140, 250, "THE RULER", family=SERIF, fontsize=58, va="center")
    T(140, 420, "A specimen of representable numbers.  Every positive finite value of each format is one tick, on a shared "
                "log2 axis:  uniform within an octave, doubling from one octave to the next.",
      family=SERIF, fontsize=15, style="italic", va="center")
    hdesc = ("tick height = significand 1 + m/2^M: each octave is one rising ramp" if mode == "sawtooth" else
             "tick height = ruler rank of mantissa field m: octave marks tallest, then halves, quarters, eighths ...")
    T(140, 500, hdesc, family=MONO, fontsize=10.5, va="center")
    for i, (name, fmt, nb, layout, note) in enumerate(ROWS):
        yb = ROW_Y0 + (i + 1) * ROW_PITCH - 40
        T(140, yb - 170, name, family=SERIF, fontsize=25, va="center")
        T(140, yb - 90, layout, family=MONO, fontsize=9.5, va="center")
        T(140, yb - 45, note, family=MONO, fontsize=9.5, va="center")
        if fmt is not None:
            full = positive_finite(fmt)["value"]
            if np.log2(full.min()) < LO:
                T(X0 - 20, yb - 20, f"from 2^{np.log2(full.min()):.0f} ", family=MONO, fontsize=8, ha="right", va="bottom")
            if np.log2(full.max()) > HI:
                T(X1 + 15, yb - 20, f"to\n2^{np.log2(full.max()):.2f}", family=MONO, fontsize=8, ha="left", va="bottom")
    ya = ROW_Y0 + len(ROWS) * ROW_PITCH - 10
    for k in range(int(LO), int(HI) + 1, 5):
        lab = {0: "1", 5: "32", 10: "1024", 15: "32768"}.get(k, f"2^{k}")
        T(lx(2.0 ** k), ya + 75, lab, family=MONO, fontsize=11, ha="center", va="center")
    T(X1, ya + 140, "log2 axis, 2^-25 … 2^18", family=MONO, fontsize=9, ha="right", va="center")
    T(140, LIN_Y0 + 40, "the same rulers on a linear axis, 0 … 8", family=SERIF, fontsize=17, style="italic", va="center")
    for j, (name, fmt) in enumerate(LIN_ROWS):
        yb = LIN_Y0 + (j + 1) * LIN_PITCH - 20
        T(X0 - 40, yb - 50, name, family=MONO, fontsize=10.5, ha="right", va="center")
    ya2 = LIN_Y0 + len(LIN_ROWS) * LIN_PITCH
    for k in range(9):
        T(linx(k), ya2 + 70, str(k), family=MONO, fontsize=11, ha="center", va="center")
    sub_note = ("subnormals (exponent field 0) in blue; colour = mantissa position m/2^M (batlow)"
                if style == "observatory" else "subnormals (exponent field 0) in the second ink")
    T(140, H - 110, sub_note, family=MONO, fontsize=9, va="center")
    T(X1, H - 110, STACK, family=MONO, fontsize=9, ha="right", va="center")
    return fig


def fig_coverage(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    plt.close(fig)
    return 1.0 - a.mean(axis=2)


def render(style, mode):
    st = STYLES[style]
    main, sub, rules = build_layers(style, mode)
    suffix = "_sawtooth" if mode == "sawtooth" else ""
    if style == "riso":
        txt = fig_coverage(text_figure(style, mode))[: H, : W]
        A = main.alpha(1.6) * 0.95
        B = np.clip(sub.alpha(1.6) + rules.alpha(1.0) + txt, 0, 1) * 0.95
        B = np.roll(np.roll(B, 5, 0), -6, 1)  # deliberate misregistration
        img = multiply(st["bg"], [(A, st["ink"]), (B, st["ink2"])])
        rng = np.random.default_rng(3)
        img = np.clip(img * (1 - 0.035 * rng.random(img.shape[:2], dtype=np.float32))[..., None], 0, 1)
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
        fig.figimage((img * 255).astype(np.uint8), 0, 0)
    else:
        gain = 1.3
        maincol = main.color(st["ink"]) if style == "observatory" else st["ink"]
        img = composite(st["bg"], [(rules.alpha(1.0) * 0.8, st["ink"]), (main.alpha(gain), maincol),
                                   (sub.alpha(gain), st["sub"])])
        if style == "engraved":
            rng = np.random.default_rng(3)
            img = np.clip(img * (1 - 0.03 * rng.random(img.shape[:2], dtype=np.float32))[..., None], 0, 1)
        fig = text_figure(style, mode, bgimg=img[::-1] if False else img, color=st["ink"])
    save(fig, f"specimen_{style}{suffix}.png", dpi=DPI)


def main(styles):
    for style in styles:
        render(style, "ruler")
        if style != "riso":
            render(style, "sawtooth")


if __name__ == "__main__":
    main(sys.argv[1:] or ["engraved", "observatory", "riso"])
