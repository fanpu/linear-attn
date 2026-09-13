"""Bit-depth series: spectrograms of the same 30 s log chirp quantized (undithered) at 2..12 bits.

Styles: (1) observatory dark column, magma; (2) sonograph paper column (ink density = dB);
(3) riso 1-bit two-ink grid; (4) hero single plate at native STFT resolution. Also a shared-scale column.
"""
import numpy as np
import render_common as rc

BITS6 = (2, 3, 4, 6, 8, 12)
BITS8 = (2, 3, 4, 5, 6, 8, 10, 12)
M = rc.META["variants"]


def plate(name, w, h, mode="per", span=70.0, gamma=1.0):
    S = rc.resample_power(rc.load_spec(name), out_w=w, out_h=h)
    if mode == "per":          # per-plate: floor at the plate's own 3rd percentile (declared)
        lo = np.percentile(S, 3)
        hi = lo + span
    else:                      # shared absolute scale, dB re full-scale sine
        lo, hi = mode
    return rc.unit(S, lo, hi, gamma), (lo, hi)


def column(style):
    PW, PH, PAD, LM, TOP = 2805, 683, 36, 250, 150
    H = TOP + len(BITS6) * (PH + PAD) + 120
    bg = (0.035, 0.03, 0.045) if style != "sonograph" else rc.PAPER
    fg = (0.85, 0.82, 0.78) if style != "sonograph" else rc.INK
    cv = rc.canvas(LM + PW + 80, H, bg)
    title = {"magma": "DITHER I  -  the error lattice of a quantized chirp",
             "shared": "DITHER I  -  the same, on one absolute dB scale",
             "sonograph": "Sona-graph  -  log chirp 20 Hz to 24 kHz, undithered, 2 to 12 bits"}[style]
    rc.text(cv, (LM, 50), title, 54, fg, rc.FONT_SERIF)
    for i, b in enumerate(BITS6):
        y0 = TOP + i * (PH + PAD)
        if style == "shared":
            v, (lo, hi) = plate(f"int{b}", PW, PH, mode=(-115.0, -5.0), gamma=1.0)
        else:
            v, (lo, hi) = plate(f"int{b}", PW, PH, gamma=1.0 if style != "sonograph" else 0.8)
        rgb = rc.cmap_rgb(v, "magma") if style != "sonograph" else rc.ink_on_paper(v, gamma=1.0)
        rc.paste(cv, rgb, LM, y0)
        rc.text(cv, (LM - 30, y0 + 10), f"{b} bits", 44, fg, rc.FONT_SERIF, anchor="ra")
        rc.text(cv, (LM - 30, y0 + 70), f"{2 ** b - 1} levels", 24, fg, rc.FONT_MONO, anchor="ra")
        rc.text(cv, (LM - 30, y0 + 104), f"SINAD {M[f'int{b}']['sinad_dB']:.1f} dB", 24, fg, rc.FONT_MONO, anchor="ra")
        rc.text(cv, (LM - 30, y0 + PH - 34), f"[{lo:.0f}, {hi:.0f}] dB", 20, fg, rc.FONT_MONO, anchor="ra")
    rc.text(cv, (LM, H - 90), "time 0 to 30 s (log chirp, so time is log of input frequency)  |  frequency 0 to 24 kHz, linear  |  "
            "STFT 4096 Blackman-Harris, hop 256, power averaged into pixels", 26, fg, rc.FONT_MONO)
    rc.text(cv, (LM, H - 50), "colour = dB re a full-scale sine; " + ("one absolute scale for all plates" if style == "shared" else
            "each plate's floor at its own 3rd percentile, 70 dB span (declared)") + "  |  " + rc.STACK, 26, fg, rc.FONT_MONO)
    cv.save(f"{rc.GAL}/bitseries_column_{style}.png", optimize=True)
    print("column", style)


def riso_grid():
    PW, PH, PAD = 1402, 683, 30
    W, H = 2 * PW + 3 * PAD, 4 * (PH + PAD) + PAD + 110
    layers_b, layers_p = np.zeros((H, W)), np.zeros((H, W))
    labels = []
    for i, b in enumerate(BITS8):
        r, c = divmod(i, 2)
        x0, y0 = PAD + c * (PW + PAD), 110 + r * (PH + PAD)
        v, _ = plate(f"int{b}", PW, PH, span=62, gamma=0.85)
        layers_b[y0:y0 + PH, x0:x0 + PW] = rc.floyd_steinberg(v)           # blue: full dB image, 1-bit error diffusion
        layers_p[y0:y0 + PH, x0:x0 + PW] = rc.floyd_steinberg(np.clip((v - 0.42) / 0.33, 0, 1) ** 1.0)  # pink: upper dB range only
        labels.append((x0 + 14, y0 + 10, f"{b}-bit"))
    rgb = rc.multiply_layers([(layers_b, rc.RISO_BLUE), (rc.shift(layers_p, 3, -2), rc.RISO_PINK)])
    cv = rc.canvas(W, H, rc.PAPER)
    rc.paste(cv, rgb, 0, 0)
    rc.text(cv, (PAD, 30), "DITHER  /  8 bit depths, 1-bit riso halftone (blue: all dB, pink: loudest lines; 3 px misregistration)",
            40, rc.INK, rc.FONT_SANS)
    for x, y, s in labels:
        rc.text(cv, (x, y), s, 34, rc.INK, rc.FONT_SERIF)
    cv.save(f"{rc.GAL}/bitseries_grid_riso.png", optimize=True)
    print("riso grid")


def hero():
    S = rc.load_spec("int3")
    S = rc.resample_power(S)  # native: 2049 x 5610
    lo = np.percentile(S, 3)
    for cm in ("magma",):
        rc.save_rgb(rc.cmap_rgb(rc.unit(S, lo, lo + 72), cm), f"{rc.GAL}/hero_int3_native_{cm}.png")
    rc.save_rgb(rc.ink_on_paper(rc.unit(S, lo + 8, lo + 72, 1.5)), f"{rc.GAL}/hero_int3_native_sonograph.png")
    print("hero")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["magma", "sonograph", "shared", "riso", "hero"]
    for s in which:
        if s in ("magma", "sonograph", "shared"):
            column(s)
        elif s == "riso":
            riso_grid()
        elif s == "hero":
            hero()
