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
    rc.save_png(cv, f"{rc.GAL}/bitseries_column_{style}.png")
    print("column", style)


def riso_grid():
    PW, PH, PAD = 1402, 683, 30
    W, H = 2 * PW + 3 * PAD, 5 * (PH + PAD) + PAD + 150
    layers_b, layers_p = np.zeros((H, W)), np.zeros((H, W))
    labels = []
    names = [(f"int{b}", f"{b}-bit") for b in BITS8] + [("int4", "4-bit, undithered"), ("int4_tpdf", "4-bit + TPDF dither: the cure")]
    for i, (nm, lab) in enumerate(names):
        r, c = divmod(i, 2)
        x0, y0 = PAD + c * (PW + PAD), 110 + r * (PH + PAD) + (40 if r == 4 else 0)
        v, _ = plate(nm, PW, PH, span=62, gamma=0.85)
        layers_b[y0:y0 + PH, x0:x0 + PW] = rc.floyd_steinberg(v)           # blue: full dB image, 1-bit error diffusion
        layers_p[y0:y0 + PH, x0:x0 + PW] = rc.floyd_steinberg(np.clip((v - 0.42) / 0.33, 0, 1) ** 1.0)  # pink: upper dB range only
        labels.append((x0 + 14, y0 + 10, lab))
    rgb = rc.multiply_layers([(layers_b, rc.RISO_BLUE), (rc.shift(layers_p, 3, -2), rc.RISO_PINK)])
    cv = rc.canvas(W, H, rc.PAPER)
    rc.paste(cv, rgb, 0, 0)
    rc.text(cv, (PAD, 30), "DITHER  /  8 bit depths, 1-bit riso halftone (blue: all dB, pink: loudest lines; 3 px misregistration)",
            40, rc.INK, rc.FONT_SANS)
    for x, y, s in labels:
        rc.text(cv, (x, y), s, 34, rc.INK, rc.FONT_SERIF)
    rc.save_png(cv, f"{rc.GAL}/bitseries_grid_riso.png")
    print("riso grid")


def hero():
    for name in ("int3", "int4"):
        S = rc.resample_power(rc.load_spec(name))  # native: 2049 x 5610
        p3, p65 = np.percentile(S, 3), np.percentile(S, 65)
        # full dynamic range (floor at 3rd pct) -- the honest "everything" version
        rc.save_rgb(rc.cmap_rgb(rc.unit(S, p3, p3 + 72), "magma"), f"{rc.GAL}/hero_{name}_native_magma_full.png")
        # glow: black point at the 65th percentile of dB (declared), so the noise floor drops to black
        rc.save_rgb(rc.cmap_rgb(rc.unit(S, p65, p65 + 50, 0.9), "magma"), f"{rc.GAL}/hero_{name}_native_magma_glow.png")
        rc.save_rgb(rc.ink_on_paper(rc.unit(S, p3 + 8, p3 + 72, 0.8)), f"{rc.GAL}/hero_{name}_native_sonograph.png")
        # log-frequency axis, 250 Hz - 24 kHz: harmonics k f(t) become parallel copies shifted by log k
        L = rc.logfreq_power(rc.load_spec(name), 2049, fmin=250.0)
        q65 = np.percentile(L, 65)
        rc.save_rgb(rc.cmap_rgb(rc.unit(L, q65, q65 + 50, 0.9), "magma"), f"{rc.GAL}/hero_{name}_logf_magma_glow.png")
        rc.save_rgb(rc.ink_on_paper(rc.unit(L, np.percentile(L, 3) + 8, np.percentile(L, 3) + 72, 0.8)),
                    f"{rc.GAL}/hero_{name}_logf_sonograph.png")
        print("hero", name, "black point p65 =", round(float(p65), 1), "dB")


def diptych():
    """Artwork diptych, no axes: undithered (top) and TPDF-dithered (bottom) 3-bit chirp, native STFT resolution,
    one shared black point (65th pct of the undithered plate) so the dithered haze is shown at its true level."""
    A = rc.resample_power(rc.load_spec("int3"))
    B = rc.resample_power(rc.load_spec("int3_tpdf"))
    lo = np.percentile(A, 65)
    for st in ("glow", "sonograph"):
        gap, strip = 60, 110
        W, Hh = A.shape[1], A.shape[0]
        bg = (0.0, 0.0, 0.0) if st == "glow" else rc.PAPER
        fg = (0.7, 0.68, 0.64) if st == "glow" else rc.INK
        cv = rc.canvas(W, 2 * Hh + gap + strip, bg)
        for i, S in enumerate((A, B)):
            if st == "glow":
                rgb = rc.cmap_rgb(rc.unit(S, lo, lo + 50, 0.9), "magma")
            else:
                rgb = rc.ink_on_paper(rc.unit(S, lo - 12, lo + 45, 0.8))
            rc.paste(cv, rgb, 0, i * (Hh + gap))
        rc.text(cv, (40, 2 * Hh + gap + 30), "above: 3-bit, rounded.   below: the same, with TPDF dither.   log chirp 20 Hz - 24 kHz, "
                "30 s; linear frequency 0 - 24 kHz; one shared dB scale.   STFT 4096 Blackman-Harris.   " + rc.STACK, 40, fg, rc.FONT_MONO)
        rc.save_png(cv, f"{rc.GAL}/diptych_int3_undithered_tpdf_{st}.png")
        print("diptych", st)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["magma", "sonograph", "shared", "riso", "hero"]
    for s in which:
        if s in ("magma", "sonograph", "shared"):
            column(s)
        elif s == "riso":
            riso_grid()
        elif s == "diptych":
            diptych()
        elif s == "hero":
            hero()
