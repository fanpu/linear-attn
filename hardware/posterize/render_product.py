"""FP8 E4M3 product rounding error, relative: r = (Q(Q(x) Q(y)) - xy) / xy  (cache/product_*.npz).
Plates: log-log axes x, y in [2^-4, 2^4) (512 px per octave: the octave tiling is exact), and linear axes (0, 8].
Style: Sohl-Dickstein Spectral split at zero (signed quantity; declared). Run: python render_product.py"""
import json
import numpy as np
import common as c

st = json.load(open(f"{c.CACHE}/product_stats.json"))
for name, lab in (("loglog", "log-log axes, x and y from 2^-4 to 2^4 (512 px per octave)"), ("linear", "linear axes, x and y in (0, 8]")):
    d = np.load(f"{c.CACHE}/product_{name}.npz")
    r = d["r_full"].astype(np.float64)[::-1]            # y up
    img = c.spectral_sd(r, r[::2, ::2])
    cv = c.canvas(4096, 4096 + 150, c.PAPER)
    c.paste(cv, img, 0, 0)
    c.text(cv, (40, 4096 + 20), f"FP8 E4M3: relative rounding error of multiplication (Q(Q(x) Q(y)) - xy)/xy, {lab}.", 40, c.INK, c.FONT_SERIF)
    extra = (f"Octave shift in x leaves every normal-range pixel bit-identical (fraction {st['loglog_full_octave_shift_exact_equal_frac_on_normal']:.4f}). "
             if name == "loglog" else "Input rounding makes rectilinear cells whose width doubles every octave; output rounding makes hyperbolae xy = const. ")
    c.text(cv, (40, 4096 + 80), extra + "Colour: Sohl-Dickstein Spectral split at 0 (declared).  " + c.STACK, 28, c.INK, c.FONT_SANS)
    c.save_png(cv, f"{c.GAL}/product_fp8_{name}_spectral.png")
    print(name)
