"""Render the recursive Bayer pieces (from cache/bayer.npz).

  bayer_genealogy_*     M_2 .. M_256 at the same print size (rank -> tone), the recursion written out
  bayer_equation_*      M_256 = 4 * tile(M_128) + expand(M_2): the substitution rule as three panels
  bayer_crosshatch_*    a 0..1 gray ramp in 1 bit, one row per order: the signature cross-hatch textures
  bayer_specimen8_*     all 65 on/off patterns of the 8x8 matrix (each tiled 4x4), a type-specimen sheet
  bayer_bitplanes_*     the 16 bit planes of the 256x256 rank matrix: stripes and checkerboards, one scale per pair
  bayer_sweep.mp4/.gif  threshold sweep 0 -> 1 on a 2x2 tiling of M_256: the recursive fill order
Styles: dark | paper | riso. Every pixel value is an exact matrix entry or an exact threshold comparison;
colours/inks are declared aesthetic choices.
"""
import os
import subprocess
import sys
import numpy as np
from PIL import Image
import common as c

D = np.load(f"{c.CACHE}/bayer.npz")


def style(st):
    if st == "dark":
        return dict(bg=(0.03, 0.03, 0.035), fg=(0.88, 0.86, 0.82), acc=(1.0, 0.72, 0.35))
    if st == "paper":
        return dict(bg=tuple(c.PAPER), fg=tuple(c.INK), acc=(0.75, 0.12, 0.1))
    return dict(bg=tuple(c.PAPER), fg=tuple(c.RISO_BLUE * 0.75), acc=tuple(c.RISO_PINK))


def tone(v, st):
    """v in [0,1] (rank / (N-1)) -> rgb."""
    if st == "dark":
        return c.cmap_rgb(v, "magma")
    if st == "paper":
        return c.ink_on_paper(1 - v)
    # riso: two inks, blue coverage = 1-bit {rank < 1/2}, pink = 1-bit {rank in [1/4, 3/4)} (exact level sets)
    blue = (v < 0.5).astype(float)
    pink = ((v >= 0.25) & (v < 0.75)).astype(float)
    return c.multiply_layers([(blue, c.RISO_BLUE), (c.shift(pink, 0, 0), c.RISO_PINK)])


def onebit(b, st):
    if st == "dark":
        return np.where(b[..., None] > 0, np.array([0.95, 0.93, 0.88]), np.array([0.03, 0.03, 0.035]))
    if st == "paper":
        return c.ink_on_paper(b.astype(float))
    return c.multiply_layers([(b.astype(float), c.RISO_BLUE)])


def up(a, s):
    return np.repeat(np.repeat(a, s, 0), s, 1)


def genealogy(st):
    P = style(st)
    S, gap, L, T = 1024, 70, 80, 230
    W = L * 2 + 4 * S + 3 * gap
    H = T + 2 * (S + 110) + 150
    cv = c.canvas(W, H, P["bg"])
    c.text(cv, (L, 50), "BAYER  -  one rule, applied eight times", 64, P["fg"], c.FONT_SERIF)
    c.text(cv, (L, 140), "M_1 = [[0, 2], [3, 1]],   M_2k = [[4 M_k, 4 M_k + 2], [4 M_k + 3, 4 M_k + 1]].   Tone = threshold rank "
           "(dark = switched on first).", 30, P["fg"], c.FONT_MONO)
    for i in range(8):
        n = 2 ** (i + 1)
        M = D[f"M{n}"]
        v = M / (n * n - 1)
        img = up(tone(v, st), S // n)
        r, col = divmod(i, 4)
        x0, y0 = L + col * (S + gap), T + r * (S + 110)
        c.paste(cv, img, x0, y0)
        c.text(cv, (x0, y0 + S + 18), f"{n} x {n}", 40, P["fg"], c.FONT_SERIF)
        c.text(cv, (x0 + S, y0 + S + 26), f"{n * n} thresholds", 26, P["fg"], c.FONT_MONO, anchor="ra")
        if n <= 8:
            for yy in range(n):
                for xx in range(n):
                    c.text(cv, (x0 + xx * S // n + S // (2 * n), y0 + yy * S // n + S // (2 * n)), str(M[yy, xx]),
                           max(18, 200 // n), P["acc"], c.FONT_MONO, anchor="mm")
    c.text(cv, (L, H - 90), "Exact: the on-set at gray level c/4^m is the order-m pattern tiled with period 2^m (checked for all "
           "87 388 (m, c) pairs at order 8).  The first 4^k pixels switched on form a square lattice of spacing 256/2^k.", 28, P["fg"], c.FONT_SANS)
    c.text(cv, (L, H - 45), c.STACK + ("  |  riso: blue = rank < 1/2, pink = rank in [1/4, 3/4)" if st == "riso" else ""), 22, P["fg"], c.FONT_MONO)
    c.save_png(cv, f"{c.GAL}/bayer_genealogy_{st}.png")
    print("genealogy", st)


def equation(st):
    P = style(st)
    S, L, T = 1024, 80, 200
    W = L * 2 + 3 * S + 2 * 260
    cv = c.canvas(W, T + S + 200, P["bg"])
    c.text(cv, (L, 50), "The substitution rule:  M_256  =  4 x tile(M_128)  +  expand(M_2)", 56, P["fg"], c.FONT_SERIF)
    M256, M128, M2 = D["M256"], D["M128"], D["M2"]
    tile = np.tile(M128, (2, 2))
    expand = up(M2, 128)
    assert np.array_equal(M256, 4 * tile + expand)
    panels = [(M256 / 65535, "M_256"), (tile / 16383, "tile(M_128), 2 x 2"), (expand / 3, "expand(M_2), 128x blocks")]
    for i, (v, lab) in enumerate(panels):
        x0 = L + i * (S + 260)
        c.paste(cv, up(tone(v, st), 4), x0, T)
        c.text(cv, (x0, T + S + 25), lab, 38, P["fg"], c.FONT_SERIF)
        if i < 2:
            c.text(cv, (x0 + S + 130, T + S // 2), "=" if i == 0 else "+", 150, P["acc"], c.FONT_SERIF, anchor="mm")
    c.text(cv, (L, T + S + 100), "Checked bit-exactly (assert).  Coarse position sets the LOW digits of the rank and fine position the "
           "HIGH digits, so adjacent pixels always receive distant thresholds.", 28, P["fg"], c.FONT_SANS)
    c.save_png(cv, f"{c.GAL}/bayer_equation_{st}.png")
    print("equation", st)


def crosshatch(st):
    P = style(st)
    s = 2
    rows = [f"bayer{2 ** k}" for k in range(1, 9)]
    rw, rh = 2048 * s, 160 * s
    L, T, gap = 230, 200, 26
    W = L + rw + 60
    H = T + len(rows) * (rh + gap) + 150
    cv = c.canvas(W, H, P["bg"])
    c.text(cv, (60, 50), "Cross-hatch  -  a gray ramp from 0 to 1, in one bit, by each order of the Bayer matrix", 54, P["fg"], c.FONT_SERIF)
    for i, key in enumerate(rows):
        b = D[f"ramp_{key}"]
        y0 = T + i * (rh + gap)
        if st == "riso":
            ink = c.RISO_BLUE if i % 2 == 0 else c.RISO_PINK
            img = c.multiply_layers([(up(b, s).astype(float), ink)])
        else:
            img = onebit(up(b, s), st)
        c.paste(cv, img, L, y0)
        n = int(key[5:])
        c.text(cv, (L - 24, y0 + rh // 2), f"{n}x{n}", 40, P["fg"], c.FONT_SERIF, anchor="rm")
        c.text(cv, (L - 24, y0 + rh // 2 + 40), f"{n * n + 1} tones", 22, P["fg"], c.FONT_MONO, anchor="rm")
    for g in (0, 0.25, 0.5, 0.75, 1):
        x = L + int(g * (rw - 1))
        c.line(cv, [(x, T - 20), (x, T - 4)], P["fg"], 3)
        c.text(cv, (x, T - 26), f"{g:g}", 26, P["fg"], c.FONT_MONO, anchor="mb")
    c.text(cv, (60, H - 90), ("Riso: rows alternate blue/pink ink (declared).  " if st == "riso" else "") + "Pixel on iff ramp value > (rank + 1/2) / N.  Each pixel is drawn as a 2x2 block (print scale, declared).  "
           "At every gray level c/4^m the texture is exactly periodic with period 2^m.", 28, P["fg"], c.FONT_SANS)
    c.text(cv, (60, H - 45), c.STACK, 22, P["fg"], c.FONT_MONO)
    c.save_png(cv, f"{c.GAL}/bayer_crosshatch_{st}.png")
    print("crosshatch", st)


def specimen(st):
    P = style(st)
    cell, pad, L, T = 384, 44, 80, 220
    cols, nrows = 13, 5
    W = L * 2 + cols * cell + (cols - 1) * pad
    H = T + nrows * (cell + pad + 40) + 110
    cv = c.canvas(W, H, P["bg"])
    c.text(cv, (L, 60), "Specimen  -  the 65 tones of the 8x8 Bayer matrix, each tiled 4x4", 60, P["fg"], c.FONT_SERIF)
    for t in range(65):
        r, col = divmod(t, cols)
        x0, y0 = L + col * (cell + pad), T + r * (cell + pad + 40)
        pat = np.tile(D["spec8"][t], (4, 4))
        if st == "riso":
            img = c.multiply_layers([(up(pat, 12).astype(float), c.RISO_PINK if (t % 2) else c.RISO_BLUE)])
        else:
            img = onebit(up(pat, 12), st)
        c.paste(cv, img, x0, y0)
        c.text(cv, (x0 + cell // 2, y0 + cell + 8), f"{t}/64", 28, P["fg"], c.FONT_MONO, anchor="ma")
    c.text(cv, (L, H - 70), "t/64 on: pattern = {M_8 < t}.  " + c.STACK, 26, P["fg"], c.FONT_MONO)
    c.save_png(cv, f"{c.GAL}/bayer_specimen8_{st}.png")
    print("specimen", st)


def bitplanes(st):
    P = style(st)
    S, pad, L, T = 768, 50, 80, 220
    W = L * 2 + 4 * S + 3 * pad
    H = T + 4 * (S + pad + 40) + 100
    cv = c.canvas(W, H, P["bg"])
    c.text(cv, (L, 60), "Bit planes of the 256x256 rank matrix  -  one scale per pair of bits", 60, P["fg"], c.FONT_SERIF)
    for b in range(16):
        bit = 15 - b
        r, col = divmod(b, 4)
        x0, y0 = L + col * (S + pad), T + r * (S + pad + 40)
        pl = D["planes"][bit]
        if st == "riso":
            img = c.multiply_layers([(up(pl, 3).astype(float), c.RISO_BLUE if bit % 2 else c.RISO_PINK)])
        else:
            img = onebit(up(pl, 3), st)
        c.paste(cv, img, x0, y0)
        lvl = (15 - bit) // 2
        what = "x XOR y" if bit % 2 else "y"
        c.text(cv, (x0, y0 + S + 8), f"bit {bit}: {what} at bit {lvl} of the coordinates (period {2 ** (lvl + 1)})", 24, P["fg"], c.FONT_MONO)
    c.text(cv, (L, H - 60), "M = sum_b (2 (x_b XOR y_b) + y_b) 4^(7-b): the top bits are the finest checkerboards, the bottom bits "
           "the coarsest.  " + c.STACK, 24, P["fg"], c.FONT_MONO)
    c.save_png(cv, f"{c.GAL}/bayer_bitplanes_{st}.png")
    print("bitplanes", st)


def sweep():
    M = np.tile(D["M256"], (2, 2))
    out = f"{c.GAL}/bayer_sweep.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1080", "-r", "30", "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    small = []
    # thresholds: geometric at the start so the lattice stages (1, 4, 16, ... pixels per tile) are visible, then linear
    ts = np.unique(np.concatenate([np.round(4 ** np.linspace(0, 6, 120)), np.linspace(4096, 65536, 240)]).astype(int))
    ts = np.concatenate([ts, [65536] * 30])
    for j, t in enumerate(ts):
        on = (M < t).astype(np.uint8)
        cv = c.canvas(1080, 1080, (0.03, 0.03, 0.035))
        c.paste(cv, onebit(up(on, 2), "dark")[: 1024, : 1024], 28, 20)
        c.text(cv, (28, 1052), f"threshold {t:>5d} / 65536   ({t / 65536:.4f})    M_256 tiled 2x2, each pixel 2x2", 24, (0.88, 0.86, 0.82), c.FONT_MONO, anchor="lm")
        arr = np.array(cv)
        proc.stdin.write(arr.tobytes())
        if j % 3 == 0:
            small.append(Image.fromarray(arr).resize((540, 540), Image.NEAREST))
    proc.stdin.close(); proc.wait()
    small[0].save("/tmp/bayer.gif", save_all=True, append_images=small[1:], duration=100, loop=0)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/bayer.gif", "-vf",
                    "split[s0][s1];[s0]palettegen=max_colors=16[p];[s1][p]paletteuse=dither=none", f"{c.GAL}/bayer_sweep.gif"], check=True)
    print("sweep", os.path.getsize(out) / 1e6, os.path.getsize(f"{c.GAL}/bayer_sweep.gif") / 1e6)


def digit_reversed(n=8):
    """R(x, y) = sum_b d_b 4^b with the SAME digits d_b = 2 (x_b XOR y_b) + y_b as the Bayer rank, read coarse-first.
    (The Bayer rank reads them fine-first: M = sum_b d_b 4^(n-1-b).)"""
    s = 2 ** n
    y, x = np.mgrid[0:s, 0:s]
    R = np.zeros((s, s), dtype=np.int64)
    M = np.zeros((s, s), dtype=np.int64)
    for b in range(n):
        d = 2 * (((x >> b) & 1) ^ ((y >> b) & 1)) + ((y >> b) & 1)
        R += d * 4 ** b
        M += d * 4 ** (n - 1 - b)
    assert np.array_equal(M, D[f"M{s}"])
    return R, x ^ y, y


def digits(st):
    """Three views of the same 8 base-4 digits of M_256."""
    P = style(st)
    S, gap, L, T = 1280, 90, 80, 240
    W = L * 2 + 3 * S + 2 * gap
    cv = c.canvas(W, T + S + 260, P["bg"])
    R, xor, yy = digit_reversed()
    c.text(cv, (L, 50), "Where the recursion hides: the Bayer rank is a base-4 number read fine-first", 58, P["fg"], c.FONT_SERIF)
    c.text(cv, (L, 140), "digit at level b:  d_b = 2 (x_b XOR y_b) + y_b.   left: M = sum d_b 4^(7-b) (the dither matrix).   middle: the same "
           "digits read coarse-first, R = sum d_b 4^b.   right: the two bit streams inside the digits, as two inks.", 28, P["fg"], c.FONT_SANS)
    M = D["M256"]
    panels = [tone(M / 65535, st) if st != "riso" else tone(M / 65535, "paper"),
              tone(R / 65535, st) if st != "riso" else tone(R / 65535, "paper"),
              c.multiply_layers([(yy / 255.0, c.RISO_BLUE), (xor / 255.0, c.RISO_PINK)]) if st != "dark" else
              np.clip(np.stack([xor / 255.0, 0.35 * xor / 255.0 + 0.35 * yy / 255.0, yy / 255.0], -1), 0, 1)]
    labs = ["M_256: fine digits first (looks like noise)", "R: the same digits, coarse first (nested quadrants)",
            "y (blue) and x XOR y (pink): the interleaved inputs"]
    for i, (img, lab) in enumerate(zip(panels, labs)):
        x0 = L + i * (S + gap)
        c.paste(cv, up(img, 5), x0, T)
        c.text(cv, (x0, T + S + 20), lab, 32, P["fg"], c.FONT_SERIF)
    c.text(cv, (L, T + S + 90), "All three panels are exact functions of (x, y) on the 256x256 grid, each pixel drawn 5x5.  The right panel's "
           "x XOR y is the self-similar 'munching squares' carpet; Bayer = bit-reversed interleave of (x XOR y, y).", 26, P["fg"], c.FONT_SANS)
    c.text(cv, (L, T + S + 135), c.STACK + ("  |  dark right panel: red = x XOR y, blue = y (declared)" if st == "dark" else ""), 22, P["fg"], c.FONT_MONO)
    c.save_png(cv, f"{c.GAL}/bayer_digits_{st}.png")
    print("digits", st)


def selfsim(st):
    """Top-left 256, 64, 16, 4 of R (coarse-first digits) and of M, each shown at the same size.
    Exact: R[:2^m, :2^m] = R_m and M[:2^m, :2^m] = 4^(8-m) M_m, so each panel is the order-m object."""
    P = style(st)
    S, gap, L, T = 1024, 70, 200, 230
    W = L + 4 * S + 3 * gap + 80
    cv = c.canvas(W, T + 2 * (S + 100) + 120, P["bg"])
    c.text(cv, (80, 50), "Self-similar by construction: the top-left corner of order 8 is order 6, 4, 2", 58, P["fg"], c.FONT_SERIF)
    R, _, _ = digit_reversed()
    M = D["M256"]
    for row, (A, name) in enumerate(((R, "R (coarse-first)"), (M, "M (Bayer)"))):
        c.text(cv, (L - 30, T + row * (S + 100) + S // 2), name.split()[0], 60, P["fg"], c.FONT_SERIF, anchor="rm")
        for i, m in enumerate((8, 6, 4, 2)):
            sub = A[: 2 ** m, : 2 ** m]
            v = (sub - sub.min()) / max(sub.max() - sub.min(), 1)
            ref = digit_reversed(m)[0] if row == 0 else D[f"M{2 ** m}"]
            exact = np.array_equal(sub, ref) if row == 0 else np.array_equal(sub, 4 ** (8 - m) * ref)
            x0, y0 = L + i * (S + gap), T + row * (S + 100)
            img = tone(v, st if st != "riso" else "paper") if st != "riso" else c.multiply_layers([((1 - v), c.RISO_BLUE if row == 0 else c.RISO_PINK)])
            c.paste(cv, up(img, S // 2 ** m), x0, y0)
            c.text(cv, (x0, y0 + S + 16), f"top-left {2 ** m}x{2 ** m}" + ("  = order " + str(m) + (" (exact)" if exact else " (MISMATCH)")), 30, P["fg"], c.FONT_MONO)
    c.save_png(cv, f"{c.GAL}/bayer_selfsim_{st}.png")
    print("selfsim", st)


def crosshatch_crops(st):
    """Native-resolution texture crops: constant gray levels dithered by several orders, 1 dot = 6x6 px."""
    P = style(st)
    grays = [(1 / 8, "1/8"), (1 / 4, "1/4"), (1 / 3, "1/3"), (3 / 8, "3/8"), (1 / 2, "1/2"), (5 / 8, "5/8")]
    orders = [4, 8, 16, 256]
    cell, sc, gap, L, T = 64, 6, 40, 260, 230
    S = cell * sc
    W = L + len(grays) * (S + gap) + 40
    H = T + len(orders) * (S + gap + 10) + 120
    cv = c.canvas(W, H, P["bg"])
    c.text(cv, (60, 50), "Cross-hatch, up close  -  constant grays, 1 dot = 6x6 px, no smoothing", 58, P["fg"], c.FONT_SERIF)
    for j, (g, lab) in enumerate(grays):
        c.text(cv, (L + j * (S + gap) + S // 2, T - 20), f"gray {lab}", 34, P["fg"], c.FONT_MONO, anchor="mb")
    for i, n in enumerate(orders):
        Mn = D[f"M{n}"]
        y0 = T + i * (S + gap + 10)
        c.text(cv, (L - 30, y0 + S // 2), f"{n}x{n}", 48, P["fg"], c.FONT_SERIF, anchor="rm")
        for j, (g, lab) in enumerate(grays):
            b = c.ordered(np.full((cell, cell), g), Mn)
            if st == "riso":
                img = c.multiply_layers([(up(b, sc).astype(float), c.RISO_BLUE if i % 2 == 0 else c.RISO_PINK)])
            else:
                img = onebit(up(b, sc), st)
            c.paste(cv, img, L + j * (S + gap), y0)
    c.text(cv, (60, H - 70), "Pixel on iff gray > (rank + 1/2)/N.  Gray c/4^m repeats with period 2^m; 1/3 and 3/8 are not dyadic "
           "quarters, so they mix two neighbouring textures." + ("  Riso: rows alternate blue/pink ink (declared)." if st == "riso" else ""), 26, P["fg"], c.FONT_SANS)
    c.save_png(cv, f"{c.GAL}/bayer_crosshatch_crops_{st}.png")
    print("crops", st)


if __name__ == "__main__":
    which = sys.argv[1:] or ["genealogy", "equation", "crosshatch", "specimen", "bitplanes", "digits", "selfsim", "crosshatch_crops", "sweep"]
    styles = [s for s in which if s in ("dark", "paper", "riso")] or ["dark", "paper", "riso"]
    for w in which:
        if w in ("genealogy", "equation", "crosshatch", "specimen", "bitplanes", "digits", "selfsim", "crosshatch_crops"):
            for st in styles:
                globals()[w](st)
        elif w == "sweep":
            sweep()
