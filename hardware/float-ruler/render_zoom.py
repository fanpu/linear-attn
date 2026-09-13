"""Octave zoom: the window [0, W] shrinks by 2x per octave; for normal floats every octave
frame is *identical* (v(e+1,m) = 2 v(e,m)), until the window reaches the subnormals and
the comb turns uniform, then runs out.

Exact: tick positions, tick heights (ruler rank of mantissa field), and the on-frame
'identical to one octave ago' flag, which is computed by comparing exact tick sets.
Aesthetic: colours, fonts, frame rate, layout.

    python render_zoom.py video   # MP4 + GIF (observatory + engraved)
    python render_zoom.py stills  # 4x4 small-multiple plates in 3 styles
"""
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from formats import FP8_E4M3, FP8_E5M2, FP16, positive_finite
from plate import GALLERY, CACHE, STYLES, trailing_zeros
from raster import InkLayer, composite, multiply
import matplotlib.pyplot as plt

FONT = "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf"
FONTI = "/usr/share/fonts/opentype/urw-base35/P052-Italic.otf"
MONO = "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf"

FMTS = [FP8_E4M3, FP8_E5M2, FP16]
DATA = {}
for f in FMTS:
    d = positive_finite(f)
    lev = f.M - trailing_zeros(d["m"], f.M)
    DATA[f.name] = dict(v=np.concatenate([[0.0], d["value"]]),
                        h=np.concatenate([[1.0], 0.12 + 0.88 * 0.7 ** lev]),
                        sub=np.concatenate([[False], d["cls"] == 1]),
                        frac=np.concatenate([[0.0], d["m"] / (1 << f.M)]))

K0 = 9.0          # window starts at [0, 2^9]
K1 = -26.0        # ends at [0, 2^-26]
FPO = 36          # frames per octave


def self_similar(fmt, k):
    """Exact: is the tick set in [W/64, W] (W=2^k) equal to 2x the tick set one octave down?"""
    v = DATA[fmt.name]["v"]
    a = v[(v <= 2.0 ** k) & (v >= 2.0 ** (k - 6))] / 2.0 ** k
    b = v[(v <= 2.0 ** (k - 1)) & (v >= 2.0 ** (k - 7))] / 2.0 ** (k - 1)
    if len(a) == 0:
        return None
    return len(a) == len(b) and np.array_equal(a, b)


def frame(t, style="observatory", W=1920, H=1080):
    st = STYLES[style]
    k = K0 - t
    Wz = 2.0 ** k
    X0, X1 = 110, W - 90
    rows_y = [410, 690, 970]
    rgb = style == "observatory"
    main = InkLayer(H, W, rgb=rgb)
    sub = InkLayer(H, W)
    rules = InkLayer(H, W)
    cmap = plt.get_cmap("plasma")
    for f, yb in zip(FMTS, rows_y):
        D = DATA[f.name]
        sel = D["v"] <= Wz * 1.0001
        x = X0 + D["v"][sel] / Wz * (X1 - X0)
        h = 175 * D["h"][sel]
        w = 1.6 if f.M < 7 else 0.5
        s = ~D["sub"][sel]
        main.ticks(x[s], yb, h[s], w_px=w, colors=cmap(0.3 + 0.7 * D["frac"][sel][s]) if rgb else None)
        if (~s).any():
            sub.ticks(x[~s], yb, h[~s], w_px=w)
        rules.hline(yb + 1, X0, X1, 1.5, 0.5)
    ink = st["ink"]
    if style == "riso":
        img = multiply(st["bg"], [(main.alpha(1.5), st["ink"]), (np.clip(sub.alpha(1.5) + rules.alpha(), 0, 1), st["ink2"])])
    else:
        mc = main.color(ink) if rgb else ink
        img = composite(st["bg"], [(rules.alpha() * 0.7, ink), (main.alpha(1.4), mc), (sub.alpha(1.4), st["sub"])])
    im = Image.fromarray((img * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    col = tuple(int(255 * c) for c in plt.matplotlib.colors.to_rgb(st["ink"] if style != "riso" else st["ink2"]))
    f_title = ImageFont.truetype(FONT, 52)
    f_it = ImageFont.truetype(FONTI, 26)
    f_m = ImageFont.truetype(MONO, 22)
    f_mb = ImageFont.truetype(MONO, 30)
    dr.text((X0, 50), "Octave zoom", font=f_title, fill=col)
    dr.text((X0, 120), "window [0, W] shrinks 2x per octave; normal floats redraw the same picture", font=f_it, fill=col)
    dr.text((X1, 62), f"W = 2^{k:+.2f}", font=f_mb, fill=col, anchor="ra")
    kk = int(np.ceil(k))
    for f, yb in zip(FMTS, rows_y):
        ss = self_similar(f, kk)
        dr.text((X0, yb - 225), f"{f.name}", font=f_m, fill=col)
        dr.text((X1, yb - 225), ("ticks in [W/64, W] identical to one octave ago" if ss else
                                ("no values in [W/64, W]: below the floor" if ss is None else
                                 "differs from one octave ago: top of range / subnormal floor")),
                font=f_m, fill=col, anchor="ra")
        for q, lab in ((0, "0"), (0.5, "W/2"), (1, "W")):
            dr.text((X0 + q * (X1 - X0), yb + 12), lab, font=f_m, fill=col, anchor="ma")
    dr.text((W - 90, H - 30), "exact decode of every bit pattern · tick height = ruler rank of mantissa · subnormals in 2nd colour",
            font=ImageFont.truetype(MONO, 16), fill=col, anchor="rs")
    return np.asarray(im)


def _frame_job(args):
    i, style = args
    t = i / FPO
    arr = frame(t, style)
    p = CACHE / f"zoom_{style}" / f"{i:05d}.png"
    Image.fromarray(arr).save(p)
    return i


def video(style):
    (CACHE / f"zoom_{style}").mkdir(parents=True, exist_ok=True)
    n = int((K0 - K1) * FPO) + 1
    with Pool(4) as pool:
        for _ in pool.imap_unordered(_frame_job, [(i, style) for i in range(n)], chunksize=8):
            pass
    mp4 = GALLERY / f"octave_zoom_{style}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(CACHE / f"zoom_{style}" / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", str(mp4)], check=True)
    print("wrote", mp4)
    if style == "observatory":
        gif = GALLERY / f"octave_zoom_{style}.gif"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(CACHE / f"zoom_{style}" / "%05d.png"),
                        "-vf", "fps=15,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=sierra2_4a",
                        str(gif)], check=True)
        print("wrote", gif)


def stills(style):
    """Octave stack: every octave [2^e, 2^(e+1)] of each format magnified to the same width, stacked.
    Normal octaves are identical rows; the top row (range end) and bottom row (subnormals, [0, min normal])
    are the only different ones."""
    st = STYLES[style]
    W, H = 5400, 7400
    cols = [(FP8_E4M3, 150, 1850), (FP8_E5M2, 1950, 3650), (FP16, 3750, 5250)]
    Y0, Y1 = 700, 7050
    rgb = style == "observatory"
    main, sub, rules = InkLayer(H, W, rgb=rgb), InkLayer(H, W), InkLayer(H, W)
    cmap = plt.get_cmap("plasma")
    labels = []
    for f, x0, x1 in cols:
        D = DATA[f.name]
        v = D["v"]
        emin = 1 - ((1 << (f.E - 1)) - 1)
        vmax = v.max()
        emax = int(np.floor(np.log2(vmax)))
        octs = list(range(emax, emin - 1, -1))
        nrows = len(octs) + 1
        pitch = (Y1 - Y0) / nrows
        hmax = pitch * 0.72
        for r, e in enumerate(octs + ["sub"]):
            yb = int(Y0 + (r + 1) * pitch - pitch * 0.12)
            if e == "sub":
                lo, hi = 0.0, 2.0 ** emin
                sel = (v >= lo) & (v <= hi)
                lab = f"[0, 2^{emin}]  subnormals"
            else:
                lo, hi = 2.0 ** e, 2.0 ** (e + 1)
                sel = (v >= lo) & (v <= hi)
                lab = f"[2^{e}, 2^{e + 1}]" + ("  top: ends at max" if hi > vmax else "")
            xx = x0 + (v[sel] - lo) / (hi - lo) * (x1 - x0)
            w = 3.0 if f.M < 7 else max(0.35, min(1.2, 0.8 * (x1 - x0) / (1 << f.M)))
            ssub = D["sub"][sel]
            hh = hmax * D["h"][sel]
            if e != "sub":  # octave start tick at 2^e is m=0; its end tick 2^(e+1) is next octave's m=0
                pass
            main.ticks(xx[~ssub], yb, hh[~ssub], w_px=w, colors=cmap(0.3 + 0.7 * D["frac"][sel][~ssub]) if rgb else None)
            if ssub.any():
                sub.ticks(xx[ssub], yb, hh[ssub], w_px=w)
            rules.hline(yb + 1, x0, x1, 1.5, 0.45)
            labels.append((x0, yb - hmax - 8, lab, pitch))
    ink = st["ink"]
    if style == "riso":
        img = multiply(st["bg"], [(main.alpha(1.5), st["ink"]), (np.clip(sub.alpha(1.5) + rules.alpha(), 0, 1), st["ink2"])])
    else:
        mc = main.color(ink) if rgb else ink
        img = composite(st["bg"], [(rules.alpha() * 0.7, ink), (main.alpha(1.4), mc), (sub.alpha(1.4), st["sub"])])
    im = Image.fromarray((img * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    col = tuple(int(255 * c) for c in plt.matplotlib.colors.to_rgb(ink if style != "riso" else st["ink2"]))
    dr.text((150, 150), "Octave stack", font=ImageFont.truetype(FONT, 150), fill=col)
    dr.text((150, 350), "each row is one octave [2^e, 2^(e+1)] of the format, magnified to the same width. "
                        "v(e+1, m) = 2 v(e, m): the rows are identical,", font=ImageFont.truetype(FONTI, 58), fill=col)
    dr.text((150, 430), "except the top (the range runs out) and the bottom (subnormals: [0, smallest normal], evenly spaced).",
            font=ImageFont.truetype(FONTI, 58), fill=col)
    for f, x0, x1 in cols:
        dr.text((x0, 590), f.name, font=ImageFont.truetype(FONT, 70), fill=col)
    for x0, y, lab, pitch in labels:
        dr.text((x0, y), lab, font=ImageFont.truetype(MONO, int(min(34, pitch * 0.2))), fill=col, anchor="ls")
    dr.text((W - 150, H - 120), "exact decode of every bit pattern · tick height = ruler rank of mantissa field · "
            "subnormals in second colour", font=ImageFont.truetype(MONO, 36), fill=col, anchor="rs")
    p = GALLERY / f"octave_stack_{style}.png"
    im.save(p, optimize=True)
    print("wrote", p)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "stills"
    styles = sys.argv[2:] or (["observatory", "engraved"] if what == "video" else ["observatory", "engraved", "riso"])
    for s in styles:
        (video if what == "video" else stills)(s)
