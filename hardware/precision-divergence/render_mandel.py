"""Render the Mandelbrot precision-floor pieces from cache (compute_mandel.py).

    python render_mandel.py video      # side-by-side float32 | float64 zoom, MP4 + GIF
    python render_mandel.py stills     # float32 | float64 | binary128 triptychs at fixed depths, 3 styles
    python render_mandel.py curve      # measured precision-floor curves

Exact/measured: escape counts, fraction of pixels that differ, distinct pixel coordinates.
Aesthetic: colour maps (log smooth count through a sequential map, normalised per frame), layout.
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cmcrameri.cm  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).parent
CACHE, GALLERY = HERE / "cache", HERE / "gallery"
FR = CACHE / "mandel_frames"
MONO = "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf"
SERIF = "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf"


def colorize(v, maxit, style="dark", lo_hi=None):
    inside = v > maxit
    lv = np.log(np.maximum(v, 1.0))
    if lo_hi is None:
        out = lv[~inside]
        lo, hi = (np.percentile(out, 1), np.percentile(out, 99.5)) if out.size else (0, 1)
    else:
        lo, hi = lo_hi
    t = np.clip((lv - lo) / max(hi - lo, 1e-9), 0, 1)
    if style == "dark":
        rgb = plt.get_cmap("cmc.batlow")(t)[..., :3]
        rgb[inside] = 0.02
    elif style == "paper":
        # ink: iso-count contour bands in one ink on paper
        bands = (np.floor(t * 14) % 2 == 0) & ~inside   # 14 iso-bands of normalised log count
        base = np.array([0.953, 0.933, 0.886])
        ink = np.array([0.106, 0.102, 0.090])
        a = (0.25 + 0.75 * t) * bands
        a[inside] = 1.0
        rgb = base * (1 - a[..., None]) + ink * a[..., None]
    else:  # riso: two spot inks by count parity, density by depth
        base = np.array([0.957, 0.937, 0.902])
        pink = np.array([1.0, 0.31, 0.545])
        blue = np.array([0.122, 0.373, 0.749])
        par = (np.floor(t * 10) % 2 == 0)
        covp = np.where(par, 0.2 + 0.7 * t, 0.0)
        covb = np.where(~par, 0.2 + 0.7 * (1 - t), 0.0)
        covb[inside] = 1.0
        covb = np.roll(np.roll(covb, 3, 0), -2, 1)
        rgb = np.ones(v.shape + (3,)) * base
        rgb *= 1 - covp[..., None] * (1 - pink)
        rgb *= 1 - covb[..., None] * (1 - blue)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def video():
    stats = json.loads((CACHE / "mandel_stats.json").read_text())
    out = CACHE / "mandel_video"
    out.mkdir(exist_ok=True)
    fm, fb, fs = ImageFont.truetype(MONO, 26), ImageFont.truetype(MONO, 20), ImageFont.truetype(SERIF, 38)
    for s in stats:
        d = np.load(FR / f"frame_{s['i']:04d}.npz")
        mi = int(d["maxit"])
        a, b = d["f32"].astype(np.float32) / 8, d["f64"].astype(np.float32) / 8
        ref = b[b <= mi]
        lh = (np.log(np.percentile(ref, 1)), np.log(np.percentile(ref, 99.5))) if ref.size else None
        img = np.concatenate([colorize(a, mi, "dark", lh), np.zeros((1080, 4, 3), np.uint8), colorize(b, mi, "dark", lh)], 1)
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        for x0, name, dx in ((0, "float32", s["distinct_x_f32"]), (964, "float64", s["distinct_x_f64"])):
            dr.rectangle([x0 + 14, 14, x0 + 470, 112], fill=(0, 0, 0))
            dr.text((x0 + 24, 20), name, font=fs, fill=(240, 236, 226))
            dr.text((x0 + 24, 70), f"{dx} distinct pixel x", font=fb, fill=(240, 236, 226))
        dr.rectangle([14, 1080 - 90, 1910, 1080 - 14], fill=(0, 0, 0))
        dr.text((24, 1080 - 80), f"width {s['width']:.2e}   maxiter {mi}   pixels whose escape count differs: "
                                 f"{100 * s['frac_differ']:.1f}%", font=fm, fill=(240, 236, 226))
        dr.text((24, 1080 - 44), "seahorse valley  c = -0.743643887037151 + 0.131825904205330i   · everything in the "
                                 "stated precision, no FMA", font=fb, fill=(200, 196, 186))
        im.save(out / f"{s['i']:05d}.png")
    mp4 = GALLERY / "mandel_floor_zoom.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(out / "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "32", "-preset", "slow", str(mp4)], check=True)  # crf 32 keeps it < 20 MB
    gif = GALLERY / "mandel_floor_zoom.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(out / "%05d.png"),
                    "-vf", "fps=8,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                    str(gif)], check=True)
    print("wrote", mp4, gif)


def stills():
    deep = json.loads((CACHE / "mandel_deep_stats.json").read_text())
    for tag in deep:
        d = np.load(CACHE / f"mandel_deep_{tag}.npz")
        mi = int(d["maxit"])
        ref = d["f128"][d["f128"] <= mi]
        lh = (np.log(np.percentile(ref, 1)), np.log(np.percentile(ref, 99.5))) if ref.size else None
        for style in ("dark", "paper", "riso"):
            panels = [colorize(d[t], mi, style, lh) for t in ("f32", "f64", "f128")]
            gap = np.full((1080, 16, 3), 255 if style != "dark" else 0, np.uint8)
            top = 150
            img = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], 1)
            bgc = (5, 6, 8) if style == "dark" else ((243, 238, 226) if style == "paper" else (244, 239, 230))
            canvas = Image.new("RGB", (img.shape[1] + 80, img.shape[0] + top + 110), bgc)
            canvas.paste(Image.fromarray(img), (40, top))
            dr = ImageDraw.Draw(canvas)
            ink = (232, 228, 216) if style == "dark" else ((27, 26, 23) if style == "paper" else (31, 95, 191))
            dr.text((40, 30), f"Precision floor at width {float(d['width']):.0e}", font=ImageFont.truetype(SERIF, 64), fill=ink)
            for k, (t, name) in enumerate((("f32", "float32"), ("f64", "float64"), ("f128", "binary128 (113-bit long double)"))):
                x0 = 40 + k * (1080 + 16)
                extra = "" if t == "f128" else f"   {100 * deep[tag][t + '_vs_f128_frac_differ']:.1f}% pixels differ from binary128"
                dr.text((x0, top + 1090), name + extra, font=ImageFont.truetype(MONO, 26), fill=ink)
            dr.text((40, top + 1130), f"seahorse valley · 1080² each · maxiter {mi} · coordinates and iteration both in the stated "
                                      "precision · colour = log smooth escape count (declared)", font=ImageFont.truetype(MONO, 22), fill=ink)
            p = GALLERY / f"mandel_floor_{tag}_{style}.png"
            canvas.save(p, optimize=True)
            print("wrote", p)


def curve():
    stats = json.loads((CACHE / "mandel_stats.json").read_text())
    w = np.array([s["width"] for s in stats])
    fd = np.array([s["frac_differ"] for s in stats])
    d32 = np.array([s["distinct_x_f32"] for s in stats])
    d64 = np.array([s["distinct_x_f64"] for s in stats])
    fig, ax = plt.subplots(1, 2, figsize=(18, 7), dpi=200, facecolor="#f3eee2")
    for a in ax:
        a.set_facecolor("#f3eee2")
        a.set_xscale("log")
        a.invert_xaxis()
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
    ax[0].plot(w, fd, color="#b8322a", lw=2)
    ax[0].set_xlabel("frame width (log, zooming right)", family="Nimbus Mono PS")
    ax[0].set_ylabel("fraction of pixels: |count32 - count64| > 0.5", family="Nimbus Mono PS")
    ax[1].plot(w, d32, color="#b8322a", lw=2, label="float32")
    ax[1].plot(w, d64, color="#2f5d8a", lw=2, label="float64")
    ax[1].axvline(960 * np.spacing(np.float32(0.7436)), color="#b8322a", ls=":", label="960 x ulp32(0.7436)")
    ax[1].axvline(960 * np.spacing(0.7436), color="#2f5d8a", ls=":", label="960 x ulp64(0.7436)")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("distinct x coordinates across 960 pixels", family="Nimbus Mono PS")
    ax[1].set_xlabel("frame width", family="Nimbus Mono PS")
    ax[1].legend(frameon=False)
    fig.suptitle("Measured precision floor of the Mandelbrot zoom", family="P052", fontsize=22, x=0.02, ha="left")
    fig.savefig(GALLERY / "mandel_floor_curve.png", facecolor="#f3eee2")
    print("wrote curve")


if __name__ == "__main__":
    {"video": video, "stills": stills, "curve": curve}[sys.argv[1]]()
