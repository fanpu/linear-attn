"""To Scale: one token's residual stream drawn as bars at true relative height.

Measured: |h_d| for every dim d of one token's residual stream (Qwen3-0.6B, doc 0 of the
fineweb-edu val shard, position 0, after decoder layer LAYER), in bf16 as the model runs.
Unit: 1 mm = the median |h| over all 7.9k tokens x 1024 dims at that layer (Sun et al.'s reference).

Declared choices:
  - bar height = |h_d| / median, exactly (anti-aliased at the top pixel, never clipped or compressed);
  - bar pitch = 1 mm, no gap (the width is not a measured quantity);
  - one ink on cream; dims in index order;
  - in the film only: bars are drawn at least 2 px wide so a 1 mm bar stays visible when the
    camera is far away. Heights are never altered.

Outputs
  gallery/banner_print_1px_per_mm.png   the object at 1 px = 1 mm (print at 1 mm/px: 1.02 m x ~39 m;
                                         at 8,600 px/m: 12 cm x ~4.6 m)
  gallery/banner_folded.png              the same bars, cut into 3 m lengths laid side by side
  gallery/banner_film.mp4                 9:16 zoom-out for the feed
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
GAL = os.path.join(HERE, "gallery")
CREAM = np.array([242, 237, 227], np.float32)
INK = np.array([27, 26, 31], np.float32)
SERIF = "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf"
SERIF_I = "/usr/share/fonts/opentype/urw-base35/C059-Italic.otf"
MONO = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"


def font(size, italic=False, mono=False):
    p = MONO if mono else (SERIF_I if italic and os.path.exists(SERIF_I) else SERIF)
    return ImageFont.truetype(p, size)


def load(name="qwen3-0.6b", layer=2, pos=0):
    r = json.load(open(os.path.join(CACHE, f"acts_{name}.json")))
    z = np.load(os.path.join(CACHE, f"acts_{name}.npz"))
    v = np.abs(z["doc0"][layer, pos]).astype(np.float64)
    med = r["layers"][layer]["median"]
    tok = str(z["tokens0"][pos])
    return v / med, med, tok, r


def bars_coverage(h_mm, px_per_mm, width_px_per_bar, H):
    """ink coverage (H, n*w) for bars of heights h_mm drawn from the bottom, anti-aliased at the top."""
    n = len(h_mm)
    cov = np.zeros((H, n * width_px_per_bar), np.float32)
    hp = h_mm * px_per_mm
    for d in range(n):
        full = int(np.floor(hp[d]))
        frac = hp[d] - full
        full = min(full, H)
        c0, c1 = d * width_px_per_bar, (d + 1) * width_px_per_bar
        if full > 0:
            cov[H - full:, c0:c1] = 1.0
        if full < H and frac > 0:
            cov[H - full - 1, c0:c1] = frac
    return cov


def to_rgb(cov):
    return (CREAM[None, None] * (1 - cov[..., None]) + INK[None, None] * cov[..., None]).astype(np.uint8)


def print_file(h, med, tok, layer):
    n = len(h)
    top = int(np.ceil(h.max())) + 400
    margin, foot = 200, 800
    H = top
    cov = bars_coverage(h, 1.0, 1, H)
    W = n + 2 * margin
    img = np.empty((H + foot + margin, W, 3), np.uint8)
    img[:] = CREAM.astype(np.uint8)
    img[margin:margin + H, margin:margin + n] = to_rgb(cov)[: H]
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    y0 = margin + H + 60
    f1, f2 = font(64), font(30)
    dr.text((margin, y0), "To Scale", font=f1, fill=tuple(INK.astype(int)))
    i = int(np.argmax(h))
    lines = [
        f"Qwen3-0.6B, after layer {layer}. The token “{tok.replace('Ġ', ' ')}”,",
        "position 0 of a fineweb-edu page.",
        "One bar per residual dimension (1,024).",
        f"1 px = the median |activation| of the layer ({med:.3f}).",
        f"Tallest bar: dim {i}, |h| = {h[i] * med:,.0f}, {h[i]:,.0f} px.",
        f"At 1 mm per px: {W / 1000:.2f} m x {(H + foot + margin) / 1000:.1f} m.",
    ]
    for k, s in enumerate(lines):
        dr.text((margin, y0 + 110 + k * 44), s, font=f2, fill=tuple(INK.astype(int)))
    out = os.path.join(GAL, "banner_print_1px_per_mm.png")
    Image.MAX_IMAGE_PIXELS = None
    im.save(out, optimize=True)
    print("print file", im.size, out, os.path.getsize(out) // 1024, "KB")


def folded(h, med, tok, layer, seg_mm=3000, scale=0.5):
    """cut the banner into seg_mm lengths, laid left to right like a folding rule."""
    n = len(h)
    nseg = int(np.ceil(h.max() / seg_mm))
    panel_w = int(n * scale)
    panel_h = int(seg_mm * scale)
    gut = int(60 * scale / 0.5)
    margin = 120
    top_txt, foot = 220, 180
    W = margin * 2 + nseg * panel_w + (nseg - 1) * gut
    Ht = margin + top_txt + panel_h + foot + margin
    im = Image.new("RGB", (W, Ht), tuple(CREAM.astype(int)))
    arr = np.array(im)
    for s in range(nseg):
        lo = s * seg_mm
        hh = np.clip(h - lo, 0, seg_mm)
        # draw at `scale` px/mm with supersampling in x for sub-pixel widths
        ss = 4
        cov = bars_coverage(hh, scale, ss, panel_h)  # (panel_h, n*ss)  width per bar = ss sub-px
        # each bar is 1 mm = scale px wide; resample columns: n*ss sub-columns -> panel_w columns
        cov = cov.reshape(panel_h, panel_w, -1).mean(axis=2) if (n * ss) % panel_w == 0 else cov
        x0 = margin + s * (panel_w + gut)
        y0 = margin + top_txt
        arr[y0:y0 + panel_h, x0:x0 + panel_w] = to_rgb(cov)
    im = Image.fromarray(arr)
    dr = ImageDraw.Draw(im)
    ink = tuple(INK.astype(int))
    fs = font(26)
    for s in range(nseg):
        x0 = margin + s * (panel_w + gut)
        y0 = margin + top_txt
        dr.line([(x0, y0 + panel_h + 6), (x0 + panel_w, y0 + panel_h + 6)], fill=ink, width=1)
        dr.text((x0, y0 + panel_h + 18), f"{s * seg_mm // 1000}–{(s + 1) * seg_mm // 1000} m", font=fs, fill=ink)
    i = int(np.argmax(h))
    dr.text((margin, margin), "To Scale", font=font(64), fill=ink)
    dr.text((margin, margin + 90),
            f"Qwen3-0.6B, layer {layer}, first token “{tok.replace('Ġ', ' ')}”: 1,024 bars, 1 mm wide, 1 mm = the median |activation|. "
            f"Dimension {i} is {h[i] / 1000:.1f} m. Cut into {seg_mm // 1000} m lengths, read left to right.",
            font=font(28), fill=ink)
    out = os.path.join(GAL, "banner_folded.png")
    im.save(out, optimize=True)
    print("folded", im.size, out)


def film(h, med, tok, layer, fps=30, secs_zoom=14, secs_hold_a=2.5, secs_hold_b=4, Wf=1080, Hf=1920):
    """camera anchored at the floor under the tallest bar, zooming out exponentially."""
    n = len(h)
    i = int(np.argmax(h))
    top = h.max()
    floor_y = Hf - 260               # floor sits above a caption strip
    s0 = (Wf - 260) / n            # start: all 1,024 bars across the frame (~0.96 px per mm)
    s1 = (floor_y - 80) / top      # end: the whole tallest bar in frame
    frames = int(fps * (secs_hold_a + secs_zoom + secs_hold_b))
    tmp = os.path.join(CACHE, "film_frames")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    ink = tuple(INK.astype(int))
    fcap, fsmall = font(40), font(30)
    xa0 = 30 + (i + 0.5) * s0      # start: bar 0 at the left margin
    for k in range(frames):
        t = k / fps
        u = np.clip((t - secs_hold_a) / secs_zoom, 0, 1)
        u = 0.5 - 0.5 * np.cos(np.pi * u)  # ease in/out of an exponential zoom
        s = s0 * (s1 / s0) ** u           # px per mm
        x_anchor = xa0 + (Wf / 2 - xa0) * u  # the tallest bar drifts to the centre as the camera pulls back
        # bars: bar d spans x in [(d - i - 0.5) * s, (d - i + 0.5) * s] + anchor, width floored at 2 px
        cov = np.zeros((Hf, Wf), np.float32)
        centers = x_anchor + (np.arange(n) - i) * s
        wpx = max(s, 2.0)
        vis = np.where((centers + wpx / 2 >= 0) & (centers - wpx / 2 < Wf))[0]
        for d in vis:
            a, b = centers[d] - wpx / 2, centers[d] + wpx / 2
            ca, cb = int(np.floor(a)), int(np.ceil(b))
            hp = h[d] * s
            full = int(np.floor(hp))
            frac = hp - full
            ytop = floor_y - full
            for c in range(max(ca, 0), min(cb, Wf)):
                wcov = min(b, c + 1) - max(a, c)
                if wcov <= 0:
                    continue
                y_lo = max(ytop, 0)
                if full > 0 and y_lo < floor_y:
                    cov[y_lo:floor_y, c] = np.maximum(cov[y_lo:floor_y, c], wcov)
                if ytop - 1 >= 0 and frac > 0 and ytop <= floor_y:
                    cov[ytop - 1, c] = max(cov[ytop - 1, c], wcov * frac)
        rgb = to_rgb(cov)
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        dr.line([(0, floor_y), (Wf, floor_y)], fill=ink, width=1)
        # ruler on the left: ticks at 1-2-5 steps chosen for the current scale
        step = None
        for cand in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
            if cand * s >= 150:
                step = cand
                break
        if step is None:
            step = 10000
        m = 0
        while floor_y - m * s > 60:
            y = floor_y - m * s
            dr.line([(Wf - 40, y), (Wf - 70, y)], fill=ink, width=2)
            lab = f"{m / 1000:g} m" if step >= 500 else (f"{m / 10:g} cm" if step >= 10 else f"{m:g} mm")
            tw = dr.textlength(lab, font=fsmall)
            dr.text((Wf - 80 - tw, y - 17), lab, font=fsmall, fill=ink)
            m += step
        dr.text((40, floor_y + 30), f"Qwen3-0.6B, layer {layer}, first token “{tok.replace('Ġ', ' ')}”", font=fcap, fill=ink)
        dr.text((40, floor_y + 90), f"one bar per dimension; 1 mm = the median |activation|", font=fsmall, fill=ink)
        dr.text((40, floor_y + 135), f"dimension {i}: {h[i] / 1000:.1f} m", font=fsmall, fill=ink)
        im.save(os.path.join(tmp, f"{k:05d}.png"))
    out = os.path.join(GAL, "banner_film.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
    # a poster frame from the start and end
    Image.open(os.path.join(tmp, f"{int(fps * 1):05d}.png")).save(os.path.join(GAL, "banner_film_first.png"))
    Image.open(os.path.join(tmp, f"{frames - 1:05d}.png")).save(os.path.join(GAL, "banner_film_last.png"))
    print("film", out, frames, "frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--what", default="print,folded,film")
    a = ap.parse_args()
    os.makedirs(GAL, exist_ok=True)
    h, med, tok, r = load(layer=a.layer)
    print(f"token {tok!r}: max {h.max():.0f} x median ({h.max() / 1000:.2f} m at 1 mm); token's own median bar {np.median(h):.2f} mm;"
          f" next bars {np.sort(h)[::-1][1:4].round(0)}")
    if "print" in a.what:
        print_file(h, med, tok, a.layer)
    if "folded" in a.what:
        folded(h, med, tok, a.layer)
    if "film" in a.what:
        film(h, med, tok, a.layer)
