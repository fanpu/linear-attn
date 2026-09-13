"""Length-as-time animation: the (T, p) partition after 1, 2, ..., L generated tokens.

Every frame is measured data. Cell colours are a proper colouring with inheritance
(analysis.stable_colours), so a cell keeps its colour until it splits and its largest
child keeps it after the split. Lead lines are drawn on every tile edge whose two
neighbouring outputs already differ at that length.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import analysis as A
import render as Rr


def plate_at(d, l, colours_l, s, pal, lead, mode="glass"):
    tk = d["tokens"][..., :l]
    L = tk.shape[2]
    P = Rr.palette(pal)
    if mode == "glass":
        img = P[colours_l]
        canvas = Rr.tiles(img, s)
        bv, bh = Rr.edge_fields(tk)
        fn = lambda v: (np.zeros(v.shape + (3,)) + 0.02, np.where(v < L, 1.0, 0.0))
        return Rr.paint_edges(canvas, bv.astype(float), bh.astype(float), s, lead, fn)
    if mode == "ink":
        return Rr.style_ink(dict(tokens=tk), s, w=lead)
    raise ValueError(mode)


def compose(plate, size, text_l, text_r, ground, ink):
    H, W = plate.shape[:2]
    canvas = np.ones((size, size, 3)) * np.array(ground)
    y0 = (size - H) // 2 - 8
    x0 = (size - W) // 2
    canvas[y0:y0 + H, x0:x0 + W] = plate
    pil = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(pil)
    f = ImageFont.truetype(Rr.FONT, 17)
    c = tuple(int(x * 255) for x in ink)
    dr.text((x0, y0 + H + 6), text_l, font=f, fill=c)
    dr.text((x0 + W, y0 + H + 6), text_r, font=f, fill=c, anchor="ra")
    return pil


def main(path, out_base, s=4, pal="glass", mode="glass", size=1080, hold=10, fade=4, fps=24,
         ground=(0.05, 0.05, 0.06), ink=(0.8, 0.78, 0.72), gif_size=540):
    d = A.load(path)
    tk = d["tokens"]
    L = tk.shape[2]
    hs = A.prefix_hashes(tk)
    cols = A.stable_colours(hs, len(Rr.palette(pal)), list(range(1, L + 1)))
    ncell = [len(np.unique(hs[l])) for l in range(L)]
    fr_dir = out_base + "_frames"
    os.makedirs(fr_dir, exist_ok=True)
    k = 0
    prev = None
    lead = max(1, s // 2)
    label = f"T {d['xs'][0] - (d['xs'][1] - d['xs'][0]) / 2:g}–{d['xs'][-1] + (d['xs'][1] - d['xs'][0]) / 2:g} →   top-p ↑"
    for l in range(1, L + 1):
        plate = plate_at(d, l, cols[l - 1], s, pal, lead, mode)
        im = compose(plate, size, f"after {l} token{'s' if l > 1 else ''}   {ncell[l - 1]} distinct outputs",
                     label, ground if mode == "glass" else Rr.INK_PAPER,
                     ink if mode == "glass" else (0.15, 0.15, 0.15))
        cur = np.asarray(im).astype(float)
        if prev is not None:
            for j in range(1, fade + 1):                    # declared crossfade between lengths
                a = j / (fade + 1)
                Image.fromarray((prev * (1 - a) + cur * a).astype(np.uint8)).save(f"{fr_dir}/{k:05d}.png"); k += 1
        reps = hold * (3 if l == L else 1)
        for _ in range(reps):
            im.save(f"{fr_dir}/{k:05d}.png"); k += 1
        prev = cur
    mp4 = out_base + ".mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{fr_dir}/%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", mp4], check=True)
    gif = out_base + ".gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{fr_dir}/%05d.png",
                    "-vf", f"fps=12,scale={gif_size}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=sierra2_4a",
                    gif], check=True)
    # keep a few stills for the README
    for l in (1, 4, 16, L):
        plate = plate_at(d, l, cols[l - 1], s, pal, lead, mode)
        Rr.save(plate, os.path.basename(out_base) + f"_L{l:03d}.png")
    subprocess.run(["rm", "-rf", fr_dir])
    print(mp4, os.path.getsize(mp4) / 1e6, "MB;", gif, os.path.getsize(gif) / 1e6, "MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], s=int(sys.argv[3]) if len(sys.argv) > 3 else 4,
         mode=sys.argv[4] if len(sys.argv) > 4 else "glass")
