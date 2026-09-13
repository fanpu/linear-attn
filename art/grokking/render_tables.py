"""The model's answer table over all (a, b) at each full checkpoint: argmax prediction coloured by residue with a
cyclic map (residues are cyclic).  Correct = anti-diagonal bands (a+b = const).  Output: gallery/table_*.mp4/.gif/.png"""
import argparse, os, subprocess, shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import colorcet as cc
from styles import STYLES, hex2rgb, P

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
A = dict(np.load("cache/analysis.npz"))
T = np.load("cache/tables.npz")
s = args.seed
tag = f"seed{int(A['init_seeds'][s])}"
fs = T["full_steps"]; pred = T["pred_tab"][:, s]; lp = T["lp_tab"][:, s].astype(np.float32)
train_mask = np.zeros(P * P, bool); train_mask[A["train_idx"][s] if "train_idx" in A else []] = True
cyc = cc.cm["CET_C6"]
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
big = 8   # each (a,b) cell -> 8x8 px


def frame(ti, style):
    st = STYLES[style]
    if style == "nocturne":
        img = cyc(pred[ti] / P)[..., :3]
    else:  # plate: probability of the correct answer, sepia density
        pc = np.exp(np.clip(lp[ti], -30, 0))
        c0, c1 = hex2rgb(st["bg"]), hex2rgb("#2a1d10")
        img = c0 + (c1 - c0) * pc[..., None]
    img = np.kron((img * 255).astype(np.uint8), np.ones((big, big, 1), np.uint8))
    H = img.shape[0]; pad = 76
    canvas = np.ones((H + 2 * pad, H + 2 * pad, 3), np.uint8) * (hex2rgb(st["bg"]) * 255).astype(np.uint8)
    canvas[pad:pad + H, pad:pad + H] = img
    im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
    tei = min(np.searchsorted(A["ev_steps"], fs[ti]), len(A["ev_steps"]) - 1)
    d.text((pad, 22), f"step {int(fs[ti]):,}   test acc {A['te_acc'][tei, s] * 100:.1f}%", fill=st["ink"], font=font)
    lab = "argmax prediction, colour = residue (cyclic)" if style == "nocturne" else "p(correct answer), dark = 1"
    d.text((pad, H + pad + 20), f"a ↓  b →   {lab}   {tag}", fill=st["muted"], font=font)
    return im


for style in ["nocturne", "plate"]:
    out = f"cache/frames_table_{style}"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    n = 0
    for ti in range(len(fs)):
        im = frame(ti, style)
        for _ in range(3):
            im.save(f"{out}/{n:05d}.png", compress_level=1); n += 1
    for _ in range(48):
        im.save(f"{out}/{n:05d}.png", compress_level=1); n += 1
    im.save(f"gallery/table_final_{tag}_{style}.png")
    frame(int(np.searchsorted(fs, 2000)), style).save(f"gallery/table_memorised_{tag}_{style}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", f"{out}/%05d.png", "-vf", "scale=1080:1080:flags=neighbor",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", f"gallery/table_{tag}_{style}.mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", f"{out}/%05d.png",
                    "-vf", "fps=12,scale=540:540:flags=neighbor,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=none",
                    f"gallery/table_{tag}_{style}.gif"], check=True)
    print(style, n, flush=True)
