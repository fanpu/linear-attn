"""Block-size sweep: the same NVFP4 recipe (E4M3 block scale x FP32 tensor scale) and MX recipe (E8M0 floor rule)
at block sizes 4..1024. Compute -> cache/blocksweep.npz ; render -> gallery/blocksweep_*.png/.mp4/.gif
   python blocksweep.py compute | render
"""
import os
import subprocess
import sys

import numpy as np
import torch

import formats as F
import styles as S
from weights import KINDS, SafeTensors, layer_name

SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
HERO = (26, "q_proj")


def compute():
    torch.set_num_threads(8)
    st = SafeTensors()
    names = [(f"L{i:02d}_{k}", layer_name(i, k)) for i in (0, 13, 26) for k in KINDS]
    rel = {"NVFP4": np.zeros((len(names), len(SIZES))), "MXFP4": np.zeros((len(names), len(SIZES)))}
    maps = {}
    for a, (tag, nm) in enumerate(names):
        W = st.tensor(nm)
        ms = float(W.pow(2).mean())
        for b, bs in enumerate(SIZES):
            for fmt, q in F.QUANTIZERS.items():
                r = q(W, block_size=bs)
                rel[fmt][a, b] = float((r["deq"] - W).pow(2).mean() / ms)
                if (int(tag[1:3]), tag[4:]) == HERO:
                    maps[f"{fmt}_{bs}"] = np.log2(r["scale"][:512].numpy()).astype(np.float32)
        print(tag, " ".join(f"{bs}:{rel['NVFP4'][a, b]:.4f}/{rel['MXFP4'][a, b]:.4f}" for b, bs in enumerate(SIZES)), flush=True)
    np.savez(os.path.join(S.CACHE, "blocksweep.npz"), tags=np.array([t for t, _ in names]), sizes=np.array(SIZES),
             rel_NVFP4=rel["NVFP4"], rel_MXFP4=rel["MXFP4"], **maps)


def curve_panel(d, h, w, cur, fg, mu):
    img = S.canvas(h, w, S.NIGHT)
    tags = d["tags"]
    lo, hi = np.log2(0.0045), np.log2(0.02)
    X = lambda b: int(60 + b / (len(SIZES) - 1) * (w - 120))
    Y = lambda v: int(h - 50 - (np.log2(v) - lo) / (hi - lo) * (h - 100))
    for fmt, col in (("NVFP4", np.array([0.55, 0.8, 1.0])), ("MXFP4", np.array([1.0, 0.55, 0.25]))):
        R = d[f"rel_{fmt}"]
        for a in range(len(tags)):
            c = col * (1.0 if "attn" in tags[a] or tags[a][4:] in ("q_proj", "k_proj", "v_proj", "o_proj") else 0.55)
            for b in range(len(SIZES) - 1):
                x0, y0, x1, y1 = X(b), Y(R[a, b]), X(b + 1), Y(R[a, b + 1])
                n = max(abs(x1 - x0), abs(y1 - y0)) + 1
                xs = np.linspace(x0, x1, n).astype(int)
                ys = np.linspace(y0, y1, n).astype(int)
                img[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)] = c
    img[:, X(cur) - 1:X(cur) + 2] = np.maximum(img[:, X(cur) - 1:X(cur) + 2], 0.35)
    items = [(X(b), h - 40, str(bs), 20, "Mono", mu, "ma") for b, bs in enumerate(SIZES)]
    for v in (0.005, 0.01, 0.015, 0.02):
        items.append((10, Y(v) - 10, f"{v:g}", 18, "Mono", mu))
    items.append((60, 8, "rel. MSE vs block size (log-log); blue NVFP4 recipe, orange MX recipe; bright = attention, dim = MLP; layers 0, 13, 26", 20, "Sans", fg))
    return img, items


def render():
    d = np.load(os.path.join(S.CACHE, "blocksweep.npz"))
    fr = os.path.join(S.CACHE, "frames_bs")
    os.makedirs(fr, exist_ok=True)
    fg, mu = (0.93, 0.91, 0.86), (0.55, 0.55, 0.6)
    ref = np.concatenate([d[f"NVFP4_{bs}"].ravel() for bs in SIZES])
    lo, hi = np.percentile(ref, 0.5), np.percentile(ref, 99.9)
    frames = []
    for b, bs in enumerate(SIZES):
        img = S.canvas(1080, 1920, S.NIGHT)
        items = [(60, 30, f"Block size {bs}", 60, "Bold", fg),
                 (60, 110, f"model.layers.26.self_attn.q_proj rows 0:512; block scale drawn over the weights it covers (log2, shared axis)", 24, "Sans", mu)]
        for j, fmt in enumerate(("NVFP4", "MXFP4")):
            m = np.repeat(d[f"{fmt}_{bs}"], bs, axis=1)[:, :1024]
            from PIL import Image
            n = np.clip((m - lo) / (hi - lo), 0, 1)
            rgb = S.to_rgb(n, S.cmap("magma"))
            x = 60 + j * 900
            small = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize((512, 256), Image.NEAREST)) / 255
            S.paste(img, S.up(small, 1), 200, x) if False else None
            big = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize((840, 262), Image.BOX)) / 255
            S.paste(img, big, 200, x)
            items.append((x, 470, f"{fmt} recipe", 26, "Mono", fg))
            items.append((x, 505, f"rel.MSE {d[f'rel_{fmt}'][list(d['tags']).index('L26_q_proj'), b]:.4f}", 26, "Mono", fg))
        panel, pit = curve_panel(d, 480, 1800, b, fg, mu)
        S.paste(img, panel, 560, 60)
        items += [(x0 + 60, y0 + 560, s, *rest) for (x0, y0, s, *rest) in pit]
        items.append((60, 1050, S.STACK + "  (scale maps box-downsampled 1024x512 -> 840x262, declared)", 18, "Sans", mu))
        img = S.text(img, items)
        path = os.path.join(fr, f"f{b:03d}.png")
        Image.fromarray((img * 255).astype(np.uint8)).save(path)
        frames.append(path)
    S.save(np.asarray(Image.open(frames[2])) / 255, "blocksweep_frame16.png")
    mp4 = os.path.join(S.GALLERY, "blocksweep.mp4")
    gif = os.path.join(S.GALLERY, "blocksweep.gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "0.8", "-i", f"{fr}/f%03d.png", "-vf", "fps=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "0.8", "-i", f"{fr}/f%03d.png", "-vf",
                    "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=none", gif], check=True)
    print("wrote", mp4, gif)


if __name__ == "__main__":
    {"compute": compute, "render": render}[sys.argv[1]]()
