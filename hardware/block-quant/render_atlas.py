"""Layer atlas: every linear matrix of every layer, from cache/atlas_<model>.npz.
   python render_atlas.py
"""
import os
import subprocess

import numpy as np

import styles as S

KIND_LAB = ["q", "k", "v", "o", "gate", "up", "down"]


def thumbs_plate(model="Qwen3-0.6B", style="night", key="NVFP4_scale", k=2):
    a = np.load(f"{S.CACHE}/atlas_{model}.npz")
    T = a[f"thumb_{key}"]            # (L, 7, 96, 96)
    L = T.shape[0]
    th = T.shape[2] * k
    gap = 10
    side, top = 150, 250
    Wd = side * 2 + 7 * th + 6 * gap
    Ht = top + L * (th + gap) + 200
    bg = S.NIGHT if style == "night" else S.PAPER
    img = S.canvas(Ht, Wd, bg)
    fg, mu = ((0.93, 0.91, 0.86), (0.55, 0.55, 0.6)) if style == "night" else ((0.1, 0.1, 0.12), (0.4, 0.38, 0.35))
    items = [(side, 40, "Layer Atlas" if style == "night" else "LAYER ATLAS", 64, "Bold", fg if style == "night" else S.RISO_BLUE),
             (side, 125, f"{model}: {'NVFP4 block scale (log2)' if 'scale' in key else 'mean |Q(W)-W|' if 'err' in key else 'mean |W|'} of every linear matrix, "
                         f"area-averaged to {T.shape[2]}x{T.shape[2]}", 24, "Sans", mu),
             (side, 160, "each tile centred on its own median, range -1.5..+2 octaves (declared); rows = layers 0 (top) .. " + str(L - 1), 24, "Sans", mu)]
    for j, lab in enumerate(KIND_LAB):
        items.append((side + j * (th + gap) + th // 2, top - 40, lab, 28, "Mono", fg, "ma"))
    for i in range(L):
        items.append((side - 20, top + i * (th + gap) + th // 2 - 12, f"L{i}", 22, "Mono", mu, "ra"))
        for j in range(7):
            t = T[i, j].astype(np.float64)
            if "scale" in key:
                n = np.clip((t - np.median(t) + 1.5) / 3.5, 0, 1)
            else:
                t = np.log2(t + 1e-12)
                n = np.clip((t - np.median(t) + 1.5) / 3.5, 0, 1)
            tile = S.up(n, k)
            rgb = S.to_rgb(tile, S.cmap("magma")) if style == "night" else S.ink(tile ** 1.3, S.RISO_BLUE)
            S.paste(img, rgb, top + i * (th + gap), side + j * (th + gap))
    items.append((side, Ht - 60, S.STACK.replace("Qwen3-0.6B", model), 20, "Sans", mu))
    if style != "night":
        img *= S.grain(img.shape[:2], 0.03, seed=4)
    img = S.text(img, items)
    return S.save(img, f"atlas_{key}_{style}_{model}.png")


METRICS = [("kurtosis", "kurtosis of W", True),
           ("NVFP4_log2scale_std", "std of log2 NVFP4 block scale", False),
           ("colmax_over_median", "column max / median", True),
           ("rowmax_over_median", "row max / median", True),
           ("NVFP4_relmse", "NVFP4 rel. MSE (RTN)", False),
           ("MXFP4_relmse", "MXFP4 rel. MSE (RTN)", False),
           ("MXFP4_blocks_clipped", "MXFP4 blocks with a clipped max", False),
           ("NVFP4_relmse_sr", "NVFP4 rel. MSE (stochastic)", False)]


def metrics_plate(models=("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B")):
    cell = 26
    side, top = 120, 300
    colw = 7 * cell
    gapm = 70
    blockw = len(models) * colw + (len(models) - 1) * 24
    Lmax = max(int(np.load(f"{S.CACHE}/atlas_{m}.npz")["n_layers"]) for m in models)
    ncol = 4
    nrow = 2
    Wd = side * 2 + ncol * blockw + (ncol - 1) * gapm
    Ht = top + nrow * (Lmax * cell + 260) + 120
    img = S.canvas(Ht, Wd, S.PAPER)
    fg, mu = (0.1, 0.1, 0.12), (0.4, 0.38, 0.35)
    items = [(side, 40, "ATLAS OF STATISTICS", 64, "Bold", fg),
             (side, 125, "rows = layers (0 at top), columns = q k v o gate up down; three model sizes per panel; colour = rank within the panel (cividis, declared) with the actual range printed", 24, "Sans", mu),
             (side, 160, "Attention vs MLP differ in the SHAPE statistics (kurtosis, scale spread, outlier ratios) but NOT in relative 4-bit error, which the micro-block scale pins near a constant (rank colour exaggerates: NVFP4 rel.MSE spans only 0.0088-0.0091).", 24, "Sans", mu)]
    data = {m: np.load(f"{S.CACHE}/atlas_{m}.npz") for m in models}
    cm = S.cmap("cividis")
    for mi, (key, lab, logv) in enumerate(METRICS):
        r, c = divmod(mi, ncol)
        x0 = side + c * (blockw + gapm)
        y0 = top + r * (Lmax * cell + 260)
        allv = np.concatenate([data[m][key] for m in models])
        vv = np.log(allv) if logv else allv
        items.append((x0, y0 - 80, lab, 28, "Bold", fg))
        items.append((x0, y0 - 44, f"range {allv.min():.4g} .. {allv.max():.4g}", 20, "Mono", mu))
        for k, m in enumerate(models):
            a = data[m]
            L = int(a["n_layers"])
            v = a[key].reshape(L, 7)
            vt = np.log(v) if logv else v
            rank = np.searchsorted(np.sort(vv), vt) / len(vv)
            xm = x0 + k * (colw + 24)
            for i in range(L):
                for j in range(7):
                    img[y0 + i * cell + 1: y0 + (i + 1) * cell - 1, xm + j * cell + 1: xm + (j + 1) * cell - 1] = cm(rank[i, j])[:3]
            items.append((xm + colw // 2, y0 + L * cell + 10, m.replace("Qwen3-", ""), 20, "Mono", mu, "ma"))
    items.append((side, Ht - 60, "Qwen3-0.6B / 1.7B / 4B bf16 weights, formats computed exactly (CPU float64), torch 2.14.0+cu130, GB10 host, 2026-09-13", 20, "Sans", mu))
    img = S.text(img, items)
    return S.save(img, "atlas_metrics_plate.png")


def depth_animation(model="Qwen3-0.6B", k=2):
    """One frame per layer: the 7 NVFP4 scale thumbnails of that layer, large. MP4 + GIF."""
    a = np.load(f"{S.CACHE}/atlas_{model}.npz")
    T = a["thumb_NVFP4_scale"]
    L = T.shape[0]
    fr_dir = os.path.join(S.CACHE, "frames_depth")
    os.makedirs(fr_dir, exist_ok=True)
    th = 240
    for i in range(L):
        img = S.canvas(1080, 1920, S.NIGHT)
        items = [(80, 60, f"Layer {i:2d} / {L - 1}", 64, "Bold", (0.93, 0.91, 0.86)),
                 (80, 150, f"{model}: NVFP4 block scales (log2, each tile centred on its median, -1.5..+2 octaves), area-averaged", 26, "Sans", (0.55, 0.55, 0.6))]
        for j in range(7):
            t = T[i, j].astype(np.float64)
            n = np.clip((t - np.median(t) + 1.5) / 3.5, 0, 1)
            from PIL import Image
            big = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((th, th), Image.NEAREST)) / 255
            x = 80 + j * (th + 20)
            S.paste(img, S.to_rgb(big, S.cmap("magma")), 360, x)
            items.append((x + th // 2, 320, KIND_LAB[j], 34, "Mono", (0.93, 0.91, 0.86), "ma"))
            items.append((x + th // 2, 360 + th + 20, f"kurt {a['kurtosis'][i * 7 + j]:.1f}", 24, "Mono", (0.55, 0.55, 0.6), "ma"))
            items.append((x + th // 2, 360 + th + 54, f"sd {a['NVFP4_log2scale_std'][i * 7 + j]:.2f} oct", 24, "Mono", (0.55, 0.55, 0.6), "ma"))
        # depth bar
        S.hline(img, 900, 80, 1840, np.array([0.3, 0.3, 0.35]), 4)
        xb = 80 + int(i / (L - 1) * 1760)
        img[880:920, xb - 4:xb + 4] = (1.0, 0.55, 0.25)
        items.append((80, 960, S.STACK, 22, "Sans", (0.55, 0.55, 0.6)))
        img = S.text(img, items)
        from PIL import Image
        Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(fr_dir, f"f{i:03d}.png"))
    mp4 = os.path.join(S.GALLERY, f"atlas_depth_{model}.mp4")
    gif = os.path.join(S.GALLERY, f"atlas_depth_{model}.gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1.5", "-i", f"{fr_dir}/f%03d.png", "-vf", "fps=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1.5", "-i", f"{fr_dir}/f%03d.png", "-vf",
                    "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=none",
                    gif], check=True)
    print("wrote", mp4, gif)


if __name__ == "__main__":
    for st in ("night", "riso"):
        thumbs_plate("Qwen3-0.6B", st, "NVFP4_scale")
    thumbs_plate("Qwen3-0.6B", "night", "absW")
    metrics_plate()
    depth_animation()
