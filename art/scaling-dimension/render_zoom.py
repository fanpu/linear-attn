"""Render the zoom film from cache/zoom/*.npz (points rasterised directly with numpy, no matplotlib).

    python render_zoom.py --data cache/zoom/zoom.npz --style dark --out gallery/zoom_dark [--frames 0,300,600] [--size 1080]
"""
import argparse, os, subprocess, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--data", default="cache/zoom/zoom.npz")
p.add_argument("--style", default="dark")
p.add_argument("--out", default="gallery/zoom_dark")
p.add_argument("--frames", default="")
p.add_argument("--size", type=int, default=1080)
p.add_argument("--kappa", type=float, default=0.5)
p.add_argument("--slab", type=float, default=0.9)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--fps", type=int, default=30)
a = p.parse_args()

Z = np.load(a.data)
H, u, rhos, widths, Ns, tests = Z["H"], Z["u"], Z["rhos"], list(Z["widths"]), list(Z["Ns"]), list(Z["tests"])
F, S, K = H.shape
W = a.size
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SERIF = "/home/fzeng/ml/research/art/.venv/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerif-Italic.ttf"

if a.style == "dark":
    bg = np.array(to_rgb("#07080c")); text = (214, 206, 186)
    cols = [np.array(to_rgb("#f1ead8"))] + [np.array(plt.get_cmap("magma")(v)[:3]) for v in np.linspace(0.3, 0.85, S - 1)]
    mode = "add"
elif a.style == "paper":
    bg = np.array(to_rgb("#f3eddf")); text = (28, 38, 51)
    inks = ["#1c2633", "#b8412c", "#c98a2a", "#3f7f6e", "#3a5ea8"]
    cols = [np.array(to_rgb(c)) for c in inks[:S]]
    mode = "ink"
elif a.style == "riso":
    bg = np.array(to_rgb("#f4f0e6")); text = (50, 85, 164)
    # teacher in Federal Blue, all students in Fluorescent Pink (two spot inks)
    cols = [np.array(to_rgb("#3255a4"))] + [np.array(to_rgb("#ff48b0"))] * (S - 1)
    mode = "ink"


def project(hh, phi, psi=np.deg2rad(58)):
    x = np.cos(phi) * u[:, 0] - np.sin(phi) * u[:, 1]
    y = np.sin(phi) * u[:, 0] + np.cos(phi) * u[:, 1]
    sy = -a.kappa * hh * np.sin(psi) + y * np.cos(psi)   # screen y grows downward
    return x, sy


def frame(fi):
    phi = np.deg2rad(20 + 50 * fi / max(F - 1, 1))       # declared: slow camera orbit, 50 degrees total
    dens = np.zeros((S, W * W))
    scale = W / 3.1
    for s in range(S):
        x, y = project(H[fi, s].astype(np.float64), phi)
        px = (x * scale + W / 2).astype(int); py = (y * scale + W / 2 + 0.02 * W).astype(int)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < W) & (np.abs(a.kappa * H[fi, s]) <= a.slab)  # declared: clip to a height slab
        dens[s] = np.bincount(py[ok] * W + px[ok], minlength=W * W)
    dens = dens.reshape(S, W, W)
    # per-pixel coverage: 1 - exp(-count / c) (declared tone curve; K points over the window)
    c = 0.9 * K / (W * W / 4.5)
    cov = 1 - np.exp(-dens / c)
    if mode == "add":
        img = bg[None, None] + sum(cov[s][..., None] * cols[s][None, None] * (0.95 if s == 0 else 1.15) for s in range(S))
    else:
        img = np.ones((W, W, 3)) * bg
        for s in range(S):
            ink = cols[s]
            img *= 1 - cov[s][..., None] * 0.85 * (1 - ink[None, None])
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    f1 = ImageFont.truetype(FONT, int(W * 0.019)); f2 = ImageFont.truetype(SERIF, int(W * 0.03))
    zoom = rhos[0] / rhos[fi]
    m = int(W * 0.04)
    dr.text((m, m), "four over d", font=f2, fill=text)
    dr.text((m, m + int(W * 0.045)), f"zoom  ×{zoom:8.1f}", font=f1, fill=text)
    dr.text((m, m + int(W * 0.072)), f"window half-width  {rhos[fi]:.2e}", font=f1, fill=text)
    # legend
    y0 = W - m - int(W * 0.028) * S
    for s in range(S):
        col = tuple(int(v * 255) for v in cols[s])
        lab = "teacher  T(z), d = 2" if s == 0 else f"student  N = {Ns[s-1]:>5d}   L = {tests[s-1]:.1e}"
        yy = y0 + s * int(W * 0.028)
        dr.rectangle([m, yy + 4, m + 14, yy + 18], fill=col)
        dr.text((m + 26, yy), lab, font=f1, fill=text)
    dr.text((W - m, W - m - int(W * 0.028)), "same axes scale in z and height", font=f1, fill=text, anchor="ra")
    return im


def save_frame(fi):
    frame(fi).save(f"{a.out}_frames/{fi:05d}.png")


if __name__ == "__main__":
    if a.frames:
        for fi in [int(v) for v in a.frames.split(",")]:
            frame(fi).save(f"{a.out}_f{fi:04d}.png")
    else:
        os.makedirs(a.out + "_frames", exist_ok=True)
        os.environ["OMP_NUM_THREADS"] = "1"
        with Pool(a.workers) as pool:
            pool.map(save_frame, range(F), chunksize=8)
        fr = a.out + "_frames/%05d.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", fr, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", a.out + ".mp4"], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", fr, "-vf",
                        "fps=15,scale=540:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
                        a.out + ".gif"], check=True)
        shutil.rmtree(a.out + "_frames")
        print("wrote", a.out + ".mp4", os.path.getsize(a.out + ".mp4") >> 20, "MB;", a.out + ".gif", os.path.getsize(a.out + ".gif") >> 20, "MB")
