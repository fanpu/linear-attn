"""Shared rendering helpers: fonts, grounds, inks, point splatting, LUTs, video writing."""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(HERE, "gallery")
os.makedirs(GAL, exist_ok=True)
EXT = (-1.45, 1.45)

URW = "/usr/share/fonts/opentype/urw-base35/"
FONTS = {
    "serif": URW + "P052-Roman.otf", "serif_it": URW + "P052-Italic.otf", "serif_b": URW + "P052-Bold.otf",
    "school": URW + "C059-Roman.otf", "school_it": URW + "C059-Italic.otf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}

INK = {  # declared aesthetic inks
    "paper": (242, 236, 222), "sepia_ink": (58, 40, 28), "iron_gall": (34, 30, 38),
    "night": (8, 8, 12), "bone": (228, 222, 205), "vermilion": (196, 64, 40), "prussian": (28, 56, 98),
}


def font(name, size):
    return ImageFont.truetype(FONTS[name], size)


def lut(name, n=1024):
    """RGB uint8 LUT [n,3] from a palettes.py scheme or matplotlib cmap name."""
    try:
        cm = P.as_cmap(name, n)
    except Exception:
        import matplotlib
        cm = matplotlib.colormaps[name]
    return (np.asarray(cm(np.linspace(0, 1, n)))[:, :3] * 255).astype(np.uint8)


def splat(X, W, H=None, ext=EXT, sigma=1.2, weights=None, ext_y=None):
    """Point cloud [M,2] -> blurred count image [H,W] (row 0 = top = max y)."""
    H = H or W
    ey = ext_y or ext
    ix = ((X[:, 0] - ext[0]) / (ext[1] - ext[0]) * W).astype(np.int64)
    iy = ((ey[1] - X[:, 1]) / (ey[1] - ey[0]) * H).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    c = np.bincount(iy[ok] * W + ix[ok], weights=None if weights is None else weights[ok], minlength=W * H)
    c = c.reshape(H, W).astype(np.float64)
    return gaussian_filter(c, sigma) if sigma > 0 else c


def tone(c, k):
    """log tone map with fixed knee k (declared): log1p(c/k)/log1p(cmax/k), cmax fixed by caller via k-scaled clip."""
    return np.log1p(c / k)


def apply_lut(v, L):
    v = np.clip(np.nan_to_num(v), 0, 1)
    return L[(v * (len(L) - 1)).astype(int)]


def mix(ground, ink, a):
    """Alpha-composite ink colour onto ground with coverage a [H,W] in [0,1]."""
    g = np.asarray(ground, float)
    i = np.asarray(ink, float)
    return (g * (1 - a[..., None]) + i * a[..., None]).astype(np.uint8)


def paper_texture(H, W, seed=0, amp=1.0):
    """Declared aesthetic: faint low-frequency paper mottling (no data)."""
    rng = np.random.default_rng(seed)
    t = gaussian_filter(rng.standard_normal((H, W)), 3) * 25 + gaussian_filter(rng.standard_normal((H, W)), 0.7) * 4
    return np.clip(np.asarray(INK["paper"], float)[None, None] + amp * t[..., None], 0, 255)


def save(img, name):
    p = os.path.join(GAL, name)
    Image.fromarray(np.asarray(img).astype(np.uint8)).save(p, optimize=True)
    print("saved", p)
    return p


def video(frame_dir, name, fps=30, gif_width=720, gif_fps=15):
    mp4 = os.path.join(GAL, name + ".mp4")
    gif = os.path.join(GAL, name + ".gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{frame_dir}/%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", mp4], check=True)
    vf = f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", vf, gif], check=True)
    print("video", mp4, os.path.getsize(mp4) >> 20, "MB; gif", os.path.getsize(gif) >> 20, "MB")
