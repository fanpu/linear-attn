"""Shared rendering helpers (no computation of measured quantities happens here)."""
import os, sys, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(HERE, "gallery")
CACHE = os.path.join(HERE, "cache")
os.makedirs(GAL, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "savefig.facecolor": "none", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42})
PAPER = "#f4efe3"
INK = "#1d1b19"


def split_rgb(sign, rank_val, pairing="sd_spectral", near_boundary="large", pastel=0.75, ref=None, seam="dark"):
    """sign: bool array True = chaotic (pos side). rank_val: positive magnitude used for the within-side rank.
    near_boundary='large' means large rank_val sits at the boundary (e.g. depth scale xi_c)."""
    x = np.where(sign, 1.0, -1.0) * np.maximum(np.asarray(rank_val, float), 1e-300)
    x[~np.isfinite(rank_val)] = np.nan
    return P.render_split(x, pairing, near_boundary=near_boundary, pastel=pastel, seam=seam, ref=ref)


def to_uint8(rgb):
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_rgb(rgb, name):
    from PIL import Image
    path = os.path.join(GAL, name)
    Image.fromarray(to_uint8(rgb) if rgb.dtype != np.uint8 else rgb).save(path, optimize=True)
    return path


def fig_px(w, h, dpi=200, bg=PAPER):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg)
    return fig


def img_axes(fig, x0, y0, w, h, W, H):
    """Axes in pixel coords of a W x H figure (origin top-left)."""
    ax = fig.add_axes([x0 / W, 1 - (y0 + h) / H, w / W, h / H])
    ax.set_axis_off()
    return ax


def savefig(fig, name, dpi=200):
    path = os.path.join(GAL, name)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def ffmpeg_mp4(frame_glob_dir, out, fps=24):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frame_glob_dir, "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", out], check=True)


def ffmpeg_gif(frame_dir, out, fps=12, width=540, step=2):
    pal = os.path.join(frame_dir, "palette.png")
    inp = ["-framerate", str(24), "-i", os.path.join(frame_dir, "f%05d.png")]
    vf = f"select='not(mod(n\\,{step}))',setpts=N/{fps}/TB,scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inp, "-vf", vf + ",palettegen=stats_mode=diff", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inp, "-i", pal, "-lavfi",
                    vf + " [x]; [x][1:v] paletteuse=dither=sierra2_4a", "-r", str(fps), out], check=True)
