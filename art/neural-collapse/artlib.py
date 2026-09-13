"""Rendering helpers: splatting, tone mapping, exact-pixel matplotlib canvases, previews."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(HERE, "gallery")
os.makedirs(GAL, exist_ok=True)
PAPER = "#f4efe3"
INK = "#1b1b1f"


def splat(xy, extent, res, weights=None):
    """Point counts on a res x res grid; extent=(x0,x1,y0,y1); row 0 is the TOP (y1)."""
    x0, x1, y0, y1 = extent
    i = ((y1 - xy[:, 1]) / (y1 - y0) * res).astype(int)
    j = ((xy[:, 0] - x0) / (x1 - x0) * res).astype(int)
    ok = (i >= 0) & (i < res) & (j >= 0) & (j < res)
    img = np.zeros((res, res))
    np.add.at(img, (i[ok], j[ok]), 1.0 if weights is None else weights[ok])
    return img


def glow(counts, sigmas=(0.8, 3.0, 12.0), amps=(1.0, 0.35, 0.12)):
    return sum(a * gaussian_filter(counts, s) for s, a in zip(sigmas, amps))


def tonemap(x, gain):
    return 1.0 - np.exp(-gain * x)


def canvas(w_px, h_px, bg, dpi=200):
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, facecolor=bg)
    return fig


def panel(fig, rect, extent, bg=None):
    ax = fig.add_axes(rect)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()
    if bg is not None:
        ax.set_facecolor(bg)
    return ax


def save(fig, name, preview=True):
    path = os.path.join(GAL, name)
    fig.savefig(path, dpi=fig.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    if preview:
        prev(path)
    return path


def prev(path, maxpx=900):
    os.makedirs(os.path.join(HERE, "scratch"), exist_ok=True)
    im = Image.open(path)
    im.thumbnail((maxpx, maxpx), Image.LANCZOS)
    out = os.path.join(HERE, "scratch", "prev_" + os.path.basename(path))
    im.convert("RGB").save(out)
    return out


def rgb(c):
    return np.array(to_rgb(c))
