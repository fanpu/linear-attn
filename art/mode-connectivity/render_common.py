"""Shared rendering helpers for One Basin (no measured quantity is computed here)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb

sys.path.insert(0, '/home/fzeng/ml/research/art/color-research')
import palettes as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(HERE, 'gallery')
CACHE = os.path.join(HERE, 'cache')
PREV = os.path.join(GAL, 'preview')
os.makedirs(PREV, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Serif', 'axes.linewidth': 0.6, 'pdf.fonttype': 42,
                     'mathtext.fontset': 'dejavuserif'})
PAPER = '#f2ecdf'
INK = '#1f1c1a'
NIGHT = '#0b0b10'
# 19th-century geological-map wash colours (declared categorical choice, one per class)
GEO = ['#c9a66b', '#8fae8b', '#d98c6a', '#a9b8c9', '#e2c98f', '#b5889a', '#7f9c9f', '#d9b3a0', '#9aa66a', '#c7b9d6']
RISO_BLUE, RISO_PINK, RISO_YEL = '#0078bf', '#ff48b0', '#ffe800'
RISO_PAPER = '#f4efe3'


def load(name):
    return dict(np.load(os.path.join(CACHE, name), allow_pickle=True))


def fig_px(W, H, dpi=200, bg=PAPER):
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg)
    return fig


def ax_px(fig, x0, y0, w, h, W, H, off=True):
    ax = fig.add_axes([x0 / W, 1 - (y0 + h) / H, w / W, h / H])
    if off:
        ax.set_axis_off()
    return ax


def save(fig, name, dpi=200, preview=900):
    path = os.path.join(GAL, name)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    make_preview(path, preview)
    return path


def make_preview(path, size=900):
    from PIL import Image
    im = Image.open(path)
    im.thumbnail((size, size), Image.LANCZOS)
    im.convert('RGB').save(os.path.join(PREV, os.path.basename(path).rsplit('.', 1)[0] + '.png'))


def save_rgb(rgb, name, preview=900):
    from PIL import Image
    arr = rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    path = os.path.join(GAL, name)
    Image.fromarray(arr).save(path, optimize=True)
    make_preview(path, preview)
    return path


def fig_to_array(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def encode_video(frames_dir, name, fps=30, gif_width=720, gif_fps=15):
    """frames_dir/%05d.png -> gallery/name.mp4 (H.264 yuv420p) and gallery/name.gif (palettegen)."""
    import subprocess
    mp4 = os.path.join(GAL, name + '.mp4')
    gif = os.path.join(GAL, name + '.gif')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{frames_dir}/%05d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow',
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', mp4], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{frames_dir}/%05d.png',
                    '-vf', f'fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];'
                           f'[b][p]paletteuse=dither=sierra2_4a', gif], check=True)
    return mp4, gif
