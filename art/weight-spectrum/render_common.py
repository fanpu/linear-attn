"""Rendering helpers (styles, saving, previews). No measurement here."""
import os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, '/home/fzeng/ml/research/art/color-research')
import palettes as P  # noqa: E402  (registers art.* cmaps)
from common import GALLERY, HERE  # noqa: E402

os.makedirs(GALLERY, exist_ok=True)
PREV = os.path.join(HERE, 'cache', 'preview'); os.makedirs(PREV, exist_ok=True)

PAPER = '#f3eee2'
INK = '#1b1a17'
RISO = P.RISO
RISO_PAPER = P.RISO_PAPER
SERIF = 'Nimbus Roman'
SANS = 'Nimbus Sans'
MONO = 'Nimbus Mono PS'

STYLES = {
    'night': dict(bg='#07070a', fg='#eeeae0', dim='#6d6a64', accent='#f2b134'),
    'ink': dict(bg=PAPER, fg=INK, dim='#8a847a', accent='#b0322a'),
    'riso': dict(bg=RISO_PAPER, fg=RISO['federal_blue'], dim='#a9a193', accent=RISO['fluo_pink']),
    'gold': dict(bg='#0b0a08', fg='#e9d49a', dim='#6f6346', accent='#c9a23f'),
}


def set_rc(font=SANS):
    plt.rcParams.update({'font.family': font, 'mathtext.fontset': 'dejavusans', 'savefig.dpi': 100,
                         'axes.linewidth': 0.6, 'pdf.fonttype': 42})


def save(fig, name, dpi=100, preview=True, **kw):
    path = os.path.join(GALLERY, name)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), **kw)
    plt.close(fig)
    if preview:
        make_preview(path)
    return path


def make_preview(path, w=900):
    im = Image.open(path)
    s = w / max(im.size)
    if s < 1:
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
    p = os.path.join(PREV, os.path.basename(path))
    im.convert('RGB').save(p)
    return p


def frames_to_video(frame_dir, out_base, fps=30, gif_width=720, gif_fps=15):
    """frame_dir/%05d.png -> gallery/out_base.mp4 (H.264 yuv420p) + .gif (palettegen)."""
    mp4 = os.path.join(GALLERY, out_base + '.mp4'); gif = os.path.join(GALLERY, out_base + '.gif')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{frame_dir}/%05d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', mp4], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{frame_dir}/%05d.png',
                    '-vf', f'fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];'
                           f'[b][p]paletteuse=dither=sierra2_4a', gif], check=True)
    return mp4, gif


def halftone_noise(shape, seed=0, amp=0.06):
    """Paper grain for riso plates (declared texture)."""
    rng = np.random.default_rng(seed)
    return 1 - amp * rng.random(shape)


def riso_composite(layers, shape, paper=RISO_PAPER, offsets=None, grain=0.05, seed=0):
    """layers: list of (coverage HxW in [0,1], ink hex). Optional integer (dy, dx) misregistration per ink."""
    covs = []
    for i, (c, ink) in enumerate(layers):
        c = np.asarray(c, float)
        if offsets is not None:
            c = np.roll(c, offsets[i], axis=(0, 1))
        rng = np.random.default_rng(seed + i)
        c = np.clip(c * (1 - grain * rng.random(c.shape)), 0, 1)
        covs.append(c)
    img = P.overprint(covs, [ink for _, ink in layers], paper=paper)
    return np.clip(img * halftone_noise(shape + (1,)[:0] if False else shape, seed=99, amp=0.03)[..., None], 0, 1)


def fig_to_array(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a
