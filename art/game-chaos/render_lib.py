"""Shared rendering helpers (no computation here: everything reads cache/)."""
import os
import subprocess
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/home/fzeng/ml/research/art/color-research')
try:
    import palettes as PAL
except Exception:  # pragma: no cover
    PAL = None

GAL = 'gallery'
os.makedirs(GAL, exist_ok=True)

PAPER = '#f3eee2'
INK = '#1d1d1f'
RISO_BLUE = '#0078bf'
RISO_PINK = '#ff48b0'
RISO_TEAL = '#00838a'
RISO_ORANGE = '#ff6c2f'

FONT_PATHS = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf']


def font(size, mono=False):
    p = FONT_PATHS[1 if mono else 0]
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()


def u8(rgb):
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def spectral(lam, ref=None, pairing='sd_spectral'):
    """Sohl-Dickstein split: lambda<0 purple->blue->green->pale, lambda>0 deep red->orange->pale,
    each side rank-normalised, dark seam at lambda = 0. Row 0 = bottom (flip for images)."""
    if PAL is not None:
        return PAL.render_split(lam, pairing, near_boundary='small', ref=ref)
    from gamelib import spectral_split
    return mpl.colormaps['Spectral'](spectral_split(lam, ref))[..., :3]


def dark_diverging(lam, scale_neg=None, scale_pos=None, cmap='cmc.berlin'):
    """Declared: cmcrameri 'berlin' (dark centre) with an asinh stretch, centred at lambda = 0;
    negative side scaled by its 99th percentile, positive side by its 99th percentile."""
    import cmcrameri.cm  # noqa
    lam = np.asarray(lam, float)
    sn = scale_neg or np.percentile(-lam[lam < 0], 99) if np.any(lam < 0) else 1
    sp = scale_pos or np.percentile(lam[lam > 0], 99) if np.any(lam > 0) else 1
    v = np.where(lam < 0, -np.arcsinh(4 * np.minimum(-lam / sn, 1)) / np.arcsinh(4),
                 np.arcsinh(4 * np.minimum(lam / sp, 1)) / np.arcsinh(4))
    return mpl.colormaps[cmap]((v + 1) / 2)[..., :3]


def save_png(rgb_or_u8, path):
    a = rgb_or_u8 if rgb_or_u8.dtype == np.uint8 else u8(rgb_or_u8)
    Image.fromarray(a).save(path, optimize=True)
    return path


def flip(a):
    return a[::-1].copy()


def to_mp4_gif(frame_dir, stem, fps=24, gif_width=720, gif_fps=15):
    mp4 = f'{GAL}/{stem}.mp4'; gif = f'{GAL}/{stem}.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{frame_dir}/%05d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow', mp4], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4, '-vf',
                    f'fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=200[p];[b][p]paletteuse=dither=sierra2_4a',
                    gif], check=True)
    for w in (600, 480, 400):
        if os.path.getsize(gif) < 15e6:
            break
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4, '-vf',
                        f'fps=12,scale={w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=sierra2_4a',
                        gif], check=True)
    print('wrote', mp4, os.path.getsize(mp4) // 1000, 'kB;', gif, os.path.getsize(gif) // 1000, 'kB')
