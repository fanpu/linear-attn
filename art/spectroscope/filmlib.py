"""Shared pieces for the spectroscope films (CPU only).

Colour conventions (declared, see README "Films"):
  * integrated colour = spectral.spectra_to_rgb of the measured power P (gamma 1 unless a file says
    otherwise): the same faithful colour as the stills, at maximum displayable brightness.
  * dispersed spectrum: wavelength runs 400 -> 700 nm left to right. Each wavelength is drawn in its
    own hue (the monochromatic colour, desaturated into sRGB and normalised to equal peak brightness,
    so hue labels position only) and with brightness EXPOSE(P) = 1 - exp(-P / 2.5): linear at low
    power, a flat (white-light) spectrum sits at 0.33 linear, and strong peaks roll off instead of
    clipping. The rainbow hue is a label for wavelength position; the data is the brightness.
  * frames are composed in linear light and encoded to sRGB, then to H.264 BT.709.
"""
import json, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import spectral as S

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'cache', 'runs')
MET = os.path.join(HERE, 'cache', 'metrics')
GAL = os.path.join(HERE, 'gallery')
FILMCACHE = os.path.join(HERE, 'cache', 'films')
os.makedirs(FILMCACHE, exist_ok=True)

L, H = 6, 6
ROWS = []
for i in range(L):
    for t in 'qkvo':
        ROWS.append(f'L{i}.{t}')
        ROWS += [f'L{i}.{t}.h{h}' for h in range(H)]
    ROWS += [f'L{i}.up', f'L{i}.down']
ROWS += ['emb', 'pos', 'head']
RI = {r: i for i, r in enumerate(ROWS)}
WHOLE_HIDDEN = [RI[r] for r in ROWS if r.startswith('L') and '.h' not in r]

EXPOSE_K = 2.5


def expose(P, k=EXPOSE_K):
    return 1.0 - np.exp(-np.asarray(P) / k)


def srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def rainbow(n=S.NW):
    """Equal-peak-brightness monochromatic hues, linear RGB, (n, 3), 400 -> 700 nm."""
    wl = np.linspace(400, 700, n)
    xyz = S.cmf(wl)
    rgb = xyz @ S.XYZ2RGB.T
    rgb = rgb - np.minimum(rgb.min(-1, keepdims=True), 0)
    return rgb / rgb.max(-1, keepdims=True)


def load_run(run, gamma=1.0):
    c = np.load(os.path.join(MET, f'{run}_shuffle.npz'))
    P = c['P1'].astype(np.float32)
    if gamma != 1.0:
        P = P ** gamma
        P = P / np.maximum(P.mean(-1, keepdims=True), 1e-30)
    d = np.load(os.path.join(RUNS, run + '.npz'))
    meta = json.load(open(os.path.join(RUNS, run + '.json')))
    steps = d['steps']
    tl = d['train_loss'] if 'train_loss' in d.files else np.full(len(steps), np.nan)
    return dict(P=P, steps=steps, train_loss=tl, meta=meta,
                lin=srgb_to_lin(S.spectra_to_rgb(P.astype(float))).astype(np.float32),
                shift=(S.centroid_nm(P.astype(float)) - 550).astype(np.float32))


def font(sz, mono=False):
    for p in (('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',) if mono else ()) + (
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def bloom(lin, strength=0.22, radius=18, down=4):
    """Declared glow: a blurred copy of the frame added back (light scattering in the room)."""
    h, w = lin.shape[:2]
    small = lin[:h - h % down, :w - w % down].reshape(h // down, down, w // down, down, 3).mean((1, 3))
    im = Image.fromarray((np.clip(small, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius / down))
    b = np.asarray(im, np.float32) / 255
    b = np.repeat(np.repeat(b, down, 0), down, 1)
    out = lin.copy()
    out[:b.shape[0], :b.shape[1]] += strength * b
    return out


class Encoder:
    def __init__(self, path, w, h, fps=30, crf=20):
        self.p = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
             '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf),
             '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
             '-movflags', '+faststart', path], stdin=subprocess.PIPE)

    def write(self, rgb8):
        self.p.stdin.write(np.ascontiguousarray(rgb8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def to8(lin):
    return (lin_to_srgb(lin) * 255 + 0.5).astype(np.uint8)
