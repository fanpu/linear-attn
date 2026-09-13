"""Rendering helpers: level-set ink from sampled fields, palettes, fonts, video writing."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import subprocess, os

FONT_DIR = "/usr/share/fonts/opentype/urw-base35/"
PAPER = np.array([244, 239, 228]) / 255      # warm plotter paper
INK = np.array([27, 27, 34]) / 255           # near-black pigment ink
RISO_PINK = np.array([255, 72, 176]) / 255   # Riso Fluorescent Pink
RISO_BLUE = np.array([0, 120, 191]) / 255    # Riso Blue
RISO_TEAL = np.array([0, 131, 138]) / 255
RISO_YELLOW = np.array([255, 232, 0]) / 255
DARK = np.array([12, 12, 16]) / 255


def font(size, kind="serif"):
    f = {"serif": "C059-Roman.otf", "italic": "C059-Italic.otf", "bold": "C059-Bold.otf",
         "mono": "NimbusMonoPS-Regular.otf", "sans": "NimbusSans-Regular.otf"}[kind]
    return ImageFont.truetype(FONT_DIR + f, size)


def boundary(f, level):
    """Boolean mask of samples adjacent (4-neighbour) to a sign change of f - level."""
    s = f > level
    e = np.zeros_like(s)
    dv = s[1:, :] != s[:-1, :]
    dh = s[:, 1:] != s[:, :-1]
    e[1:, :] |= dv; e[:-1, :] |= dv
    e[:, 1:] |= dh; e[:, :-1] |= dh
    return e


def ink_coverage(f, levels, factor, weight=1, weights=None):
    """Anti-aliased ink coverage in [0,1] at resolution f.shape // factor.
    weight: dilation radius (in samples) of the 1-sample boundary before box-downsampling."""
    if np.isscalar(levels):
        levels = [levels]
    cov = np.zeros(f.shape, dtype=np.float32)
    for i, lv in enumerate(levels):
        w = weight if weights is None else weights[i]
        e = boundary(f, lv)
        if w > 1:
            e = ndimage.binary_dilation(e, structure=np.ones((w, w), bool))
        cov = np.maximum(cov, e.astype(np.float32))
    return downsample(cov, factor)


def downsample(a, k):
    if k == 1:
        return a
    H, W = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    a = a[:H, :W]
    return a.reshape(H // k, k, W // k, k, *a.shape[2:]).mean((1, 3))


def mix(bg, fg, alpha):
    """bg, fg: rgb (3,) or [H,W,3]; alpha [H,W]."""
    return bg * (1 - alpha[..., None]) + fg * alpha[..., None]


def multiply_ink(paper, inks_alphas):
    """Subtractive (multiply) layering of spot inks."""
    out = np.broadcast_to(paper, inks_alphas[0][1].shape + (3,)).copy()
    for ink, a in inks_alphas:
        out = out * (1 - a[..., None] * (1 - ink))
    return out


def to_img(rgb):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def grain(shape, seed=0, amp=0.03):
    """Declared aesthetic: faint paper grain."""
    rng = np.random.default_rng(seed)
    g = ndimage.gaussian_filter(rng.standard_normal(shape), 1.2)
    return 1 + amp * g / g.std()


def write_video(frames_dir, out_mp4, fps=30, gif=None, gif_width=540, gif_fps=15):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    f"{frames_dir}/%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                    "-preset", "slow", out_mp4], check=True)
    if gif:
        pal = frames_dir + "/palette.png"
        vf = f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out_mp4, "-vf",
                        vf + ",palettegen=max_colors=128:stats_mode=diff", pal], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out_mp4, "-i", pal, "-lavfi",
                        vf + " [x]; [x][1:v] paletteuse=dither=sierra2_4a", gif], check=True)


def spectral_split(f, level, mask=None, mode="seam", buffer=0.25):
    """Sohl-Dickstein 'Spectral' split colouring of a signed quantity f - level (declared aesthetic).
    Each side is rank (CDF) normalised separately.
    mode 'seam'  (user-requested): dark ends meet at the level set. f<level: pale yellow (far) ->
                 green -> blue -> purple #5e4fa2 (at the boundary); f>level: pale yellow (far) ->
                 orange -> deep red #9e0142 (at the boundary).
    mode 'colab' (exactly cdf_img in github.com/Sohl-Dickstein/fractal, readout='loss'): negatives by
                 rank to [-1,-buffer], non-negatives to [buffer,1], y = -v, Spectral on [-1,1]; dark
                 ends at the extremes, pastel jump at the boundary."""
    import matplotlib
    cmap = matplotlib.colormaps["Spectral"]
    x = np.asarray(f, dtype=np.float64) - level
    sel = np.ones(x.shape, bool) if mask is None else mask
    vals = x[sel]
    lo = np.sort(vals[vals < 0]); hi = np.sort(vals[vals >= 0])
    t = np.full(x.shape, 0.5)
    neg = sel & (x < 0); pos = sel & (x >= 0)
    q_lo = np.searchsorted(lo, x[neg], side="right") / max(len(lo), 1)     # 0 far below -> 1 at boundary
    q_hi = np.searchsorted(hi, x[pos], side="left") / max(len(hi), 1)      # 0 at boundary -> 1 far above
    if mode == "seam":
        t[neg] = 0.5 + 0.5 * q_lo
        t[pos] = 0.5 * q_hi
    else:
        v_neg = -1 + (1 - buffer) * q_lo           # -1 far below ... -buffer at boundary
        v_pos = buffer + (1 - buffer) * q_hi       # buffer at boundary ... 1 far above
        t[neg] = (1 - v_neg) / 2                   # y = -v, then [-1,1] -> [0,1]
        t[pos] = (1 - v_pos) / 2
    return cmap(t)[..., :3]
