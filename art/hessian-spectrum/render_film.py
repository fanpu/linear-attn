"""Spectrograph over training: exact Hessian spectra at 49 log-spaced checkpoints (film_mlps_C10, N=2000).

Stills:  gallery/spectrograph_<style>.png   (rows = training time on log(step+1), columns = symlog eigenvalue)
Film:    gallery/spectrograph_film.mp4/.gif (current plate on top, the spectrograph trail accumulating below)

Between checkpoints the sorted eigenvalue lists are linearly interpolated in symlog space (declared tween).

python render_film.py --stills --film
"""
import argparse, glob, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_common import n_structural, slog, exposure_profile, log_density, wavelength_rgb, hexrgb, GAL, CACHE
import sys
sys.path.insert(0, '/home/fzeng/ml/research/art/color-research')

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='film_mlps_C10_pc200')
ap.add_argument('--stills', action='store_true')
ap.add_argument('--film', action='store_true')
ap.add_argument('--styles', default='hue,silver,magma,paper,split,aurora_ember,cyanotype_vandyke')
ap.add_argument('--film_style', default='hue', help='hue or split')
ap.add_argument('--frames', type=int, default=720)
args = ap.parse_args()

fs = sorted(glob.glob(os.path.join(CACHE, 'exact', args.run, 'step_*.npz')))
ck = [np.load(f) for f in fs]
steps = np.array([int(d['step']) for d in ck])
S = np.stack([np.sort(slog(d['H_eig'])) for d in ck])        # (T, P) sorted symlog eigenvalues
loss = np.array([float(d['loss']) for d in ck]); acc = np.array([float(d['acc']) for d in ck])
count = np.array([n_structural(d) for d in ck])          # eigvecs in span{d_c}
gapcount = np.array([int(d['count_H'][0]) for d in ck])
u = np.log10(steps + 1)
x0, x1 = np.floor(S.min() * 2) / 2 - 0.2, np.ceil(S.max() * 2) / 2 + 0.2
print('checkpoints', len(steps), 'x range', x0, x1, 'counts', count)

FONT = '/usr/share/fonts/opentype/urw-base35/P052-Roman.otf'
FONTI = '/usr/share/fonts/opentype/urw-base35/P052-Italic.otf'
MONO = '/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf'


def spectrum_at(uu):
    i = int(np.clip(np.searchsorted(u, uu) - 1, 0, len(u) - 2))
    a = float(np.clip((uu - u[i]) / (u[i + 1] - u[i]), 0, 1))
    return (1 - a) * S[i] + a * S[i + 1], (i if a < 0.5 else i + 1)


def islog_(s, tau=1e-4):
    return np.sign(s) * tau * (10 ** np.abs(s) - 1)


DREF = 60.0   # reference line density for the log texture term (declared tone curve)


def tone(D, d0=0.7):
    """Declared tone curve: 60% photographic exposure (a lone eigenvalue reaches ~0.46) + 40% log density
    (keeps texture inside the bulk instead of saturating it)."""
    return np.clip(0.6 * (1 - np.exp(-D / d0)) + 0.4 * np.log1p(D) / np.log1p(DREF), 0, 1)


def dens_row(s, W, sigma=1.1):
    return exposure_profile(islog_(s), x0, x1, W, sigma_px=sigma)[1]


def expo_row(s, W, sigma=1.1):
    return tone(dens_row(s, W, sigma))


def waterfall(W, Hrows):
    uu = np.linspace(u[0], u[-1], Hrows)
    D = np.stack([dens_row(spectrum_at(v)[0], W) for v in uu])
    return tone(D), D, uu


def colorize(E, style, W, D=None):
    xs = np.linspace(x0, x1, W)
    if style == 'hue':
        col = wavelength_rgb(np.interp(xs, [x0, x1], [395, 690])) * 0.85 + 0.15
        return E[..., None] * col[None]
    if style == 'silver':
        return E[..., None] * hexrgb('#efe6d2')[None, None]
    if style == 'magma':
        import matplotlib.cm as cm
        return cm.magma(E ** 0.9)[..., :3]
    if style == 'paper':
        return hexrgb('#f3eee3') * (1 - E[..., None]) + hexrgb('#1c1a17') * E[..., None]
    if style in ('split', 'aurora_ember', 'cyanotype_vandyke'):
        # Signed field: sign(lambda) * log line density; per-side rank normalisation (palettes.render_split,
        # Sohl-Dickstein convention). Empty spectrum = |field| 0 = the dark ends of both halves; the densest bulk
        # glows pale. Negative eigenvalues take the purple/cool half, positive the red/warm half.
        import palettes as P
        M = np.where(xs[None] < 0, -1.0, 1.0) * np.log1p(D)
        M = np.where(D < 1e-3, 0.0, M)
        pairing = 'sd_spectral' if style == 'split' else style
        rgb = P.render_split(M, pairing, near_boundary='small')
        return rgb * (0.08 + 0.92 * np.clip(D / 0.05, 0, 1)[..., None])  # fade to black where no eigenvalue falls
    raise ValueError(style)


if args.stills:
    W, Hr = 2400, 1600
    E, D, uu = waterfall(W, Hr)
    for style in args.styles.split(','):
        rgb = colorize(E, style, W, D)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        ground = (243, 238, 227) if style == 'paper' else (6, 6, 6)
        ink = (28, 26, 23) if style == 'paper' else (220, 214, 200)
        faint = (130, 124, 112)
        can = Image.new('RGB', (W + 420, Hr + 330), ground)
        can.paste(img, (300, 200))
        dr = ImageDraw.Draw(can)
        dr.text((300, 60), 'Spectrograph of a Hessian during training', font=ImageFont.truetype(FONT, 56), fill=ink)
        dr.text((300, 130), 'MNIST, 10 classes · MLP 100-32-32-10 · exact eigenvalues at 49 checkpoints (time runs down)',
                font=ImageFont.truetype(FONTI, 28), fill=faint)
        fm = ImageFont.truetype(MONO, 24)
        for st in [0, 10, 100, 1000, 4680]:
            if st <= steps[-1]:
                y = 200 + (np.log10(st + 1) - u[0]) / (u[-1] - u[0]) * (Hr - 1)
                dr.line([(270, y), (292, y)], fill=faint, width=2)
                dr.text((260, y), f'step {st}', font=fm, fill=faint, anchor='rm')
        for t in [-1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1, 1, 10]:
            xs = slog(t)
            if x0 < xs < x1:
                X = 300 + (xs - x0) / (x1 - x0) * (W - 1)
                dr.line([(X, Hr + 205), (X, Hr + 225)], fill=faint, width=2)
                dr.text((X, Hr + 250), '0' if t == 0 else f'{t:g}', font=fm, fill=ink, anchor='mt')
        dr.text((300 + W, Hr + 290), 'eigenvalue λ (symlog, τ = 1e-4)', font=fm, fill=faint, anchor='rt')
        # outlier counts per checkpoint in the right margin
        for j in range(len(steps)):
            y = 200 + (u[j] - u[0]) / (u[-1] - u[0]) * (Hr - 1)
            if j % 3 == 0 or j == len(steps) - 1:
                dr.text((W + 330, y), f'{count[j]}', font=fm, fill=faint, anchor='mm')
        dr.text((W + 330, 175), 'lines', font=fm, fill=faint, anchor='mm')
        out = f'{GAL}/spectrograph_{style}.png'
        can.save(out, optimize=True); print('wrote', out)

if args.film:
    FW, FH = 1920, 1080
    L, W = 120, 1680
    trace_y, strip_y, strip_h = 120, 250, 130
    wf_y, wf_h = 470, 520
    E, D, uu = waterfall(W, wf_h)
    wf = (np.clip(colorize(E, args.film_style, W, D), 0, 1) * 255).astype(np.uint8)
    Dref = D
    xs = np.linspace(x0, x1, W)
    hue = wavelength_rgb(np.interp(xs, [x0, x1], [395, 690])) * 0.85 + 0.15
    slit = np.clip(1.25 - np.abs(np.linspace(-1, 1, strip_h)) ** 6 * 1.25, 0, 1)
    fT, fM, fI = ImageFont.truetype(FONT, 40), ImageFont.truetype(MONO, 24), ImageFont.truetype(FONTI, 24)
    tmp = os.path.join(CACHE, 'film_frames_' + args.film_style); os.makedirs(tmp, exist_ok=True)
    nF, hold = args.frames, 72
    for fi in range(nF + hold):
        v = u[0] + (u[-1] - u[0]) * min(fi, nF - 1) / (nF - 1)
        s, jn = spectrum_at(v)
        stepv = 10 ** v - 1
        can = np.zeros((FH, FW, 3), np.uint8) + 6
        ex = expo_row(s, W, sigma=1.1)
        if args.film_style == 'hue':
            strip = ex[None, :, None] * hue[None] * slit[:, None, None]
        else:   # same per-side rank normalisation as the waterfall (ref = whole waterfall)
            import palettes as P
            dr_ = dens_row(s, W, 1.1)[None]
            M = np.where(xs[None] < 0, -1.0, 1.0) * np.log1p(dr_); M = np.where(dr_ < 1e-3, 0.0, M)
            Mref = np.where(xs[None] < 0, -1.0, 1.0) * np.log1p(Dref); Mref = np.where(Dref < 1e-3, 0.0, Mref)
            row = P.render_split(M, 'sd_spectral', near_boundary='small', ref=Mref)
            row = row * (0.08 + 0.92 * np.clip(dr_ / 0.05, 0, 1)[..., None])
            strip = row * slit[:, None, None]
        can[strip_y:strip_y + strip_h, L:L + W] = (np.clip(strip, 0, 1) * 255).astype(np.uint8)
        nrow = int(round((v - u[0]) / (u[-1] - u[0]) * (wf_h - 1))) + 1
        can[wf_y:wf_y + nrow, L:L + W] = wf[:nrow]
        im = Image.fromarray(can); dr = ImageDraw.Draw(im)
        dens = np.log10(1 + log_density(islog_(s), x0, x1, W // 2, sigma_px=2.0))
        pts = [(L + 2 * k, trace_y + 110 - dens[k] / 3.8 * 105) for k in range(W // 2)]
        dr.line(pts, fill=(200, 194, 182), width=2)
        top = np.sort(s)[::-1][:count[jn]]
        for t in top:
            X = L + (t - x0) / (x1 - x0) * (W - 1)
            dr.line([(X, strip_y - 22), (X, strip_y - 8)], fill=(220, 214, 200), width=2)
        for t in [-1e-2, 0, 1e-3, 1e-2, 1e-1, 1, 10]:
            X = L + (slog(t) - x0) / (x1 - x0) * (W - 1)
            if x0 < slog(t) < x1:
                dr.line([(X, strip_y + strip_h + 6), (X, strip_y + strip_h + 18)], fill=(120, 116, 106), width=2)
                dr.text((X, strip_y + strip_h + 24), '0' if t == 0 else f'{t:g}', font=fM, fill=(160, 154, 142),
                        anchor='mt')
        dr.text((L, 40), 'Bulk and Outliers', font=fT, fill=(225, 219, 205))
        dr.text((L + W, 52), 'exact Hessian spectrum · MNIST 10 classes · MLP 100-32-32-10', font=fI,
                fill=(130, 124, 112), anchor='rt')
        li = int(np.clip(np.searchsorted(u, v), 0, len(u) - 1))
        dr.text((L, FH - 60), f'step {stepv:7.0f}    train loss {np.interp(v, u, loss):.3f}    '
                f'lines (eigvecs in class-mean span): {count[jn]}', font=fM, fill=(200, 194, 182))
        dr.text((L + W, FH - 60), 'time ↓  symlog λ →', font=fM, fill=(130, 124, 112), anchor='rt')
        im.save(os.path.join(tmp, f'f{fi:04d}.png'))
    sfx = '' if args.film_style == 'hue' else '_' + args.film_style
    mp4 = f'{GAL}/spectrograph_film{sfx}.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-i', os.path.join(tmp, 'f%04d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow', mp4], check=True)
    gif = f'{GAL}/spectrograph_film{sfx}.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4, '-vf',
                    'fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=sierra2_4a',
                    gif], check=True)
    print('wrote', mp4, gif, os.path.getsize(gif) / 1e6, 'MB')
