"""Render single-window pieces from cache/windows: architecture diptychs, semantic-axis
plates, the step-count animation, and relief/riso/line variants. CPU only.

  python render_windows.py diptych B
  python render_windows.py semantic
  python render_windows.py steps steps_kf2_256
  python render_windows.py styles <npz-name|zoomTag:k> <outname>
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

import styles as S
from common_render import cdf_img, edges
from pages import font, MONO, SERIF, SERIF_B, PAPER, INKC, GREY, fmt_log_centre, plate_page

os.makedirs('gallery', exist_ok=True)


def load(name):
    if ':' in name:
        tag, k = name.split(':')
        d = np.load(f'cache/zoom_{tag}/kf_{int(k):03d}.npz')
        return dict(M=d['measure'], c0=float(d['c0']), c1=float(d['c1']), hw=float(d['hw']),
                    hwy=float(d['hw']), res=int(d['res']), steps=int(d['steps']), nonlin=str(d['nonlin']),
                    axes='lr_lr', minibatch=-1)
    d = np.load(f'cache/windows/{name}.npz')
    out = {k: d[k] for k in d.files}
    for k in ['c0', 'c1', 'hw', 'hwy']:
        out[k] = float(out[k])
    for k in ['res', 'steps', 'minibatch']:
        out[k] = int(out[k])
    out['M'] = out.pop('measure')
    out['nonlin'] = str(out['nonlin']); out['axes'] = str(out['axes'])
    return out


def label_of(w):
    if w['minibatch'] > 0:
        return f'tanh, minibatch {w["minibatch"]}'
    return {'tanh': 'tanh, full batch', 'relu': 'ReLU, full batch', 'sin': 'sin, full batch',
            'quadratic': 'quadratic null', 'liu_cos': 'quadratic + cosine ripple (Liu-type toy)'}.get(w['nonlin'], w['nonlin'])


# --------------------------------------------------------------------------- diptych
def diptych(win, style='spectral'):
    colf = {'spectral': S.spectral, 'magma': S.dark_magma}[style]
    names = [f'dip_{win}_tanh', f'dip_{win}_relu', f'dip_{win}_sin', f'dip_{win}_mb16']
    if win == 'A':
        names[0] = 'zoomA:0'
    if win == 'OV':   # full overview, tanh vs ReLU at 1024^2 float64
        names = ['hero_overview_tanh_1024_f64', 'ov_relu_1024_f64']
    ws = [load(n) for n in names if (':' in n) or os.path.exists(f'cache/windows/{n}.npz')]
    if len(ws) < 2:
        print('not enough panels'); return
    P = 768 if n > 2 else 1024; pad = 40; top = 170; bot = 230
    n = len(ws)
    Wd = n * P + (n + 1) * pad
    # --- (1) dark magma panels
    page = Image.new('RGB', (Wd, top + P + bot), (14, 12, 16))
    d = ImageDraw.Draw(page)
    w0 = ws[0]
    d.text((pad, 40), f'Same window, {["","one","two","three","four"][n]} architectures', font=font(SERIF_B, 52), fill=(235, 228, 215))
    d.text((pad, 110), f'log10 eta0 in [{w0["c0"]-w0["hw"]:.3f}, {w0["c0"]+w0["hw"]:.3f}]   '
                       f'log10 eta1 in [{w0["c1"]-w0["hw"]:.3f}, {w0["c1"]+w0["hw"]:.3f}]   '
                       f'{w0["res"]}x{w0["res"]} nets per panel, 500 steps, {w0.get("dtype", "float64")}',
           font=font(MONO, 26), fill=(170, 160, 150))
    for i, w in enumerate(ws):
        x = pad + i * (P + pad)
        page.paste(Image.fromarray(colf(w['M'])).resize((P, P), Image.NEAREST), (x, top))
        E = edges(w['M'])
        d.text((x, top + P + 24), label_of(w), font=font(SERIF, 38), fill=(235, 228, 215))
        d.text((x, top + P + 80), f'trainable {100*(w["M"]<0).mean():.1f}%   boundary px {100*E.mean():.2f}%',
               font=font(MONO, 26), fill=(170, 160, 150))
    page.save(f'gallery/diptych_{win}_{style}.png')
    # --- (2) overlay line drawing: each architecture's boundary in its own ink on paper
    inks = [(200, 40, 70), (20, 90, 160), (30, 130, 80), (120, 80, 20)]
    R = ws[0]['res']; sc = max(1, 1024 // R)
    canvas = np.ones((R * sc, R * sc, 3)) * np.array(PAPER) / 255
    from scipy import ndimage
    for w, ink in zip(ws, inks):
        E = np.zeros((R, R), bool); E[:-1, :-1] = edges(w['M'])
        E = S.upscale(E[::-1].astype(float), sc)
        E = np.clip(ndimage.gaussian_filter(E, 0.8) * 2.0, 0, 1)
        canvas = canvas * (1 - 0.85 * E[..., None] * (1 - np.array(ink) / 255))
    img = Image.fromarray((canvas * 255).astype(np.uint8))
    page2 = Image.new('RGB', (R * sc + 2 * pad, R * sc + 2 * pad + 260), PAPER)
    page2.paste(img, (pad, pad))
    dd = ImageDraw.Draw(page2)
    dd.rectangle([pad - 2, pad - 2, pad + R * sc + 1, pad + R * sc + 1], outline=INKC, width=2)
    y = pad + R * sc + 30
    dd.text((pad, y), 'Architecture fingerprints: trainability boundaries overprinted', font=font(SERIF_B, 40), fill=INKC)
    for i, (w, ink) in enumerate(zip(ws, inks)):
        dd.rectangle([pad + i * 250, y + 80, pad + i * 250 + 40, y + 100], fill=ink)
        dd.text((pad + i * 250 + 50, y + 76), label_of(w), font=font(SERIF, 28), fill=INKC)
    dd.text((pad, y + 140), f'window centre (log10 eta0, log10 eta1) = ({w0["c0"]:.3f}, {w0["c1"]:.3f}), '
                            f'half-width {w0["hw"]:.3f} decades', font=font(MONO, 24), fill=GREY)
    page2.save(f'gallery/diptych_{win}_overprint.png')
    print('diptych', win, [label_of(w) for w in ws])


# --------------------------------------------------------------------------- semantic
def semantic():
    specs = [('sem_sigma_lr_384', 'log10 sigma  (init scale of both layers) ->',
              'log10 eta  (shared learning rate) ^', 'Trainability over init scale and learning rate'),
             ('sem_wd_lr_384', 'log10 lambda  (L2 weight decay) ->',
              'log10 eta  (shared learning rate) ^', 'Trainability over weight decay and learning rate')]
    for name, xl, yl, title in specs:
        if not os.path.exists(f'cache/windows/{name}.npz'):
            continue
        w = load(name)
        M = w['M']
        for style, fn in [('spectral', S.spectral), ('magma', S.dark_magma), ('riso', lambda m: S.riso_two_ink(m, scale=2)),
                          ('line', lambda m: S.line_boundary(m, scale=2))]:
            img = fn(M)
            lines = [f'x centre {w["c0"]:.3f}, half-width {w["hw"]:.2f} decades;  '
                     f'y centre {w["c1"]:.3f}, half-width {w["hwy"]:.2f} decades',
                     f'{w["res"]}x{w["res"]} independent 16-unit tanh networks, {w["steps"]} steps full-batch GD, float64',
                     f'trainable fraction {100*(M<0).mean():.1f}%',
                     '~same data, same base init, same convergence measure as the (eta0, eta1) plates']
            page = plate_page(img, dict(c0=w['c0'], c1=w['c1'], hw=w['hw'], hwy=w['hwy'], xlabel=xl, ylabel=yl),
                              lines, title)
            page.save(f'gallery/{name}_{style}.png')
        print('semantic', name)


# --------------------------------------------------------------------------- steps
def steps(name, fps=12, style='spectral'):
    colf = {'spectral': S.spectral, 'magma': S.dark_magma}[style]
    w = load(name)
    MT = w['measure_T'].astype(np.float64)
    cps = w['checkpoints']
    # normalise by T so frames are comparable: mean v (converged) / mean 1/v (diverged)
    MTn = MT / cps[:, None, None]
    ref = MTn[-1]
    frames_dir = f'cache/frames_{name}_{style}'
    os.makedirs(frames_dir, exist_ok=True)
    N = 1080
    idx = 0
    E_prev = None
    for i, T in enumerate(cps):
        img = colf(MTn[i], ref)
        im = Image.fromarray(img).resize((N, N), Image.NEAREST)
        canvas = Image.new('RGB', (N, N + 120), (14, 12, 16))
        canvas.paste(im, (0, 0))
        dd = ImageDraw.Draw(canvas)
        dd.text((30, N + 22), f'T = {T:4d} steps of gradient descent', font=font(MONO, 40), fill=(235, 228, 215))
        conv = (MT[i] < 0).mean()
        dd.text((30, N + 74), f'trainable {100*conv:5.1f}%   boundary px {100*edges(MT[i]).mean():5.2f}%',
                font=font(MONO, 28), fill=(160, 150, 140))
        hold = 3 if i < len(cps) - 1 else 36
        for _ in range(hold):
            canvas.save(f'{frames_dir}/f_{idx:05d}.png'); idx += 1
    mp4 = f'gallery/steps_{name}_{style}.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{frames_dir}/f_%05d.png',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', mp4], check=True)
    gif = f'gallery/steps_{name}_{style}.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4, '-vf',
                    'fps=8,scale=540:-1:flags=neighbor,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=none',
                    gif], check=True)
    # small multiples still
    sel = [0, 1, 2, 4, 7, 11, 17, 24, 36, len(cps) - 1]
    sel = [s for s in sel if s < len(cps)]
    T = 360; pad = 16
    sheet = Image.new('RGB', (5 * (T + pad) + pad, 2 * (T + pad + 50) + pad), (14, 12, 16))
    dd = ImageDraw.Draw(sheet)
    for j, s in enumerate(sel[:10]):
        r, c = divmod(j, 5)
        x = pad + c * (T + pad); y = pad + r * (T + pad + 50)
        sheet.paste(Image.fromarray(colf(MTn[s], ref)).resize((T, T), Image.NEAREST), (x, y))
        dd.text((x, y + T + 8), f'T = {cps[s]}', font=font(MONO, 28), fill=(220, 210, 200))
    sheet.save(f'gallery/steps_{name}_{style}_multiples.png')
    print('steps', mp4, os.path.getsize(mp4) / 1e6, 'MB', gif, os.path.getsize(gif) / 1e6, 'MB')


# --------------------------------------------------------------------------- styles
def styles_set(name, out):
    w = load(name)
    M = w['M']
    R = w['res']
    sc = max(1, 2048 // R)
    S_ = {
        'magma': S.upscale(S.dark_magma(M), sc),
        'fireice': S.upscale(S.dark_fire_ice(M), sc),
        'riso': S.riso_two_ink(M, scale=sc, period=max(4.0, sc * 1.1)),
        'line': S.line_boundary(M, scale=sc, weight=max(1.0, sc / 3)),
        'relief': S.upscale(S.hillshade(M), sc),
    }
    for k, v in S_.items():
        Image.fromarray(v).save(f'gallery/{out}_{k}.png')
    print('styles', out, list(S_))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'diptych':
        diptych(sys.argv[2], 'spectral'); diptych(sys.argv[2], 'magma')
    elif cmd == 'semantic':
        semantic()
    elif cmd == 'steps':
        steps(sys.argv[2], style=sys.argv[3] if len(sys.argv) > 3 else 'spectral')
    elif cmd == 'styles':
        styles_set(sys.argv[2], sys.argv[3])


# --------------------------------------------------------------------------- Liu toy
def liu(style='spectral'):
    fn = {'spectral': S.spectral, 'magma': S.dark_magma}[style]
    from boxcount import dimension_of_measure
    panels = [('liu_eps0_1024', 'pure quadratic  (eps = 0)'),
              ('liu_eps05_1024', 'quadratic + cosine ripple  (eps = 0.05, lam = 0.2)')]
    P = 1024; pad = 60; top = 220; bot = 330
    page = Image.new('RGB', (2 * P + 3 * pad, top + P + bot), PAPER)
    d = ImageDraw.Draw(page)
    d.text((pad, 50), 'Trivial non-convexity, same pipeline', font=font(SERIF_B, 56), fill=INKC)
    d.text((pad, 130), 'L(a,b) = a^2 + 0.6ab + b^2 + eps(1 + cos(2pi(a-b)/lam)),  a0=b0=1,  eta0 for a, eta1 for b, '
                       '500 GD steps, float64', font=font(MONO, 26), fill=GREY)
    for i, (n, lab) in enumerate(panels):
        w = load(n)
        f, s, c = dimension_of_measure(w['M'], 2, 256)
        x = pad + i * (P + pad)
        page.paste(Image.fromarray(fn(w['M'])), (x, top))
        d.text((x, top + P + 24), lab, font=font(SERIF, 38), fill=INKC)
        d.text((x, top + P + 84), f'box-counting D = {f["D"]:.2f} +- {f["se"]:.2f} over {f["decades"]:.1f} decades (r2 {f["r2"]:.4f})', font=font(MONO, 25), fill=GREY)
    w = load(panels[0][0])
    d.text((pad, top + P + 160), f'axes: log10 eta0 (x) and log10 eta1 (y), each in [{w["c0"]-w["hw"]:.1f}, {w["c0"]+w["hw"]:.1f}]. '
                                 'Colour: the same restretched convergence measure as every network plate.',
           font=font(SERIF, 30), fill=INKC)
    d.text((pad, top + P + 210), 'After Liu (2024), arXiv:2406.13971: a fractal-looking trainability boundary needs only a '
                                 'rough, non-convex loss, not a neural network.', font=font(SERIF, 30), fill=INKC)
    page.save(f'gallery/deflation_liu_diptych_{style}.png')
    Image.fromarray(S.line_boundary(load('liu_eps05_2048')['M'], scale=1, weight=0.8)).save('gallery/deflation_liu_2048_line.png')
    print('liu done')


if __name__ == '__main__' and sys.argv[1] == 'liu':
    liu('spectral'); liu('magma')
