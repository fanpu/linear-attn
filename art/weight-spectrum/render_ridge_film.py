"""Film: an Unknown-Pleasures stack that is printed while training runs.

The bottom ridge is the live ESD of one layer (x = singular value s/sqrt(N) = sqrt(lambda), linear); every PER frames a
copy is left behind and drifts up the stack, so the picture at any moment is the recent history of the spectrum.
Measured: sorted eigenvalues at checkpoints. Declared: time between checkpoints interpolates the k-th largest
eigenvalue linearly in log lambda; KDE bandwidth; height = density^GAMMA; fade of old ridges; colours.

usage: python render_ridge_film.py run layer style   (style: joy | gold | ink)
"""
import argparse, os, shutil
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from common import load_run, shape_NM, CACHE
import render_common as R
from render_ridgeline import lin_kde, BW_SV, GAMMA

G = {}
PER = 6          # frames between printed ridges
K = 56           # ridges visible
NF = 900         # frames of training time (30 s at 30 fps)
HOLD = 90


def setup(run, layer):
    r = load_run(run)
    lam = np.sort(r[f'{layer}/lam'], axis=1)
    L = np.log10(lam)
    st = r['step'].astype(float); T = st.size
    lin = bool(r['meta'].get('n_lin'))
    if lin:
        steps = np.linspace(0, st[-1], NF)
    else:
        steps = np.concatenate([[0], np.geomspace(1, st[-1], NF - 1)])
    ci = np.interp(steps, st, np.arange(T))
    i0 = np.minimum(np.floor(ci).astype(int), T - 2); u = ci - i0
    S = np.sqrt(10 ** ((1 - u)[:, None] * L[i0] + u[:, None] * L[i0 + 1]))
    grid = np.linspace(0, S.max() * 1.04, 1500)
    bw = BW_SV * np.sqrt(lam[0].max())
    D = np.stack([lin_kde(s, grid, bw) for s in S])
    H = (D / D.max()) ** GAMMA
    acc = np.interp(steps, st, r['test_acc'])
    G.update(r=r, grid=grid, H=H, steps=steps, acc=acc, layer=layer, run=run, N=shape_NM(r, layer)[0],
             M=shape_NM(r, layer)[1], lin=lin)


STY = {'joy': dict(bg='#000000', fg=(0.96, 0.95, 0.92), txt='#8d8a84'),
       'gold': dict(bg='#0b0a08', fg=None, txt='#8f7f58'),
       'ink': dict(bg=R.PAPER, fg=(0.11, 0.10, 0.09), txt='#8a847a')}


def frame(k):
    f = min(k, NF - 1)
    s = STY[G['style']]
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=s['bg'])
    ax = fig.add_axes([0.1, 0.12, 0.8, 0.76]); ax.set_facecolor(s['bg'])
    grid, H = G['grid'], G['H']
    height, spacing = 7.0, 1.0
    last_print = (f // PER) * PER
    items = [(f, 0.0)] + [(p, (f - p) / PER + 1e-9) for p in range(last_print, -1, -PER)][:K]
    cm = plt.get_cmap('art.klimt_gold') if G['style'] == 'gold' else None
    bgc = np.array(R.P.hex2rgb(s['bg']))
    for z, (p, age) in enumerate(items[::-1]):   # oldest first (back), live ridge last (front)
        y0 = age * spacing
        if p != f and age < 0.999:
            continue
        fade = np.clip(1 - (age - (K - 8)) / 8, 0, 1)
        y = y0 + H[p] * height
        ax.fill_between(grid, y0 - 0.05, y, color=s['bg'], lw=0, zorder=2 * z)
        col = np.array(cm(0.3 + 0.65 * p / NF)[:3]) if cm else np.array(s['fg'])
        col = bgc + (col - bgc) * fade
        ax.plot(grid, y, color=col, lw=1.05 if p != f else 1.5, zorder=2 * z + 1)
    ax.set_xlim(grid[0], grid[-1]); ax.set_ylim(-1.0, K * spacing + height * 0.6); ax.axis('off')
    step = G['steps'][f]
    ep = step / G['r']['meta']['steps_per_epoch']
    fig.text(0.1, 0.06, f'step {int(step):>7,}   epoch {ep:5.2f}   test acc {G["acc"][f]:.3f}', color=s['txt'],
             fontsize=13, family='DejaVu Sans Mono')
    fig.text(0.9, 0.06, f'{G["layer"]} {G["N"]}×{G["M"]} singular values', color=s['txt'], fontsize=13,
             family='DejaVu Sans Mono', ha='right')
    fig.text(0.1, 0.03, f'bottom ridge: live ESD (x = s/√N linear); a copy is left every {PER} frames and drifts up · '
             f'{"linear" if G["lin"] else "log"} time · between checkpoints: interpolated (declared)', color=s['txt'],
             fontsize=8.5, family='DejaVu Sans')
    fig.savefig(os.path.join(G['fdir'], f'{k:05d}.png'), dpi=100, facecolor=s['bg'])
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('layer'); ap.add_argument('style')
    ap.add_argument('--workers', type=int, default=3); ap.add_argument('--only', type=int, default=-1)
    a = ap.parse_args()
    R.set_rc(); setup(a.run, a.layer); G['style'] = a.style
    fdir = os.path.join(CACHE, 'frames', f'ridge_{a.run}_{a.layer}_{a.style}')
    if a.only < 0 and os.path.exists(fdir):
        shutil.rmtree(fdir)
    os.makedirs(fdir, exist_ok=True); G['fdir'] = fdir
    if a.only >= 0:
        frame(a.only); print(R.make_preview(os.path.join(fdir, f'{a.only:05d}.png'))); raise SystemExit
    with Pool(a.workers) as p:
        p.map(frame, range(NF + HOLD), chunksize=8)
    print(R.frames_to_video(fdir, f'ridge_film_{a.run}_{a.layer}_{a.style}', gif_width=600))
