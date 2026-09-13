"""Film: the ESD of one layer growing a tail, against Marchenko-Pastur and the shuffled-entries null.

Measured per checkpoint: eigenvalues of W^T W/N (float64 SVD), eigenvalues of the element-shuffled W,
element variance sigma^2 (-> MP law), power-law fit (Clauset MLE, xmin by KS).
Declared: between two measured checkpoints the *sorted* eigenvalues are interpolated linearly in log10(lambda)
(TWEEN frames per interval); fit numbers shown are from the last measured checkpoint, never interpolated.

usage: python render_esd_film.py run layer style [--frames-per 5]
"""
import argparse, os, shutil
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from common import load_run, metrics, shape_NM, mp_density_log10, mp_edges, CACHE
import render_common as R

G = {}


def setup(run, layer, style, fpi):
    r = load_run(run); m = metrics(run, layer)
    lam = r[f'{layer}/lam']; sh = r[f'{layer}/lam_shuf']; var = r[f'{layer}/elem_var']
    N, M = shape_NM(r, layer); Q = N / M
    lo = np.log10(np.percentile(np.concatenate([lam.ravel(), sh.ravel()]), 1.0)) - 0.15
    hi = np.log10(lam.max()) + 0.25
    T = lam.shape[0]
    frames = [(i, u) for i in range(T - 1) for u in np.linspace(0, 1, fpi, endpoint=False)] + [(T - 1, 0.0)] * 75
    G.update(r=r, m=m, lam=lam, sh=sh, var=var, Q=Q, N=N, M=M, lo=lo, hi=hi, frames=frames, style=style,
             layer=layer, run=run, T=T)


STY = {
    'night': dict(bg='#08080b', fg='#ece6d6', dim='#57544e', esd='#f2a93b', null='#4f7fb8', mp='#9fc3e8', tick='#f7d38a'),
    'ink': dict(bg=R.PAPER, fg=R.INK, dim='#a39c8f', esd='#b3342b', null='#7d93ab', mp='#2b4a6f', tick='#1b1a17'),
}


def frame(k):
    i, u = G['frames'][k]
    s = STY[G['style']]
    lam, sh = G['lam'], G['sh']
    if u > 0:
        a = np.log10(np.sort(lam[i])); b = np.log10(np.sort(lam[i + 1]))
        cur = 10 ** ((1 - u) * a + u * b)
        a = np.log10(np.sort(sh[i])); b = np.log10(np.sort(sh[i + 1]))
        cur_sh = 10 ** ((1 - u) * a + u * b)
        var = (1 - u) * G['var'][i] + u * G['var'][i + 1]
        step = 10 ** ((1 - u) * np.log10(max(G['r']['step'][i], 0.5)) + u * np.log10(G['r']['step'][i + 1]))
    else:
        cur, cur_sh, var, step = lam[i], sh[i], G['var'][i], G['r']['step'][i]
    lo, hi = G['lo'], G['hi']
    bins = np.linspace(lo, hi, 170); w = bins[1] - bins[0]
    h, _ = np.histogram(np.log10(cur), bins); hs, _ = np.histogram(np.log10(cur_sh), bins)
    dens = h / (G['M'] * w); dens_s = hs / (G['M'] * w)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=s['bg'])
    ax = fig.add_axes([0.06, 0.25, 0.62, 0.62]); ax.set_facecolor(s['bg'])
    floor = 0.5 / (G['M'] * w)
    xc = np.repeat(bins, 2)[1:-1]
    ax.fill_between(xc, floor, np.maximum(np.repeat(dens_s, 2), floor), color=s['null'], alpha=0.35, lw=0, step=None)
    ax.fill_between(xc, floor, np.maximum(np.repeat(dens, 2), floor), color=s['esd'], alpha=0.85, lw=0)
    ax.plot(xc, np.maximum(np.repeat(dens, 2), floor), color=s['esd'], lw=1.2)
    g = np.linspace(lo, hi, 3000)
    mp = mp_density_log10(10 ** g, var, G['Q'])
    mp0 = mp_density_log10(10 ** g, G['var'][0], G['Q'])
    ax.plot(g, np.where(mp0 > 0, mp0, np.nan), color=s['dim'], lw=1.0, ls=(0, (2, 3)))
    ax.plot(g, np.where(mp > 0, mp, np.nan), color=s['mp'], lw=1.8)
    m = G['m']; ci = i if u == 0 else i  # last measured checkpoint
    al, xmin = m['alpha'][ci], m['xmin'][ci]
    # power-law fit drawn in log10-density units: rho(log10 l) ∝ l^(1-alpha), anchored to the tail count
    ntail = m['n_tail'][ci]
    gx = np.linspace(np.log10(xmin), np.log10(lam[ci].max()), 50)
    dfit = (ntail / G['M']) * (al - 1) * np.log(10) * (10 ** gx / xmin) ** (1 - al)
    ax.plot(gx, dfit, color=s['fg'], lw=1.3, ls=(0, (6, 3)))
    ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(floor * 0.9, max(dens.max(), mp.max()) * 3)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color(s['dim'])
    ax.tick_params(colors=s['dim'], labelsize=11)
    ax.set_xlabel('log₁₀ λ    (eigenvalues of WᵀW / N)', color=s['fg'], fontsize=13, family='DejaVu Sans')
    ax.set_ylabel('density in log₁₀ λ', color=s['fg'], fontsize=13, family='DejaVu Sans')
    # spectral-line strip: every eigenvalue as one line
    ax2 = fig.add_axes([0.06, 0.09, 0.62, 0.09]); ax2.set_facecolor(s['bg'])
    lp = mp_edges(var, G['Q'])[1]
    lc = np.log10(cur)
    out = cur > cur_sh.max()
    ax2.vlines(lc[~out], 0, 1, color=s['dim'], lw=0.35, alpha=0.5)
    ax2.vlines(lc[out], 0, 1, color=s['tick'], lw=1.4)
    ax2.axvline(np.log10(lp), color=s['mp'], lw=1.0, ls=(0, (2, 2)))
    ax2.set_xlim(lo, hi); ax2.set_ylim(0, 1); ax2.axis('off')
    fig.text(0.06, 0.195, 'every eigenvalue, one line   ·   bright: above the largest eigenvalue of the shuffled-entries null   ·   dashed: MP edge λ₊',
             color=s['dim'], fontsize=10.5, family='DejaVu Sans')
    # right column: text + alpha trace
    r = G['r']
    ep = step / r['meta']['steps_per_epoch']
    fig.text(0.72, 0.87, f"{G['layer']}  ·  {G['N']}×{G['M']}", color=s['fg'], fontsize=26, family=R.SERIF)
    fig.text(0.72, 0.83, 'a weight matrix departing from randomness', color=s['dim'], fontsize=15, family=R.SERIF,
             style='italic')
    rows = [('step', f'{int(round(step)):,}'), ('epoch', f'{ep:.2f}'), ('test acc', f"{r['test_acc'][ci]:.3f}"),
            ('α (tail fit)', f'{al:.2f}'), ('λmax / null max', f"{m['lmax_over_null'][ci]:.1f}"),
            ('eigs above null', f"{m['n_out'][ci]}")]
    for j, (kk, vv) in enumerate(rows):
        y = 0.75 - j * 0.045
        fig.text(0.72, y, kk, color=s['dim'], fontsize=15, family='DejaVu Sans')
        fig.text(0.93, y, vv, color=s['fg'], fontsize=17, family='DejaVu Sans Mono', ha='right')
    ax3 = fig.add_axes([0.72, 0.12, 0.21, 0.26]); ax3.set_facecolor(s['bg'])
    st = np.maximum(r['step'], 1)
    ax3.plot(st, m['alpha_shuf'], color=s['null'], lw=1.2, alpha=0.7)
    ax3.plot(st, m['alpha'], color=s['dim'], lw=1.0)
    ax3.plot(st[:ci + 1], m['alpha'][:ci + 1], color=s['esd'], lw=2.2)
    ax3.set_xscale('log'); ax3.set_ylim(1.5, max(12, np.nanmax(m['alpha']) + 0.5))
    ax3.axhspan(2, 4, color=s['esd'], alpha=0.08, lw=0)
    for sp in ['top', 'right']:
        ax3.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax3.spines[sp].set_color(s['dim'])
    ax3.tick_params(colors=s['dim'], labelsize=10)
    ax3.set_title('α over training (blue: shuffled null)', color=s['dim'], fontsize=12, loc='left', family='DejaVu Sans')
    ax3.set_xlabel('step', color=s['dim'], fontsize=11)
    fig.text(0.06, 0.93, f"orange: measured ESD   ·   blue fill: same entries shuffled   ·   solid curve: Marchenko–Pastur at current σ²   ·   dotted: MP at init   ·   dashed: power-law fit λ^−α",
             color=s['dim'], fontsize=11, family='DejaVu Sans')
    fig.text(0.06, 0.025, f"MLP 784-1024-1024-1024-10 on FashionMNIST · SGD lr {r['meta']['lr']} momentum {r['meta']['momentum']} batch {r['meta']['bs']} · "
             f"frames between checkpoints: sorted eigenvalues interpolated in log λ (declared); fit numbers from last measured checkpoint",
             color=s['dim'], fontsize=10, family='DejaVu Sans')
    fdir = G['fdir']
    fig.savefig(os.path.join(fdir, f'{k:05d}.png'), dpi=100, facecolor=s['bg'])
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('layer'); ap.add_argument('style')
    ap.add_argument('--fpi', type=int, default=5); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only', type=int, default=-1, help='render a single frame index to gallery preview')
    a = ap.parse_args()
    R.set_rc()
    setup(a.run, a.layer, a.style, a.fpi)
    fdir = os.path.join(CACHE, 'frames', f'esd_{a.run}_{a.layer}_{a.style}')
    if os.path.exists(fdir) and a.only < 0:
        shutil.rmtree(fdir)
    os.makedirs(fdir, exist_ok=True); G['fdir'] = fdir
    if a.only >= 0:
        frame(a.only); print(os.path.join(fdir, f'{a.only:05d}.png')); raise SystemExit
    with Pool(a.workers) as p:
        p.map(frame, range(len(G['frames'])), chunksize=4)
    print(R.frames_to_video(fdir, f'esd_film_{a.run}_{a.layer}_{a.style}'))
    # hero still = last frame
    shutil.copy(os.path.join(fdir, f'{len(G["frames"]) - 1:05d}.png'),
                os.path.join(R.GALLERY, f'esd_final_{a.run}_{a.layer}_{a.style}.png'))
