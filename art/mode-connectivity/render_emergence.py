"""Emergence: linear mode connectivity after matching is a product of training, not of initialisation.

Measured (compute_hero.py): at 28 checkpoints (epoch 0 ... 20) of both runs, train loss along 25-point lerps
A_k -> B_k (naive) and A_k -> pi_k(B_k) (pi_k = weight matching of those checkpoints).
  emergence_<ds>_strata.png   one profile per checkpoint, stacked (paper) and night variant
  emergence_<ds>.mp4/.gif     profile morphing through training + barrier-vs-epoch trace
Between checkpoints, frames interpolate log-loss linearly in log(epoch + 0.01) (declared).

usage: python render_emergence.py mnist [--pieces strata,film]
"""
import argparse, shutil, textwrap
from PIL import Image
from render_common import *

ap = argparse.ArgumentParser()
ap.add_argument('ds')
ap.add_argument('--tag', default='')
ap.add_argument('--pieces', default='strata,film')
ap.add_argument('--frames', type=int, default=660)
args = ap.parse_args()
D = load(f'hero_{args.ds}{args.tag}.npz')
DSNAME = {'mnist': 'MNIST', 'fmnist': 'Fashion-MNIST'}[args.ds]
E = D['ep_epochs']
le = D['ep_lams']
NV, MT = D['ep_naive'][:, :, 0], D['ep_match'][:, :, 0]
K = len(E)


def blin(L):
    return (L - ((1 - le) * L[0] + le * L[-1])).max()


bn = np.array([blin(x) for x in NV])
bm = np.array([blin(x) for x in MT])

if 'strata' in args.pieces:
    for style in ['paper', 'night']:
        bg, ink, sub = (PAPER, INK, '#5b554d') if style == 'paper' else (NIGHT, '#ece5d5', '#8f887c')
        cm = plt.get_cmap('cmc.batlow') if style == 'paper' else plt.get_cmap('magma')
        W, H = 2600, 3200
        fig = fig_px(W, H, bg=bg)
        for j, (M, t) in enumerate([(NV, 'Aₖ → Bₖ'), (MT, 'Aₖ → πₖ(Bₖ)')]):
            ax = ax_px(fig, 250 + j * 1150, 380, 1000, 2350, W, H)
            xs = np.linspace(0, 1, 300)
            ytop = -1
            step = 1 / K
            amp = step * 7 / np.log(NV.max() / NV.min())
            lo = np.log(NV.min())
            for k in range(K):
                base = (K - 1 - k) * step
                y = base + amp * (np.log(np.interp(xs, le, M[k])) - lo)
                col = cm(0.1 + 0.8 * k / (K - 1)) if style == 'paper' else cm(0.25 + 0.72 * k / (K - 1))
                ytop = max(ytop, y.max())
                ax.fill_between(xs, -1, y, color=bg, lw=0, zorder=2 * k)
                ax.plot(xs, y, color=col, lw=1.4, zorder=2 * k + 1)
                if j == 0 and k % 3 == 0:
                    ax.text(-0.04, base + amp * (np.log(M[k][0]) - lo), f'{E[k]:g}', ha='right', va='center', color=sub, fontsize=10)
            ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.3 * step, ytop + step)
            fig.text((250 + j * 1150 + 500) / W, 1 - 330 / H, t, ha='center', color=ink, fontsize=18, style='italic')
        fig.text(250 / W - 0.01, 1 - 330 / H, 'epoch', ha='right', color=sub, fontsize=11, style='italic')
        fig.text(0.5, 1 - 130 / H, 'THE BASIN IS GROWN, NOT FOUND', ha='center', color=ink, fontsize=26)
        fig.text(0.5, 1 - 210 / H, f'two {DSNAME} networks interpolated at 28 moments of their training', ha='center', color=ink,
                 fontsize=15, style='italic')
        txt = (f'Each line: train loss (log scale, same in every line) along the straight path between the two checkpoints taken at '
               f'that epoch (top: initialisation, bottom: epoch {E[-1]:g}); later profiles occlude earlier ones. Left: raw. '
               f'Right: after weight matching those checkpoints. At initialisation both are flat (barrier {bn[0]:.3f} / {bm[0]:.3f}); '
               f'the naive barrier peaks at {bn.max():.2f} nats, while the matched barrier ends at {bm[-1]:.3f}. '
               f'Colour encodes epoch ({"batlow" if style == "paper" else "magma"}).')
        fig.text(0.5, 1 - 2830 / H, '\n'.join(textwrap.wrap(txt, 125)), ha='center', va='top', color=sub, fontsize=11, linespacing=1.6)
        save(fig, f'emergence_{args.ds}{args.tag}_strata_{style}.png')
    print('strata done')

if 'film' in args.pieces:
    W, H = 1920, 1080
    BG, INKc, SUB, ACC = '#08080c', '#e8e2d4', '#8f887c', '#f4a259'
    fd = os.path.join(HERE, 'frames', f'emerg_{args.ds}')
    shutil.rmtree(fd, ignore_errors=True); os.makedirs(fd)
    u = np.log(E + 0.01)
    nf = args.frames
    us = np.concatenate([np.linspace(u[0], u[-1], nf), np.full(90, u[-1])])
    lNV, lMT = np.log(NV), np.log(MT)
    ylo, yhi = np.log(min(NV.min(), MT.min()) * 0.6), np.log(NV.max() * 1.6)
    for f, uu in enumerate(us):
        k = min(np.searchsorted(u, uu, side='right') - 1, K - 2)
        a = np.clip((uu - u[k]) / (u[k + 1] - u[k]), 0, 1)
        n_ = np.exp(lNV[k] * (1 - a) + lNV[k + 1] * a)
        m_ = np.exp(lMT[k] * (1 - a) + lMT[k + 1] * a)
        ep = np.exp(uu) - 0.01
        fig = fig_px(W, H, dpi=100, bg=BG)
        ax = ax_px(fig, 120, 170, 1000, 760, W, H)
        ax.fill_between(le, m_, n_, color=INKc, alpha=0.13, lw=0)
        ax.plot(le, n_, color=INKc, lw=1.4, ls=(0, (4, 3)))
        ax.fill_between(le, np.exp(ylo), m_, color=ACC, alpha=0.07, lw=0)
        ax.plot(le, m_, color=ACC, lw=2.2)
        ax.set_yscale('log'); ax.set_ylim(np.exp(ylo), np.exp(yhi)); ax.set_xlim(0, 1)
        for yv in [0.01, 0.1, 1]:
            if np.exp(ylo) < yv < np.exp(yhi):
                ax.axhline(yv, color=INKc, lw=0.4, alpha=0.25)
                ax.text(-0.015, yv, f'{yv:g}', ha='right', va='center', color=SUB, fontsize=11)
        ax.text(0, np.exp(ylo) * 0.8, 'A', color=INKc, ha='center', va='top', fontsize=14)
        ax.text(1, np.exp(ylo) * 0.8, 'B', color=INKc, ha='center', va='top', fontsize=14)
        ax.text(0.02, np.exp(yhi) * 0.8, 'train loss (log)', color=SUB, fontsize=11, va='top')
        # barrier trace
        ax2 = ax_px(fig, 1260, 250, 580, 520, W, H, off=False)
        ax2.set_facecolor(BG)
        sel = u <= uu + 1e-9
        bnt = np.interp(uu, u, bn); bmt = np.interp(uu, u, bm)
        ax2.plot(np.r_[E[sel], ep] + 0.01, np.r_[bn[sel], bnt], color=INKc, lw=1.4, ls=(0, (4, 3)))
        ax2.plot(np.r_[E[sel], ep] + 0.01, np.r_[bm[sel], bmt], color=ACC, lw=2.2)
        ax2.plot(ep + 0.01, bnt, 'o', color=INKc, ms=5); ax2.plot(ep + 0.01, bmt, 'o', color=ACC, ms=6)
        ax2.set_xscale('log'); ax2.set_yscale('symlog', linthresh=0.01)
        ax2.set_xlim(0.009, E[-1] * 1.3); ax2.set_ylim(0, bn.max() * 1.5)
        for s in ax2.spines.values():
            s.set_color(SUB)
        ax2.tick_params(colors=SUB, labelsize=10)
        ax2.set_xlabel('epoch + 0.01', color=SUB, fontsize=11); ax2.set_ylabel('barrier (nats, symlog)', color=SUB, fontsize=11)
        fig.text(1260 / W, 1 - 215 / H, f'naive {bnt:.3f}', color=INKc, fontsize=13, family='DejaVu Sans Mono')
        fig.text(1560 / W, 1 - 215 / H, f'matched {bmt:.3f}', color=ACC, fontsize=13, family='DejaVu Sans Mono')
        fig.text(0.5, 1 - 55 / H, 'THE BASIN IS GROWN, NOT FOUND', ha='center', color=INKc, fontsize=22)
        fig.text(120 / W, 1 - 130 / H, f'epoch {ep:6.3f}', color=INKc, fontsize=20, family='DejaVu Sans Mono')
        fig.text(0.5, 1 - 1020 / H, f'Two {DSNAME} MLPs (width {int(D["width"])}) interpolated during training.  Dashed: Aₖ → Bₖ.  '
                 f'Orange: Aₖ → πₖ(Bₖ), weight-matched at that epoch.  28 measured checkpoints; frames between them '
                 'interpolate log-loss in log-epoch.', ha='center', color=SUB, fontsize=11)
        Image.fromarray(fig_to_array(fig)).save(os.path.join(fd, f'{f:05d}.png'))
    print(encode_video(fd, f'emergence_{args.ds}{args.tag}', fps=30, gif_width=900, gif_fps=12))
    shutil.rmtree(fd)
