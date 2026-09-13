"""Triptych 'One Basin': loss along three paths between two trained MLPs, as geological cross-sections.

Measured: surface = full-train-set cross-entropy along the path (201 points); strata = per-class
contributions (1/N) sum_{i in class k} CE_i, stacked in class order so they sum exactly to the surface;
dashed line = test loss.  Aesthetic: bedrock band below zero, stratum colours, hachures.

usage: python render_triptych.py mnist [--styles survey,night,spectral,riso]
"""
import argparse
from render_common import *
from matplotlib.patches import Rectangle

ap = argparse.ArgumentParser()
ap.add_argument('ds')
ap.add_argument('--styles', default='survey,night,spectral,riso')
ap.add_argument('--tag', default='')
args = ap.parse_args()
D = load(f'hero_{args.ds}{args.tag}.npz')
lams = D['lams']
DSNAME = {'mnist': 'MNIST', 'fmnist': 'Fashion-MNIST'}[args.ds]
PANELS = [('naive', 'I', 'straight line  A → B'), ('bezier', 'II', 'learned Bézier curve  A → C → B'),
          ('matched', 'III', 'straight line  A → π(B)')]


def barrier_lin(L):
    return float((L - ((1 - lams) * L[0] + lams * L[-1])).max())


STY = {
    'survey': dict(bg=PAPER, ink=INK, cols=GEO, edge=INK, elw=0.35, hatch=True, sub='#6b635a'),
    'night': dict(bg=NIGHT, ink='#e9e2d0', cols=[plt.get_cmap('magma')(v) for v in np.linspace(0.25, 0.97, 10)],
                  edge=None, elw=0, hatch=False, sub='#8a8478'),
    'spectral': dict(bg='#faf7f0', ink=INK, cols=[plt.get_cmap('Spectral')(v) for v in np.linspace(0.02, 0.98, 10)],
                     edge='#faf7f0', elw=0.5, hatch=True, sub='#6b635a'),
    'riso': dict(bg=RISO_PAPER, ink=RISO_BLUE, cols=None, edge=None, elw=0, hatch=True, sub=RISO_BLUE),
}


def draw_panel(ax, key, st, ymax, exag=False):
    r = D['path_' + key]
    L, Lte = r[:, 0], r[:, 2]
    cls = D.get('cls_' + key)
    ax.set_xlim(-0.04, 1.04)
    base = -0.30 * ymax if not exag else -0.12 * ymax
    ax.set_ylim(base, ymax)
    # bedrock (aesthetic)
    if st['hatch']:
        for xx in np.arange(-0.4, 1.4, 0.012):
            ax.plot([xx, xx + 0.1], [base, 0], color=st['ink'], lw=0.25, alpha=0.55, zorder=1)
        ax.add_patch(Rectangle((-0.04, base), 1.08, -base, color=st['bg'], alpha=0.0))
    else:
        yy = np.linspace(base, 0, 60)
        for i, y0 in enumerate(yy):
            ax.axhline(y0, color=st['ink'], lw=0.3, alpha=0.05 + 0.25 * (i / len(yy)) ** 2, zorder=1)
    ax.axhline(0, color=st['ink'], lw=0.8, zorder=5)
    # strata
    if cls is not None:
        c = cls[:, 0, :]
        cum = np.concatenate([np.zeros((len(lams), 1)), np.cumsum(c, 1)], 1)
        for k in range(10):
            if args_style == 'riso':
                ink = RISO_BLUE if k % 2 == 0 else RISO_PINK
                dx = 0.0 if k % 2 == 0 else 0.0025  # deliberate misregistration of the pink drum
                ax.fill_between(lams + dx, cum[:, k], cum[:, k + 1], color=ink, alpha=0.35 + 0.5 * ((k // 2) % 2 == 0),
                                lw=0, zorder=3)
            else:
                ax.fill_between(lams, cum[:, k], cum[:, k + 1], color=st['cols'][k], lw=0, zorder=3)
                if st['edge'] is not None and k > 0:
                    ax.plot(lams, cum[:, k], color=st['edge'], lw=st['elw'], zorder=4)
    else:
        ax.fill_between(lams, 0, L, color=st['cols'][3] if st['cols'] else RISO_BLUE, lw=0, zorder=3)
    ax.plot(lams, np.minimum(L, ymax), color=st['ink'], lw=1.3, zorder=6, solid_capstyle='round')
    if not exag:
        ax.plot(lams, Lte, color=st['ink'], lw=0.8, ls=(0, (2.5, 2.5)), zorder=6, alpha=0.9)
    ax.set_clip_on(True)
    # boreholes
    for x0 in (0, 1):
        ax.plot([x0, x0], [base * 0.92, ymax * (1.0 if exag else 0.93)], color=st['ink'], lw=0.5, zorder=7)
    ax.set_axis_off()
    return barrier_lin(L), barrier_lin(Lte), r[:, 3].min()


for args_style in args.styles.split(','):
    st = STY[args_style]
    W, H = 3600, 1900
    fig = fig_px(W, H, bg=st['bg'])
    ymax = 1.18 * max(D['path_naive'][:, 0].max(), D['path_naive'][:, 2].max())
    low = 1.25 * max(D['path_matched'][:, 0].max(), D['path_bezier'][:, 0].max())
    EX = [1, 2, 5][int(np.searchsorted([1, 2, 5], (ymax / low) / 10 ** np.floor(np.log10(ymax / low)), side='right')) - 1] * 10 ** np.floor(np.log10(ymax / low))
    pw, gap, x0, y0, ph = 1040, 90, 190, 250, 900
    eh, ey = 330, 250 + 900 + 190
    for i, (key, num, title) in enumerate(PANELS):
        ax = ax_px(fig, x0 + i * (pw + gap), y0, pw, ph, W, H)
        b, bte, amin = draw_panel(ax, key, st, ymax)
        axe = ax_px(fig, x0 + i * (pw + gap), ey, pw, eh, W, H)
        draw_panel(axe, key, st, ymax / EX, exag=True)
        fx0 = (x0 + i * (pw + gap) + pw / 2) / W
        fig.text(fx0, 1 - (ey - 22) / H, f'same section, vertical exaggeration ×{EX:g}', ha='center', va='bottom',
                 color=st['sub'], fontsize=9.5, style='italic')
        endB = 'π(B)' if key == 'matched' else 'B'
        ax.text(0, ymax * 0.96, 'A', ha='center', va='bottom', color=st['ink'], fontsize=15)
        ax.text(1, ymax * 0.96, endB, ha='center', va='bottom', color=st['ink'], fontsize=15)
        fx = (x0 + i * (pw + gap) + pw / 2) / W
        fig.text(fx, 1 - (y0 - 110) / H, num, ha='center', va='center', color=st['ink'], fontsize=22)
        fig.text(fx, 1 - (y0 + ph + 45) / H, title, ha='center', va='center', color=st['ink'], fontsize=13, style='italic')
        fig.text(fx, 1 - (y0 + ph + 100) / H,
                 f'barrier  {b:.3f} nats (train)   {bte:.3f} (test)   worst test accuracy {100 * amin:.1f}%',
                 ha='center', va='center', color=st['sub'], fontsize=9.5)
    # scale bar
    axs = ax_px(fig, 95, y0, 40, ph, W, H)
    axs.set_ylim(-0.30 * ymax, ymax)
    axs.set_xlim(0, 1)
    step = 0.5 if ymax > 1.2 else 0.1
    for v in np.arange(0, ymax * 0.95, step):
        axs.plot([0.55, 1], [v, v], color=st['ink'], lw=0.6)
        axs.text(0.4, v, f'{v:g}', ha='right', va='center', color=st['ink'], fontsize=8)
    axs.plot([1, 1], [0, np.arange(0, ymax * 0.95, step)[-1]], color=st['ink'], lw=0.6)
    axs.text(-1.3, ymax * 0.45, 'cross-entropy (nats)', rotation=90, ha='center', va='center', color=st['ink'], fontsize=9)
    axs2 = ax_px(fig, 95, ey, 40, eh, W, H)
    axs2.set_ylim(-0.12 * ymax / EX, ymax / EX); axs2.set_xlim(0, 1)
    tv = step / EX
    for v in [0, tv]:
        axs2.plot([0.55, 1], [v, v], color=st['ink'], lw=0.6)
        axs2.text(0.4, v, f'{v:g}', ha='right', va='center', color=st['ink'], fontsize=8)
    axs2.plot([1, 1], [0, tv], color=st['ink'], lw=0.6)
    fig.text(0.5, 1 - 60 / H, f'ONE BASIN  ·  three sections between two {DSNAME} networks', ha='center', va='center',
             color=st['ink'], fontsize=17)
    fig.text(0.5, 1 - (H - 55) / H - 0.0,
             f'MLP 784-{int(D["width"])}-{int(D["width"])}-{int(D["width"])}-10, seeds 0 and 1.  Surface: train loss (60k images); '
             'strata: each ' + ('digit' if args.ds == 'mnist' else 'garment') + ' class\'s share of it, stacked class 0 (bottom) to 9; dashed: test loss.  '
             'Bedrock and colours are decorative.', ha='center', va='center', color=st['sub'], fontsize=9)
    save(fig, f'triptych_{args.ds}{args.tag}_{args_style}.png')
    print('saved', args_style)
