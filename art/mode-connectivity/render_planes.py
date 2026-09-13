"""2-D loss planes through {A, B, pi(B)} (and {A, B, C}) as survey plates.

Measured: train cross-entropy on a regular grid in the affine plane (compute_planes.py; fixed 10k train subset).
Displayed field is bicubic-upsampled log-loss (declared).  Split styles put the seam at a declared level c:
  'basin'  c = highest train loss on the straight segment A -> pi(B)  (sublevel set {L <= c} contains that
           whole segment; whether B joins it is what the image shows)
  'chance' c = ln 10 (uniform guessing)
Each side is rank-normalised separately (Sohl-Dickstein / palettes.render_split).

usage: python render_planes.py mnist --plane perm --styles spectral,aurora,topo,night
"""
import argparse, textwrap
import matplotlib.patheffects as pe
from scipy.ndimage import zoom
from render_common import *

ap = argparse.ArgumentParser()
ap.add_argument('ds')
ap.add_argument('--plane', default='perm')
ap.add_argument('--styles', default='spectral,aurora,topo,night')
ap.add_argument('--tag', default='')
ap.add_argument('--split', default='basin')
ap.add_argument('--px', type=int, default=2200)
ap.add_argument('--split_field', default='train')
args = ap.parse_args()
D = load(f'planes_hero_{args.ds}{args.tag}.npz')
H = load(f'hero_{args.ds}{args.tag}.npz')
xs, ys, pts = D[f'{args.plane}_xs'], D[f'{args.plane}_ys'], D[f'{args.plane}_pts']
Lg = D[f'{args.plane}_{args.split_field}']
DSNAME = {'mnist': 'MNIST', 'fmnist': 'Fashion-MNIST'}[args.ds]
z = args.px / Lg.shape[0]
F_ = zoom(np.log(Lg), z, order=3)  # log loss, bicubic
N = F_.shape[0]
ext = [xs[0], xs[-1], ys[0], ys[-1]]
names = {'perm': ['A', 'B', 'π(B)'], 'bezier': ['A', 'B', 'C'], 'bezm': ['A', 'π(B)', "C'"]}[args.plane]
seg_key = {'perm': 'path_matched', 'bezier': 'path_bezier', 'bezm': 'path_bezier_matched'}[args.plane]
if args.split == 'basin':
    # highest loss on the segment A -> pi(B) (or on the curve) measured in this plane's own field (same image subset)
    from scipy.ndimage import map_coordinates
    if args.plane == 'perm':
        seg = pts[0][None] + np.linspace(0, 1, 400)[:, None] * (pts[2] - pts[0])[None]
    else:
        seg = D[f'{args.plane}_curve']
    rr = (seg[:, 1] - ys[0]) / (ys[1] - ys[0]); cc_ = (seg[:, 0] - xs[0]) / (xs[1] - xs[0])
    c = float(np.exp(map_coordinates(np.log(Lg), [rr, cc_], order=3).max()))
    seam_txt = f'seam: L = {c:.3f}, the highest train loss on the {"segment A–π(B)" if args.plane == "perm" else "curve"}'
else:
    c = np.log(10)
    seam_txt = 'seam: L = ln 10, the loss of uniform guessing'
S = F_ - np.log(c)


def marks(ax, ink, bg, lw=1.0, fs=15):
    A, B, Q = pts
    if args.plane == 'perm':
        ax.plot([A[0], B[0]], [A[1], B[1]], color=ink, lw=lw, ls=(0, (4, 3)), alpha=0.9)
        ax.plot([A[0], Q[0]], [A[1], Q[1]], color=ink, lw=lw)
    else:
        cur = D[f'{args.plane}_curve']
        ax.plot(cur[:, 0], cur[:, 1], color=ink, lw=lw)
        ax.plot([A[0], B[0]], [A[1], B[1]], color=ink, lw=lw, ls=(0, (4, 3)), alpha=0.9)
    for p, nm, off in zip(pts, names, [(-0.9, -1.2), (0.9, -1.2), (0, 1.1)]):
        ax.plot(*p, marker='^', ms=10, mfc=bg, mec=ink, mew=1.4)
        ax.text(p[0] + off[0] * (xs[-1] - xs[0]) / 40, p[1] + off[1] * (xs[-1] - xs[0]) / 40, nm, color=ink,
                fontsize=fs, ha='center', va='center', path_effects=[pe.withStroke(linewidth=3.5, foreground=bg, alpha=0.8)])


def plate(rgb, style, ink, bg, sub, caption, contours=None):
    W, Hh = 2600, 3000
    fig = fig_px(W, Hh, bg=bg)
    ax = ax_px(fig, 200, 330, 2200, 2200, W, Hh)
    if rgb is not None:
        ax.imshow(rgb, origin='lower', extent=ext, interpolation='lanczos')
    if contours is not None:
        contours(ax)
    marks(ax, ink, bg)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    fig.text(0.5, 1 - 150 / Hh, 'ONE BASIN', ha='center', va='center', color=ink, fontsize=30)
    t = {'perm': f'the plane through A, B and π(B)  ·  {DSNAME}',
         'bezier': f'the plane through A, B and the Bézier control point C  ·  {DSNAME}',
         'bezm': f'the plane through A, π(B) and control point C\'  ·  {DSNAME}'}[args.plane]
    fig.text(0.5, 1 - 235 / Hh, t, ha='center', va='center', color=ink, fontsize=15, style='italic')
    # scale bar in weight-space units
    span = xs[-1] - xs[0]
    bar = 10 ** np.floor(np.log10(span / 4))
    bx = 200 + 2200 * 0.04
    fig.add_artist(plt.Line2D([bx / W, (bx + 2200 * bar / span) / W], [1 - 2590 / Hh] * 2, color=ink, lw=1.5))
    fig.text(bx / W, 1 - 2625 / Hh, f'{bar:g} (L2 distance in weight space)', color=ink, fontsize=10, va='top')
    caption = '\n'.join(textwrap.fill(par, 150) for par in caption.split('\n'))
    fig.text(0.5, 1 - 2720 / Hh, caption, ha='center', va='top', color=sub, fontsize=11, linespacing=1.6)
    return save(fig, f'plane_{args.ds}{args.tag}_{args.plane}_{style}.png')


common = (f'Measured: train cross-entropy (10k fixed images) on a {Lg.shape[0]}² grid, bicubic-upsampled log-loss. '
          f'|A−B| = {np.hypot(*pts[1]):.1f}.')
for style in args.styles.split(','):
    if style in ('spectral', 'aurora', 'hubble', 'indigo'):
        pair = {'spectral': 'sd_spectral', 'aurora': 'aurora_ember', 'hubble': 'hubble_sho', 'indigo': 'indigo_madder'}[style]
        rgb = P.render_split(S, pair, near_boundary='small')
        if style == 'spectral':
            ink, bg, sub = INK, '#faf7f0', '#5b554d'
        else:
            ink, bg, sub = '#efe8d8', '#07090d', '#9a9385'
        pname = 'matplotlib Spectral' if style == 'spectral' else P.PAIRINGS[pair]['title']
        cap = (common + f'\n{seam_txt}.  Below the seam: purple→pale ({pname} low half); above: deep red→pale. '
               'Each side rank-normalised separately (declared aesthetic).')
        if style != 'spectral':
            cap = cap.replace('purple→pale', 'cool ramp').replace('deep red→pale', 'warm ramp')
        plate(rgb, f'{style}_{args.split}', ink, bg, sub, cap)
    elif style == 'topo':
        levels = np.log(np.geomspace(Lg.min() * 1.02, Lg.max(), 44))
        idx = levels[::4]
        yy = np.linspace(ext[2], ext[3], N); xx = np.linspace(ext[0], ext[1], N)

        def cont(ax):
            ax.contour(xx, yy, F_, levels=levels, colors=INK, linewidths=0.45, alpha=0.85, negative_linestyles='solid')
            ax.contour(xx, yy, F_, levels=idx, colors=INK, linewidths=1.1, negative_linestyles='solid')
            ax.contour(xx, yy, F_, levels=[np.log(c)], colors='#b2182b', linewidths=1.8, negative_linestyles='solid')
        cap = (common + f'\nInk contours at 44 log-spaced loss levels (every 4th heavy, factor '
               f'{(Lg.max() / Lg.min()) ** (4 / 43):.2f} apart); red contour: {seam_txt[6:]}.')
        plate(None, 'topo', INK, PAPER, '#5b554d', cap, contours=cont)
    elif style == 'night':
        cm = plt.get_cmap('cmc.lajolla_r') if 'cmc.lajolla_r' in plt.colormaps() else plt.get_cmap('magma')
        v = (F_ - F_.min()) / (F_.max() - F_.min())
        rgb = plt.get_cmap('magma')(1 - v)[..., :3] * 0.98
        yy = np.linspace(ext[2], ext[3], N); xx = np.linspace(ext[0], ext[1], N)

        def cont(ax):
            ax.contour(xx, yy, F_, levels=np.log(np.geomspace(Lg.min() * 1.02, Lg.max(), 30)), colors='#ffffff',
                       linewidths=0.35, alpha=0.35, negative_linestyles='solid')
        cap = common + '\nColour: magma, reversed, linear in log-loss (bright = low loss). White: 30 log-spaced contours.'
        plate(rgb, 'night', '#efe8d8', '#050507', '#9a9385', cap, contours=cont)  # labels haloed
    print('saved', style)
