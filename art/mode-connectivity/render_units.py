"""Unit-level prints from cached hero weights + analyze_units.py output.

  perm_<ds>_ink.png         the layer-1 permutation matrix (one dot per row), black on paper
  perm_<ds>_riso.png        all three hidden-layer permutations overprinted in three riso inks (misregistered)
  similarity_<ds>_<s>.png   cosine similarity of A vs B first-layer units: B's order (noise) | reordered by pi (diagonal)
  quilt_<ds>_<s>.png        first-layer receptive fields of A | pi(B) | B (each tile = one unit's 784 weights)

usage: python render_units.py mnist [--pieces perm,similarity,quilt]
"""
import argparse, textwrap, torch
from render_common import *

ap = argparse.ArgumentParser()
ap.add_argument('ds')
ap.add_argument('--tag', default='')
ap.add_argument('--pieces', default='perm,similarity,quilt')
args = ap.parse_args()
U = load(f'units_{args.ds}{args.tag}.npz')
Wt = torch.load(os.path.join(CACHE, f'hero_{args.ds}{args.tag}_weights.pt'), weights_only=False)
DSNAME = {'mnist': 'MNIST', 'fmnist': 'Fashion-MNIST'}[args.ds]
p0 = U['perm0']
n = len(p0)
sfx = f'{args.ds}{args.tag}'


def title(fig, H, t1, t2, ink, y1=120, y2=200, fs1=28, fs2=14):
    fig.text(0.5, 1 - y1 / H, t1, ha='center', va='center', color=ink, fontsize=fs1)
    fig.text(0.5, 1 - y2 / H, t2, ha='center', va='center', color=ink, fontsize=fs2, style='italic')


def caption(fig, H, y, txt, col, width=140, fs=11):
    txt = '\n'.join(textwrap.fill(p, width) for p in txt.split('\n'))
    fig.text(0.5, 1 - y / H, txt, ha='center', va='top', color=col, fontsize=fs, linespacing=1.6)


if 'perm' in args.pieces:
    W, H = 2600, 3000
    fig = fig_px(W, H, bg='#f7f4ec')
    ax = ax_px(fig, 250, 350, 2100, 2100, W, H)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ms = 2100 / n * 2.4 * 72 / 200  # declared: dot diameter 2.4 cells
    ax.plot(p0, np.arange(n), 's', ms=ms, color='#111111', mew=0)
    for s in ['top', 'bottom', 'left', 'right']:
        pass
    ax.add_patch(plt.Rectangle((-0.5, -0.5), n, n, fill=False, ec='#111111', lw=1.2))
    fixed = int((p0 == np.arange(n)).sum())
    title(fig, H, f'π₁  ∈  S{n}'.translate(str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')),
          f'the entire difference between two {DSNAME} networks, first hidden layer', '#111111')
    caption(fig, H, 2560, f'Row i (unit i of network A) has one dot, in column π₁(i): the unit of network B that weight matching '
            f'(Git Re-Basin, Algorithm 1) paired with it. {n} dots, {fixed} on the diagonal (1 expected by chance). '
            'Applying π to every layer of B leaves its function exactly unchanged, yet makes the straight path from A '
            'nearly free of loss barriers.', '#444444', width=120)
    save(fig, f'perm_{sfx}_ink.png')

    fig = fig_px(W, H, bg=RISO_PAPER)
    ax = ax_px(fig, 250, 350, 2100, 2100, W, H)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    inks = [RISO_BLUE, RISO_PINK, '#ffb511']
    offs = [(0, 0), (0.9, -0.6), (-0.7, 0.8)]
    for l, (ink, (dx, dy)) in enumerate(zip(inks, offs)):
        pl = U[f'perm{l}']
        ax.plot(pl + dx, np.arange(n) + dy, 'o', ms=ms * 0.8, color=ink, mew=0, alpha=0.82)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), n, n, fill=False, ec=RISO_BLUE, lw=1.2))
    title(fig, H, 'π₁  π₂  π₃', 'three hidden layers, three drums', RISO_BLUE)
    caption(fig, H, 2560, 'Blue: π₁, pink: π₂, yellow: π₃, the permutations weight matching found for each hidden layer '
            f'of B ({DSNAME}, width {n}). Dots are exact positions; the offset between drums (under one cell) is '
            'deliberate misregistration.', RISO_BLUE, width=120)
    save(fig, f'perm_{sfx}_riso.png')
    print('perm done')

if 'similarity' in args.pieces:
    S = U['sim0']
    Sp = S[:, p0]
    for style in ['night', 'paper']:
        W, H = 3600, 2300
        bg, ink, sub = (NIGHT, '#ece5d5', '#8f887c') if style == 'night' else ('#f7f4ec', INK, '#5b554d')
        fig = fig_px(W, H, bg=bg)
        vmax = float(np.percentile(np.diag(Sp), 90))
        for i, (M, t) in enumerate([(S, 'columns in B\'s own order'), (Sp, 'columns reordered by π₁')]):
            ax = ax_px(fig, 250 + i * 1650, 320, 1450, 1450, W, H)
            if style == 'night':
                ax.imshow(np.clip(M, 0, None), cmap='inferno', vmin=0, vmax=vmax, interpolation='nearest')
            else:
                ax.imshow(np.clip(M, 0, None), cmap='Greys', vmin=0, vmax=vmax, interpolation='nearest')
            fig.text((250 + i * 1650 + 725) / W, 1 - 1830 / H, t, ha='center', color=ink, fontsize=15, style='italic')
        title(fig, H, 'THE SAME UNITS, LISTED DIFFERENTLY',
              f'cosine similarity between the first-layer units of two {DSNAME} networks', ink)
        caption(fig, H, 1920, f'Cell (i, j) = cosine similarity of the incoming weights (784 pixels + bias) of unit i of A and unit j of B, '
                f'{n}×{n}. Left: B in its trained order. Right: the same matrix with columns permuted by π₁. '
                f'Mean matched cosine {U["matchcos0"].mean():.2f}; negative values shown as 0; linear scale to {vmax:.2f} '
                f'({"inferno" if style == "night" else "grey ink"}). No cell values changed between panels.', sub, width=170)
        save(fig, f'similarity_{sfx}_{style}.png')
    print('similarity done')

if 'quilt' in args.pieces:
    side = int(round(np.sqrt(Wt['A']['W0'].shape[1])))
    order = np.argsort(-U['matchcos0'])  # declared: tiles sorted by match quality, best first
    ncol = 16
    nrow = int(np.ceil(n / ncol))

    def tiles(Wm):
        T = Wm.numpy().reshape(-1, side, side)
        return T / np.abs(T).max(axis=(1, 2), keepdims=True)

    TA = tiles(Wt['A']['W0'])[order]
    TBp = tiles(Wt['Bp']['W0'])[order]
    TB = tiles(Wt['B']['W0'])  # B's own order

    def mosaic(T, fill=np.nan, gap=3):
        s = side + gap
        M = np.full((nrow * s - gap, ncol * s - gap), fill)
        for k, t in enumerate(T):
            r, c = divmod(k, ncol)
            M[r * s:r * s + side, c * s:c * s + side] = t
        return M

    mos = [mosaic(TA), mosaic(TBp), mosaic(TB)]
    for style in ['spectral', 'vik', 'riso']:
        if style == 'spectral':
            allv = np.concatenate([m[np.isfinite(m)] for m in mos])
            rgbs = [P.render_split(m, 'sd_spectral', near_boundary='small', nan_color='#faf7f0', ref=allv) for m in mos]
            bg, ink, sub = '#faf7f0', INK, '#5b554d'
            cdesc = ('Colour: weight sign split at 0, matplotlib Spectral; negative weights purple half, positive red half, '
                     'each side rank-normalised over all tiles (declared aesthetic).')
        elif style == 'vik':
            cm = plt.get_cmap('cmc.vik') if 'cmc.vik' in plt.colormaps() else plt.get_cmap('RdBu_r')
            rgbs = []
            for m in mos:
                r = cm((np.nan_to_num(m) + 1) / 2)[..., :3]
                r[~np.isfinite(m)] = to_rgb('#0d0d12')
                rgbs.append(r)
            bg, ink, sub = '#0d0d12', '#ece5d5', '#8f887c'
            cdesc = 'Colour: Crameri vik, linear, symmetric about 0 (blue negative, orange positive).'
        else:
            rgbs = []
            for m in mos:
                pos = np.clip(np.nan_to_num(m), 0, 1) ** 0.8
                neg = np.clip(-np.nan_to_num(m), 0, 1) ** 0.8
                r = np.ones(m.shape + (3,)) * to_rgb(RISO_PAPER)
                r *= 1 - pos[..., None] * (1 - np.array(to_rgb(RISO_BLUE)))
                negs = np.roll(neg, (1, 1), (0, 1))  # 1-px misregistration of the pink drum
                r *= 1 - negs[..., None] * (1 - np.array(to_rgb(RISO_PINK)))
                rgbs.append(r)
            bg, ink, sub = RISO_PAPER, RISO_BLUE, RISO_BLUE
            cdesc = 'Two inks: positive weights blue, negative pink (multiplicative overprint, pink drum offset 1 px).'
        mh, mw = mos[0].shape
        scale = 2.0
        pw, ph = mw * scale, mh * scale
        gap = 150
        W = int(3 * pw + 2 * gap + 400)
        H = int(ph + 700)
        fig = fig_px(W, H, bg=bg)
        for i, (rgb, t) in enumerate(zip(rgbs, ['A', 'π(B)', 'B'])):
            ax = ax_px(fig, 200 + i * (pw + gap), 330, pw, ph, W, H)
            ax.imshow(rgb, interpolation='nearest')
            fig.text((200 + i * (pw + gap) + pw / 2) / W, 1 - 290 / H, t, ha='center', va='bottom', color=ink, fontsize=22)
        title(fig, H, 'RECEPTIVE-FIELD QUILT', f'all {n} first-layer units of two {DSNAME} networks', ink, y1=110, y2=190)
        caption(fig, H, ph + 400, f'Each tile: one unit\'s 784 incoming weights as a 28×28 image, scaled by its own max |w|. '
                f'A and π(B) are shown in the same order (sorted by matched cosine similarity, best top-left, a declared ordering); '
                f'B in its trained order. ' + cdesc, sub, width=int(W / 22))
        save(fig, f'quilt_{sfx}_{style}.png')
    print('quilt done')

if 'pairs' in args.pieces:
    side = int(round(np.sqrt(Wt['A']['W0'].shape[1])))
    mc = U['matchcos0']
    order = np.argsort(-mc)
    cm = plt.get_cmap('cmc.vik')
    ncol, nrow, sc = 6, 4, 8
    ts = side * sc
    W, H = ncol * (2 * ts + 16 + 110) + 200, nrow * (ts + 120) + 560
    fig = fig_px(W, H, bg='#0d0d12')
    for k in range(ncol * nrow):
        u = order[k * 3]  # every 3rd of the best 72 pairs (declared selection)
        r, c = divmod(k, ncol)
        x0 = 150 + c * (2 * ts + 16 + 110)
        y0 = 330 + r * (ts + 120)
        for j, key in enumerate(['A', 'Bp']):
            t = Wt[key]['W0'][u].numpy().reshape(side, side)
            ax = ax_px(fig, x0 + j * (ts + 16), y0, ts, ts, W, H)
            ax.imshow(cm((t / np.abs(t).max() + 1) / 2), interpolation='nearest')
        fig.text((x0 + ts + 8) / W, 1 - (y0 + ts + 40) / H, f'cos {mc[u]:.2f}', ha='center', color='#8f887c', fontsize=11,
                 family='DejaVu Sans Mono')
    title(fig, H, 'TWINS', f'first-layer units of network A (left of each pair) and the unit of B matched to them  ·  {DSNAME}',
          '#ece5d5', y1=120, y2=205)
    caption(fig, H, H - 190, f'24 pairs: every third of the 72 best-matched units by cosine similarity of incoming weights '
            f'(the best {int(np.round(100 * 72 / n))}% of {n}; median over all units {np.median(mc):.2f}). Pixels are the raw 28×28 weights, '
            'Crameri vik, symmetric per tile about 0 (blue negative). The two networks never saw each other; B is from a different seed.',
            '#8f887c', width=170)
    save(fig, f'twins_{sfx}.png')
    print('pairs done')
