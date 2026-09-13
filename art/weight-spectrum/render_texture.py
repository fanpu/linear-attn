"""W texture and eigenvector pieces.

Measured: full weight matrices at step 0 and at the final step (train.py --n_full), top-32 singular vectors
and IPR (inverse participation ratio) of all singular vectors at every checkpoint.
Declared: colour maps; the permutation of rows/columns (sorted by the leading singular vectors of the learned
update dW = W_T - W_0, a relabelling of neurons that leaves the function unchanged); 28x28 reshaping of FC1 rows.

usage: python render_texture.py run [pieces...]   pieces: weave fields garments ipr
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from common import load_run, full_W, shape_NM, GALLERY
import render_common as R


def svd_order(D):
    U, S, Vt = np.linalg.svd(D.astype(np.float64), full_matrices=False)
    # order rows by angle in the (u1,u2) plane -> continuous "weave" rather than a monotone ramp
    ro = np.argsort(np.arctan2(U[:, 1] * S[1], U[:, 0] * S[0]))
    co = np.argsort(np.arctan2(Vt[1] * S[1], Vt[0] * S[0]))
    return ro, co, S


def to_img(rgb, path, scale=2):
    a = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a).resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    im.save(path); R.make_preview(path)
    return path


def weave(run, layer='FC2'):
    steps = full_W(run, layer)
    W0 = full_W(run, layer, steps[0]); WT = full_W(run, layer, steps[-1])
    D = WT - W0
    ro, co, S = svd_order(D)
    Ds = D[ro][:, co]
    W0s = W0[ro][:, co]; WTs = WT[ro][:, co]
    out = []
    # 1 Spectral split on the sign of dW (Sohl-Dickstein style): each sign rank-normalised separately
    out.append(to_img(P_split(Ds, 'sd_spectral'), f'{GALLERY}/weave_{run}_{layer}_dW_spectral.png'))
    out.append(to_img(P_split(Ds, 'indigo_madder'), f'{GALLERY}/weave_{run}_{layer}_dW_indigo_madder.png'))
    out.append(to_img(lin_split(Ds, 'sd_spectral'), f'{GALLERY}/weave_{run}_{layer}_dW_spectral_linear.png'))
    # 2 riso: positive update in fluo pink, negative in blue, coverage = |dW| / p99
    c = np.clip(np.abs(Ds) / np.percentile(np.abs(Ds), 99), 0, 1)
    pos = np.where(Ds > 0, c, 0); neg = np.where(Ds < 0, c, 0)
    img = R.riso_composite([(np.kron(neg, np.ones((2, 2))), R.RISO['blue']), (np.kron(pos, np.ones((2, 2))), R.RISO['fluo_pink'])],
                           (2 * D.shape[0], 2 * D.shape[1]), offsets=[(0, 0), (2, 3)], grain=0.15)
    out.append(to_img(img, f'{GALLERY}/weave_{run}_{layer}_dW_riso.png', scale=1))
    # 3 triptych in ink: W0 | W_T | dW, same permutation, grey on paper, shared linear scale
    s = np.percentile(np.abs(WT), 99.5)
    def ink(M, sc):
        v = np.clip(M / sc, -1, 1)
        paper = P.hex2rgb(R.PAPER); inkc = P.hex2rgb(R.INK); red = P.hex2rgb('#a8322a')
        a = np.abs(v)[..., None]
        col = np.where((v > 0)[..., None], inkc, red)
        return paper * (1 - a) + col * a
    crop = slice(0, 384)
    panels = [ink(W0s[crop, crop], s), ink(WTs[crop, crop], s), ink(Ds[crop, crop], np.percentile(np.abs(Ds), 99.5))]
    gap = np.ones((384, 24, 3)) * P.hex2rgb(R.PAPER)
    trip = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], 1)
    out.append(to_img(trip, f'{GALLERY}/weave_{run}_{layer}_triptych_ink.png', scale=3))
    return out, S


def P_split(x, pairing, mode='rank'):
    if mode == 'rank':
        return R.P.render_split(x, pairing, near_boundary='small')
    return lin_split(x, pairing)


def lin_split(x, pairing, q=99.5, pastel=0.8):
    """Linear symmetric normalisation (x / q-th percentile of |x|, clipped) onto a dark-seam split map:
    zero is dark, large |x| is pale. Unlike rank normalisation this does not inflate small values."""
    p = R.P.PAIRINGS[pairing]
    cm = R.P.split_cmap(p['neg'], p['pos'])
    x = np.asarray(x, float)
    v = np.clip(x / np.nanpercentile(np.abs(x), q), -1, 1) * pastel
    return cm((v + 1) / 2)[..., :3]


def fields(run, n=24):
    """FC1 rows as 28x28 receptive fields: learned update dW rows, the n*n neurons with the largest |dW| row norm."""
    steps = full_W(run, 'FC1')
    W0 = full_W(run, 'FC1', steps[0]); WT = full_W(run, 'FC1', steps[-1]); D = WT - W0
    nrm = np.linalg.norm(D, axis=1); sel = np.argsort(-nrm)[:n * n]
    tiles = D[sel].reshape(n, n, 28, 28)
    pad = 3
    big = np.full((n * (28 + pad) + pad, n * (28 + pad) + pad), np.nan)
    for i in range(n):
        for j in range(n):
            t = tiles[i, j]; t = t / np.abs(t).max()
            big[pad + i * (28 + pad):pad + i * (28 + pad) + 28, pad + j * (28 + pad):pad + j * (28 + pad) + 28] = t
    outs = []
    for name, pairing in [('spectral', 'sd_spectral'), ('hubble', 'hubble_sho')]:
        rgb = lin_split(np.nan_to_num(big, nan=0.0), pairing, q=100)
        rgb[np.isnan(big)] = P.hex2rgb('#0c0c10')
        outs.append(to_img(rgb, f'{GALLERY}/fields_{run}_FC1_dW_{name}.png', scale=3))
    v = np.clip(big, -1, 1)
    paper = P.hex2rgb(R.PAPER)
    a = np.abs(np.nan_to_num(v))[..., None]
    rgb = paper * (1 - a) + np.where((np.nan_to_num(v) > 0)[..., None], P.hex2rgb(R.INK), P.hex2rgb('#a8322a')) * a
    outs.append(to_img(rgb, f'{GALLERY}/fields_{run}_FC1_dW_ink.png', scale=3))
    return outs


def garments(run, k=12, n_rows=9):
    """Top-k right singular vectors of FC1 (784-d = 28x28 input images) across checkpoints (sign-aligned)."""
    r = load_run(run)
    V = r['FC1/V_top']  # T x k_vec x 784
    T = V.shape[0]
    s0 = np.searchsorted(r['step'], 30)
    rows = np.unique(np.round(np.geomspace(s0, T - 1, n_rows)).astype(int))
    rows = np.concatenate([[0], rows])
    fig = plt.figure(figsize=(k * 1.05 + 1.6, len(rows) * 1.05 + 1.3), facecolor='#0c0c10')
    for a, t in enumerate(rows):
        for b in range(k):
            v = V[t, b].reshape(28, 28).astype(float)
            if v.flat[np.abs(v).argmax()] < 0:
                v = -v
            ax = fig.add_axes([1.3 / fig.get_figwidth() + b * 1.05 / fig.get_figwidth(),
                               1 - (0.9 + (a + 1) * 1.05) / fig.get_figheight(), 1.0 / fig.get_figwidth(), 1.0 / fig.get_figheight()])
            ax.imshow(lin_split(v, 'sd_spectral', q=100), interpolation='nearest'); ax.axis('off')
        fig.text(1.2 / fig.get_figwidth(), 1 - (0.9 + (a + 0.5) * 1.05) / fig.get_figheight(), f"step {int(r['step'][t]):,}",
                 color='#cfc8b8', ha='right', va='center', fontsize=8, family='DejaVu Sans Mono')
    fig.text(0.5, 1 - 0.45 / fig.get_figheight(), 'FC1 right singular vectors v1 … v12 (input space, 28×28), over training',
             ha='center', color='#ece6d6', fontsize=14, family=R.SERIF)
    fig.text(0.5, 0.25 / fig.get_figheight(), 'each tile: sign fixed so largest |entry| > 0; linear scale ±max|entry| on a dark-seam Spectral split (0 = dark; declared)',
             ha='center', color='#8a8578', fontsize=8, family='DejaVu Sans')
    return R.save(fig, f'garments_{run}_FC1_spectral.png', dpi=200)


def ipr(run):
    r = load_run(run)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=R.PAPER)
    T = r['step'].size
    sel = np.unique(np.round(np.geomspace(1, T, 7)).astype(int) - 1)
    cm = plt.get_cmap('art.cyanotype_r') if 'art.cyanotype_r' in plt.colormaps() else plt.get_cmap('Blues')
    for ax, L in zip(axs, ['FC1', 'FC2', 'FC3']):
        N, M = shape_NM(r, L)
        key = 'ipr_in' if L == 'FC1' else 'ipr_out'
        d = 784 if L == 'FC1' else 1024
        for j, t in enumerate(sel):
            ax.plot(np.arange(1, r[f'{L}/{key}'].shape[1] + 1), r[f'{L}/{key}'][t], lw=1.0,
                    color=cm(0.25 + 0.75 * j / (len(sel) - 1)), label=f"step {int(r['step'][t]):,}")
        ax.axhline(3 / (d + 2), color='#a8322a', lw=1, ls='--', label='random unit vector 3/(d+2)')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_facecolor(R.PAPER)
        ax.set_title(f'{L}: IPR of singular vectors ({"input side, d=784" if L == "FC1" else "output side, d=1024"})',
                     fontsize=10, family='DejaVu Sans')
        ax.set_xlabel('rank (1 = largest singular value)', fontsize=9)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel('IPR = Σ vᵢ⁴  (1/d delocalised … 1 one-hot)', fontsize=9)
    axs[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return R.save(fig, f'ipr_{run}.png', dpi=160)


P = R.P
if __name__ == '__main__':
    run = sys.argv[1] if len(sys.argv) > 1 else 'mlp_bs16_s0'
    pieces = sys.argv[2:] or ['weave', 'fields', 'garments', 'ipr']
    R.set_rc()
    for p in pieces:
        print(p, globals()[p](run))
