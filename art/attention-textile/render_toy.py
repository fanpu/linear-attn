"""Part A renders: the toy 2-layer attention-only transformer forming induction heads.

  python render_toy.py [curves] [strip] [loom] [movie]
"""
import json, os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from textile import *

OUT = 'gallery'
TAG = 'main'
log = {k: v for k, v in np.load(f'cache/toy_{TAG}.npz').items() if k not in ('mean_attn_rep', 'probe_attn')}
an = dict(np.load(f'cache/toy_{TAG}_analysis.npz'))      # load fully (lazy npz is slow and not fork-safe)
summ = json.load(open(f'cache/toy_{TAG}_summary.json'))
steps = log['step']
L, H = log['ind_score'].shape[1:]
IND = tuple(summ['induction_head'])                         # (layer, head)
PREV = (0, int(np.argmax(log['prev_score'][-1][0])))
PH = summ['phase_step_half_drop']
P10, P90 = summ['phase_step_10pct'], summ['phase_step_90pct']
SER = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']            # validated categorical slots (light)
INK, MUTED, SURF = '#26251f', '#77756b', '#f7f5f0'


def style_ax(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#b9b6aa')
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color='#e4e1d8', lw=0.6)
    ax.set_facecolor(SURF)
    ax.axvspan(P10, P90, color='#e9e3d2', lw=0, zorder=0)


def load_other(tag):
    p = f'cache/toy_{tag}.npz'
    return np.load(p) if os.path.exists(p) else None


def curves():
    others = {t: load_other(t) for t in ['seed1', 'seed2', 'onelayer']}
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True, facecolor=SURF)
    ax = axes[0]
    style_ax(ax)
    ax.plot(steps, log['loss_rep_later'], color=SER[0], lw=2)
    ax.plot(steps, log['loss_markov_copied'], color=SER[1], lw=2)
    ax.plot(steps, log['loss_markov_fresh'], color=SER[2], lw=2)
    for t, o in others.items():
        if o is not None:
            ax.plot(o['step'], o['loss_rep_later'], color=SER[0], lw=1, ls='--' if t != 'onelayer' else ':', alpha=0.8)
    x = steps[-1]
    for y, lab, dy in [(log['loss_rep_later'][-1], 'repeated random, repeats 2-4', -7),
                       (log['loss_markov_copied'][-1], 'Markov text, copied spans', 7),
                       (log['loss_markov_fresh'][-1], 'Markov text, fresh tokens', 0)]:
        ax.annotate(lab, (x, y), xytext=(6, dy), textcoords='offset points', color=INK, fontsize=9, va='center')
    if others['onelayer'] is not None:
        o = others['onelayer']
        ax.annotate('1-layer control (dotted)', (o['step'][-1], o['loss_rep_later'][-1]), xytext=(6, 0),
                    textcoords='offset points', color=MUTED, fontsize=9, va='center')
    ax.set_ylabel('eval loss (nats)', color=INK)
    ax.set_title('The phase change: loss', loc='left', color=INK, fontsize=12)

    ax = axes[1]
    style_ax(ax)
    ind = log['ind_score']
    for h in range(H):
        c = SER[0] if (L - 1, h) == IND else '#9fb4cf'
        ax.plot(steps, ind[:, L - 1, h], color=c, lw=2 if (L - 1, h) == IND else 1)
    prev = log['prev_score']
    for h in range(H):
        c = SER[1] if (0, h) == PREV else '#e8b9a2'
        ax.plot(steps, prev[:, 0, h], color=c, lw=2 if (0, h) == PREV else 1)
    ax.annotate(f'induction score, L1 heads (L1H{IND[1]} bold)', (x, ind[-1, L - 1, IND[1]]), xytext=(6, 0),
                textcoords='offset points', color=INK, fontsize=9, va='center')
    ax.annotate(f'previous-token score, L0 heads (L0H{PREV[1]} bold)', (x, prev[-1, 0, PREV[1]]), xytext=(6, 0),
                textcoords='offset points', color=INK, fontsize=9, va='center')
    ax.set_ylabel('mean attention', color=INK)
    ax.set_title('Heads specialise at the same moment', loc='left', color=INK, fontsize=12)

    ax = axes[2]
    style_ax(ax)
    pm = log['pm_score']
    for h in range(H):
        ax.plot(steps, pm[:, L - 1, h], color=SER[0] if (L - 1, h) == IND else '#9fb4cf', lw=2 if (L - 1, h) == IND else 1)
    ax.axhline(float(log['pm_baseline'][-1][0][0]), color=MUTED, lw=1, ls='--')
    ax.annotate('uniform-attention baseline', (steps[len(steps) // 20], float(log['pm_baseline'][-1][0][0])), xytext=(0, 5),
                textcoords='offset points', color=MUTED, fontsize=9)
    abl = log['ablate_dloss']
    ax.annotate(f'prefix-matching score on Markov text, L1 heads', (x, pm[-1, L - 1, IND[1]]), xytext=(6, 0),
                textcoords='offset points', color=INK, fontsize=9, va='center')
    ax.set_ylabel('attention mass', color=INK)
    ax.set_title('Olsson prefix matching (attention to the token after an earlier copy of the current token)',
                 loc='left', color=INK, fontsize=12)

    ax = axes[3]
    style_ax(ax)
    ax.plot(steps, log['icl_score'], color=SER[3], lw=2)
    for t, o in others.items():
        if o is not None:
            ax.plot(o['step'], o['icl_score'], color=SER[3], lw=1, ls='--' if t != 'onelayer' else ':', alpha=0.8)
    ax.annotate('in-context learning score\n(loss at tokens 100-120 minus tokens 8-16)', (x, log['icl_score'][-1]),
                xytext=(6, 0), textcoords='offset points', color=INK, fontsize=9, va='center')
    ax.set_ylabel('nats', color=INK)
    ax.set_xlabel('training step (shaded: 10%-90% of the loss drop; dashed: other seeds; dotted: 1-layer model)', color=INK)
    fig.subplots_adjust(right=0.72, hspace=0.25, left=0.08, top=0.96, bottom=0.05)
    fig.savefig(f'{OUT}/toy_phase_change_curves.png', dpi=200, facecolor=SURF)
    plt.close(fig)

    # in-context loss vs token position at selected steps
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURF)
    style_ax(ax)
    ax.patches[0].remove() if ax.patches else None
    sel = [0, P10 - 400, P10, PH, P90, int(steps[-1])]
    ramp = ['#cde2fb', '#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
    for s_, c in zip(sel, ramp):
        k = int(np.argmin(np.abs(steps - max(0, s_))))
        y = log['loss_pos_markov'][k]
        ax.plot(np.arange(1, len(y) + 1), y, color=c, lw=2)
        ax.annotate(f'step {steps[k]}', (len(y), y[-1]), xytext=(6, 0), textcoords='offset points', color=INK, fontsize=9,
                    va='center')
    ax.set_xlabel('token position in context (Markov text with one copied span)', color=INK)
    ax.set_ylabel('loss (nats)', color=INK)
    ax.set_title('In-context loss vs. token position: after the phase change, later tokens get cheaper', loc='left',
                 color=INK, fontsize=12)
    fig.subplots_adjust(right=0.86)
    fig.savefig(f'{OUT}/toy_loss_vs_position.png', dpi=200, facecolor=SURF)
    plt.close(fig)


def probe_W(k, l, h):
    return to_unit(an['probe_pats'][k, l, h].astype(np.float64), 0.5)


def strip():
    """Film strip: the induction head's probe pattern at checkpoints across the phase change."""
    ps = an['probe_steps']
    targets = [0, 500, P10 - 200, P10 - 60, P10, (P10 + PH) // 2, PH, (PH + P90) // 2, P90, P90 + 200, P90 + 1000, int(ps[-1])]
    ks = [int(np.argmin(np.abs(ps - t))) for t in targets]
    for style in ['weave', 'riso', 'indigo', 'stitch']:
        tiles = []
        for k in ks:
            Wi, Wp = probe_W(k, *IND), probe_W(k, *PREV)
            if style == 'weave':
                tiles.append(weave(Wi, s=6, warp_lo='#3a2f45', warp_hi='#ffd27f', weft='#1f2f5c', gap='#0a0c14', float_thr=0.3, seed=k))
            elif style == 'indigo':
                tiles.append(indigo(Wi, s=6, bleed=0.3, seed=k))
            elif style == 'riso':
                up = np.ones((6, 6))
                tiles.append(riso([np.kron(Wp, up) * 0.8, np.kron(Wi, up)], ['#0078bf', '#ff48b0'], paper='#f4efe4',
                                  offsets=[(0, 0), (2, -2)], grain=0.15, dot=2, seed=k))
            else:
                A = an['probe_pats'][k, IND[0], IND[1]].astype(float)
                tiles.append(cross_stitch(np.digitize(A, [0.03, 0.1, 0.25, 0.5, 0.8]),
                                          ['#d9c7a7', '#8fa9c4', '#3f6e9a', '#b8452f', '#5a1e1b'], s=6, th=0.22, seed=k))
        bg, fg = {'weave': ('#0a0c14', '#e6dcc0'), 'indigo': ('#f1eee6', '#15306b'), 'riso': ('#f4efe4', '#2a2a2a'),
                  'stitch': ('#e7dcc4', '#3b2f25')}[style]
        gut, out = 60, 40
        img = tile(tiles, 2, 6, gutter=gut, bg=bg, outer=out)
        img = label_tiles(img, [f'step {ps[k]}' for k in ks], 2, 6, tiles[0].shape[:2], gut, out, fg=fg, size=20)
        note = {'riso': f'pink = L1H{IND[1]} (induction head), blue = L0H{PREV[1]} (previous-token head), same probe',
                'weave': f'L1H{IND[1]} attention^0.5 woven', 'indigo': f'L1H{IND[1]} attention^0.5, indigo',
                'stitch': f'L1H{IND[1]}, floss bins 0.03/0.10/0.25/0.50/0.80'}[style]
        img = frame(img, f'The stripe arrives: toy 2-layer attention-only model, 64-token probe (16 random tokens x 4)',
                    f'{note};  loss drop 10%-90% between steps {P10} and {P90}', bg=bg, fg=fg,
                    margin=(110, 40, 80, 40), title_size=30, font_size=19)
        save(img, f'{OUT}/toy_strip_{style}.png')


def loom_img(S, l, h, upto=None, s=(4, 4), style='weave'):
    """Loom record: rows = logged training steps (every 20 steps), columns = key offset 0..127."""
    X = S[::4, l, h, :]                                  # every 20 steps
    X = np.clip(X / 0.4, 0, 1) ** 0.75                   # declared: near-uniform attention recedes
    if upto is not None:
        X = X.copy()
        X[upto:] = 0
    if style == 'weave':
        return weave(X, s=s[0], warp_lo='#3a2f45', warp_hi='#ffd27f', weft='#1f2f5c', gap='#0a0c14', float_thr=0.3, seed=l * 4 + h)
    return indigo(X, s=s[0], bleed=0.0, mottle=0.0, absorb=3.0, seed=l * 4 + h)


def loss_strip_vertical(h_px, w_px, bg, fg, line):
    """Loss on repeats 2-4 vs training step, time running DOWN on the same axis as the loom rows."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new('RGB', (w_px, h_px), tuple(int(c * 255) for c in hexrgb(bg)))
    dr = ImageDraw.Draw(im)
    ys = steps[::4]
    ls = log['loss_rep_later'][::4]
    lo, hi = 0.0, 6.0
    pts = [(10 + (w_px - 20) * (v - lo) / (hi - lo), h_px * i / len(ys)) for i, v in enumerate(ls)]
    col = tuple(int(c * 255) for c in hexrgb(line))
    dr.line(pts, fill=col, width=3)
    f = ImageFont.truetype(FONT_MONO, 18)
    fc = tuple(int(c * 255) for c in hexrgb(fg))
    dr.line([(10, 0), (10, h_px)], fill=fc, width=1)
    dr.text((14, 4), '0', font=f, fill=fc)
    dr.text((w_px - 14, 4), '6 nats', font=f, fill=fc, anchor='ra')
    return np.asarray(im).astype(float) / 255


def loom():
    S = an['offset_spectrum'].astype(np.float64)
    s = 3
    nrows = S[::4].shape[0]
    ph_row = int(np.searchsorted(steps[::4], PH))
    for style in ['weave', 'indigo']:
        bg, fg = ('#0a0c14', '#e6dcc0') if style == 'weave' else ('#f1eee6', '#15306b')
        rule = hexrgb('#ff6b5a') if style == 'weave' else hexrgb('#9b2d1f')
        tiles = [loom_img(S, l, h, style=style, s=(s, s)) for l in range(L) for h in range(H)]
        th, tw = tiles[0].shape[:2]
        strip_ = loss_strip_vertical(th, 220, bg, fg, '#ffd27f' if style == 'weave' else '#15306b')
        gut, out = 40, 40
        Wd = out * 2 + len(tiles) * tw + (len(tiles) - 1) * gut + gut + strip_.shape[1]
        canvas = np.ones((th + 2 * out + 40, Wd, 3)) * hexrgb(bg)
        xs = []
        for k, t in enumerate(tiles):
            x = out + k * (tw + gut)
            canvas[out:out + th, x:x + tw] = t
            xs.append(x)
        xstrip = out + len(tiles) * (tw + gut)
        canvas[out:out + th, xstrip:xstrip + strip_.shape[1]] = strip_
        yr = out + ph_row * s
        canvas[yr - 1:yr + 1, out - 10:xstrip + strip_.shape[1]] = rule
        items = [(x, out + th + 8, f'L{l}H{h}', 'la') for x, (l, h) in zip(xs, [(l, h) for l in range(L) for h in range(H)])]
        items += [(xstrip, out + th + 8, 'loss, repeats 2-4', 'la'), (xstrip + strip_.shape[1], yr - 6, f'step {PH}', 'rd')]
        canvas = text_at(canvas, items, fg=fg, size=20)
        img = frame(canvas, 'Loom record: each weft row is a moment of training, each warp thread a look-back distance',
                    f'mean attention to the key k tokens back (k = 0..127, left to right), period-32 random probe; rows every 20 steps, '
                    f'top = step 0, bottom = step {int(steps[-1])};\n'
                    f'display (min(a/0.4,1))^0.75;  rule = half of the loss drop (step {PH});  k = 31, 63, 95: induction;  k = 1: previous token;  k = 0: current token',
                    bg=bg, fg=fg, margin=(110, 40, 120, 40), title_size=30, font_size=18)
        save(img, f'{OUT}/toy_loom_{style}.png')


_CTX = {}


def _render_frame(fi):
    c = _CTX
    ps, fr, tmp, looms, base, x0, x1, Hs, Ws = (c[k] for k in ['ps', 'fr', 'tmp', 'looms', 'base', 'x0', 'x1', 'Hs', 'Ws'])
    st = fr[fi]
    k = int(np.where(ps == st)[0][0])
    img = np.ones((1080, 1920, 3)) * hexrgb('#0a0c14')
    tiles = [weave(probe_W(k, l, h), s=5, warp_lo='#3a2f45', warp_hi='#ffd27f', weft='#1f2f5c', gap='#0a0c14',
                   float_thr=0.3, seed=l * 4 + h) for l in range(L) for h in range(H)]
    grid = tile(tiles, L, H, gutter=24, bg='#0a0c14', outer=0)
    img[110:110 + grid.shape[0], 40:40 + grid.shape[1]] = grid
    row = int(np.searchsorted(steps[::4], st, side='right'))
    for j in range(2):
        lm = looms[j].copy()
        lm[row * 2:] = hexrgb('#0a0c14')
        lm = lm[:680, :]
        img[120:120 + lm.shape[0], 1400 + j * 260:1400 + j * 260 + lm.shape[1]] = lm
    strip_ = base.copy()
    cx = int(x0 + (x1 - x0) * st / steps[-1])
    strip_[:, max(0, cx - 1):cx + 2] = hexrgb('#ff6b5a')
    img[1080 - Hs:, (1920 - Ws) // 2:(1920 - Ws) // 2 + Ws] = strip_
    img = text_at(img, [(40, 50, 'Induction heads forming  |  2-layer attention-only transformer', 'lm'),
                        (40, 88, f'step {st:5d}   rows = query, columns = key, probe = 16 random tokens x 4', 'lm'),
                        (40, 110 + grid.shape[0] + 30, f'top: layer 0, bottom: layer 1, heads 0-3;  '
                         f'L1H{IND[1]} = induction, L0H{PREV[1]} = previous token', 'lm'),
                        (1400, 50, f'loom  L1H{IND[1]}      L0H{PREV[1]}', 'lm'),
                        (1400, 88, 'down = time, across = look-back', 'lm')],
                  fg='#e6dcc0', size=22)
    save(img, f'{tmp}/f{fi:04d}.png')


def movie(fps=10):
    ps = an['probe_steps']
    # frame schedule: every 5 steps inside [P10-300, P90+300], every 40 steps outside
    fr = [s_ for s_ in ps if (P10 - 300 <= s_ <= P90 + 300) or s_ % 40 == 0]
    tmp = 'cache/frames_toy'
    os.makedirs(tmp, exist_ok=True)
    S = an['offset_spectrum'].astype(np.float64)
    # loss strip base (matplotlib once)
    fig, ax = plt.subplots(figsize=(18.4, 2.6), dpi=100, facecolor='#0a0c14')
    ax.set_facecolor('#0a0c14')
    ax.plot(steps, log['loss_rep_later'], color='#ffd27f', lw=2)
    ax.plot(steps, log['loss_markov_copied'], color='#8fb3e8', lw=2)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#55534c')
    ax.tick_params(colors='#9d998c', labelsize=10)
    ax.set_xlim(0, steps[-1])
    ax.text(steps[-1] * 0.99, 4.6, 'loss on repeats 2-4 of random tokens', color='#ffd27f', ha='right', fontsize=11)
    ax.text(steps[-1] * 0.99, 3.6, 'loss on copied spans in Markov text', color='#8fb3e8', ha='right', fontsize=11)
    ax.set_xlabel('training step', color='#9d998c', fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.25)
    fig.canvas.draw()
    base = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255
    x0, x1 = ax.transData.transform((0, 0))[0], ax.transData.transform((steps[-1], 0))[0]
    plt.close(fig)
    Hs, Ws = base.shape[:2]
    looms = [loom_img(S, l, h, s=(2, 2)) for (l, h) in [IND, PREV]]
    _CTX.update(dict(ps=ps, fr=fr, tmp=tmp, looms=looms, base=base, x0=x0, x1=x1, Hs=Hs, Ws=Ws))
    import multiprocessing as mp
    with mp.get_context('fork').Pool(4) as pool:
        pool.map(_render_frame, range(len(fr)), chunksize=4)
    n = len(fr)
    for j in range(fps * 2):                                  # hold last frame 2 s
        os.link(f'{tmp}/f{n - 1:04d}.png', f'{tmp}/f{n + j:04d}.png')
    mp4, gif = f'{OUT}/toy_induction_forming.mp4', f'{OUT}/toy_induction_forming.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{tmp}/f%04d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', mp4], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{tmp}/f%04d.png',
                    '-vf', 'fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
                    gif], check=True)
    subprocess.run(['rm', '-rf', tmp])
    print('movie frames', n, os.path.getsize(mp4) // 1024, 'KB', os.path.getsize(gif) // 1024, 'KB')


if __name__ == '__main__':
    which = sys.argv[1:] or ['curves', 'strip', 'loom', 'movie']
    for w in which:
        globals()[w]()
