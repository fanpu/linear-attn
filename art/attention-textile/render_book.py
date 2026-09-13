"""Pattern book: one induction head of Qwen3-0.6B on many repetition rhythms.

Same head (L20H14, the strongest induction head on the main probe) for every
swatch. Swatches share a thread gauge (px per token) so longer sequences make
physically larger swatches.

  python render_book.py
"""
import json
import numpy as np
from textile import *

OUT = 'gallery'
book = np.load('cache/qwen_book.npz')
meta = json.loads(str(book['meta']))
HEAD = (20, 14)
G = 0.45                                   # declared display gamma


def att(i):
    return book[f'attn_{i}'][HEAD].astype(np.float64)


def ideal_mask(tok):
    """Where a perfect induction head would look: key j such that tok[j-1] == tok[query]."""
    T = len(tok)
    M = np.zeros((T, T))
    M[:, 1:] = tok[:, None] == tok[None, :-1]
    return np.tril(M)


def ind_score(i):
    A = att(i)
    M = ideal_mask(book[f'tokens_{i}'])
    rows = M.sum(1) > 0
    return float((A * M).sum(1)[rows].mean()) if rows.any() else float('nan')


groups = {
    'rhythm': [0, 1, 2, 3, 4, 5, 9, 13, 14, 15, 16, 17],       # ~200-token swatches
    'repeats': [6, 7, 8, 9],
    'length': [10, 11, 12],
}
labels = {}
for i, m in enumerate(meta):
    s = ind_score(i)
    labels[i] = f"{m['name']}" + (f"  |  prefix-match mass {s:.2f}" if s == s else "  |  (nothing to match)")
    print(i, labels[i])


def swatch(i, style, s=4, crop=200):
    A = att(i)
    tok = book[f'tokens_{i}']
    if crop:
        A = A[:crop, :crop]
        tok = tok[:crop]
    T = A.shape[0]
    W = to_unit(A, G)
    if style == 'weave':
        return weave(W, s=s, warp_lo='#3a2f45', warp_hi='#ffd27f', weft='#1f2f5c', gap='#0a0c14', float_thr=0.3, seed=i)
    if style == 'dark':
        return cmap_img(W, 'magma', s)
    if style == 'indigo':
        return indigo(W, s=s, bleed=0.3, seed=i)
    if style == 'riso':
        up = np.ones((s, s))
        lb = np.kron(ideal_mask(tok) * 0.55, up)
        lp = np.kron(W, up)
        return riso([lb, lp], ['#0078bf', '#ff48b0'], paper='#f4efe4', offsets=[(0, 0), (2, -2)], grain=0.2, dot=2, seed=i)
    raise ValueError(style)


STY = {'weave': ('#0a0c14', '#e6dcc0'), 'dark': ('#000000', '#cfc8b8'),
       'indigo': ('#f1eee6', '#15306b'), 'riso': ('#f4efe4', '#2a2a2a')}
NOTE = {'weave': 'warp colour/width/float = attention^0.45 (weave declared)',
        'dark': 'magma, attention^0.45',
        'indigo': 'single indigo dye, attention^0.45, bleed 0.3 cell (declared)',
        'riso': 'blue = where an ideal induction head looks (from tokens); pink = measured attention'}


def page_rhythm(style, s=4):
    idx = groups['rhythm']
    bg, fg = STY[style]
    imgs = []
    for i in idx:
        im = swatch(i, style, s)
        pad = np.ones((200 * s, 200 * s, 3)) * hexrgb(bg)
        pad[:im.shape[0], :im.shape[1]] = im
        imgs.append(pad)
    gut, out = 90, 60
    img = tile(imgs, 3, 4, gutter=gut, bg=bg, outer=out)
    img = label_tiles(img, [labels[i] for i in idx], 3, 4, imgs[0].shape[:2], gut, out, fg=fg, size=20)
    img = frame(img, 'Pattern book I: rhythms  (Qwen3-0.6B, head L20H14, first 200 tokens)',
                NOTE[style], bg=bg, fg=fg, margin=(120, 40, 80, 40))
    save(img, f'{OUT}/book_rhythm_{style}.png')


def page_series(style, s=4):
    bg, fg = STY[style]
    rows = []
    for key, title in [('repeats', 'repeats: period 20 x 2, 3, 6, 10'), ('length', 'length: period 16, T = 64, 128, 256')]:
        imgs = [swatch(i, style, s, crop=None) for i in groups[key]]
        canvas, boxes = row_bottom(imgs, gap=80, bg=bg, pad=60)
        canvas = text_at(canvas, [(x, y + h + 10, f"T={meta[i]['T']}  mass={ind_score(i):.2f}", 'la')
                                  for (x, y, w, h), i in zip(boxes, groups[key])] +
                         [(60, 30, title, 'la')], fg=fg, size=22)
        extra = np.ones((50, canvas.shape[1], 3)) * hexrgb(bg)
        rows.append(np.concatenate([canvas, extra]))
    W = max(r.shape[1] for r in rows)
    rows = [np.concatenate([r, np.ones((r.shape[0], W - r.shape[1], 3)) * hexrgb(bg)], 1) for r in rows]
    img = np.concatenate(rows)
    img = frame(img, 'Pattern book II: same thread gauge, different lengths', NOTE[style], bg=bg, fg=fg,
                margin=(120, 40, 80, 40))
    save(img, f'{OUT}/book_series_{style}.png')


def page_stitch(s=14, n=56):
    """Cross-stitch: a 56x56 window (rows 144-200, cols 100-156) of each rhythm swatch."""
    idx = groups['rhythm']
    cols = ['#d9c7a7', '#8fa9c4', '#3f6e9a', '#b8452f', '#5a1e1b']
    edges = [0.03, 0.1, 0.25, 0.5, 0.8]
    imgs = []
    for i in idx:
        A = att(i)
        A = A[A.shape[0] - n:, A.shape[0] - n - 44:A.shape[0] - 44]
        imgs.append(cross_stitch(np.digitize(A, edges), cols, s=s, seed=i))
    gut, out = 90, 60
    img = tile(imgs, 3, 4, gutter=gut, bg='#e7dcc4', outer=out)
    img = label_tiles(img, [meta[i]['name'] for i in idx], 3, 4, imgs[0].shape[:2], gut, out, fg='#3b2f25', size=22)
    img = frame(img, 'Pattern book III: sampler (last 56 queries x keys T-100..T-45)',
                'floss bins on raw attention 0.03 / 0.10 / 0.25 / 0.50 / 0.80 (declared quantisation)',
                bg='#e7dcc4', fg='#3b2f25', margin=(120, 40, 80, 40))
    save(img, f'{OUT}/book_sampler_crossstitch.png')


if __name__ == '__main__':
    for st in ['weave', 'dark', 'indigo', 'riso']:
        page_rhythm(st)
        page_series(st)
    page_stitch()
