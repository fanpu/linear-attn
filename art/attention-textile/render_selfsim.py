"""Self-similar words: Qwen3-0.6B induction heads reading Thue-Morse, period-doubling,
Fibonacci, Cantor, Sierpinski-doubling and nested words (letters -> blocks of 8 random tokens).

Display matrix (letter level) = mean over 4 random token draws and the top-4 induction
heads of [query-block mean, key-block sum] attention; the first-letter sink column and the
self-letter diagonal are blanked; scaled by its 99.5th percentile, ** 0.6 (all declared).

  python render_selfsim.py
"""
import json
import numpy as np
from textile import *
from analyze_selfsim import ideal  # noqa: E402  (pure function; module runs its analysis on import)

OUT = 'gallery'
K = 8
d = np.load(f'cache/qwen_selfsim_k{K}.npz')
meta = json.loads(str(d['meta']))
metrics = json.load(open('cache/selfsim_metrics.json'))
Q = {'thue-morse': 2, 'period-doubling': 2, 'fibonacci': 2, 'cantor': 3, 'sierpinski-doubling': 2,
     'nested': 3, 'control iid a/b': 2, 'control cantor-shuffled': 3}


def letter_matrix(wi):
    A = d[f'w{wi}_letter'].astype(np.float64)[:, :4].mean((0, 1))
    A[:, 0] = 0
    np.fill_diagonal(A, 0)
    tri = np.tril_indices(A.shape[0], -1)
    A = A / np.quantile(A[tri], 0.995)
    return np.clip(A, 0, 1) ** 0.6


def up(W, size):
    s = max(1, size // W.shape[0])
    return np.kron(W, np.ones((s, s))), s


def zooms(W, q, levels=4):
    n = W.shape[0]
    return [W[:max(4, n // q ** z), :max(4, n // q ** z)] for z in range(levels)]


def render_cell(W, style, size, C=None, seed=0):
    n = W.shape[0]
    s = max(1, size // n)
    if style == 'dark':
        img = cmap_img(W, 'magma', s)
    elif style == 'weave':
        img = weave(W, s=s, warp_lo='#3a2f45', warp_hi='#ffd27f', weft='#1f2f5c', gap='#0a0c14',
                    float_thr=0.3, seed=seed, fibre=0.05)
    elif style == 'indigo':
        img = indigo(W, s=s, bleed=0.25, absorb=2.5, seed=seed)
    elif style == 'riso':
        ub = np.kron(C[:n, :n].astype(float) * 0.5, np.ones((s, s)))
        up_ = np.kron(W, np.ones((s, s)))
        img = riso([ub, up_], ['#0078bf', '#ff48b0'], paper='#f4efe4', offsets=[(0, 0), (max(1, s // 3), -max(1, s // 3))],
                   grain=0.2, dot=2, seed=seed)
    out = np.ones((size, size, 3)) * hexrgb(STY[style][0])
    h = min(size, img.shape[0])
    out[:h, :h] = img[:h, :h]
    return out


STY = {'weave': ('#0a0c14', '#e6dcc0'), 'dark': ('#000000', '#cfc8b8'),
       'indigo': ('#f1eee6', '#15306b'), 'riso': ('#f4efe4', '#2a2a2a')}
NOTE = {'weave': 'weave: warp colour/width/float encode the display value (declared)',
        'dark': 'magma',
        'indigo': 'single indigo dye, bleed 0.25 cell (declared)',
        'riso': 'blue = ideal "every previous occurrence" matrix from the word; pink = measured attention'}


def plate(style, size=520, words=None):
    words = words or list(range(len(meta)))
    tiles, labels = [], []
    for wi in words:
        m = meta[wi]
        W = letter_matrix(wi)
        C = ideal(m['word'])
        q = Q[m['name']]
        for z, Wz in enumerate(zooms(W, q)):
            tiles.append(render_cell(Wz, style, size, C, seed=wi * 10 + z))
            n = Wz.shape[0]
            if z == 0:
                mt = metrics[f"k{K}/{m['name']}"]
                extra = f"  decimation r={mt['decim_r_top4']:.2f}" if 'decim_r_top4' in mt else ''
                labels.append(f"{m['name']}  ({m['T']} tok){extra}")
            else:
                labels.append(f"first {n} letters (x{q ** z})")
    bg, fg = STY[style]
    gut, out = 70, 40
    img = tile(tiles, len(words), 4, gutter=gut, bg=bg, outer=out)
    img = label_tiles(img, labels, len(words), 4, (size, size), gut, out, fg=fg, size=19)
    img = frame(img, 'Self-similar words, read by induction heads  (Qwen3-0.6B, top-4 heads, letter level)',
                NOTE[style] + ';  columns zoom into the top-left corner', bg=bg, fg=fg, margin=(120, 40, 80, 40),
                title_size=34)
    save(img, f'{OUT}/selfsim_plate_{style}.png')


def hero_single(name, style, s=8):
    wi = [m['name'] for m in meta].index(name)
    W = letter_matrix(wi)
    C = ideal(meta[wi]['word'])
    n = W.shape[0]
    img = render_cell(W, style, n * s, C, seed=wi)
    bg, fg = STY[style]
    img = frame(img, f'{name}: {n} letters x {K} random tokens, Qwen3-0.6B',
                f'letter-level attention of the top-4 induction heads, 4 token draws; {NOTE[style]}',
                bg=bg, fg=fg, margin=(110, 40, 80, 40), title_size=34, font_size=18)
    save(img, f'{OUT}/selfsim_{name.replace(" ", "_")}_{style}.png')


if __name__ == '__main__':
    for st in ['dark', 'riso', 'weave', 'indigo']:
        plate(st)
    hero_single('cantor', 'weave')
    hero_single('cantor', 'riso')
    hero_single('thue-morse', 'weave')
    hero_single('sierpinski-doubling', 'indigo')
