"""Dedicated plates for the quasi-periodic words (Thue-Morse, Fibonacci) at TOKEN level,
plus the exact self-similarity of the ideal Thue-Morse matrix.

Data: cache/qwen_selfsim_k4.npz (letters -> blocks of 4 random tokens, 2048 tokens,
head L20H14 = strongest induction head on the P=50 probe, first token draw).
Display: per-head 99.5th-percentile scaling over causal non-sink cells, ^0.5; the
first-token sink column is blanked (declared). Coarsest zoom is 2x2 area-averaged.

  python render_words.py
"""
import json
import numpy as np
from textile import *

OUT = 'gallery'
d = np.load('cache/qwen_selfsim_k4.npz')
meta = json.loads(str(d['meta']))
names = [m['name'] for m in meta]


def prefix_mask(tok):
    T = len(tok)
    M = np.zeros((T, T), np.float32)
    M[:, 1:] = tok[:, None] == tok[None, :-1]
    return np.tril(M)


def disp(A):
    T = A.shape[0]
    tri = np.tril(np.ones((T, T), bool))
    tri[:, 0] = False
    v = np.percentile(A[tri], 99.5)
    W = np.clip(A / v, 0, 1) ** 0.5
    W[:, 0] = 0
    return W


def pool2(X):
    n = X.shape[0] // 2 * 2
    return X[:n, :n].reshape(n // 2, 2, n // 2, 2).mean((1, 3))


def zoom_panels(W, M, size=1024):
    T = W.shape[0]
    out = []
    for z in range(4):
        n = T // 2 ** z
        w, m = W[:n, :n], M[:n, :n]
        while w.shape[0] > size:
            w, m = pool2(w), pool2(m)
        s = size // w.shape[0]
        out.append((np.kron(w, np.ones((s, s))), np.kron(m, np.ones((s, s))), n))
    return out


def word_plate(name, head=0):
    wi = names.index(name)
    A = d[f'w{wi}_top'][head].astype(np.float64)
    tok = d[f'w{wi}_tokens']
    W = disp(A)
    M = prefix_mask(tok)
    tot = (A * M).sum(1)
    rows = M.sum(1) > 0
    mass = float(tot[rows].mean())
    panels = zoom_panels(W, M)
    for style in ['riso', 'dark']:
        tiles, labels = [], []
        for k, (w, m, n) in enumerate(panels):
            if style == 'riso':
                off = max(1, w.shape[0] // n // 3) if n < 1024 else 1
                im = riso([m * 0.5, w], ['#0078bf', '#ff48b0'], paper='#f4efe4', offsets=[(0, 0), (off, -off)],
                          grain=0.2, dot=2, seed=k)
            else:
                im = cmap_img(w, 'magma')
                blue = np.array([0.15, 0.35, 0.75])
                im = np.clip(im + (m[..., None] * 0.22) * blue * (w[..., None] < 0.15), 0, 1)
            tiles.append(im)
            labels.append(f'first {n} tokens' + ('  (2x2 area-averaged)' if n == 2048 else f'  (x{2048 // n} zoom)'))
        bg, fg = ('#f4efe4', '#2a2a2a') if style == 'riso' else ('#000000', '#cfc8b8')
        gut, out = 80, 50
        img = tile(tiles, 2, 2, gutter=gut, bg=bg, outer=out)
        img = label_tiles(img, labels, 2, 2, (1024, 1024), gut, out, fg=fg, size=22)
        note = ('blue = ideal induction target (key j with token[j-1] = token[query], from the tokens);  pink = measured attention'
                if style == 'riso' else 'magma = measured attention (per-head 99.5 pct, ^0.5);  faint blue = ideal induction target where attention is low')
        img = frame(img, f'{name} word, letters = 4 random tokens, 2048 tokens  |  Qwen3-0.6B L20H14',
                    f'{note};  attention mass on ideal targets = {mass:.2f}', bg=bg, fg=fg,
                    margin=(120, 40, 80, 40), title_size=34, font_size=19)
        save(img, f'{OUT}/word_{name}_{style}.png')
    print(name, 'mass on prefix-match', mass)
    return mass


def tm_word(n):
    return np.array([bin(i).count('1') % 2 for i in range(n)])


def tm_ideal_plate(size=1024):
    """Ideal letter coincidence matrix of Thue-Morse at 8..512 letters, each drawn at the same size."""
    tiles, labels = [], []
    prev = None
    exact = True
    for p in range(3, 10):
        n = 2 ** p
        w = tm_word(n)
        C = (w[:, None] == w[None, :]).astype(float)
        if prev is not None:
            blk = np.block([[prev, 1 - prev], [1 - prev, prev]])
            exact &= bool(np.array_equal(blk, C))
        prev = C
        Cl = np.tril(C, -1)
        s = size // n
        W = np.kron(Cl, np.ones((s, s)))
        im = riso([W * 0.85], ['#0078bf'], paper='#f4efe4', grain=0.12, dot=2, seed=p)
        tiles.append(im)
        labels.append(f'{n} letters')
    # add the measured letter-level matrix at 512 letters for comparison
    wi = names.index('thue-morse')
    L = d[f'w{wi}_letter'].astype(np.float64)[:, :4].mean((0, 1))
    L[:, 0] = 0
    np.fill_diagonal(L, 0)
    tri = np.tril_indices(L.shape[0], -1)
    Wm = np.clip(L / np.quantile(L[tri], 0.995), 0, 1) ** 0.6
    Wm = np.kron(Wm, np.ones((size // Wm.shape[0], size // Wm.shape[0])))
    tiles.append(riso([Wm], ['#ff48b0'], paper='#f4efe4', grain=0.12, dot=2, seed=99))
    labels.append('measured, 512 letters (top-4 heads)')
    gut, out = 80, 50
    img = tile(tiles, 2, 4, gutter=gut, bg='#f4efe4', outer=out)
    img = label_tiles(img, labels, 2, 4, (size, size), gut, out, fg='#2a2a2a', size=22)
    img = frame(img, 'Thue-Morse: the ideal "same letter" matrix is exactly self-similar',
                f'C(2n) = [[C(n), 1-C(n)], [1-C(n), C(n)]] verified for n = 8..256: {exact};  '
                'blue = ideal (lower triangle), pink = measured letter-level attention', bg='#f4efe4', fg='#2a2a2a',
                margin=(120, 40, 80, 40), title_size=34, font_size=19)
    save(img, f'{OUT}/word_thue-morse_ideal_selfsimilar.png')
    print('TM block recursion exact:', exact)
    return exact


if __name__ == '__main__':
    res = {'thue-morse': word_plate('thue-morse'), 'fibonacci': word_plate('fibonacci'),
           'tm_recursion_exact': tm_ideal_plate()}
    json.dump(res, open('cache/words_summary.json', 'w'), indent=1)
