"""Render Part B (Qwen3-0.6B) tapestries from cache/. No model needed.

  python render_qwen.py [piece ...]      pieces: tapestry hero detail stitch
"""
import sys, json
import numpy as np
from textile import *

OUT = 'gallery'
tap = np.load('cache/qwen_tapestry.npz')
main = np.load('cache/qwen_main.npz')
NL, NH = 28, 16
GAMMA = 0.5          # declared: display value = attention ** 0.5


def head_classes(ind, prev, sink):
    """Categorical class per head from MEASURED scores (thresholds declared)."""
    cls = np.full(ind.shape, 3)                      # 3 = mixed/other
    cls[sink > 0.7] = 2                              # sink / 'selvedge'
    cls[prev > 0.35] = 1                             # previous-token
    cls[ind > 0.2] = 0                               # induction
    return cls


def tapestry(key='P12R4', s=5, styles=('weave', 'dark', 'riso', 'indigo', 'quilt')):
    A = tap[f'{key}_attn'].astype(np.float64)        # single sequence, L,H,T,T
    M = tap[f'{key}_mean_attn'].astype(np.float64)
    ind, prev = tap[f'{key}_ind'], tap[f'{key}_prev']
    sink = M[:, :, 1:, 0].mean(-1)
    T = A.shape[-1]
    P = int(key[1:].split('R')[0])
    g = 6
    rl = [f'L{l}' for l in range(NL)]
    cl = [f'H{h}' for h in range(NH)]
    tile_hw = (T * s, T * s)
    mask = np.tril(np.ones((T, T)))
    sub = f'Qwen3-0.6B, random tokens period {P} x {T // P}; rows = query, columns = key; value = attention^{GAMMA}'
    if 'weave' in styles:
        imgs = [weave(to_unit(A[l, h], 0.4) * mask, s=s, warp_lo='#34304a', warp_hi='#ffd98a', weft='#233466',
                      gap='#090b12', float_thr=0.3, seed=l * 16 + h) for l in range(NL) for h in range(NH)]
        img = tile(imgs, NL, NH, gutter=g, bg='#090b12')
        img = frame(img, 'Attention, woven', sub.replace(f'{GAMMA}', '0.4') + '  |  weave = declared aesthetic',
                    rl, cl, tile_hw, g, g, bg='#090b12', fg='#e6dcc0')
        save(img, f'{OUT}/qwen_tapestry_weave_{key}.png')
    if 'dark' in styles:
        imgs = [cmap_img(to_unit(A[l, h], GAMMA), 'inferno', s) for l in range(NL) for h in range(NH)]
        img = tile(imgs, NL, NH, gutter=g, bg='#000000')
        img = frame(img, 'Attention, all 448 heads', sub + '  |  colormap inferno',
                    rl, cl, tile_hw, g, g, bg='#000000', fg='#cfc8b8')
        save(img, f'{OUT}/qwen_tapestry_dark_{key}.png')
    if 'riso' in styles:
        # ink 1 (fluorescent pink): the head's attention pattern
        # ink 2 (blue): flat tint = the head's induction score (mean over 16 draws)
        # ink 3 (yellow): flat tint = previous-token score
        imgs_p, imgs_b, imgs_y = [], [], []
        for l in range(NL):
            for h in range(NH):
                imgs_p.append(np.repeat(np.kron(to_unit(A[l, h], GAMMA), np.ones((s, s)))[..., None], 3, -1))
                imgs_b.append(np.ones((T * s, T * s, 3)) * min(1, ind[l, h] / 0.6) * 0.75)
                imgs_y.append(np.ones((T * s, T * s, 3)) * min(1, prev[l, h] / 0.8) * 0.6)
        lp = tile(imgs_p, NL, NH, gutter=g, bg='#000000')[..., 0]
        lb = tile(imgs_b, NL, NH, gutter=g, bg='#000000')[..., 0]
        ly = tile(imgs_y, NL, NH, gutter=g, bg='#000000')[..., 0]
        img = riso([ly, lb, lp], ['#ffe800', '#0078bf', '#ff48b0'], paper='#f4efe4',
                   offsets=[(0, 0), (4, -3), (-2, 3)], grain=0.25, seed=3)
        img = frame(img, 'Attention, three-ink riso',
                    'pink = attention pattern; blue tint = induction score; yellow tint = previous-token score (all measured)',
                    rl, cl, tile_hw, g, g, bg='#f4efe4', fg='#2a2a2a')
        save(img, f'{OUT}/qwen_tapestry_riso_{key}.png')
    if 'indigo' in styles:
        imgs = [indigo(to_unit(A[l, h], GAMMA), s=s, bleed=0.35, seed=l * 16 + h) for l in range(NL) for h in range(NH)]
        img = tile(imgs, NL, NH, gutter=g, bg='#f1eee6')
        img = frame(img, 'Attention, indigo', sub + '  |  dye bleed 0.35 cell (declared)',
                    rl, cl, tile_hw, g, g, bg='#f1eee6', fg='#15306b')
        save(img, f'{OUT}/qwen_tapestry_indigo_{key}.png')
    if 'quilt' in styles:
        cls = head_classes(ind, prev, sink)
        fabrics = ['#b5412f', '#2f4b7c', '#c99a2e', '#7d8f69']     # declared categorical palette
        light = ['#f0d5cc', '#d3dcea', '#f2e4c2', '#dfe5d6']
        imgs = []
        for l in range(NL):
            for h in range(NH):
                c = cls[l, h]
                W = np.kron(to_unit(A[l, h], GAMMA), np.ones((s, s)))
                base = hexrgb(light[c]) * (1 - W[..., None]) + hexrgb(fabrics[c]) * W[..., None]
                # printed-cotton texture + quilting stitch border (declared)
                yy, xx = np.mgrid[0:T * s, 0:T * s]
                tex = 1 - 0.035 * ((yy + xx) % 3 == 0)
                im = base * tex[..., None]
                b = 3
                dash = ((xx + yy) // 6) % 2 == 0
                edge = ((yy == b) | (yy == T * s - 1 - b) | (xx == b) | (xx == T * s - 1 - b)) & dash
                im[edge] = hexrgb('#fbf6ea')
                imgs.append(im)
        img = tile(imgs, NL, NH, gutter=g, bg='#e9e0cc')
        img = frame(img, 'Attention, patchwork',
                    'patch colour = head class from measured scores: red induction>0.2, blue prev-token>0.35, ochre sink>0.7, sage other',
                    rl, cl, tile_hw, g, g, bg='#e9e0cc', fg='#3b2f25')
        save(img, f'{OUT}/qwen_tapestry_quilt_{key}.png')
    print('tapestry', key, 'classes:', np.bincount(head_classes(ind, prev, sink).ravel(), minlength=4))


def hero_main():
    """P=50 x 4 repeats, all heads, 1 px per attention cell."""
    A = main['attn'].astype(np.float64)
    T = A.shape[-1]
    g = 4
    imgs = [cmap_img(to_unit(A[l, h], GAMMA), 'magma') for l in range(NL) for h in range(NH)]
    img = tile(imgs, NL, NH, gutter=g, bg='#000000')
    img = frame(img, 'Qwen3-0.6B  |  200 random tokens, period 50, four times',
                'every head of every layer; 1 px = one attention weight (attention^0.5, magma)',
                [f'L{l}' for l in range(NL)], [f'H{h}' for h in range(NH)], (T, T), g, g,
                bg='#000000', fg='#cfc8b8', font_size=18)
    save(img, f'{OUT}/qwen_hero_dark_P50.png')


def detail_top(s=4):
    A = main['attn'].astype(np.float64)
    ind = main['score_ind']
    order = np.argsort(-ind.ravel())[:8]
    heads = [(int(i // NH), int(i % NH)) for i in order]
    T = A.shape[-1]
    imgs = [weave(to_unit(A[l, h], 0.4), s=s, warp_lo='#34304a', warp_hi='#ffd98a', weft='#233466',
                  gap='#090b12', float_thr=0.3, seed=k) for k, (l, h) in enumerate(heads)]
    img = tile(imgs, 2, 4, gutter=24, bg='#090b12', outer=24)
    img = frame(img, 'The eight strongest induction heads, woven',
                'period 50 x 4; induction score = mean attention to the token after the previous occurrence',
                None, [f'L{l}H{h}  ind={ind[l, h]:.2f}' for l, h in heads[:4]], (T * s, T * s), 24, 24,
                bg='#090b12', fg='#e6dcc0')
    save(img, f'{OUT}/qwen_top8_weave_P50.png')
    return heads


def stitch_top(key='P12R4', s=14):
    A = tap[f'{key}_attn'].astype(np.float64)
    ind = tap[f'{key}_ind']
    order = np.argsort(-ind.ravel())[:8]
    heads = [(int(i // NH), int(i % NH)) for i in order]
    T = A.shape[-1]
    cols = ['#d9c7a7', '#8fa9c4', '#3f6e9a', '#b8452f', '#5a1e1b']   # 5 floss colours, declared
    edges = [0.03, 0.1, 0.25, 0.5, 0.8]                                 # on raw attention
    imgs = [cross_stitch(np.digitize(A[l, h], edges), cols, s=s, seed=k) for k, (l, h) in enumerate(heads)]
    img = tile(imgs, 2, 4, gutter=36, bg='#e7dcc4', outer=36)
    img = frame(img, 'Cross-stitch sampler: eight induction heads',
                'period 12 x 4; floss by raw attention bins 0.03/0.10/0.25/0.50/0.80 (declared quantisation)',
                None, [f'L{l}H{h}' for l, h in heads[:4]], (T * s, T * s), 36, 36, bg='#e7dcc4', fg='#3b2f25')
    save(img, f'{OUT}/qwen_top8_crossstitch_P12.png')


if __name__ == '__main__':
    which = sys.argv[1:] or ['tapestry', 'hero', 'detail', 'stitch']
    if 'tapestry' in which:
        tapestry()
    if 'hero' in which:
        hero_main()
    if 'detail' in which:
        detail_top()
    if 'stitch' in which:
        stitch_top()
