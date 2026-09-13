"""'Descent' poster: every zoom keyframe as a panel in snake order, with the next window
outlined on each panel and connector lines to the next panel (classic zoom-figure idiom).

  python render_descent.py zoomAB [--style spectral|aurora_ember] [--cols 7] [--panel 480]
Measured: sign + within-plate rank of the convergence measure per panel, window geometry.
Aesthetic: layout, ground, line colour, colour map.
"""
import argparse
import glob

import numpy as np
from PIL import Image, ImageDraw

import styles as S
from pages import font, MONO, SERIF_B
from render_windows import load

p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('--style', default='spectral')
p.add_argument('--cols', type=int, default=7)
p.add_argument('--panel', type=int, default=480)
args = p.parse_args()

if args.style == 'spectral':
    colf = S.spectral
else:
    import sys
    sys.path.insert(0, '/home/fzeng/ml/research/art/color-research')
    import palettes as P
    colf = lambda m: (np.clip(P.render_split(m, args.style, near_boundary='large')[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)

ks = sorted(int(f[-7:-4]) for f in glob.glob(f'cache/zoom_{args.tag}/kf_*.npz'))
ws = [load(f'{args.tag}:{k}') for k in ks]
n = len(ws); C = args.cols; Rr = (n + C - 1) // C
Pn = args.panel; gx = 110; gy = 150; top = 200; left = 90
W = left * 2 + C * Pn + (C - 1) * gx
H = top + Rr * Pn + (Rr - 1) * gy + 120
BG = (14, 12, 16); LINE = (232, 226, 214); DIM = (150, 142, 134)
page = Image.new('RGB', (W, H), BG)
d = ImageDraw.Draw(page)
hw0 = ws[0]['hw']


def pos(i):
    r, c = divmod(i, C)
    if r % 2 == 1:
        c = C - 1 - c
    return left + c * (Pn + gx), top + r * (Pn + gy), r, c


def box_px(w, v):
    """window v inside panel of window w -> (x0, y0, x1, y1) in panel pixels (top = high eta1)"""
    s = Pn / (2 * w['hw'])
    x0 = (v['c0'] - v['hw'] - (w['c0'] - w['hw'])) * s
    x1 = (v['c0'] + v['hw'] - (w['c0'] - w['hw'])) * s
    y0 = ((w['c1'] + w['hw']) - (v['c1'] + v['hw'])) * s
    y1 = ((w['c1'] + w['hw']) - (v['c1'] - v['hw'])) * s
    return x0, y0, x1, y1


d.text((left, 50), 'Descent into the trainability boundary', font=font(SERIF_B, 64), fill=LINE)
d.text((left, 128), f'{n} nested windows, magnification 10^0 to 10^{np.log10(hw0 / ws[-1]["hw"]):.1f}; '
       f'each panel 256x256 tanh networks, 500 steps, float64; outline = next window',
       font=font(MONO, 28), fill=DIM)
boxes = []
for i, w in enumerate(ws):
    x, y, r, c = pos(i)
    mag = np.log10(hw0 / w['hw'])
    d.text((x, y + Pn + 14), f'{i + 1:02d}   x10^{mag:.1f}', font=font(MONO, 30), fill=LINE)
    if i + 1 < n:
        bx0, by0, bx1, by1 = box_px(w, ws[i + 1])
        B = [x + bx0, y + by0, x + bx1, y + by1]
        # small boxes: draw at least 7 px so they stay visible
        cx, cy = (B[0] + B[2]) / 2, (B[1] + B[3]) / 2
        hb = max((B[2] - B[0]) / 2, 7)
        B = [cx - hb, cy - hb, cx + hb, cy + hb]
        boxes.append(B)
        nx, ny, nr, nc = pos(i + 1)
        if nr == r:   # same row: connect box edge to facing edge of next panel
            if nc > c:
                pairs = [((B[2], B[1]), (nx, ny)), ((B[2], B[3]), (nx, ny + Pn))]
            else:
                pairs = [((B[0], B[1]), (nx + Pn, ny)), ((B[0], B[3]), (nx + Pn, ny + Pn))]
        else:         # row change: box bottom to next panel top
            pairs = [((B[0], B[3]), (nx, ny)), ((B[2], B[3]), (nx + Pn, ny))]
        for a, b in pairs:
            d.line([a, b], fill=DIM, width=2)
for i, w in enumerate(ws):
    x, y, r, c = pos(i)
    page.paste(Image.fromarray(colf(w['M'])).resize((Pn, Pn), Image.NEAREST), (x, y))
for B in boxes:
    d.rectangle(B, outline=(0, 0, 0), width=5)
    d.rectangle(B, outline=LINE, width=2)
page.save(f'gallery/descent_{args.tag}_{args.style}.png')
print('saved', f'gallery/descent_{args.tag}_{args.style}.png', page.size)
