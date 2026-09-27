"""Spectroscope hero: the four runs as lit panels on a dark ground (light seen in a dark room).
Same cells as wall_g*.png (nearest-neighbour only); the ground and labels are the declared choices.

    .venv/bin/python spectroscope/hero.py
"""
import os, json
import numpy as np
from PIL import Image, ImageDraw
from render import load, strip, font, GAL, HERE, key_commit

RUNS = [('adamw', 'AdamW'), ('muon', 'Muon'), ('sgd', 'SGD + momentum'), ('null', 'NULL: AdamW-sized steps, random directions')]
GROUND = (17, 16, 15)
GREY = (150, 146, 140)
m = json.load(open(os.path.join(HERE, 'metrics.json')))

for g in (1.0, 2.0):
    panels = [strip(load(r)['P1'], g, sx=4, sy=5) for r, _ in RUNS]
    ph, pw = panels[0].shape[:2]
    mx, my, gap, lab = 160, 150, 70, 34
    W = pw + 2 * mx
    Hh = my * 2 + len(panels) * (ph + lab) + (len(panels) - 1) * gap
    im = Image.new('RGB', (W, Hh), GROUND)
    d = ImageDraw.Draw(im)
    y = my
    for (r, name), p in zip(RUNS, panels):
        im.paste(Image.fromarray(p), (mx, y))
        s = m[r]
        d.text((mx, y + ph + 10), f'{name}    final val loss {s["final_val"]:.2f}    mean hidden-matrix shift {s["hidden(whole)"]["shift_nm"]:+.0f} nm',
               fill=GREY, font=font(20))
        y += ph + lab + gap
    d.text((mx, my - 70), 'SPECTROSCOPE   every weight matrix of a 6-layer GPT, at every checkpoint, seen as the light of its singular-value spectrum',
           fill=GREY, font=font(22))
    d.text((mx, Hh - my + 30), f'rows: 183 matrices (per layer q, k, v, o and their 6 heads each; mlp up, down; then emb, pos, head)    columns: 509 checkpoints, step 0 to 2000, log time    '
           + ('gamma 1: faithful' if g == 1 else 'gamma 2: declared exaggeration of spectral shape'), fill=GREY, font=font(17))
    out = os.path.join(GAL, f'hero_dark_g{g:g}.png')
    im.save(out, optimize=True)
    print(out, im.size, 'key commitment %.3f' % key_commit(np.asarray(im)))
