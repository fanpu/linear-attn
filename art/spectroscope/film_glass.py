"""Film 1, "Stained glass": every recorded matrix of the GPT as a pane in a window, one window per
optimiser, lit by its own spectrum and animated over the recorded checkpoints.

    .venv/bin/python spectroscope/film_glass.py [--set main|long] [--gamma 1] [--still IDX]

Window layout (declared): read bottom-up like the network. Bottom row: emb, pos (input). Then layers
0..5 upward. Each layer band: four rows q, k, v, o; in each row the whole matrix first, a gap, then
its six head slices h0..h5. To the right, the MLP: up and down as two tall panes. Top row: the
output head. Every pane is the integrated colour of its spectrum (faithful, gamma 1 unless labelled)
with the dispersed spectrum itself set along its lower edge (400 nm left, 700 nm right, brightness =
expose(power); see filmlib). Leading between panes is the ground. One frame per checkpoint.
"""
import argparse, os
import numpy as np
from PIL import Image, ImageDraw
import filmlib as F

ap = argparse.ArgumentParser()
ap.add_argument('--set', default='main')
ap.add_argument('--gamma', type=float, default=1.0)
ap.add_argument('--still', type=int, nargs='*', default=None, help='only write PNG stills at these frame indices')
ap.add_argument('--fps', type=int, default=30)
ap.add_argument('--crf', type=int, default=21)
ap.add_argument('--bloom', type=float, default=0.15)
ap.add_argument('--pane', default='profile', choices=['flat', 'profile'])
ap.add_argument('--energy', type=int, default=0, help='pane brightness also scales with log energy growth')
ap.add_argument('--line', type=int, default=8, help='rainbow line height (px), drawn only on panes taller than 5x this')
ap.add_argument('--lead', type=float, default=200, help='leading width = window width / lead')
args = ap.parse_args()

W, Hh = 2560, 1440
GROUND = np.array([17, 16, 15]) / 255.0
GROUND_LIN = F.srgb_to_lin(GROUND)
GREY = (150, 146, 140)
DIM = (96, 93, 89)

if args.set == 'main':
    RUNSPEC = [('adamw', 'AdamW', 80, 150, 900, 1120),
               ('muon', 'Muon', 1060, 150, 900, 1120),
               ('sgd', 'SGD + momentum', 2040, 150, 440, 520),
               ('null', 'NULL: AdamW-sized steps, random directions', 2040, 750, 440, 520)]
    TITLE = 'SPECTROSCOPE / STAINED GLASS    every weight matrix of a 6-layer GPT as a pane lit by its own singular-value spectrum'
else:
    RUNSPEC = [('adamw10k_wd', 'AdamW + weight decay 0.1', 300, 150, 900, 1120),
               ('muon10k_wd', 'Muon + weight decay 0.015 (same per-step shrink)', 1360, 150, 900, 1120)]
    TITLE = 'SPECTROSCOPE / STAINED GLASS, 10,000 steps with weight decay    every weight matrix of a 6-layer GPT as a pane lit by its own singular-value spectrum'


def window_tiles(x0, y0, w, h):
    """[(row_index, x, y, w, h, band_h)] for one window, in pixels."""
    g = max(2, int(round(w / args.lead)))
    top_h = bot_h = int(0.06 * h)
    G = 3 * g
    band_total = h - top_h - bot_h - 2 * G
    bh = (band_total - 5 * G) / 6
    sub = (bh - 3 * g) / 4
    u = (w - g * (5 + G / g * 2 + 1)) / (7 + 2.8)
    band_px = args.line
    out = []
    ys = y0 + top_h + G
    for li in range(6):
        lay = 5 - li
        by = ys + li * (bh + G)
        for si, t in enumerate('qkvo'):
            yy = by + si * (sub + g)
            xx = x0
            out.append((F.RI[f'L{lay}.{t}'], xx, yy, u, sub, band_px))
            xx += u + G
            for hd in range(6):
                out.append((F.RI[f'L{lay}.{t}.h{hd}'], xx, yy, u, sub, band_px))
                xx += u + g
        mx = x0 + 7 * u + 5 * g + 2 * G
        out.append((F.RI[f'L{lay}.up'], mx, by, 1.4 * u, bh, band_px))
        out.append((F.RI[f'L{lay}.down'], mx + 1.4 * u + g, by, w - (mx - x0) - 1.4 * u - g, bh, band_px))
    out.append((F.RI['head'], x0, y0, w, top_h, band_px))
    ey = y0 + h - bot_h
    out.append((F.RI['emb'], x0, ey, 0.7 * w, bot_h, band_px))
    out.append((F.RI['pos'], x0 + 0.7 * w + G, ey, w - 0.7 * w - G, bot_h, band_px))
    return [(r, int(round(x)), int(round(y)), int(round(x + ww)) - int(round(x)), int(round(y + hh)) - int(round(y)), b)
            for r, x, y, ww, hh, b in out]


runs = [F.load_run(r, args.gamma) for r, *_ in RUNSPEC]
T = min(len(R['steps']) for R in runs)
NB = F.S.NW
RB = F.rainbow(NB).astype(np.float32)

# pixel index maps: which (global tile, wavelength bin) each pixel shows
body_pix, body_g, body_k, band_pix, band_g, band_k = [], [], [], [], [], []
tiles = []   # (run_i, row)
for ri, (run, name, x0, y0, w, h) in enumerate(RUNSPEC):
    for (row, x, y, tw, th, b) in window_tiles(x0, y0, w, h):
        gi = len(tiles); tiles.append((ri, row))
        yy, xx = np.mgrid[y:y + th, x:x + tw]
        isband = (yy >= y + th - b) & (th > 5 * b)
        flat = yy * W + xx
        body_pix.append(flat[~isband]); body_g.append(np.full((~isband).sum(), gi))
        k = ((xx - x + 0.5) / tw * NB).astype(int).clip(0, NB - 1)
        body_k.append(k[~isband])
        band_pix.append(flat[isband]); band_g.append(np.full(isband.sum(), gi)); band_k.append(k[isband])
body_pix, body_g, body_k = np.concatenate(body_pix), np.concatenate(body_g), np.concatenate(body_k)
band_pix, band_g, band_k = np.concatenate(band_pix), np.concatenate(band_g), np.concatenate(band_k)
tile_run = np.array([t[0] for t in tiles]); tile_row = np.array([t[1] for t in tiles])
print(f'{len(tiles)} panes, {len(body_pix) + len(band_pix)} lit pixels, {T} checkpoints')

ticks_steps = [1, 10, 100, 1000, 10000]


def frame(t):
    lin = np.empty((Hh * W, 3), np.float32); lin[:] = GROUND_LIN
    col = np.stack([runs[r]['lin'][row, t] for r, row in zip(tile_run, tile_row)])            # (G, 3)
    P = np.stack([runs[r]['P'][row, t] for r, row in zip(tile_run, tile_row)])                # (G, NB)
    if args.pane == 'flat':
        lin[body_pix] = col[body_g]
    else:   # the pane's own integrated colour, brightness profile across it = the power at each wavelength
        # brightness across the pane = power / the pane's own 98th-percentile power (clipped at 1): linear,
        # so a flat (white) spectrum is an evenly lit pane and a blue-shifted one fades toward 700 nm
        ref = np.quantile(P, 0.98, axis=1)[:, None]
        prof = np.minimum(P / np.maximum(ref, 1e-6), 1.0)
        lin[body_pix] = col[body_g] * prof[body_g, body_k][:, None]
    lin[band_pix] = F.expose(P[band_g, band_k])[:, None] * RB[band_k]
    lin = lin.reshape(Hh, W, 3)
    if args.bloom > 0:
        lin = F.bloom(lin, args.bloom)
    im = Image.fromarray(F.to8(lin))
    d = ImageDraw.Draw(im)
    d.text((80, 50), TITLE, fill=GREY, font=F.font(26))
    for R, (run, name, x0, y0, w, h) in zip(runs, RUNSPEC):
        big = w > 500
        fs = 30 if big else 20
        d.text((x0, y0 - fs - 14), name, fill=(215, 210, 202), font=F.font(fs))
        sh = R['shift'][F.WHOLE_HIDDEN, t].mean()
        tl = R['train_loss'][t]
        d.text((x0, y0 + h + 12), f'hidden-matrix shift {sh:+5.1f} nm' + (f'    train loss {tl:.2f}' if np.isfinite(tl) else ''),
               fill=GREY, font=F.font(20 if big else 16, mono=True))
    # clock: checkpoint index is ~log time; ticks at decades
    st = runs[0]['steps']
    y = 1360; x0, x1 = 80, 2480
    d.line([(x0, y), (x1, y)], fill=DIM, width=2)
    fx = x0 + (x1 - x0) * t / (T - 1)
    d.line([(x0, y), (fx, y)], fill=(215, 210, 202), width=4)
    for s in ticks_steps:
        if s <= st[T - 1]:
            i = int(np.searchsorted(st[:T], s))
            tx = x0 + (x1 - x0) * i / (T - 1)
            d.line([(tx, y - 8), (tx, y + 8)], fill=DIM, width=2)
            d.text((tx - 10, y + 12), f'{s:,}', fill=DIM, font=F.font(16))
    d.text((x1 - 330, y - 56), f'step {int(st[t]):>6,d} of {int(st[T-1]):,}', fill=(215, 210, 202), font=F.font(28, mono=True))
    d.text((x0, y + 40 if y + 40 < Hh - 20 else Hh - 30),
           'pane = integrated colour (gamma %g%s), lit across its width by the power at 400 -> 700 nm; lower edge of the large panes = the dispersed spectrum. bottom-up: emb, pos | layers 0-5: q,k,v,o whole + 6 heads, mlp up, down | head. time: log after step 150'
           % (args.gamma, ', faithful' if args.gamma == 1 else ', DECLARED EXAGGERATION'), fill=DIM, font=F.font(16))
    return np.asarray(im)


tag = f'{args.set}_g{args.gamma:g}_{args.pane}'
if args.still is not None:
    for i in args.still:
        p = os.path.join(F.FILMCACHE, f'glass_{tag}_t{i}.png')
        Image.fromarray(frame(i)).save(p)
        print(p)
else:
    out = os.path.join(F.GAL, f'film_glass_{tag}.mp4')
    enc = F.Encoder(out, W, Hh, args.fps, args.crf)
    hold0, hold1 = int(1.0 * args.fps), int(3.5 * args.fps)
    order = [0] * hold0 + list(range(T)) + [T - 1] * hold1
    last = None
    for n, t in enumerate(order):
        if t != last:
            fr = frame(t); last = t
        enc.write(fr)
        if n % 100 == 0:
            print(n, len(order), flush=True)
    enc.close()
    Image.fromarray(fr).save(os.path.join(F.GAL, f'still_glass_{tag}_final.png'), optimize=True)
    print(out, os.path.getsize(out) / 1e6, 'MB')
