"""Film 2, "Spectrograph": six chosen matrices, each read out like a spectrometer over training.

    .venv/bin/python spectroscope/film_spectrograph.py [--still IDX ...]

Per column (one matrix):
  * swatch: the integrated colour now (faithful, gamma 1), with its shift in nm;
  * trace: the measured spectral power P(lambda) now, log scale, dashed line at P = 1 (the shuffled
    null: white light);
  * band: the dispersed spectrum now, 400 -> 700 nm, each wavelength in its hue with brightness
    min(P / 5, 1): linear, a flat spectrum (white light) at 0.2, clipped at five times flat (declared);
  * waterfall: the same band for every checkpoint so far, time running down the page; the narrow
    strip on its left is the integrated colour at each of those checkpoints.
"""
import argparse, os
import numpy as np
from PIL import Image, ImageDraw
import filmlib as F

ap = argparse.ArgumentParser()
ap.add_argument('--still', type=int, nargs='*', default=None)
ap.add_argument('--fps', type=int, default=30)
ap.add_argument('--crf', type=int, default=21)
ap.add_argument('--set', default='main')
args = ap.parse_args()

W, Hh = 2560, 1440
PSAT = 5.0   # linear brightness = P / 5, clipped at 1: white light (P = 1) sits at 0.2
GROUND_LIN = F.srgb_to_lin(np.array([12, 11, 10]) / 255.0)
GREY, DIM, LIGHT = (150, 146, 140), (90, 87, 83), (220, 215, 207)

if args.set == 'main':
    PICK = [('adamw', 'L1.up', 'AdamW', 'layer 1 mlp up'), ('muon', 'L1.up', 'Muon', 'layer 1 mlp up'),
            ('null', 'L1.up', 'NULL', 'layer 1 mlp up'), ('sgd', 'L1.o', 'SGD + momentum', 'layer 1 attn out (o)'),
            ('adamw', 'head', 'AdamW', 'output head'), ('muon', 'L0.v.h4', 'Muon', 'layer 0 value, head 4 (reddest)')]
    TITLE = 'SPECTROSCOPE / SPECTROGRAPH    six weight matrices of a 6-layer GPT, their singular-value spectra dispersed as light, over training'
else:
    PICK = [('adamw10k_wd', 'L1.up', 'AdamW + wd', 'layer 1 mlp up'), ('muon10k_wd', 'L1.up', 'Muon + wd', 'layer 1 mlp up'),
            ('adamw10k_wd', 'L0.o', 'AdamW + wd', 'layer 0 attn out (o)'), ('muon10k_wd', 'L0.o', 'Muon + wd', 'layer 0 attn out (o), reddest'),
            ('adamw10k_wd', 'head', 'AdamW + wd', 'output head'), ('muon10k_wd', 'head', 'Muon + wd', 'output head (AdamW-trained)')]
    TITLE = 'SPECTROSCOPE / SPECTROGRAPH, 10,000 steps with weight decay    six weight matrices, their singular-value spectra dispersed as light'

cache = {}
for run, *_ in PICK:
    if run not in cache:
        cache[run] = F.load_run(run)
T = min(len(cache[r]['steps']) for r, *_ in PICK)
NC = len(PICK)
mx, gap = 80, 50
cw = (W - 2 * mx - (NC - 1) * gap) // NC
strip_w = 22
band_w = cw - strip_w - 8
RB = F.rainbow(band_w).astype(np.float32)
Y_SW, SW_H = 190, 120
Y_TR, TR_H = 340, 170
Y_BAND, BAND_H = 530, 70
Y_WF = 625
WF_H = Hh - 90 - Y_WF
rows_of_t = np.minimum((np.arange(WF_H) / WF_H * T).astype(int), T - 1)   # waterfall pixel row -> checkpoint
LOGLO, LOGHI = -1.5, 2.3


def resample(P):
    """(..., NW) -> (..., band_w) by nearest bin."""
    k = ((np.arange(band_w) + 0.5) / band_w * F.S.NW).astype(int)
    return P[..., k]


cols = []
for run, row, name, what in PICK:
    R = cache[run]; i = F.RI[row]
    Pb = resample(R['P'][i, :T])                                  # (T, band_w)
    cols.append(dict(Pb=Pb, band=(np.minimum(Pb / PSAT, 1.0)[..., None] * RB[None]).astype(np.float32), col=R['lin'][i, :T],
                     shift=R['shift'][i, :T], P=R['P'][i, :T], name=name, what=what, run=run))


def frame(t):
    lin = np.empty((Hh, W, 3), np.float32); lin[:] = GROUND_LIN
    for c, C in enumerate(cols):
        x0 = mx + c * (cw + gap); xb = x0 + strip_w + 8
        lin[Y_SW:Y_SW + SW_H, x0:x0 + cw] = C['col'][t]
        lin[Y_BAND:Y_BAND + BAND_H, xb:xb + band_w] = C['band'][t]
        lin[Y_BAND:Y_BAND + BAND_H, x0:x0 + strip_w] = C['col'][t]
        n = int(np.searchsorted(rows_of_t, t, side='right'))           # waterfall rows revealed so far
        if n:
            lin[Y_WF:Y_WF + n, xb:xb + band_w] = C['band'][rows_of_t[:n]]
            lin[Y_WF:Y_WF + n, x0:x0 + strip_w] = C['col'][rows_of_t[:n]][:, None]
    lin = F.bloom(lin, 0.12, radius=12)
    im = Image.fromarray(F.to8(lin)); d = ImageDraw.Draw(im)
    d.text((mx, 60), TITLE, fill=GREY, font=F.font(26))
    st = cache[PICK[0][0]]['steps']
    d.text((W - mx - 380, 52), f'step {int(st[t]):>6,d} of {int(st[T-1]):,}', fill=LIGHT, font=F.font(30, mono=True))
    for c, C in enumerate(cols):
        x0 = mx + c * (cw + gap); xb = x0 + strip_w + 8
        d.text((x0, 125), C['name'], fill=LIGHT, font=F.font(26))
        d.text((x0, 158), C['what'], fill=GREY, font=F.font(18))
        sh = C['shift'][t]
        lum = (F.to8(C['col'][t][None, None])[0, 0] @ np.array([0.2126, 0.7152, 0.0722])) / 255
        d.text((x0 + 12, Y_SW + SW_H - 36), f'{sh:+6.1f} nm', fill=(20, 20, 20) if lum > 0.5 else LIGHT, font=F.font(24, mono=True))
        # trace box
        d.rectangle([xb, Y_TR, xb + band_w, Y_TR + TR_H], outline=(40, 38, 36))
        yl = lambda v: Y_TR + TR_H - (np.log10(v) - LOGLO) / (LOGHI - LOGLO) * TR_H
        for v, lab in ((0.1, '0.1'), (1, '1'), (10, '10'), (100, '100')):
            y = yl(v)
            for xx in range(xb, xb + band_w, 10 if v == 1 else 4):
                d.line([(xx, y), (xx + (5 if v == 1 else 1), y)], fill=(110, 106, 100) if v == 1 else (45, 43, 41))
            d.text((x0 - 4, y - 8), lab, fill=DIM, font=F.font(13))
        P = np.clip(C['Pb'][t], 10 ** LOGLO, 10 ** LOGHI)
        pts = [(xb + j, yl(p)) for j, p in enumerate(P)]
        d.line(pts, fill=LIGHT, width=2)
        d.text((xb, Y_TR + TR_H + 4), '400', fill=DIM, font=F.font(13))
        d.text((xb + band_w // 2 - 12, Y_TR + TR_H + 4), '550 nm', fill=DIM, font=F.font(13))
        d.text((xb + band_w - 26, Y_TR + TR_H + 4), '700', fill=DIM, font=F.font(13))
    # waterfall time ticks on the far left
    for s in (10, 100, 1000, 10000):
        if s <= st[T - 1]:
            i = int(np.searchsorted(st[:T], s)); y = Y_WF + i / T * WF_H
            d.text((18, y - 8), f'{s:,}', fill=DIM, font=F.font(14))
    d.text((18, Y_WF - 26), 'step', fill=DIM, font=F.font(14))
    d.text((mx, Hh - 60), 'top to bottom: integrated colour now (gamma 1, faithful) | spectral power now, log scale, dashed = 1 = the shuffled null (white light) | '
           'the dispersed spectrum now, linear brightness P/5 (clipped at 1) | every checkpoint so far, time downward (log after step 150); left strip = integrated colour',
           fill=DIM, font=F.font(16))
    return np.asarray(im)


tag = args.set
if args.still is not None:
    for i in args.still:
        p = os.path.join(F.FILMCACHE, f'spectrograph_{tag}_t{i}.png'); Image.fromarray(frame(i)).save(p); print(p)
else:
    out = os.path.join(F.GAL, f'film_spectrograph_{tag}.mp4')
    enc = F.Encoder(out, W, Hh, args.fps, args.crf)
    order = [0] * int(1.0 * args.fps) + list(range(T)) + [T - 1] * int(4 * args.fps)
    last = None
    for n, t in enumerate(order):
        if t != last:
            fr = frame(t); last = t
        enc.write(fr)
        if n % 100 == 0:
            print(n, len(order), flush=True)
    enc.close()
    Image.fromarray(fr).save(os.path.join(F.GAL, f'still_spectrograph_{tag}_final.png'), optimize=True)
    print(out, os.path.getsize(out) / 1e6, 'MB')
