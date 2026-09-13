"""Render the deep-zoom sequence: plate-book pages, a contact sheet and the continuous
zoom video (MP4 + GIF). CPU only, reads cache/zoom_<tag>/.

  python render_zoom.py zoomA --plates --video
"""
import argparse
import glob
import json
import math
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import styles as S
from common_render import cdf_img
from pages import plate_page, fmt_log_centre, font, MONO, SERIF, SERIF_B

p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('--plates', action='store_true')
p.add_argument('--video', action='store_true')
p.add_argument('--style', default='magma')
p.add_argument('--fps', type=int, default=30)
p.add_argument('--sec', type=float, default=4.0, help='seconds per keyframe transition')
p.add_argument('--size', type=int, default=1080)
p.add_argument('--maxk', type=int, default=None)
p.add_argument('--floor_cut', type=float, default=0.5,
               help='stop the sequence at the first keyframe whose 1-ulp flip fraction of boundary px exceeds this')
args = p.parse_args()
torch.set_num_threads(4)
os.makedirs('gallery', exist_ok=True)

meta = json.load(open(f'cache/zoom_{args.tag}/path.json'))
kfs = []
for f in sorted(glob.glob(f'cache/zoom_{args.tag}/kf_*.npz')):
    d = np.load(f)
    k = int(f[-7:-4])
    e = dict(k=k, M=d['measure'], c0=float(d['c0']), c1=float(d['c1']), hw=float(d['hw']),
             res=int(d['res']), steps=int(d['steps']), nonlin=str(d['nonlin']))
    if k < len(meta['keyframes']):
        e.update({a: b for a, b in meta['keyframes'][k].items() if a.startswith('flip')})
    kfs.append(e)
if args.maxk is not None:
    kfs = kfs[:args.maxk + 1]
floor_k = None
for e in kfs:
    if e.get('flip_frac_edge') is not None and e['flip_frac_edge'] > args.floor_cut:
        floor_k = e['k']
        break
print('keyframes', len(kfs), 'precision-floor keyframe', floor_k)
hw0 = kfs[0]['hw']


def colour(M, ref=None):
    if args.style == 'magma':
        return S.dark_magma(M, ref)
    if args.style == 'fireice':
        return S.dark_fire_ice(M)
    raise ValueError


# ------------------------------------------------------------------ plates
if args.plates:
    outdir = f'gallery/plates_{args.tag}'
    os.makedirs(outdir, exist_ok=True)
    thumbs = []
    prev = None
    for i, e in enumerate(kfs):
        img = colour(e['M'])
        zoom_dec = math.log10(hw0 / e['hw'])
        eta0 = 10 ** e['c0']; eta1 = 10 ** e['c1']
        fl = e.get('flip_frac_edge')
        seeing = (f'seeing: a 1-ulp nudge of both learning rates flips {100*fl:.1f}% of boundary pixels'
                  if fl is not None else '')
        lines = [
            f'centre   log10 eta0 = {fmt_log_centre(e["c0"], e["hw"])}   (eta0 = {eta0:.10e})',
            f'centre   log10 eta1 = {fmt_log_centre(e["c1"], e["hw"])}   (eta1 = {eta1:.10e})',
            f'field width {2*e["hw"]:.3e} decades on each axis   magnification 10^{zoom_dec:.1f} = {hw0/e["hw"]:.3g}x',
            f'{e["res"]}x{e["res"]} independent networks (one per cell), 16-unit {e["nonlin"]}, '
            f'{e["steps"]} steps full-batch GD, float64',
            '~colour: Sohl-Dickstein restretch of sum-of-losses (converged, light) / sum of inverse losses (diverged, dark), magma',
            '~' + seeing,
        ]
        loc = None
        if prev is not None:
            R = prev['res']
            x0 = (e['c0'] - e['hw'] - (prev['c0'] - prev['hw'])) / (2 * prev['hw'])
            x1 = (e['c0'] + e['hw'] - (prev['c0'] - prev['hw'])) / (2 * prev['hw'])
            y0 = (e['c1'] - e['hw'] - (prev['c1'] - prev['hw'])) / (2 * prev['hw'])
            y1 = (e['c1'] + e['hw'] - (prev['c1'] - prev['hw'])) / (2 * prev['hw'])
            loc = (colour(prev['M']), (x0, y0, x1, y1))
        page = plate_page(img, dict(c0=e['c0'], c1=e['c1'], hw=e['hw'],
                                    xlabel='log10 eta0  (input-layer learning rate) ->',
                                    ylabel='log10 eta1  (output-layer learning rate) ^'),
                          lines, 'Trainability, 10^%.1f' % zoom_dec, plate_no=i + 1, locator=loc)
        page.save(f'{outdir}/plate_{i+1:02d}.png')
        thumbs.append(img)
        prev = e
    # contact sheet: all plates, 6 per row
    n = len(thumbs); cols = 6; rows = math.ceil(n / cols)
    T = 384; pad = 24; top = 90
    sheet = Image.new('RGB', (cols * (T + pad) + pad, rows * (T + pad + 40) + pad + top), (244, 240, 230))
    dd = ImageDraw.Draw(sheet)
    dd.text((pad, 30), f'Trainability boundary, zoom sequence "{args.tag}": one plate per half-decade',
            font=font(SERIF_B, 40), fill=(28, 26, 30))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (T + pad); y = top + pad + r * (T + pad + 40)
        sheet.paste(Image.fromarray(t).resize((T, T), Image.NEAREST), (x, y))
        z = math.log10(hw0 / kfs[i]['hw'])
        mark = '  (precision floor)' if floor_k is not None and kfs[i]['k'] >= floor_k else ''
        dd.text((x, y + T + 6), f'{i+1:02d}  10^{z:.1f}{mark}', font=font(MONO, 24), fill=(90, 84, 80))
    sheet.save(f'gallery/zoom_{args.tag}_contact_sheet.png')
    print('plates done')

# ------------------------------------------------------------------ video
if args.video:
    K = len(kfs) if floor_k is None else min(len(kfs), floor_k + 1)
    ref = kfs[K - 1]
    # centres as offsets from the deepest centre (exact float64 differences)
    for e in kfs:
        e['d0'] = e['c0'] - ref['c0']; e['d1'] = e['c1'] - ref['c1']
    rgbs = [torch.from_numpy(colour(e['M'])[::-1].copy()).permute(2, 0, 1).float() / 255 for e in kfs[:K]]
    N = args.size
    frames_dir = f'cache/frames_{args.tag}'
    os.makedirs(frames_dir, exist_ok=True)
    fpt = int(round(args.sec * args.fps))
    grid_off = (torch.arange(N, dtype=torch.float64) + 0.5) / N * 2 - 1

    def window(a, j):
        """his interpolate_history: exponential zoom from window j to j+1 about the fixed
        point of the similarity map."""
        e1, e2 = kfs[j], kfs[j + 1]
        w1, w2 = e1['hw'], e2['hw']
        g = (w2 / w1) ** a
        r = w2 / w1
        cs0 = (e2['d0'] - e1['d0'] * r) / (1 - r)
        cs1 = (e2['d1'] - e1['d1'] * r) / (1 - r)
        return cs0 + (e1['d0'] - cs0) * g, cs1 + (e1['d1'] - cs1) * g, w1 * g

    def sample(j, d0, d1, hw):
        e = kfs[j]
        R = e['res']
        # frame pixel -> normalised coords of keyframe j in [-1,1]
        gx = ((d0 - e['d0']) + grid_off * hw) / e['hw']
        gy = ((d1 - e['d1']) + grid_off * hw) / e['hw']
        GY, GX = torch.meshgrid(gy, gx, indexing='ij')
        grid = torch.stack([GX, GY], -1).float()[None]
        img = F.grid_sample(rgbs[j][None], grid, mode='bilinear', padding_mode='border', align_corners=False)[0]
        # feather mask: 1 inside, falling to 0 over 3 keyframe pixels at the edge
        m = torch.clamp((1 - torch.maximum(GX.abs(), GY.abs())) * R / 2 / 3, 0, 1).float()
        return img, m

    idx = 0
    for j in range(K - 1):
        for t in range(fpt):
            a = t / fpt
            d0, d1, hw = window(a, j)
            base, _ = sample(j, d0, d1, hw)
            out = base
            for jj, wgt in ((j + 1, 1.0), (j + 2, a)):
                if jj < K:
                    im, m = sample(jj, d0, d1, hw)
                    w = m * wgt
                    out = out * (1 - w) + im * w
            arr = (out.clamp(0, 1).permute(1, 2, 0).numpy()[::-1] * 255 + 0.5).astype(np.uint8)
            Image.fromarray(arr).save(f'{frames_dir}/f_{idx:05d}.png')
            idx += 1
    # hold last keyframe 2 s
    for t in range(2 * args.fps):
        Image.fromarray(arr).save(f'{frames_dir}/f_{idx:05d}.png'); idx += 1
    mp4 = f'gallery/zoom_{args.tag}_{args.style}.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps), '-i', f'{frames_dir}/f_%05d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', mp4], check=True)
    gif = f'gallery/zoom_{args.tag}_{args.style}.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4, '-vf',
                    'fps=12,scale=480:480:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=sierra2_4a',
                    gif], check=True)
    print('video', mp4, os.path.getsize(mp4) / 1e6, 'MB; gif', os.path.getsize(gif) / 1e6, 'MB')
