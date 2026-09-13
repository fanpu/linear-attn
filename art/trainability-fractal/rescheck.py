"""Resolution check (fractals doc s11): the central 32x32 px of zoomA kf5 (10^2.5) recomputed
at 2x (64^2) and 4x (128^2) grid density over the identical lr window. If the boundary is a
resolved curve of dimension D, boundary-pixel counts grow like r^D (r = density factor) and
the converged fraction is stable. Writes cache/verify_res.json and gallery/verify_rescheck_spectral.png."""
import json
import numpy as np
from PIL import Image, ImageDraw
import styles as S
from common_render import edges
from boxcount import box_counts, fit_dimension
from pages import font, MONO

kf = np.load('cache/zoom_zoomA/kf_005.npz')['measure']
blocks = {1: kf[112:144, 112:144],
          2: np.load('cache/windows/res_kf5_block_64.npz')['measure'],
          4: np.load('cache/windows/res_kf5_block_128.npz')['measure']}
out = {}
for r, M in blocks.items():
    E = edges(M)
    out[r] = dict(res=M.shape[0], conv=float((M < 0).mean()), edge_px=int(E.sum()))
rs = np.array([1, 2, 4.]); ne = np.array([out[r]['edge_px'] for r in (1, 2, 4)], float)
D_res = float(np.polyfit(np.log(rs), np.log(ne), 1)[0])
s, c = box_counts(edges(blocks[4]))
fit = fit_dimension(s, c, 2, 32)
# coarse-grain agreement: 4x block sampled at every 4th px vs the 1x block (centres offset by 1.5 fine px)
agree = float(((blocks[4][1::4, 1::4] < 0) == (blocks[1] < 0)).mean())
agree2 = float(((blocks[2][::2, ::2] < 0) == (blocks[1] < 0)).mean())
res = dict(blocks=out, D_from_edge_count_scaling=D_res, boxcount_128=fit,
           label_agreement_4x_subsampled_vs_1x=agree, label_agreement_2x_subsampled_vs_1x=agree2)
json.dump(res, open('cache/verify_res.json', 'w'), indent=1, default=float)
print(json.dumps(res, default=float))
P = 640; pad = 30
sheet = Image.new('RGB', (3 * P + 4 * pad, P + 2 * pad + 60), (244, 240, 230))
dd = ImageDraw.Draw(sheet)
for j, r in enumerate((1, 2, 4)):
    im = Image.fromarray(S.spectral(blocks[r])).resize((P, P), Image.NEAREST)
    x = pad + j * (P + pad)
    sheet.paste(im, (x, pad))
    dd.text((x, pad + P + 12), f'{blocks[r].shape[0]}^2 ({r}x)  boundary px {out[r]["edge_px"]}  trainable {100*out[r]["conv"]:.1f}%',
            font=font(MONO, 24), fill=(40, 36, 40))
sheet.save('gallery/verify_rescheck_spectral.png')
