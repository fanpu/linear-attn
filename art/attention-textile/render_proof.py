"""Self-similarity, shown and measured.

1. Proof plates (indigo): for Sierpinski-doubling, Thue-Morse and Cantor words,
   row 1 = measured letter-level attention at full size, top-left 1/2, 1/4, 1/8
   (each upsampled to the same size); row 2 = the ideal "every previous occurrence"
   matrix for the same blocks (madder ink). Caption: AUC of measured attention for the
   ideal cells, in the annulus that each zoom adds (i.e. at that scale), and the
   decimation correlation where the word has an exact decimation rule.
2. Row-normalised Thue-Morse weave (each query row scaled to its max; declared).
3. Zoom animations (MP4 + GIF): continuous zoom into the top-left corner, the letter
   count halving every 2 s, measured (left) next to ideal (right).

  python render_proof.py
"""
import json, os, subprocess
import numpy as np
from scipy.ndimage import map_coordinates
from textile import *
from analyze_selfsim import ideal, auc, offset_norm

OUT = 'gallery'
K = 8
d = np.load(f'cache/qwen_selfsim_k{K}.npz')
meta = json.loads(str(d['meta']))
names = [m['name'] for m in meta]
metrics = json.load(open('cache/selfsim_metrics.json'))


def measured(name, rownorm=False, enrich=False):
    wi = names.index(name)
    A = d[f'w{wi}_letter'].astype(np.float64)[:, :4].mean((0, 1))
    A[:, 0] = 0
    np.fill_diagonal(A, 0)
    A = np.tril(A)
    if enrich:
        # enrichment over uniform attention: A[i,j] * (number of letters available to query i)
        E = A * (np.arange(A.shape[0])[:, None] + 1)
        return np.clip(np.log2(np.maximum(E, 1e-9)) / 5.0, 0, 1)      # 1 = 32x uniform
    if rownorm:
        return A / (A.max(1, keepdims=True) + 1e-12)
    tri = np.tril_indices(A.shape[0], -1)
    return np.clip(A / np.quantile(A[tri], 0.995), 0, 1) ** 0.6


def scale_aucs(A, C, q, levels=4):
    n = A.shape[0]
    I, J = np.indices(A.shape)
    valid = (I > J) & (J >= 1)
    out = []
    for z in range(levels):
        hi = n // q ** z
        lo = n // q ** (z + 1)
        ann = valid & (np.maximum(I, J) < hi) & (np.maximum(I, J) >= lo)
        out.append(auc(A[ann & C], A[ann & ~C]))
    return out


def up_to(W, size):
    """Nearest-neighbour resample of a square matrix to exactly size x size."""
    idx = np.minimum((np.arange(size) * W.shape[0] / size).astype(int), W.shape[0] - 1)
    return W[idx][:, idx]


def proof_plate(name, q, size=720):
    A = measured(name, enrich=True)
    C = ideal(meta[names.index(name)]['word'])
    C[:, 0] = False
    n = A.shape[0]
    aucs = scale_aucs(A, C, q)
    tiles, labels = [], []
    for z in range(4):
        m = n // q ** z
        tiles.append(indigo(up_to(A[:m, :m], size), s=1, bleed=0, absorb=2.6, seed=z))
        labels.append(f'measured, first {m} letters' + (f'  AUC {aucs[z]:.3f}' if aucs[z] == aucs[z] else ''))
    for z in range(4):
        m = n // q ** z
        tiles.append(indigo(up_to(C[:m, :m].astype(float) * 0.8, size), s=1, bleed=0, absorb=2.6, ink='#8a2a1c', seed=10 + z))
        labels.append(f'ideal, first {m} letters')
    gut, out = 80, 50
    img = tile(tiles, 2, 4, gutter=gut, bg='#f1eee6', outer=out)
    img = label_tiles(img, labels, 2, 4, (size, size), gut, out, fg='#15306b', size=20)
    mt = metrics.get(f'k{K}/{name}', {})
    dec = (f"decimation r (measured) = {mt['decim_r_top4']:.2f} vs ideal {mt['decim_r_ideal']:.2f};  "
           if 'decim_r_top4' in mt else '')
    img = frame(img, f'{name}: zooming in by {q}x repeats the pattern  (Qwen3-0.6B, top-4 induction heads, 8 tokens per letter)',
                f'{dec}AUC = how well measured attention picks out the ideal cells in the ring each zoom adds '
                f'(1.0 perfect, 0.5 chance);  indigo = measured enrichment over uniform attention, log2 scale 1x..32x (declared);  madder = ideal',
                bg='#f1eee6', fg='#15306b', margin=(120, 40, 80, 40), title_size=30, font_size=19)
    save(img, f'{OUT}/proof_{name}_indigo.png')
    print(name, 'scale AUCs', np.round(aucs, 3))
    return aucs


def tm_rownorm_weave(s=8):
    A = measured('thue-morse', rownorm=True) ** 0.8
    n = A.shape[0]
    img = weave(A, s=s, warp_lo='#3a2f45', warp_hi='#ffd27f', weft='#1f2f5c', gap='#0a0c14', float_thr=0.3, seed=5, fibre=0.05)
    img = frame(img, f'thue-morse, row-normalised: {n} letters x {K} random tokens, Qwen3-0.6B',
                'each query row scaled to its own max, ^0.8 (declared) so the lower lattice reads; top-4 induction heads, 4 draws',
                bg='#0a0c14', fg='#e6dcc0', margin=(110, 40, 80, 40), title_size=34, font_size=18)
    save(img, f'{OUT}/selfsim_thue-morse_rownorm_weave.png')
    raw = measured('thue-morse')
    img = weave(raw, s=s, warp_lo='#3a2f45', warp_hi='#ffd27f', weft='#1f2f5c', gap='#0a0c14', float_thr=0.3, seed=5, fibre=0.05)


def zoom_movie(name, q, fps=30, sec_per_level=2.0, levels=3, size=900):
    """Continuous zoom into the top-left corner; letters shown shrink by q every sec_per_level."""
    A = measured(name, enrich=True)
    C = ideal(meta[names.index(name)]['word']).astype(float) * 0.8
    C[:, 0] = 0
    n = A.shape[0]
    nframes = int(fps * sec_per_level * levels)
    tmp = f'cache/frames_zoom_{name}'
    os.makedirs(tmp, exist_ok=True)
    for f in range(nframes + int(fps)):
        t = min(f, nframes) / (fps * sec_per_level)
        m = n / q ** t                                        # letters visible
        coords = (np.arange(size) + 0.5) / size * m - 0.5
        yy, xx = np.meshgrid(coords, coords, indexing='ij')
        a = map_coordinates(A, [yy, xx], order=0, mode='nearest')
        c = map_coordinates(C, [yy, xx], order=0, mode='nearest')
        L = indigo(a, s=1, bleed=0, absorb=2.6, seed=0)
        R = indigo(c, s=1, bleed=0, absorb=2.6, ink='#8a2a1c', seed=1)
        img = np.ones((1080, 1920, 3)) * hexrgb('#f1eee6')
        img[90:90 + size, 40:40 + size] = L
        img[90:90 + size, 980:980 + size] = R
        img = text_at(img, [(40, 45, f'{name}  measured enrichment (Qwen3-0.6B top-4 induction heads)', 'lm'),
                            (980, 45, 'ideal: every previous occurrence of the same letter', 'lm'),
                            (40, 1030, f'showing the first {m:6.1f} of {n} letters  (x{n / m:5.2f})', 'lm')],
                      fg='#15306b', size=26)
        save(img, f'{tmp}/f{f:04d}.png')
    mp4 = f'{OUT}/zoom_{name}.mp4'
    gif = f'{OUT}/zoom_{name}.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{tmp}/f%04d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', mp4], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{tmp}/f%04d.png',
                    '-vf', 'fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
                    gif], check=True)
    subprocess.run(['rm', '-rf', tmp])
    print('movie', mp4, os.path.getsize(mp4) // 1024, 'KB', gif, os.path.getsize(gif) // 1024, 'KB')


if __name__ == '__main__':
    res = {}
    for name, q in [('sierpinski-doubling', 2), ('thue-morse', 2), ('cantor', 3)]:
        res[name] = proof_plate(name, q)
    json.dump(res, open('cache/proof_scale_aucs.json', 'w'), indent=1)
    tm_rownorm_weave()
    import sys
    if 'movies' in sys.argv:
        zoom_movie('sierpinski-doubling', 2, levels=3)
        zoom_movie('cantor', 3, levels=3)
