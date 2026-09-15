"""Window choices for M2 (CPU).
  choose_windows.py null   -> cache/vol/null_window.json : hw = (1,1,1) decades, centred where the quad2 null's
                              128^3 toy-(a)-chart volume (nullA128) holds the most edge cells in a window of that size
  choose_windows.py B      -> cache/vol/B_window.json    : the probe with the highest D3(b=1-16) - D3(null probe)
Existing JSON files are never overwritten (manual override)."""
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter

from se_analysis import dimension3d, edges3d

HERE = os.environ.get('SE_ROOT', os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(HERE, 'cache', 'vol')
os.makedirs(V, exist_ok=True)
HW = [1.0, 1.0, 1.0]

mode = sys.argv[1]
if mode == 'candidates':
    # top-3 8^3 blocks by edge-cell count in the (measure-fixed) M1 toy (a) volume, >= 2 blocks apart
    out = os.path.join(V, 'candidates.json')
    if os.path.exists(out):
        print('exists', out); raise SystemExit
    d = np.load(os.path.join(HERE, 'cache', 'toys', 'toy_a_64_float32_fix.npz'))
    L = d['measure'] < 0
    le, ls = d['log10_eta'], d['third_log']
    B = edges3d(L).reshape(8, 8, 8, 8, 8, 8).sum(axis=(1, 3, 5))
    order = np.argsort(B.ravel())[::-1]
    picks = []
    for f in order:
        kij = np.array(np.unravel_index(f, B.shape))
        if all(np.abs(kij - p).max() >= 2 for p in picks):
            picks.append(kij)
        if len(picks) == 3:
            break
    cands = {}
    for n, (k, i, j) in enumerate(picks):
        c = [float(le[8 * j:8 * j + 8].mean()), float(le[8 * i:8 * i + 8].mean()), float(ls[8 * k:8 * k + 8].mean())]
        cands[f'c{n+1}'] = dict(center=c, hw=HW, block_kij=[int(k), int(i), int(j)], edge_cells=int(B[k, i, j]))
        json.dump(dict(center=c, hw=HW), open(os.path.join(V, f'cand_c{n+1}.json'), 'w'))
    json.dump(cands, open(out, 'w'), indent=1)
    print(json.dumps(cands, indent=1))
elif mode == 'null':
    out = os.path.join(V, 'null_window.json')
    if os.path.exists(out):
        print('exists', out); raise SystemExit
    F = np.load(os.path.join(V, 'nullA128_final.npz'))
    L = F['measure'] < 0
    R = L.shape[0]
    le0, le1, ls = np.log10(F['eta0']), np.log10(F['eta1']), np.log10(F['sigma'])
    size = [max(3, int(round(2 * HW[2] / (ls[1] - ls[0])))), int(round(2 * HW[1] / (le1[1] - le1[0]))), int(round(2 * HW[0] / (le0[1] - le0[0])))]
    S = uniform_filter(edges3d(L).astype(np.float32), size=size, mode='constant')
    k, i, j = np.unravel_index(int(np.argmax(S)), S.shape)
    w = dict(center=[float(le0[j]), float(le1[i]), float(ls[k])], hw=HW, rule='argmax edge-cell density of nullA128 over a window of this size',
             filter_size_kij=size, edge_cells_in_window=float(S[k, i, j] * np.prod(size)))
    json.dump(w, open(out, 'w'), indent=1)
    print(w)
elif mode == 'B':
    out = os.path.join(V, 'B_window.json')
    PROBES = list(json.load(open(os.path.join(V, 'candidates.json'))))
    rows = {}
    for nm in PROBES + ['null']:
        d = os.path.join(V, f'probe_{nm}')
        M = np.load(os.path.join(d, 'f32.npy'))
        f, s, c, E = dimension3d(M < 0, 1, 16)
        g = json.load(open(os.path.join(d, 'grid.json')))
        rows[nm] = dict(center=g['center'], hw=g['hw'], conv=float((M < 0).mean()), edge_cells=int(E.sum()),
                        D3_b1_16=f['D'], se=f['se'], counts=c.tolist())
    Dn = rows['null']['D3_b1_16']
    for nm in PROBES:
        rows[nm]['D_minus_null'] = rows[nm]['D3_b1_16'] - Dn
    best = max(PROBES, key=lambda n: rows[n]['D_minus_null'])
    json.dump(rows, open(os.path.join(V, 'probe_table.json'), 'w'), indent=1)
    for n, r in rows.items():
        print(n, r['center'], f"conv={r['conv']:.3f} edge={r['edge_cells']} D={r['D3_b1_16']:.3f}+-{r['se']:.3f}", r.get('D_minus_null'))
    if os.path.exists(out):
        print('exists (kept)', out, json.load(open(out))); raise SystemExit
    json.dump(dict(center=rows[best]['center'], hw=rows[best]['hw'], chosen=best, rule='max D3(b=1-16) - D3(null probe)'), open(out, 'w'), indent=1)
    print('chosen', best)
