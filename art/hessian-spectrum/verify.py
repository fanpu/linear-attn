"""Outlier count vs C, for both measurement routes; writes cache/verify.json and prints a markdown table.

count  = number of eigenvalues above the widest multiplicative gap among the top 3C+1 (positive) eigenvalues
gap    = lambda_k / lambda_{k+1} at that gap;  2nd = runner-up gap (k, ratio)
ov_dc  = mean over the top-k eigenvectors of ||proj onto span{d_c}||^2 (Papyan class-mean directions, C vectors);
         chance level for a random unit vector is C/P.
ov_next= same for eigenvectors k+1..k+C (the first bulk / mini-bulk directions)
"""
import json, os
import numpy as np
from render_common import SERIES, load_exact, load_lanczos, CACHE, n_structural

rows = []
for src, loader, key in (('exact', load_exact, 'H_eig'), ('lanczos', load_lanczos, 'H_ritz')):
    for C in SERIES:
        try:
            d = loader(C)
        except (IndexError, FileNotFoundError):
            continue
        k, g, k2, g2 = d['count_H']
        k = int(k)
        lam = np.sort(d[key])[::-1]
        ks = n_structural(d)
        r = dict(src=src, C=C, k_span=ks, ov_k=[float(v) for v in d['ov_dc'][max(ks - 1, 0):ks + 1]], P=int(d['P']), N=int(d['N']), loss=float(d['loss']), count=k, gap=float(g),
                 second=(int(k2), float(g2)), top=[float(v) for v in lam[:C + 2]],
                 ov_dc=float(np.mean(d['ov_dc'][:k])), ov_next=float(np.mean(d['ov_dc'][k:k + C])),
                 chance=C / int(d['P']), n_neg=int((d[key] < 0).sum()) if src == 'exact' else None,
                 count_G=int(d['count_G'][0]) if 'count_G' in d else None,
                 G1=[float(v) for v in d['pap_G1'][:C]])
        if src == 'exact':
            r['H_resid_max'] = 0.0
        else:
            r['H_resid_top'] = float(np.max(d['H_resid'][-(k + 1):]))
        rows.append(r)

json.dump(rows, open(os.path.join(CACHE, 'verify.json'), 'w'), indent=1)
print('| route | C | P | train loss (subset) | **k_span** (eigvecs with overlap > 0.5) | = C-1? | overlap of k-th / (k+1)-th | gap count k |  gap λk/λk+1 | runner-up gap | count on G | '
      'overlap top-k with span{d_c} | next C | chance C/P |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print(f"| {r['src']} | {r['C']} | {r['P']:,} | {r['loss']:.4f} | **{r['k_span']}** | {'yes' if r['k_span'] == r['C'] - 1 else 'no'} "
          f"| {r['ov_k'][0]:.2f} / {r['ov_k'][1]:.2f} | {r['count']} "
          f"| {r['gap']:.2f} | k={r['second'][0]}: {r['second'][1]:.2f} | {r['count_G']} | {r['ov_dc']:.2f} | "
          f"{r['ov_next']:.2f} | {r['chance']:.1e} |")
