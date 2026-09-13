"""How much boundary survives quotienting by symmetry? Box counting of raw vs canonical label maps.
Writes cache/quotient_<tag>.json"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractal import boundary, box_counts, fit_dimension


def between(L):
    b = np.zeros(L.shape, bool)
    for ax in (0, 1):
        a = np.take(L, range(L.shape[ax] - 1), axis=ax); c = np.take(L, range(1, L.shape[ax]), axis=ax)
        d = (a != c) & (a >= 0) & (c >= 0)
        pad = [(0, 0), (0, 0)]; pad[ax] = (0, 1); b |= np.pad(d, pad); pad[ax] = (1, 0); b |= np.pad(d, pad)
    return b


def stats(path):
    d = np.load(path)
    R = d['raw'].shape[0]
    out = {}
    for key in ('raw', 'canon'):
        lab = d[key]
        for name, b in (('all', boundary(lab)), ('between_solutions', between(lab))):
            s, c = box_counts(b, 1, R // 8)
            D, se, n = fit_dimension(s, c, 1, R // 16)
            out[f'{key}_{name}'] = dict(boundary_frac=float(b.mean()), D=D, se=se)
    conv = d['status'] == 0
    out['n_raw_classes'] = int(len(np.unique(d['raw'][conv])))
    out['n_canon_classes'] = int(len(np.unique(d['canon'][conv])))
    out['canon_over_raw_between_px'] = out['canon_between_solutions']['boundary_frac'] / max(out['raw_between_solutions']['boundary_frac'], 1e-12)
    return out


if __name__ == '__main__':
    res = {p: stats(p) for p in sys.argv[1:]}
    json.dump(res, open('cache/quotient_' + os.path.basename(sys.argv[1])[:-4] + '.json', 'w'), indent=1)
    for p, r in res.items():
        print(p, json.dumps(r))
