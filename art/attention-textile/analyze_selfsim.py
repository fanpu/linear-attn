"""Measurements on the self-similar-word attention (no GPU).

For every word and block size k, working at LETTER level
(query-block mean, key-block sum of token attention):

1. ideal matrix C[i,j] = [w_i == w_j and w_i != filler], j < i
   (where a perfect "attend to every previous occurrence" head would put mass)
2. agreement: AUC of measured attention for separating C=1 from C=0 cells,
   in offset bands (near / mid / far) - does the structure survive at long range?
3. decimation self-similarity (TM, period-doubling: q=2; Cantor: q=3):
   Pearson r between offset-normalised A[::q, ::q] and A[:n/q, :n/q]
   (lower triangle, diagonal excluded), vs the same statistic on controls.
4. box counting (Cantor, Cantor-shuffled control, periodic "smooth" null):
   binarise measured attention keeping the top-N cells, N = #ideal cells;
   count occupied boxes at eps = 3^0..3^4 letters, fit slope.

  python analyze_selfsim.py  -> cache/selfsim_metrics.json
"""
import json
import numpy as np


def ideal(word):
    w = np.array(list(word))
    C = (w[:, None] == w[None, :]) & (w[:, None] != 'z')
    return np.tril(C, -1)


def offset_norm(M):
    n = M.shape[0]
    I, J = np.indices(M.shape)
    off = I - J
    oc = np.clip(off, 0, None)
    s = np.bincount(oc.ravel(), weights=M.ravel(), minlength=n)
    c = np.bincount(oc.ravel(), minlength=n)
    mu = s / np.maximum(c, 1)
    out = M / (mu[oc] + 1e-12)
    out[off < 1] = 0
    return out


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def decimation_r(M, q, start=0):
    n = M.shape[0]
    m = (n - start) // q
    R = offset_norm(M)
    a = R[start::q, start::q][:m, :m]
    b = R[:m, :m]
    tri = np.tril_indices(m, -1)
    x, y = a[tri], b[tri]
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def box_count(B, eps_list):
    n = B.shape[0]
    out = []
    for e in eps_list:
        m = n // e
        bb = B[:m * e, :m * e].reshape(m, e, m, e).any((1, 3))
        out.append(int(bb.sum()))
    return out


def fit(eps, N):
    x = np.log(1 / np.array(eps, float))
    y = np.log(np.array(N, float))
    p = np.polyfit(x, y, 1)
    resid = y - np.polyval(p, x)
    return float(p[0]), float(np.sqrt(np.mean(resid ** 2)))


def binarise_topN(M, N):
    tri = np.tril(np.ones_like(M, bool), -1)
    tri[:, 0] = False                       # exclude the first-letter sink column
    v = M[tri]
    thr = np.sort(v)[-N]
    return (M >= thr) & tri


res = {}
for K in (8, 4):
    d = np.load(f'cache/qwen_selfsim_k{K}.npz')
    meta = json.loads(str(d['meta']))
    tops = [tuple(x) for x in d['top_heads'].tolist()]
    for wi, m in enumerate(meta):
        L = d[f'w{wi}_letter'].astype(np.float64)            # draws, 8, n, n
        n = L.shape[-1]
        C = ideal(m['word'])
        C[:, 0] = False
        key = f"k{K}/{m['name']}"
        r = dict(letters=n, T=m['T'], block=K)
        I, J = np.indices((n, n))
        off = I - J
        for hi, name in [(0, 'L%dH%d' % tops[0]), ('top4', 'top4')]:
            A = L[:, hi].mean(0) if hi != 'top4' else L[:, :4].mean((0, 1))
            A = A.copy()
            A[:, 0] = 0
            valid = (off >= 1) & (J >= 1)
            bands = {}
            for bn, (lo, hi_) in {'near(1-8)': (1, 9), 'mid(9-64)': (9, 65), 'far(65+)': (65, 10 ** 9)}.items():
                sel = valid & (off >= lo) & (off < hi_)
                bands[bn] = auc(A[sel & C], A[sel & ~C])
            r[f'auc_{name}'] = bands
            q = {'thue-morse': 2, 'period-doubling': 2, 'cantor': 3, 'control iid a/b': 2,
                 'control cantor-shuffled': 3}.get(m['name'])
            if q:
                st = 1 if m['name'] == 'period-doubling' else 0   # PD: w_{2n+1} codes w_n
                r[f'decim_r_{name}'] = decimation_r(A, q, st)
                r[f'decim_r_ideal'] = decimation_r(C.astype(float) + 1e-3, q, st)
        if m['name'] in ('cantor', 'control cantor-shuffled'):
            A = L[:, :4].mean((0, 1))
            A[:, 0] = 0
            N = int(C.sum())
            B = binarise_topN(A, N)
            eps = [1, 3, 9, 27, 81]
            nb = box_count(B, eps)
            nb_ideal = box_count(C, eps)
            r['box_eps_letters'] = eps
            r['box_N_measured'] = nb
            r['box_N_ideal'] = nb_ideal
            r['box_slope_measured'], r['box_rmse_measured'] = fit(eps, nb)
            r['box_slope_ideal'], r['box_rmse_ideal'] = fit(eps, nb_ideal)
            r['box_slope_measured_fine(1-9)'] = fit(eps[:3], nb[:3])[0]
            r['box_slope_measured_coarse(9-81)'] = fit(eps[2:], nb[2:])[0]
            r['box_slope_ideal_fine(1-9)'] = fit(eps[:3], nb_ideal[:3])[0]
            r['box_slope_ideal_coarse(9-81)'] = fit(eps[2:], nb_ideal[2:])[0]
            r['precision_at_N'] = float((B & C).sum() / N) if m['name'] == 'cantor' else None
            r['theory_log4_log3'] = float(np.log(4) / np.log(3))
        res[key] = r
        print(key, json.dumps({k: v for k, v in r.items() if k.startswith(('auc_top4', 'decim', 'box_slope', 'precision'))}))

# resolution check: k=2 run has depth-6 Cantor (729 letters, single draw, top-4 heads)
d2 = np.load('cache/qwen_selfsim_k2.npz')
meta2 = json.loads(str(d2['meta']))
for wi, m in enumerate(meta2):
    if m['name'] not in ('cantor', 'control cantor-shuffled'):
        continue
    top = d2[f'w{wi}_top'][:4].astype(np.float64)
    k = m['block']
    n = top.shape[-1] // k
    A = top.reshape(4, n, k, n, k).sum(4).mean(2).mean(0)
    A[:, 0] = 0
    C = ideal(m['word'])
    C[:, 0] = False
    eps = [1, 3, 9, 27, 81, 243]
    B = binarise_topN(A, int(C.sum()))
    nb, nbi = box_count(B, eps), box_count(C, eps)
    res[f"k2/{m['name']}"] = dict(letters=n, box_eps_letters=eps, box_N_measured=nb, box_N_ideal=nbi,
                                   box_slope_measured=fit(eps, nb)[0], box_slope_ideal=fit(eps, nbi)[0],
                                   box_slope_measured_fine=fit(eps[:4], nb[:4])[0], box_slope_measured_coarse=fit(eps[2:], nb[2:])[0],
                                   precision_at_N=float((B & C).sum() / C.sum()))
    print('k2', m['name'], res[f"k2/{m['name']}"])

# resolution check / rendering null for box counting: a periodic word (stripes -> dim 1)
n = 243
per = np.array([i % 9 for i in range(n)])
Cp = np.tril(per[:, None] == per[None, :], -1)
nbp = box_count(Cp, [1, 3, 9, 27, 81])
res['null_periodic_ideal'] = dict(box_N=nbp, slope=fit([1, 3, 9, 27, 81], nbp)[0],
                                  note='ideal matrix of a period-9 word; expected dim ~1 (lines) at eps < 9, 2 above')
print('null periodic', res['null_periodic_ideal'])
json.dump(res, open('cache/selfsim_metrics.json', 'w'), indent=1)
