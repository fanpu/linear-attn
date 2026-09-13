"""Verification numbers and figures for the README.
python analyze_verify.py lyap | saf | icmap | basins
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'cache/verify.json'


def save(key, val):
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d[key] = val
    json.dump(d, open(OUT, 'w'), indent=1)
    print(key, json.dumps(val, indent=1)[:2000])


def edges(B):
    """Boundary pixels of a boolean field: 4-neighbourhood contains the other value."""
    e = np.zeros_like(B)
    e[:-1] |= B[:-1] != B[1:]; e[1:] |= B[:-1] != B[1:]
    e[:, :-1] |= B[:, :-1] != B[:, 1:]; e[:, 1:] |= B[:, :-1] != B[:, 1:]
    return e


def boxcount(E, sizes):
    R = E.shape[0]
    N = []
    for k in sizes:
        m = R // k
        b = E[:m * k, :m * k].reshape(m, k, m, k).any(axis=(1, 3))
        N.append(int(b.sum()))
    return np.array(N)


def fit(sizes, N, R):
    eps = np.array(sizes) / R
    s, c = np.polyfit(np.log(1 / eps), np.log(N), 1)
    return float(s)


def lyap():
    import torch
    from gamelib import gpu_setup, lyap_cong1d, DT
    gpu_setup()
    res = {}
    d = np.load('cache/cong_rescheck.npz')
    sizes = [1, 2, 4, 8, 16, 32, 64]
    for R in (500, 1000, 2000):
        B = d[f'L{R}'] > 0
        E = edges(B)
        N = boxcount(E, [k for k in sizes if R // k >= 8])
        res[f'shrimp_window_R{R}'] = dict(edge_frac=float(E.mean()), N=N.tolist(),
                                          D=fit([k for k in sizes if R // k >= 8], N, R))
    a, b = d['L500'], d['L500_long']
    res['iter_check_500'] = dict(sign_agree=float(np.mean((a > 0) == (b > 0))),
                                 corr=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
    # null model: same pipeline on the first period-doubling curve (smooth), s in [3,12], y* in [0.15,0.85]
    for R in (500, 1000, 2000):
        s = torch.linspace(3, 12, R, dtype=DT); y = torch.linspace(0.15, 0.85, R, dtype=DT)
        Y, S = torch.meshgrid(y, s, indexing='ij')
        L = lyap_cong1d(S, Y, T0=3000, T1=5000)
        # smooth threshold: period-1 -> period-2 boundary = sign of (s y*(1-y*) - 2)
        B = L < -0.05  # 'strongly stable' vs near-marginal; also test the analytic curve
        B2 = S.numpy() * Y.numpy() * (1 - Y.numpy()) < 2
        for nm, BB in (('lyap', B), ('analytic', B2)):
            E = edges(BB); ss = [k for k in sizes if R // k >= 8]; N = boxcount(E, ss)
            res[f'null_{nm}_R{R}'] = dict(edge_frac=float(E.mean()), N=N.tolist(), D=fit(ss, N, R))
    save('lyap_boxcount', res)


def saf():
    d = np.load('cache/saf_repro.npz')
    paper = {'0.25': [49.0, 35.3, 16.6, 0.4, 0.4], '0.50': [61.6, 35.0, 28.1, 12.1, 0.2], '0.00': [1.0, 1.4, 0.4, 0.4, 0.4]}
    res = {}
    for e in ('0.00', '0.25', '0.50'):
        res[e] = dict(ours_x1e3=np.round(d[f'lyap_{e}'][:5] * 1e3, 2).tolist(), saf_table1_x1e3=paper[e],
                      chaotic_k=[int(k) for k in d['k'][d[f'lyap_{e}'] > 5e-3]], max_H_drift=float(d[f'Hd_{e}'].max()))
    save('saf_table1', res)
    # convergence figure
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    h = d['hist_0.50']; T = np.arange(1, h.shape[1] + 1) * 100
    for k in range(8):
        ax.loglog(T, np.abs(h[k]) + 1e-7, lw=1, label=f'k={k+1}')
    ax.loglog(T, 3 / T, 'k--', lw=0.8, label='∝ 1/T')
    ax.set_xlabel('integration time T'); ax.set_ylabel('finite-time λ₁'); ax.legend(fontsize=7, ncol=2)
    ax.set_title('SAF initial conditions, ε = 0.5: chaotic orbits plateau, regular ones decay as 1/T', fontsize=9)
    fig.tight_layout(); fig.savefig('gallery/verify_lyap_convergence.png'); plt.close(fig)


if __name__ == '__main__':
    for w in sys.argv[1:]:
        globals()[w]()
