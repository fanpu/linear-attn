"""EXACT full spectra for small nets (P ~ 4.6k): materialise H and G = J^T A J / N column by column
(float64, vmapped HVP / GN-VP), diagonalise with LAPACK. No SLQ smoothing, no Lanczos convergence caveat.

Per checkpoint saves cache/exact/<run>/step_XXXXXX.npz with
  H_eig, G_eig (all P eigenvalues), [E_eig of E = H - G with --E], top-k H/G eigenvectors,
  Papyan class/cross-class quantities, overlap of top H eigenvectors with span{d_c} (C vectors) and span{d_cc'} (C^2).

python exact_analyze.py --run s_mlps_C10 --steps final --E
"""
import argparse, json, os, time, glob
import numpy as np
import torch
from common import (load_dataset, class_subset, ARCHS, HessianOps, full_matrix, papyan_decomposition,
                    count_outliers, CACHE)

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--steps', default='final')
ap.add_argument('--per_class', type=int, default=500)
ap.add_argument('--E', action='store_true')
ap.add_argument('--noG', action='store_true')
ap.add_argument('--kvec', type=int, default=30)
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--overwrite', action='store_true')
args = ap.parse_args()
torch.set_num_threads(args.threads)

rdir = os.path.join(CACHE, 'runs', args.run)
meta = json.load(open(os.path.join(rdir, 'meta.json')))
tag = '' if args.per_class == 500 else f'_pc{args.per_class}'
odir = os.path.join(CACHE, 'exact', args.run + tag)
os.makedirs(odir, exist_ok=True)
C, ds = meta['C'], meta['ds']
x, y = load_dataset(ds, train=True, down=meta.get('down') or None)
x, y = class_subset(x, y, C, per_class=args.per_class, seed=1234)

files = sorted(glob.glob(os.path.join(rdir, 'step_*.pt')))
if args.steps == 'final':
    files = files[-1:]
elif args.steps != 'all':
    want = set(int(s) for s in args.steps.split(','))
    files = [f for f in files if int(os.path.basename(f)[5:11]) in want]


def overlap(U, B):
    """U: (k,P) orthonormal rows; B: (m,P) spanning set. Returns ||proj_span(B) u_i||^2 per row."""
    Q, _ = torch.linalg.qr(B.T)
    return ((U @ Q) ** 2).sum(1).numpy()


for f in files:
    step = int(os.path.basename(f)[5:11])
    outp = os.path.join(odir, f'step_{step:06d}.npz')
    if os.path.exists(outp) and not args.overwrite:
        print('skip', outp); continue
    t0 = time.time()
    model = ARCHS[meta['arch']](ds, C)
    model.load_state_dict(torch.load(f, map_location='cpu'))
    ops = HessianOps(model, x, y, chunk=5000)
    loss, acc = ops.loss_acc()
    res = dict(step=step, C=C, P=ops.P, N=ops.N, loss=loss, acc=acc)
    k = args.kvec
    H = full_matrix(ops.Hv_batched, ops.P, block=512)
    eH, UH = torch.linalg.eigh(H)
    res.update(H_eig=eH.numpy(), H_vec=UH[:, -k:].flip(1).T.float().numpy())
    UHk = UH[:, -k:].flip(1).T.contiguous()
    if not args.noG:
        G = full_matrix(ops.Gv_batched, ops.P, block=512)
        eG, UG = torch.linalg.eigh(G)
        res.update(G_eig=eG.numpy(), G_vec=UG[:, -k:].flip(1).T.float().numpy())
        if args.E:
            res.update(E_eig=torch.linalg.eigvalsh(H - G).numpy())
        del G, UG
    del H, UH
    D = ops.class_means(C)
    dec = papyan_decomposition(D)
    mask = ~torch.eye(C, dtype=torch.bool)
    dc = torch.stack([D[c][mask[c]].mean(0) for c in range(C)])
    res['ov_dc'] = overlap(UHk, dc)                       # C vectors
    res['ov_dcc'] = overlap(UHk, D.reshape(C * C, -1))    # C^2 vectors
    res.update({f'pap_{kk}': v for kk, v in dec.items()})
    res['count_H'] = count_outliers(eH.numpy(), 3 * C)
    if not args.noG:
        res['count_G'] = count_outliers(res['G_eig'], 3 * C)
    res['wall'] = time.time() - t0
    np.savez(outp, **res)
    top = np.sort(res['H_eig'])[::-1][:C + 3]
    print(f'[{args.run} {step}] loss {loss:.4f} acc {acc:.4f} {res["wall"]:.0f}s  H top {np.round(top, 4)}  '
          f'count_H {res["count_H"]}  ov_dc {np.round(res["ov_dc"][:C + 2], 2)}', flush=True)
