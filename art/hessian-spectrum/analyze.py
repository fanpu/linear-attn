"""Spectral measurements for checkpoints of a run.

For each requested checkpoint:
  * long Lanczos (m_top iterations, full reorth, float64) on H and on G  -> extreme eigenvalues + residuals
  * SLQ (m_slq iterations x nv probes) on H, G, and (optionally) E = H - G -> Ritz nodes and weights
  * Papyan class / cross-class decomposition eigenvalues (G0, G1, G1+2) from exact class means
Saves cache/spectra/<run>/step_XXXXXX.npz

python analyze.py --run mnist_mlp --steps final --m_top 300 --m_slq 100 --nv 8 --E
"""
import argparse, json, os, time, glob
import numpy as np
import torch
from common import (load_dataset, class_subset, ARCHS, HessianOps, top_eigs, slq,
                    papyan_decomposition, count_outliers, CACHE)

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--steps', default='final', help="'final', 'all', or comma list")
ap.add_argument('--per_class', type=int, default=1000)
ap.add_argument('--m_top', type=int, default=300)
ap.add_argument('--m_slq', type=int, default=100)
ap.add_argument('--nv', type=int, default=8)
ap.add_argument('--E', action='store_true', help='also SLQ of E = H - G')
ap.add_argument('--noG', action='store_true', help='skip Lanczos/SLQ on G (class means still computed)')
ap.add_argument('--chunk', type=int, default=2500)
ap.add_argument('--overwrite', action='store_true')
ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()

torch.set_num_threads(args.threads)
rdir = os.path.join(CACHE, 'runs', args.run)
meta = json.load(open(os.path.join(rdir, 'meta.json')))
odir = os.path.join(CACHE, 'spectra', args.run)
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

for f in files:
    step = int(os.path.basename(f)[5:11])
    outp = os.path.join(odir, f'step_{step:06d}.npz')
    if os.path.exists(outp) and not args.overwrite:
        print('skip', outp); continue
    t0 = time.time()
    model = ARCHS[meta['arch']](ds, C)
    model.load_state_dict(torch.load(f, map_location='cpu'))
    ops = HessianOps(model, x, y, chunk=args.chunk)
    P = ops.P
    loss, acc = ops.loss_acc()
    res = dict(step=step, C=C, P=P, N=ops.N, loss=loss, acc=acc,
               m_top=args.m_top, m_slq=args.m_slq, nv=args.nv)
    th, r, UHk = top_eigs(ops.Hv, P, args.m_top, 'cpu', seed=7, k_vec=3 * C)
    res.update(H_ritz=th, H_resid=r, count_H=count_outliers(th, 3 * C))
    print(f'[{args.run} {step}] loss {loss:.4f} acc {acc:.4f}  H top {th[-1]:.4g} bottom {th[0]:.4g}  {time.time()-t0:.0f}s', flush=True)
    n, w = slq(ops.Hv, P, args.m_slq, args.nv, 'cpu', seed=11)
    res.update(H_slq_nodes=np.stack(n), H_slq_w=np.stack(w))
    if not args.noG:
        th, r = top_eigs(ops.Gv, P, args.m_top, 'cpu', seed=7)
        res.update(G_ritz=th, G_resid=r, count_G=count_outliers(th, 3 * C))
        n, w = slq(ops.Gv, P, args.m_slq, args.nv, 'cpu', seed=11)
        res.update(G_slq_nodes=np.stack(n), G_slq_w=np.stack(w))
    if args.E:
        n, w = slq(ops.Ev, P, args.m_slq, args.nv, 'cpu', seed=11)
        res.update(E_slq_nodes=np.stack(n), E_slq_w=np.stack(w))
    D = ops.class_means(C)
    dec = papyan_decomposition(D)
    mask = ~torch.eye(C, dtype=torch.bool)
    dc = torch.stack([D[c][mask[c]].mean(0) for c in range(C)])
    for nm, B in (('ov_dc', dc), ('ov_dcc', D.reshape(C * C, -1))):
        Q, _ = torch.linalg.qr(B.T)
        res[nm] = ((UHk @ Q) ** 2).sum(1).numpy()
    del D, UHk
    res.update({f'pap_{k}': v for k, v in dec.items()})
    res['wall'] = time.time() - t0
    np.savez(outp, **res)
    print(f'[{args.run} {step}] done {res["wall"]:.0f}s  H top8 {np.round(res["H_ritz"][-8:][::-1],4)}  G1 {np.round(dec["G1"][:C],4)} count_H {res["count_H"]} ov_dc {np.round(res["ov_dc"][:C+2],2)}', flush=True)
    del ops, model
