"""M1 hyperparameter toys: one res^3 volume per toy, checkpointed per chunk.

  toy a     net2 (source network), axes (log eta0, log eta1, log sigma), sigma scales W0 and W1
  toy b     net3 (two hidden layers), axes (log eta0, log eta1, log eta2)
  toy null  quad3 (frozen-feature quadratic of net3), axes (log eta0, log eta1, log eta2)

eta axes: every (1024/res)-th pixel centre of the source 1024^2 overview window
(se_engine.overview_lr_axis). Volume index order [k, i, j] = [third axis, eta1, eta0].

  gpu1.sh ../.venv/bin/python toys_compute.py --toy a
  ../.venv/bin/python toys_compute.py --toy a --device cpu --res 4 --dtype float64 --tag smoke   (CPU smoke)

Outputs: cache/toys/<name>/chunk_XXX.npz (resume), cache/toys/<name>.npz (assembled).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import se_engine as se

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument('--toy', required=True, choices=['a', 'b', 'null'])
p.add_argument('--res', type=int, default=64)
p.add_argument('--steps', type=int, default=500)
p.add_argument('--chunk', type=int, default=None)
p.add_argument('--dtype', default='float32')
p.add_argument('--device', default='cuda')
p.add_argument('--tag', default='')
p.add_argument('--no_compile', action='store_true')
p.add_argument('--sigma_plane_only', action='store_true', help='toy a: only the sigma=1 plane (res^2 voxels)')
args = p.parse_args()

KIND = {'a': 'net2', 'b': 'net3', 'null': 'quad3'}[args.toy]
DEF_CHUNK = {'a': 32768, 'b': 16384, 'null': 32768}[args.toy]
dt = getattr(torch, args.dtype)
dev = args.device
R = args.res
chunk = args.chunk or min(DEF_CHUNK, R ** (2 if args.sigma_plane_only else 3))
name = f'toy_{args.toy}_{R}_{args.dtype}{"_plane" if args.sigma_plane_only else ""}{args.tag}'
outdir = os.path.join(HERE, 'cache', 'toys', name)
final = os.path.join(HERE, 'cache', 'toys', f'{name}.npz')
os.makedirs(outdir, exist_ok=True)
if os.path.exists(final):
    print('exists', final)
    raise SystemExit

if dev == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.10)
    torch.backends.cuda.matmul.allow_tf32 = False
    gpu = torch.cuda.get_device_name(0)
else:
    gpu = 'cpu'

eta, pix = se.overview_lr_axis(R)                     # float64 CPU
if args.toy == 'a':
    third, third_log, k1 = se.sigma_axis(R)
    third_name = 'log10_sigma'
else:
    third, third_log, k1 = eta, np.log10(eta.numpy()), None
    third_name = 'log10_eta2'

K, I, J = torch.meshgrid(torch.arange(R), torch.arange(R), torch.arange(R), indexing='ij')
K, I, J = K.reshape(-1), I.reshape(-1), J.reshape(-1)
shape = (R, R, R)
if args.sigma_plane_only:
    assert args.toy == 'a'
    sel = K == k1
    K, I, J = K[sel], I[sel], J[sel]
    shape = (R, R)
prob = se.make_problem(KIND, device=dev, dtype=dt)
nvox = K.numel()
nch = (nvox + chunk - 1) // chunk
print(f'{name}: kind={KIND} res={R} chunk={chunk} nch={nch} dtype={args.dtype} device={dev} ({gpu})', flush=True)
t_all = time.time()
for c in range(nch):
    fn = os.path.join(outdir, f'chunk_{c:03d}.npz')
    if os.path.exists(fn):
        continue
    s, e = c * chunk, min(nvox, (c + 1) * chunk)
    j, i, k = J[s:e], I[s:e], K[s:e]
    e0, e1, e3 = eta[j].to(dev), eta[i].to(dev), third[k].to(dev)
    if dev == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        if args.toy == 'a':
            r = se.train_chunk(prob, [e0, e1], sigma=e3, steps=args.steps, compiled=not args.no_compile)
        else:
            r = se.train_chunk(prob, [e0, e1, e3], steps=args.steps, compiled=not args.no_compile)
    m = r['measure'].to(torch.float64).cpu().numpy()
    sec = time.time() - t0
    np.savez(fn + '.tmp.npz', measure=m, seconds=sec, shapes=np.array(sorted(r['shapes'])))
    os.replace(fn + '.tmp.npz', fn)
    print(f'  chunk {c+1}/{nch}  {sec:.0f}s  {(e-s)/sec:.0f} px/s  conv={np.mean(m<0):.3f}  shapes={sorted(r["shapes"])}', flush=True)
    del r
if dev == 'cuda':
    torch.cuda.empty_cache()

ms, secs = [], []
for c in range(nch):
    d = np.load(os.path.join(outdir, f'chunk_{c:03d}.npz'))
    ms.append(d['measure']); secs.append(float(d['seconds']))
M = np.concatenate(ms).reshape(shape)
tot = float(np.sum(secs))
steady = [(min(nvox, (c + 1) * chunk) - c * chunk) / s_ for c, s_ in enumerate(secs)][1:] or [nvox / tot]
meta = dict(name=name, toy=args.toy, kind=KIND, res=R, steps=args.steps, dtype=args.dtype, chunk=chunk,
            seconds=tot, px_per_s=nvox / tot, px_per_s_excl_first=float(nvox - chunk) / float(np.sum(secs[1:])) if nch > 1 else nvox / tot,
            chunk_seconds=secs, device=gpu, torch=torch.__version__, date=time.strftime('%F %T'),
            third_axis=third_name, sigma_plane=k1, overview_pixels=pix.tolist())
np.savez(final, measure=M, log10_eta=np.log10(eta.numpy()), third_log=np.asarray(third_log), pix=pix,
         meta=json.dumps(meta))
print(f'{name}: conv={np.mean(M<0):.4f}  {tot:.0f}s  {nvox/tot:.0f} px/s (excl. first chunk {meta["px_per_s_excl_first"]:.0f})', flush=True)
