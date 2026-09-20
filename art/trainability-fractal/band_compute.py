"""Compute a large window in horizontal bands, saving each band as it finishes.

window_compute.py holds the whole grid in one process and writes a single npz at the
end, so a 2048^2 float64 run is a ~20 h all-or-nothing exposure. This driver cuts the
same grid into bands, writes cache/bands/<name>/band_XXX.npz per band, and skips bands
that already exist, so the job is resumable and its progress is inspectable.

Grid coordinates are taken from the same formula as tfractal.log_grid, so band rows are
exactly the rows of the full res x res grid -- no resampling, no seams.

  ../_shared/gpu_run.sh ../.venv/bin/python band_compute.py \
      --name deep_zoomA4_2048_f64 --res 2048 --bands 16 \
      --c0 1.0801625442981355 --c1 2.301518970647778 --hw 0.045 --hwy 0.045
"""
import argparse, os, time, json
import numpy as np, torch
import tfractal as tf

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--res', type=int, required=True)
p.add_argument('--bands', type=int, default=16)
p.add_argument('--c0', type=float, required=True)
p.add_argument('--c1', type=float, required=True)
p.add_argument('--hw', type=float, required=True)
p.add_argument('--hwy', type=float, default=None)
p.add_argument('--nonlin', default='tanh')
p.add_argument('--steps', type=int, default=500)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--chunk', type=int, default=32768)
p.add_argument('--dtype', default='float64')
p.add_argument('--merge_only', action='store_true')
a = p.parse_args()

tf.setup_gpu(0.10)
tf.DT = getattr(torch, a.dtype)
DT = tf.DT
R, B = a.res, a.bands
assert R % B == 0, 'res must divide evenly into bands'
rows = R // B
hwy = a.hwy if a.hwy is not None else a.hw
bdir = f'cache/bands/{a.name}'
os.makedirs(bdir, exist_ok=True)

# exact full-grid axis values, same construction as tfractal.log_grid
off = (torch.arange(R, dtype=DT) + 0.5) / R * 2.0 - 1.0
ten = torch.tensor(10.0, dtype=DT)
ex_full = (10.0 ** torch.tensor(a.c0, dtype=DT)) * torch.pow(ten, off * a.hw)
ey_full = (10.0 ** torch.tensor(a.c1, dtype=DT)) * torch.pow(ten, off * hwy)

prob = tf.make_problem(a.seed, nonlin=a.nonlin)

if not a.merge_only:
    for b in range(B):
        fn = f'{bdir}/band_{b:03d}.npz'
        if os.path.exists(fn):
            print(f'band {b:3d}/{B} exists, skip', flush=True); continue
        ey = ey_full[b*rows:(b+1)*rows]
        EY, EX = torch.meshgrid(ey, ex_full, indexing='ij')
        h0, h1 = EX.reshape(-1).cuda(), EY.reshape(-1).cuda()
        t0 = time.time()
        r = tf.run_grid(prob, h0, h1, steps=a.steps, chunk=a.chunk, verbose=False)
        dt = time.time() - t0
        np.savez(fn, measure=r.reshape(rows, R), row0=b*rows, rows=rows, seconds=dt)
        done = b + 1
        print(f'band {b:3d}/{B}  {dt:6.0f}s  conv={np.mean(r<0):.3f}  '
              f'eta {(B-done)*dt/3600:5.2f} h left', flush=True)

# merge
parts = sorted(os.listdir(bdir))
if len(parts) == B:
    M = np.empty((R, R), np.float64); tot = 0.0
    for f in parts:
        d = np.load(f'{bdir}/{f}')
        M[int(d['row0']):int(d['row0'])+int(d['rows'])] = d['measure']; tot += float(d['seconds'])
    out = f'cache/windows/{a.name}.npz'
    np.savez(out, c0=a.c0, c1=a.c1, hw=a.hw, hwy=hwy, res=R, steps=a.steps,
             nonlin=a.nonlin, axes='lr_lr', minibatch=-1, seconds=tot,
             seed=a.seed, dtype=a.dtype, measure=M)
    print(f'merged -> {out}  conv={np.mean(M<0):.4f}  total {tot/3600:.2f} h', flush=True)
else:
    print(f'{len(parts)}/{B} bands present; rerun to continue', flush=True)
