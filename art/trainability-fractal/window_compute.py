"""Compute one 2D window for one configuration and cache it.

axes:
  lr_lr     x = log10 eta0 (input layer lr),  y = log10 eta1 (output layer lr)
  sigma_lr  x = log10 sigma (init scale, multiplies both W0 and W1 at init),
            y = log10 eta (same lr for both layers)
  wd_lr     x = log10 lambda (L2 weight decay added to the gradient), y = log10 eta
nonlin: tanh | relu | sin | identity | quadratic (null model, lr_lr only)

  gpu_run.sh python window_compute.py --name overview_tanh_1024 --res 1024
"""
import argparse, os, time
import numpy as np, torch
import tfractal as tf

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--nonlin', default='tanh')
p.add_argument('--axes', default='lr_lr')
p.add_argument('--res', type=int, default=256)
p.add_argument('--steps', type=int, default=500)
p.add_argument('--c0', type=float, default=1.5)
p.add_argument('--c1', type=float, default=1.5)
p.add_argument('--hw', type=float, default=4.5)
p.add_argument('--hwy', type=float, default=None)
p.add_argument('--minibatch', type=int, default=None)
p.add_argument('--checkpoints', default=None, help='start:stop:step, e.g. 20:500:20')
p.add_argument('--from_zoom', default=None, help='tag:k -> take window from a zoom keyframe')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--chunk', type=int, default=32768)
p.add_argument('--dtype', default='float64')
args = p.parse_args()
tf.setup_gpu(0.10)
tf.DT = getattr(torch, args.dtype)
os.makedirs('cache/windows', exist_ok=True)
fn = f'cache/windows/{args.name}.npz'
if os.path.exists(fn):
    print('exists', fn); raise SystemExit
c0, c1, hw = args.c0, args.c1, args.hw
if args.from_zoom:
    tag, k = args.from_zoom.split(':')
    d = np.load(f'cache/zoom_{tag}/kf_{int(k):03d}.npz')
    c0, c1, hw = float(d['c0']), float(d['c1']), float(d['hw'])
hwy = args.hwy if args.hwy is not None else hw
prob = tf.make_problem(args.seed, nonlin=args.nonlin)
hx, hy = tf.log_grid(c0, c1, (hw, hwy), args.res)
kw = {}
if args.axes == 'lr_lr':
    h0, h1 = hx, hy
elif args.axes == 'sigma_lr':
    h0, h1 = hy, hy
    kw = dict(sigma0=hx, sigma1=hx)
elif args.axes == 'wd_lr':
    h0, h1 = hy, hy
    kw = dict(wd=hx)
cps = None
if args.checkpoints:
    a, b, s = map(int, args.checkpoints.split(':'))
    cps = list(range(a, b + 1, s))
t0 = time.time()
r = tf.run_grid(prob, h0, h1, steps=args.steps, chunk=args.chunk, minibatch=args.minibatch,
                checkpoints=cps, **kw)
dt = time.time() - t0
R = args.res
save = dict(c0=c0, c1=c1, hw=hw, hwy=hwy, res=R, steps=args.steps, nonlin=args.nonlin,
            axes=args.axes, minibatch=-1 if args.minibatch is None else args.minibatch,
            seconds=dt, seed=args.seed, dtype=args.dtype)
if cps:
    save['measure'] = r[0].reshape(R, R); save['measure_T'] = r[1].reshape(len(cps), R, R)
    save['checkpoints'] = np.array(cps)
else:
    save['measure'] = r.reshape(R, R)
np.savez(fn, **save)
M = save['measure']
print(f'{args.name}: conv={np.mean(M<0):.3f} {dt:.0f}s ({R*R/dt:.0f} px/s)', flush=True)
