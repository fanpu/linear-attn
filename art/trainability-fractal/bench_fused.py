"""Throughput of the fused kernel vs the torch engine on the *mixed* hero window,
plus a block-size / launch-size sweep."""
import argparse, time
import numpy as np, torch
import tfractal as tf
import tfast

p = argparse.ArgumentParser()
p.add_argument('--res', type=int, default=256)
p.add_argument('--steps', type=int, default=500)
a = p.parse_args()
tf.setup_gpu(0.30)
HERO = dict(c0=1.0801625442981355, c1=2.301518970647778, hw=0.045)
prob = tf.make_problem(0, nonlin='tanh')
h0, h1 = tf.log_grid(HERO['c0'], HERO['c1'], HERO['hw'], a.res)
P = h0.numel()
tfast.load()

def timeit(fn, label):
    torch.cuda.synchronize(); t0 = time.time()
    m = fn()
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f'{label:44s} {dt:8.2f}s  {P/dt:8.1f} px/s  conv={float((m<0).mean()):.6f}', flush=True)
    return m, dt

ref = None
for blk in (128, 160, 192, 224, 256):
    for L in (131072,):
        m, dt = timeit(lambda blk=blk, L=L: tf.run_grid(
            prob, h0, h1, steps=a.steps, chunk=L, verbose=False,
            trainer=lambda *A, **K: tfast.train_chunk(*A, block=blk, launch=L, **K)),
            f'fused block={blk} launch={L}')
        if ref is None:
            ref = m
        else:
            print(f'      vs first fused: flips {int(((m<0)!=(ref<0)).sum())}, '
                  f'max|rel| {np.nanmax(np.abs(m-ref)/np.maximum(np.abs(ref),1e-300)):.2e}')

for ee in (True, False):
    m, dt = timeit(lambda ee=ee: tf.run_grid(prob, h0, h1, steps=a.steps, chunk=131072,
                                             verbose=False, engine='fused', early_exit=ee),
                   f'fused early_exit={ee}')
mt, _ = timeit(lambda: tf.run_grid(prob, h0, h1, steps=a.steps, chunk=32768, verbose=False,
                                   engine='torch'), 'torch compiled+early_exit (production)')
print(f'fused vs torch: flips {int(((ref<0)!=(mt<0)).sum())}/{P}  '
      f'max|rel| {np.nanmax(np.abs(ref-mt)/np.maximum(np.abs(mt),1e-300)):.2e}')
