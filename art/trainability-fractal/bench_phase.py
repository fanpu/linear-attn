"""PERF-BRIEF s4, phase 2: is the all-converged band really faster than a mixed window
on an *idle* GPU, at identical P / chunk / settings?

band_000 of deep_zoomA4_2048_f64 is rows 0..127 of the 2048^2 grid (100% converged),
262144 px, timed at 868 s -> 302 px/s on a *shared* GPU. Rebuild exactly that pixel
set and compare against an equally large mixed set from the same window.
"""
import time
import numpy as np, torch
import tfractal as tf

tf.setup_gpu(0.30)
DT = tf.DT
C0, C1, HW = 1.0801625442981355, 2.301518970647778, 0.045
R = 2048
off = (torch.arange(R, dtype=DT) + 0.5) / R * 2.0 - 1.0
ten = torch.tensor(10.0, dtype=DT)
ex = (10.0 ** torch.tensor(C0, dtype=DT)) * torch.pow(ten, off * HW)
ey = (10.0 ** torch.tensor(C1, dtype=DT)) * torch.pow(ten, off * HW)
prob = tf.make_problem(0, nonlin='tanh')

def band(b, rows=128):
    EY, EX = torch.meshgrid(ey[b*rows:(b+1)*rows], ex, indexing='ij')
    return EX.reshape(-1).cuda(), EY.reshape(-1).cuda()

for label, b in [('band_000 (all-converged)', 0), ('band_008 (mixed)', 8)]:
    for ee in (True, False):
        h0, h1 = band(b)
        torch.cuda.synchronize(); t0 = time.time()
        m = tf.run_grid(prob, h0, h1, steps=500, chunk=32768, verbose=False, early_exit=ee)
        torch.cuda.synchronize(); dt = time.time() - t0
        print(f'{label:26s} early_exit={ee!s:5s} {dt:7.1f}s {h0.numel()/dt:7.1f} px/s '
              f'conv={float((m<0).mean()):.4f}', flush=True)
