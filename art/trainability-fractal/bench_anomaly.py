"""Evidence for PERF-BRIEF s4: is the mixed-window slowdown caused by early-exit
compaction changing shapes and forcing torch.compile recompiles?

Runs one 256^2 window from the hero (mixed-phase) region and one all-converged
window, over {early_exit} x {compiled}, and reports wall time and px/s.

  TORCH_LOGS=recompiles ../.venv/bin/python bench_anomaly.py
"""
import argparse, time
import numpy as np, torch
import tfractal as tf

p = argparse.ArgumentParser()
p.add_argument('--res', type=int, default=256)
p.add_argument('--steps', type=int, default=500)
p.add_argument('--chunk', type=int, default=32768)
p.add_argument('--which', default='all')
a = p.parse_args()

tf.setup_gpu(0.30)
prob = tf.make_problem(0, nonlin='tanh')

# hero (mixed, conv_frac ~0.512) and band-0 style all-converged window
WINDOWS = {
    'mixed': dict(c0=1.0801625442981355, c1=2.301518970647778, hw=0.045),
}

def run(tag, c0, c1, hw, early_exit, compiled):
    h0, h1 = tf.log_grid(c0, c1, hw, a.res)
    torch.cuda.synchronize(); t0 = time.time()
    m = tf.run_grid(prob, h0, h1, steps=a.steps, chunk=a.chunk, verbose=False,
                    early_exit=early_exit, compiled=compiled)
    torch.cuda.synchronize(); dt = time.time() - t0
    P = h0.numel()
    print(f'{tag:8s} early_exit={early_exit!s:5s} compiled={compiled!s:5s} '
          f'{dt:7.1f}s  {P/dt:7.1f} px/s  conv={float((m<0).mean()):.4f}', flush=True)
    return m, dt

for tag, w in WINDOWS.items():
    res = {}
    for compiled in (True, False):
        for ee in (True, False):
            if a.which != 'all' and a.which != f'{int(ee)}{int(compiled)}':
                continue
            m, dt = run(tag, w['c0'], w['c1'], w['hw'], ee, compiled)
            res[(ee, compiled)] = (m, dt)
    ms = list(res.values())
    if len(ms) > 1:
        base = ms[0][0]
        for k, (m, dt) in res.items():
            print(f'   {k}: sign flips vs first = {int(((m<0)!=(base<0)).sum())}, '
                  f'max rel = {float(np.max(np.abs(m-base)/np.maximum(np.abs(base),1e-300))):.3e}')
