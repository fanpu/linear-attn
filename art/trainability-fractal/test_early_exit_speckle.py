"""Check the early-exit approximation in the high-eta0 'catapult' regime."""
import numpy as np, torch
import tfractal as tf
tf.setup_gpu(0.10)
prob = tf.make_problem(0, nonlin='tanh')
R = 40
for name, c0, c1, hw in [('speckle_z10', 5.548942834915078, 1.7992672733723396, 0.045),
                         ('kf2', 5.548942834915078, 1.7992672733723396, 0.45)]:
    e0, e1 = tf.log_grid(c0, c1, hw, R)
    ex = tf.run_grid(prob, e0, e1, steps=500, early_exit=False, verbose=False)
    ee = tf.run_grid(prob, e0, e1, steps=500, early_exit=True, verbose=False)
    ee2 = tf.run_grid(prob, e0, e1, steps=500, early_exit=True, exit_loss=1e300, verbose=False)
    print(name, 'conv exact', (ex<0).mean(), 'flips(1e100)', int(((ex<0)!=(ee<0)).sum()),
          'flips(1e300)', int(((ex<0)!=(ee2<0)).sum()), 'of', ex.size,
          'min measure', ex.min(), flush=True)
