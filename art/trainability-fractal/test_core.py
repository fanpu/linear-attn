"""Sanity tests: compiled manual gradients vs autograd; early-exit+checkpoints vs exact."""
import math, time, torch, numpy as np
import tfractal as tf
tf.setup_gpu(0.10)
for nl in ['tanh', 'relu', 'sin']:
    prob = tf.make_problem(0, nonlin=nl)
    P = 3
    W0 = (prob['W0'].expand(P,16,16) * torch.tensor([0.5,1,2],dtype=tf.DT,device='cuda').view(P,1,1)).clone().requires_grad_()
    W1 = prob['W1'].expand(P,16,1).clone().requires_grad_()
    X, Y = prob['X'], prob['Y']
    Z = X @ W0 / 4
    h = {'tanh': lambda z: torch.tanh(z*math.sqrt(2)), 'relu': lambda z: torch.relu(z)*math.sqrt(2), 'sin': lambda z: torch.sin(z*math.sqrt(2))}[nl](Z)
    L = (((h @ W1) / 16 - Y)**2).mean(dim=(1,2)); L.sum().backward()
    lg, _, _ = tf.get_steps(nl, 16, compiled=False)
    with torch.no_grad():
        l2, g0, g1 = lg(W0.detach(), W1.detach(), X, Y)
    print(nl, 'loss err', (L-l2).abs().max().item(), 'g0 rel err', ((W0.grad-g0).abs().max()/W0.grad.abs().max()).item(),
          'g1 rel err', ((W1.grad-g1).abs().max()/W1.grad.abs().max()).item(), flush=True)

prob = tf.make_problem(0, nonlin='tanh')
R = 64
e0, e1 = tf.log_grid(1.5, 1.5, 4.5, R)
cps = [100, 250]
t = time.time()
m_ex, mT_ex = tf.run_grid(prob, e0, e1, steps=500, early_exit=False, checkpoints=cps, verbose=False)
print('exact', time.time()-t)
t = time.time()
m_ee, mT_ee = tf.run_grid(prob, e0, e1, steps=500, early_exit=True, checkpoints=cps, verbose=False)
print('early', time.time()-t)
for k in range(mT_ex.shape[0]):
    a, b = mT_ex[k], mT_ee[k]
    print('cp', k, 'sign flips', int(((a<0)!=(b<0)).sum()), 'max rel', float(np.max(np.abs(a-b)/np.abs(a))))
# separate short runs must equal checkpoints (same engine on both sides: the
# checkpoint path is torch-only, so the standalone run has to be torch too)
for k, T in enumerate(cps):
    mT = tf.run_grid(prob, e0, e1, steps=T, early_exit=False, verbose=False, engine='torch')
    print('T', T, 'checkpoint vs standalone max rel', float(np.max(np.abs(mT - mT_ex[k])/np.abs(mT))))
# compare with old eager toy at 128 (first 64? different grid) -> recompute toy 128 exact check against file
# cache/toy_128_tanh.npy was computed by the eager torch engine; the default engine
# is now the fused CUDA kernel, so a handful of boundary-pixel flips is expected here
toy = np.load('cache/toy_128_tanh.npy')
e0, e1 = tf.log_grid(1.5, 1.5, 4.5, 128)
m128 = tf.run_grid(prob, e0, e1, steps=500, verbose=False)
print('toy128 file vs compiled+early: sign flips', int(((toy<0)!=(m128<0)).sum()), 'frac conv', (m128<0).mean())
