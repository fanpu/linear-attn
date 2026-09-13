"""Sanity tests: manual gradients vs autograd; early-exit vs exact; throughput."""
import math, time, torch, numpy as np
import tfractal as tf
tf.setup_gpu(0.10)
for nl in ['tanh', 'relu']:
    prob = tf.make_problem(0, nonlin=nl)
    P = 3
    W0 = (prob['W0'].expand(P,16,16) * torch.tensor([0.5,1,2],dtype=tf.DT,device='cuda').view(P,1,1)).clone().requires_grad_()
    W1 = prob['W1'].expand(P,16,1).clone().requires_grad_()
    X, Y = prob['X'], prob['Y']
    Z = X @ W0 / 4
    h = torch.tanh(Z*math.sqrt(2)) if nl=='tanh' else torch.relu(Z)*math.sqrt(2)
    out = h @ W1 / 4 / 4
    L = ((out - Y)**2).mean(dim=(1,2))
    L.sum().backward()
    with torch.no_grad():
        l2, g0, g1 = tf._loss_and_grad(W0.detach(), W1.detach(), X, Y, nl, 16)
    print(nl, 'loss err', (L-l2).abs().max().item(), 'g0 err', (W0.grad-g0).abs().max().item()/W0.grad.abs().max().item(),
          'g1 err', (W1.grad-g1).abs().max().item()/W1.grad.abs().max().item())

prob = tf.make_problem(0, nonlin='tanh')
e0, e1 = tf.log_grid(1.5, 1.5, 4.5, 128)
for ee in [False, True]:
    torch.cuda.synchronize(); t=time.time()
    m = tf.run_grid(prob, e0, e1, steps=500, chunk=16384, early_exit=ee, verbose=False)
    torch.cuda.synchronize(); dt=time.time()-t
    print('early_exit', ee, f'{dt:.1f}s', f'{m.size/dt:.0f} px/s', 'frac conv', (m<0).mean())
    if not ee: m_exact = m
    else:
        print(' sign mismatch', ((m<0)!=(m_exact<0)).sum(), 'max rel diff', np.max(np.abs(m-m_exact)/np.abs(m_exact)))
np.save('cache/toy_128_tanh.npy', m_exact)
