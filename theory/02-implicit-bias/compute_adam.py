"""Build-on #1: which margin does Adam reach on separable logistic regression, with a realistic epsilon?

Same 8 datasets as the Soudry reproduction (d=50, n=40), plus the 2D geometry dataset.  Full-batch Adam with bias
correction, beta1 = 0.9, beta2 = 0.999, constant lr, eps in {0, 1e-16, 1e-12, 1e-8, 1e-6, 1e-4}; float64.
Gradients are carried in log-scale (g = e^C * ghat, moments rescaled every step) so exp(-margin) underflow can never
freeze or corrupt the eps = 0 runs.  Writes cache/adam_<name>.npz.

    python compute_adam.py gauss|geom [T] [lr] [const|sqrt]
"""
import sys, time
import numpy as np
from core import gaussian_separable, geometry_2d, svm, log_checkpoints

name = sys.argv[1]
T = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10 ** 7
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
sched = sys.argv[4] if len(sys.argv) > 4 else "const"   # const | sqrt  (lr_t = lr / sqrt(t), as in Zhang et al. with a = 1/2)
EPS = np.array([0.0, 1e-16, 1e-12, 1e-8, 1e-6, 1e-4])
b1, b2 = 0.9, 0.999

if name == "gauss":
    data = [gaussian_separable(40, 50, s) for s in range(8)]
elif name == "geom":
    data = [geometry_2d()]
Zs = np.stack([y[:, None] * X for X, y in data])  # S, n, d
S_, n, d = Zs.shape
E = len(EPS)
Z = np.repeat(Zs, E, axis=0)  # R = S*E runs
eps = np.tile(EPS, S_)
R = Z.shape[0]
w = np.zeros((R, d))
M = np.zeros((R, d)); V = np.zeros((R, d)); C = np.zeros((R, 1))
ts = log_checkpoints(T, 40)
rec = np.empty((len(ts), R, d))
recC = np.empty((len(ts), R))       # log gradient scale
recV = np.empty((len(ts), R))       # log10 of mean sqrt(vhat)  (to compare with eps)
k = 0
t0 = time.time()
logeps = np.where(eps > 0, np.log(np.maximum(eps, 1e-300)), -np.inf)[:, None]
for t in range(1, T + 1):
    m = np.einsum("rnd,rd->rn", Z, w)
    lw = -np.logaddexp(0.0, m)
    Cn = lw.max(1, keepdims=True)
    gh = -np.einsum("rn,rnd->rd", np.exp(lw - Cn), Z)
    fac = np.exp(C - Cn)
    M = b1 * M * fac + (1 - b1) * gh
    V = b2 * V * fac ** 2 + (1 - b2) * gh ** 2
    C = Cn
    Mh = M / (1 - b1 ** t)
    Vh = V / (1 - b2 ** t)
    # eps in ghat units: eps * e^{-C}; exp overflow -> inf -> the update is exactly the eps-dominated limit 0
    with np.errstate(over="ignore"):
        eps_h = np.exp(np.minimum(logeps - C, 700.0))
    lr_t = lr if sched == "const" else lr / np.sqrt(t)
    w -= lr_t * Mh / (np.sqrt(Vh) + eps_h)
    if t == ts[k]:
        rec[k] = w; recC[k] = C[:, 0]
        recV[k] = (C[:, 0] + np.log(np.sqrt(Vh).mean(1))) / np.log(10)
        k += 1
        if k % 40 == 0:
            print(name, t, f"{time.time() - t0:.0f}s", flush=True)

W2 = np.array([svm(X, y, "l2")[0] for X, y in data])
Winf = np.array([svm(X, y, "linf")[0] for X, y in data])
np.savez(f"cache/adam_{name}_lr{lr:g}_{sched}.npz", steps=ts, W=rec.reshape(len(ts), S_, E, d), logC=recC.reshape(len(ts), S_, E),
         log10_sqrtv=recV.reshape(len(ts), S_, E), eps=EPS, lr=lr, sched=sched, Z=Zs, W_l2=W2, W_linf=Winf)
print("done", time.time() - t0)
