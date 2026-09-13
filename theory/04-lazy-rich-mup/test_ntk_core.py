"""Checks for ntk_core: torch.func NTK == explicit Jacobian; wide NTK -> analytic; linearized GD == iterate."""
import math

import numpy as np
import torch

from ntk_core import cifar_binary, empirical_ntk, init_params, linearized_gd, net, relu_ntk_analytic

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.08)

xtr, ytr, xte, yte = cifar_binary(n_train=200, n_test=100)
assert xtr.shape == (200, 192) and abs(np.linalg.norm(xtr[0]) - 1) < 1e-12 and set(ytr) == {-1.0, 1.0}
X = torch.tensor(xtr[:40], device=dev)

# 1) empirical NTK via jvp/vjp matches explicit Jacobian product
p = init_params(192, 50, 0, dev)
J = torch.func.jacrev(lambda W, a: net({"W": W, "a": a}, X), argnums=(0, 1))(p["W"], p["a"])
Jf = torch.cat([J[0].reshape(40, -1), J[1].reshape(40, -1)], 1)
K = empirical_ntk(p, X, chunk=7)
err = (K - Jf @ Jf.T).abs().max().item()
print(f"jvp/vjp NTK vs explicit Jacobian: max err {err:.2e}")
assert err < 1e-10

# 2) wide empirical NTK approaches closed form, error ~ 1/sqrt(m)
Ka = relu_ntk_analytic(xtr[:40], xtr[:40])
errs = []
for m in [1000, 16000]:
    Ke = empirical_ntk(init_params(192, m, 1, dev), X).cpu().numpy()
    errs.append(np.linalg.norm(Ke - Ka) / np.linalg.norm(Ka))
print(f"|K_emp - K_inf|/|K_inf|: m=1000 {errs[0]:.4f}, m=16000 {errs[1]:.4f}, ratio {errs[0]/errs[1]:.2f} (expect ~4)")
assert 2.5 < errs[0] / errs[1] < 6

# 3) linearized GD closed form == explicit iteration
rng = np.random.default_rng(0)
A = rng.standard_normal((30, 30)); Ktt = A @ A.T / 30
Kat = np.concatenate([Ktt, rng.standard_normal((10, 30))])
y = rng.standard_normal(30); f0t = rng.standard_normal(30) * .1; f0a = np.concatenate([f0t, rng.standard_normal(10)])
lr = 0.5 * 30 / np.linalg.eigvalsh(Ktt).max()
fa, ft = f0a.copy(), f0t.copy()
for _ in range(137):
    r = ft - y
    fa, ft = fa - lr / 30 * Kat @ r, ft - lr / 30 * Ktt @ r
got = linearized_gd(Kat, Ktt, f0a, f0t, y, lr, 137)
print(f"linearized GD closed form vs iteration: {np.abs(got - fa).max():.2e}")
assert np.abs(got - fa).max() < 1e-9
print("all ok")

# 4) toy hand-written gradients == autograd
from toy_core import forward, grads_np, init as toy_init, spiral
xs, ys = spiral(50)
pt = toy_init(16, 0)
for k in pt:
    pt[k].requires_grad_(True)
X_, Y_ = torch.tensor(xs), torch.tensor(ys)
loss = 0.5 * ((forward(pt, X_, 3.0) - Y_) ** 2).mean() / 9.0
g_auto = torch.autograd.grad(loss, [pt["w"], pt["b"], pt["a"]])
g_np = grads_np(*(pt[k].detach().numpy() for k in "wba"), xs, ys, 3.0)[:3]
e = max(np.abs(ga.numpy() - gn).max() for ga, gn in zip(g_auto, g_np))
print(f"toy grads numpy vs autograd: {e:.2e}")
assert e < 1e-12
print("toy ok")
