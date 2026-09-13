"""Deflationary companion (Liu 2024, arXiv:2406.13971): trivially non-convex 2-parameter
losses through exactly the same pipeline (two learning rates, his convergence measure,
same edge set and box counting). CPU, float64.

  L(a,b) = a^2 + 2 rho a b + b^2 + eps * (1 + cos(2 pi (a - b) / lam))      (>= 0)
  a <- a - eta0 dL/da,  b <- b - eta1 dL/db,  a0 = b0 = 1, 500 steps.
eps = 0 is the pure quadratic (smooth stability boundary). Liu's roughness for the
cosine term is eps/lam^2 (non-convex once 4 pi^2 eps / lam^2 > 2(1-|rho|) roughly).

  OMP_NUM_THREADS=4 python liu_toy.py --res 512 --eps 0.05 --lam 0.2 --name liu_eps05
"""
import argparse, os, time
import numpy as np, torch

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--res', type=int, default=512)
p.add_argument('--steps', type=int, default=500)
p.add_argument('--eps', type=float, default=0.05)
p.add_argument('--lam', type=float, default=0.2)
p.add_argument('--rho', type=float, default=0.3)
p.add_argument('--c0', type=float, default=-0.3)
p.add_argument('--c1', type=float, default=-0.3)
p.add_argument('--hw', type=float, default=0.6)
args = p.parse_args()
torch.set_num_threads(4)
DT = torch.float64
R = args.res
off = (torch.arange(R, dtype=DT) + 0.5) / R * 2 - 1
e0 = 10 ** torch.tensor(args.c0, dtype=DT) * torch.pow(torch.tensor(10., dtype=DT), off * args.hw)
e1 = 10 ** torch.tensor(args.c1, dtype=DT) * torch.pow(torch.tensor(10., dtype=DT), off * args.hw)
E1, E0 = torch.meshgrid(e1, e0, indexing='ij')
a = torch.ones_like(E0); b = torch.ones_like(E0)
k = 2 * np.pi / args.lam
eps, rho = args.eps, args.rho
S = torch.zeros_like(E0); Sinv = torch.zeros_like(E0); Slast = torch.zeros_like(E0)
t0 = time.time()
for t in range(args.steps):
    u = a - b
    L = a * a + 2 * rho * a * b + b * b + eps * (1 + torch.cos(k * u))
    L = torch.where(torch.isfinite(L), L, torch.full_like(L, 1e6))
    if t == 0:
        L0 = L.clone()
    v = torch.clamp(L / L0, max=1e6)
    S += v; Sinv += 1 / v
    if t >= args.steps - 20:
        Slast += v
    s = -eps * k * torch.sin(k * u)
    ga = 2 * a + 2 * rho * b + s
    gb = 2 * b + 2 * rho * a - s
    a = a - E0 * ga
    b = b - E1 * gb
M = torch.where(Slast / 20 < 1, -S, Sinv).numpy()
os.makedirs('cache/windows', exist_ok=True)
np.savez(f'cache/windows/{args.name}.npz', measure=M, c0=args.c0, c1=args.c1, hw=args.hw, hwy=args.hw,
         res=R, steps=args.steps, nonlin='liu_cos', axes='lr_lr', minibatch=-1, eps=eps, lam=args.lam,
         rho=rho, seconds=time.time() - t0, seed=0)
print(args.name, 'conv', (M < 0).mean(), f'{time.time()-t0:.1f}s')
