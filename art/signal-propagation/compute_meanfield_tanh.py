"""Part A, analytic: mean-field phase diagram of a deep tanh MLP (Poole et al. 2016;
Schoenholz et al. 2017) on the (sigma_w^2, sigma_b^2) plane, by Gauss-Hermite quadrature.

  q*    : sigma_w^2 E[tanh(sqrt(q*) z)^2] + sigma_b^2 = q*               (bisection)
  chi_1 : sigma_w^2 E[tanh'(sqrt(q*) z)^2]
  c*    : largest root in [0,1] of C(c) = c,  C(c) = (sigma_w^2 E[phi(u1)phi(u2)] + sigma_b^2)/q*
  xi_q  : -1 / log(chi_1 + sigma_w^2 E[phi''(u) phi(u)])
  xi_c  : -1 / log(sigma_w^2 E[phi'(u1) phi'(u2)])  at c*   (= -1/log chi_1 in the ordered phase)

  python compute_meanfield_tanh.py --W 2400 --H 1200
"""
import argparse, math, os, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--W", type=int, default=2400)
ap.add_argument("--H", type=int, default=1200)
ap.add_argument("--sw2", type=float, nargs=2, default=[0.5, 4.5])
ap.add_argument("--sb2", type=float, nargs=2, default=[0.0, 2.0])
ap.add_argument("--n1", type=int, default=121)
ap.add_argument("--n2", type=int, default=48)
ap.add_argument("--out", default="cache/meanfield_tanh.npz")
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.10)
dev, dt = "cuda", torch.float64
here = os.path.dirname(os.path.abspath(__file__))


def gh(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return torch.tensor(x, device=dev, dtype=dt), torch.tensor(w / math.sqrt(2 * math.pi), device=dev, dtype=dt)


z1, w1 = gh(args.n1)
za, wa = gh(args.n2)
Za, Zb = torch.meshgrid(za, za, indexing="ij")
Wab = (wa[:, None] * wa[None, :]).reshape(-1)
Za, Zb = Za.reshape(-1), Zb.reshape(-1)

xs = args.sw2[0] + (np.arange(args.W) + 0.5) * (args.sw2[1] - args.sw2[0]) / args.W
ys = args.sb2[0] + (np.arange(args.H) + 0.5) * (args.sb2[1] - args.sb2[0]) / args.H
SW2, SB2 = np.meshgrid(xs, ys)
sw2_all = torch.tensor(SW2.ravel(), device=dev, dtype=dt)
sb2_all = torch.tensor(SB2.ravel(), device=dev, dtype=dt)
P = sw2_all.numel()
keys = ["qstar", "chi1", "cstar", "chic", "xi_q", "xi_c", "fallback"]
res = {k: np.zeros(P) for k in keys}
t0 = time.time()
CH = 16384
for s in range(0, P, CH):
    sw2, sb2 = sw2_all[s:s + CH, None], sb2_all[s:s + CH, None]
    # q* by bisection on g(q) = sw2 E tanh^2 + sb2 - q (g(0) >= 0, g(sw2+sb2) <= 0)
    lo = torch.zeros_like(sw2); hi = sw2 + sb2 + 1e-12
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        g = sw2 * (w1 * torch.tanh(mid.sqrt() * z1) ** 2).sum(1, keepdim=True) + sb2 - mid
        lo = torch.where(g > 0, mid, lo); hi = torch.where(g > 0, hi, mid)
    q = 0.5 * (lo + hi)
    u = q.sqrt() * z1
    sech2 = 1 / torch.cosh(u) ** 2
    chi1 = sw2 * (w1 * sech2 ** 2).sum(1, keepdim=True)
    th = torch.tanh(u)
    dq = chi1 + sw2 * (w1 * (-2 * th * sech2) * th).sum(1, keepdim=True)
    xi_q = -1 / torch.log(dq)

    sq = q.sqrt().clamp_min(1e-300)

    chaotic = (chi1 > 1).squeeze(1)
    cstar = torch.ones_like(q)
    chic = chi1.clone()
    fallback = torch.zeros_like(q)
    idx = chaotic.nonzero().squeeze(1)
    if idx.numel():
        sqc, qc, sw2c, sb2c, chi1c = sq[idx], q[idx], sw2[idx], sb2[idx], chi1[idx]

        def Cmap(c):
            u1 = sqc * Za
            u2 = sqc * (c * Za + torch.sqrt((1 - c * c).clamp_min(0)) * Zb)
            return (sw2c * (Wab * torch.tanh(u1) * torch.tanh(u2)).sum(1, keepdim=True) + sb2c) / qc

        # subtract the quadrature bias at c = 1 (2D rule vs the 1D rule that defined q*), so that
        # the trivial fixed point c = 1 is exact and the near-critical root is resolvable
        bias = Cmap(torch.ones_like(qc)) - 1
        lo = torch.zeros_like(qc); hi = torch.full_like(qc, 1 - 1e-15)
        for _ in range(44):
            mid = 0.5 * (lo + hi)
            f = Cmap(mid) - mid - bias
            lo = torch.where(f > 0, mid, lo); hi = torch.where(f > 0, hi, mid)
        cs = 0.5 * (lo + hi)
        u1 = sqc * Za
        u2 = sqc * (cs * Za + torch.sqrt((1 - cs ** 2).clamp_min(0)) * Zb)
        cc = sw2c * (Wab / (torch.cosh(u1) ** 2 * torch.cosh(u2) ** 2)).sum(1, keepdim=True)
        # transcritical first-order fallback C'(c*) ~ 2 - chi_1 if quadrature still fails (counted)
        bad = cc >= 1 - 1e-12
        cc = torch.where(bad, 2 - chi1c, cc)
        cstar[idx], chic[idx], fallback[idx] = cs, cc, bad.double()
    xi_c = -1 / torch.log(chic)
    for k, v in zip(keys, [q, chi1, cstar, chic, xi_q, xi_c, fallback]):
        res[k][s:s + CH] = v.squeeze(1).cpu().numpy()
    if (s // CH) % 20 == 0:
        print(f"{s + CH}/{P}  {time.time() - t0:.0f}s", flush=True)

# critical line sigma_w^2(sigma_b^2) by 1D bisection on chi_1 = 1 at high precision (CPU)
zc, wc = np.polynomial.hermite_e.hermegauss(201); wc /= math.sqrt(2 * math.pi)


def chi1_1d(sw2, sb2):
    lo, hi = 0.0, sw2 + sb2
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if sw2 * np.sum(wc * np.tanh(np.sqrt(m) * zc) ** 2) + sb2 - m > 0: lo = m
        else: hi = m
    q = 0.5 * (lo + hi)
    return sw2 * np.sum(wc / np.cosh(np.sqrt(q) * zc) ** 4)


line_sb2 = np.linspace(0, args.sb2[1], 401)
line_sw2 = []
for b in line_sb2:
    lo, hi = 0.5, 20.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if chi1_1d(m, b) < 1: lo = m
        else: hi = m
    line_sw2.append(0.5 * (lo + hi))
np.savez_compressed(os.path.join(here, args.out), sw2=xs, sb2=ys, line_sb2=line_sb2, line_sw2=np.array(line_sw2),
                    wall=time.time() - t0, **{k: v.reshape(args.H, args.W) for k, v in res.items()})
print("done", time.time() - t0, "critical sw2 at sb2=0.05:", np.interp(0.05, line_sb2, line_sw2))
