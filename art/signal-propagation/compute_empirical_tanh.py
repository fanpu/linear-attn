"""Part A, empirical: push inputs through wide random tanh MLPs on the (sigma_w^2, sigma_b^2)
plane and *measure* the depth scales. No mean-field formula is used anywhere in here.

Per pixel and per realisation k = 0..K-1 (CRN: realisation k's standard-normal W, b are
shared by every pixel, scaled by that pixel's sigma_w, sigma_b), five inputs are propagated:
  A        |h0|^2/N = 1
  B        independent of A              -> pair (A,B) starts at c0 ~ 0
  C        0.99 A + sqrt(1-0.99^2) B'    -> pair (A,C) starts at c0 = 0.99
  Dp       A + 1e-3 relative noise       -> infinitesimal pair: growth rate = chi_1
  E        2 A (|h0|^2/N = 4)            -> length transient from above
Recorded per layer (pre-activations z): q_A, q_E, c_AB, c_AC, d_AD = |z_A - z_Dp|^2/N,
averaged over K realisations. Fits (all from these curves):
  xi_c  : joint fit  log|c^l - c*_emp| = a_pair - l / xi_c, c*_emp = tail mean (last 100 layers)
  xi_q  : joint fit  log|q^l - q*_emp| = a_vec  - l / xi_q
  chi_1 : exp(slope of log d_AD over the linear-growth/decay window)

  python compute_empirical_tanh.py --W 480 --H 240 --N 1000 --D 400 --K 4
"""
import argparse, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_core import layer_draw, inputs_draw

ap = argparse.ArgumentParser()
ap.add_argument("--W", type=int, default=480)
ap.add_argument("--H", type=int, default=240)
ap.add_argument("--sw2", type=float, nargs=2, default=[0.5, 4.5])
ap.add_argument("--sb2", type=float, nargs=2, default=[0.0, 2.0])
ap.add_argument("--N", type=int, default=1000)
ap.add_argument("--D", type=int, default=400)
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--out", default="cache/empirical_tanh.npz")
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.10)
dev, dt = "cuda", torch.float32
here = os.path.dirname(os.path.abspath(__file__))
N, D, K = args.N, args.D, args.K

xs = args.sw2[0] + (np.arange(args.W) + 0.5) * (args.sw2[1] - args.sw2[0]) / args.W
ys = args.sb2[0] + (np.arange(args.H) + 0.5) * (args.sb2[1] - args.sb2[0]) / args.H
SW2, SB2 = np.meshgrid(xs, ys)
P = SW2.size
series = {k: np.zeros((D + 1, P), dtype=np.float32) for k in ("qA", "qE", "cAB", "cAC", "dAD")}


@torch.compile(dynamic=False)
def step(h, Wt, b, w, bb):
    z = torch.mm(h, Wt) * w + bb * b
    return z, torch.tanh(z)


t0 = time.time()
for k in range(K):
    Ws = [layer_draw(1000 + k, N, l, dev, torch.float64) for l in range(1, D + 1)]
    Ws = [((W.T / math.sqrt(N)).to(dt).contiguous(), b.to(dt)) for W, b in Ws]
    X = inputs_draw(1000 + k, N, 3, dev, torch.float64)
    A, Bv, Bp = X[0], X[1], X[2]
    C = 0.99 * A + math.sqrt(1 - 0.99 ** 2) * Bp
    g = torch.Generator(device="cpu").manual_seed(777 + k)
    Dp = A + 1e-3 * torch.randn(N, generator=g, dtype=torch.float64).to(dev)
    E = 2 * A
    V0 = torch.stack([A, Bv, C, Dp, E]).to(dt)  # (5, N)
    for s in range(0, P, args.chunk):
        e = min(P, s + args.chunk); n = e - s; ch = args.chunk
        w = np.zeros(ch); w[:n] = np.sqrt(SW2.ravel()[s:e])
        bbv = np.zeros(ch); bbv[:n] = np.sqrt(SB2.ravel()[s:e])
        w = torch.tensor(np.tile(w, 5), device=dev, dtype=dt)[:, None]
        bbv = torch.tensor(np.tile(bbv, 5), device=dev, dtype=dt)[:, None]
        h = V0.repeat_interleave(ch, 0)  # (5*ch, N): blocks A,B,C,Dp,E
        buf = {kk: torch.zeros(D + 1, ch, device=dev, dtype=torch.float64) for kk in series}
        # layer 0 "pre-activations" are the inputs themselves
        z = h
        for l in range(D + 1):
            if l > 0:
                z, h = step(h, Ws[l - 1][0], Ws[l - 1][1], w, bbv)
            zA, zB, zC, zD, zE = z[:ch].double(), z[ch:2 * ch].double(), z[2 * ch:3 * ch].double(), z[3 * ch:4 * ch].double(), z[4 * ch:].double()
            nA, nB, nC = (zA ** 2).sum(1), (zB ** 2).sum(1), (zC ** 2).sum(1)
            buf["qA"][l] = nA / N
            buf["qE"][l] = (zE ** 2).sum(1) / N
            buf["cAB"][l] = (zA * zB).sum(1) / torch.sqrt(nA * nB)
            buf["cAC"][l] = (zA * zC).sum(1) / torch.sqrt(nA * nC)
            buf["dAD"][l] = ((z[:ch] - z[3 * ch:4 * ch]).double() ** 2).sum(1) / N
        for kk in series:
            series[kk][:, s:e] += (buf[kk][:, :n] / K).cpu().numpy().astype(np.float32)
        print(f"k={k} {e}/{P} {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ fits (numpy, vectorised)
def joint_slope(ys_list, masks):
    """Common slope of y vs layer with a separate intercept per curve, per pixel."""
    L = np.arange(D + 1, dtype=np.float64)[:, None]
    num = np.zeros(P); den = np.zeros(P)
    npts = np.zeros(P)
    for y, m in zip(ys_list, masks):
        m = m.astype(np.float64)
        S = m.sum(0); Sx = (m * L).sum(0); Sy = (m * y).sum(0)
        Sxx = (m * L * L).sum(0); Sxy = (m * L * y).sum(0)
        ok = S >= 2
        num += np.where(ok, Sxy - Sx * Sy / np.maximum(S, 1), 0)
        den += np.where(ok, Sxx - Sx ** 2 / np.maximum(S, 1), 0)
        npts += S
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), npts


def first_below(delta, floor, start):
    """mask of layers start <= l < first layer where delta < floor."""
    below = (delta < floor[None, :]) & (np.arange(D + 1)[:, None] >= start)
    below[-1] = True
    idx = np.argmax(below, axis=0)
    Lr = np.arange(D + 1)[:, None]
    return (Lr >= start) & (Lr < idx[None, :]) & (Lr < D - 100)


tail = slice(D - 99, D + 1)
cAB = series["cAB"].astype(np.float64)
cstar = cAB[tail].mean(0)
cnoise = cAB[tail].std(0)
Lr = np.arange(D + 1)[:, None]


def fit_xi_c(fhi, kf, minpts):
    delta = np.abs(cAB - cstar[None, :])
    floor = np.maximum(kf * cnoise, 2e-6)
    below = (delta < floor[None, :]) | (Lr == D)
    stop = np.argmax(below, axis=0)
    m = (delta <= fhi * delta[0][None, :]) & (Lr < stop[None, :]) & (Lr < D - 100) & (Lr >= 1)
    sl, npts = joint_slope([np.log(np.maximum(delta, 1e-300))], [m])
    return np.where((sl < 0) & (npts >= minpts), -1 / sl, np.nan), npts


# window chosen on a toy run against theory (README): the asymptotic regime |c - c*| <= 0.3 |c0 - c*|,
# stopped at 2 sigma of the finite-width noise floor; pixels with too few points use the full window
xi_c, npts_c = fit_xi_c(0.3, 2, 3)
xi_c_full, _ = fit_xi_c(1.0, 2, 2)
xi_c_flag = np.isnan(xi_c) & np.isfinite(xi_c_full)
xi_c = np.where(np.isnan(xi_c), xi_c_full, xi_c)

qA, qE = series["qA"].astype(np.float64), series["qE"].astype(np.float64)
qstar = 0.5 * (qA[tail].mean(0) + qE[tail].mean(0))
qfloor = np.maximum(4 * 0.5 * (qA[tail].std(0) + qE[tail].std(0)), 1e-6 * np.maximum(qstar, 1e-6))
ys_q, ms_q = [], []
for q in (qA, qE):
    delta = np.abs(q - qstar[None, :])
    m = first_below(delta, qfloor, 1)
    ys_q.append(np.log(np.maximum(delta, 1e-300))); ms_q.append(m)
slq, npts_q = joint_slope(ys_q, ms_q)
xi_q = np.where((slq < 0) & (npts_q >= 2), -1 / slq, np.nan)

# chi_1: growth/decay of an infinitesimal perturbation, after the length transient (l >= 8),
# while it stays in the linear regime (d < 1e-3 q) and above the float32 floor (d > 1e-10 q)
dAD = series["dAD"].astype(np.float64)
ok = (dAD < 1e-3 * qA) & (dAD > 1e-10 * np.maximum(qA, 1e-12)) & (Lr >= 8) & (Lr <= 120)
bad = ~ok & (Lr >= 8)
bad[-1] = True
stop = np.argmax(bad, axis=0)
mchi = ok & (Lr < stop[None, :])
slc, npts_x = joint_slope([np.log(np.maximum(dAD, 1e-300))], [mchi])
chi1 = np.where(npts_x >= 3, np.exp(slc), np.nan)

sub = max(1, P // 200000)
np.savez_compressed(os.path.join(here, args.out), sw2=xs, sb2=ys, N=N, D=D, K=K, wall=time.time() - t0,
                    xi_c=xi_c.reshape(args.H, args.W), xi_q=xi_q.reshape(args.H, args.W),
                    chi1=chi1.reshape(args.H, args.W), cstar=cstar.reshape(args.H, args.W),
                    qstar=qstar.reshape(args.H, args.W), npts_c=npts_c.reshape(args.H, args.W), xi_c_flag=xi_c_flag.reshape(args.H, args.W),
                    cAB=series["cAB"].reshape(D + 1, args.H, args.W).astype(np.float16),
                    cAC=series["cAC"].reshape(D + 1, args.H, args.W).astype(np.float16),
                    qE=series["qE"].reshape(D + 1, args.H, args.W).astype(np.float16),
                    qA=series["qA"].reshape(D + 1, args.H, args.W).astype(np.float16),
                    dAD=series["dAD"].reshape(D + 1, args.H, args.W))
print("done", time.time() - t0)
