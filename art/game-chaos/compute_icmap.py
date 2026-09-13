"""Initial-condition Lyapunov map for continuous-time replicator learning on zero-sum
generalised RPS (SAF 2002): player 1's initial mixed strategy x0 sweeps the simplex, player 2
starts at SAF's y0 = (0.5, 0.25, 0.25). Each pixel = one orbit integrated to time T; colour =
finite-time largest Lyapunov exponent. Regular (KAM) orbits give lambda ~ 1/T, chaotic ones plateau.
python compute_icmap.py R T eps [x_lo_P x_lo_S width]   (C/OpenMP, CPU)
"""
import sys
import time

import numpy as np

import replicator_c as rc


def main():
    R = int(sys.argv[1]); T = float(sys.argv[2]); eps = float(sys.argv[3])
    lo = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (0.0, 0.0)
    width = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    a = np.linspace(0, 1, R + 2)[1:-1] * width
    S, P = np.meshgrid(a + lo[1], a + lo[0], indexing='ij')
    X = np.stack([1 - P - S, P, S], -1)
    mask = (X > 1e-6).all(-1)
    xs = X[mask]; ys = np.tile([0.5, 0.25, 0.25], (len(xs), 1))
    t = time.time()
    r = rc.integrate(rc.probs_to_logits(xs, ys), eps, -eps, h=0.01, T=T, every=100,
                     hist_every=int(T / 0.01 / 4), nhist=4)
    L = np.full((R, R), np.nan, np.float32); L[mask] = r['lyap']
    H4 = np.full((R, R, 4), np.nan, np.float32); H4[mask] = r['lyap_hist']
    E = np.full((R, R), np.nan, np.float32); E[mask] = rc.energy(xs, ys)
    tag = f'R{R}_T{int(T)}_eps{eps:.2f}' + ('' if width == 1.0 else f'_zoom{lo[0]:.3f}_{lo[1]:.3f}_{width:.3f}')
    np.savez_compressed(f'cache/icmap_{tag}.npz', L=L, hist=H4, energy=E, lo=lo, width=width, eps=eps, T=T,
                        Hdrift=float(r['Hdrift'].max()))
    print(f'icmap {tag}: {len(xs)} orbits, {time.time()-t:.0f}s, frac(L>5e-3)={np.mean(r["lyap"]>5e-3):.3f}, '
          f'Hdrift {r["Hdrift"].max():.1e}', flush=True)


if __name__ == '__main__':
    main()
