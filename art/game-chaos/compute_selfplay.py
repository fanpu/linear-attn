"""Self-play framing: two tiny softmax policies (3 logits each) trained against each other on
zero-sum generalised RPS (SAF payoffs, eps_x = -eps_y = eps).

  PG  (simultaneous policy-gradient ascent, exact expected-payoff gradient):
        theta_1 += eta * (diag(x) - x x^T) A y ,   theta_2 += eta * (diag(y) - y y^T) B x
  MWU (= FTRL-entropic = 'logit += eta * payoff vector'):
        theta_1 += eta * A y ,                      theta_2 += eta * B x

Both are Euler discretisations of continuous learning flows; MWU's flow is exactly the SAF replicator.
For a sweep over eta we record the finite-time largest Lyapunov exponent (tangent propagation, per step),
the minimum probability reached (distance to the simplex boundary), and a few trajectories.
CPU numpy, float64.  python compute_selfplay.py
"""
import time

import numpy as np

EPS = 0.5


def A_mul(e, z):
    return np.stack([e * z[..., 0] - z[..., 1] + z[..., 2], z[..., 0] + e * z[..., 1] - z[..., 2],
                     -z[..., 0] + z[..., 1] + e * z[..., 2]], -1)


def softmax(q):
    q = q - q.max(-1, keepdims=True); e = np.exp(q); return e / e.sum(-1, keepdims=True)


def run(etas, rule, T0=5000, T1=20000, keep=0, x0=(0.5, 0.01, 0.49), y0=(0.5, 0.25, 0.25)):
    n = len(etas); eta = np.asarray(etas)[:, None]
    th1 = np.tile(np.log(x0), (n, 1)); th2 = np.tile(np.log(y0), (n, 1))
    rng = np.random.default_rng(0)
    d1 = rng.normal(size=(n, 3)); d2 = rng.normal(size=(n, 3))
    L = np.zeros(n); pmin = np.ones(n); traj = []
    for t in range(T0 + T1):
        x = softmax(th1); y = softmax(th2)
        dx = x * (d1 - (x * d1).sum(-1, keepdims=True)); dy = y * (d2 - (y * d2).sum(-1, keepdims=True))
        Ay = A_mul(EPS, y); Bx = A_mul(-EPS, x); dAy = A_mul(EPS, dy); dBx = A_mul(-EPS, dx)
        if rule == 'mwu':
            g1, g2, dg1, dg2 = Ay, Bx, dAy, dBx
        else:  # exact policy gradient J_softmax^T payoff, and its derivative
            m1 = (x * Ay).sum(-1, keepdims=True); m2 = (y * Bx).sum(-1, keepdims=True)
            g1 = x * (Ay - m1); g2 = y * (Bx - m2)
            dg1 = dx * (Ay - m1) + x * (dAy - (dx * Ay).sum(-1, keepdims=True) - (x * dAy).sum(-1, keepdims=True))
            dg2 = dy * (Bx - m2) + y * (dBx - (dy * Bx).sum(-1, keepdims=True) - (y * dBx).sum(-1, keepdims=True))
        th1 = th1 + eta * g1; th2 = th2 + eta * g2
        d1 = d1 + eta * dg1; d2 = d2 + eta * dg2
        th1 -= th1.mean(-1, keepdims=True); th2 -= th2.mean(-1, keepdims=True)
        d1 -= d1.mean(-1, keepdims=True); d2 -= d2.mean(-1, keepdims=True)
        nr = np.sqrt((d1 ** 2).sum(-1) + (d2 ** 2).sum(-1)); d1 /= nr[:, None]; d2 /= nr[:, None]
        if t >= T0:
            L += np.log(nr); pmin = np.minimum(pmin, np.minimum(x.min(-1), y.min(-1)))
        if keep and t % keep == 0:
            traj.append(np.concatenate([x, y], -1))
    return L / T1, pmin, (np.stack(traj, 1) if keep else None)


def main():
    out = {}
    etas = np.geomspace(0.01, 3.0, 1200)
    for rule in ('pg', 'mwu'):
        t = time.time()
        L, pmin, _ = run(etas, rule)
        out[f'L_{rule}'] = L; out[f'pmin_{rule}'] = pmin
        print(f'{rule}: {time.time()-t:.0f}s  frac(L>5e-3)={np.mean(L>5e-3):.3f}  max L={L.max():.3f}  '
              f'frac(pmin<1e-8)={np.mean(pmin<1e-8):.3f}', flush=True)
    out['etas'] = etas
    show = np.array([0.02, 0.1, 0.4, 1.0])
    for rule in ('pg', 'mwu'):
        L, pmin, tr = run(show, rule, T0=0, T1=30000, keep=1)
        out[f'traj_{rule}'] = tr.astype(np.float32); out[f'trajL_{rule}'] = L; out[f'trajpmin_{rule}'] = pmin
        print(rule, 'traj lyap', np.round(L, 4), 'pmin', pmin)
    out['show_etas'] = show
    np.savez_compressed('cache/selfplay.npz', **out)


if __name__ == '__main__':
    main()
