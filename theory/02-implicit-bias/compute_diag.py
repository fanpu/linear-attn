"""Woodworth et al. (2020) diagonal linear networks.  Writes cache/diag_*.npz.

Gradient flow on L = 1/(2n) ||X (u^2 - v^2) - y||^2 from u = v = alpha*1, integrated with LSODA (rtol 1e-11)
and an analytic Jacobian.  Compared to: min-L2 interpolant, basis pursuit (min-L1, LP), and the closed-form
Q_alpha minimizer (solved independently through its convex dual).
Also: plain discrete GD at a few step sizes (the step-size gap) and a 2D n=1 toy for the widget/figure.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from core import q_alpha_min, basis_pursuit, min_l2_interp, diag_net_gd, Q_alpha


def flow(X, y, alpha, T=1e8, t_eval=None, depth=2):
    n, d = X.shape
    H = X.T @ X / n
    D = depth

    def rhs(t, th):
        u, v = th[:d], th[d:]
        g = X.T @ (X @ (u ** D - v ** D) - y) / n
        return np.r_[-D * u ** (D - 1) * g, D * v ** (D - 1) * g]

    def jac(t, th):
        u, v = th[:d], th[d:]
        g = X.T @ (X @ (u ** D - v ** D) - y) / n
        du, dv = D * u ** (D - 1), D * v ** (D - 1)
        J = np.empty((2 * d, 2 * d))
        J[:d, :d] = -np.diag(D * (D - 1) * u ** (D - 2) * g) - du[:, None] * H * du[None, :]
        J[:d, d:] = du[:, None] * H * dv[None, :]
        J[d:, :d] = dv[:, None] * H * du[None, :]
        J[d:, d:] = np.diag(D * (D - 1) * v ** (D - 2) * g) - dv[:, None] * H * dv[None, :]
        return J

    atol = min(1e-14, 1e-4 * alpha ** D)
    sol = solve_ivp(rhs, (0, T), np.full(2 * d, alpha), method="LSODA", jac=jac, rtol=1e-11, atol=atol, t_eval=t_eval)
    W = (sol.y[:d] ** D - sol.y[d:] ** D).T
    return sol.t, W


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(0)
    n, d, k = 40, 100, 5
    X = rng.standard_normal((n, d))
    w_star = np.zeros(d)
    idx = np.arange(k)  # support = first 5 coordinates (permuted later for display only)
    w_star[idx] = rng.choice([-1, 1], k) * (1 + rng.random(k))
    y = X @ w_star
    bp = basis_pursuit(X, y)
    l2 = min_l2_interp(X, y)

    # ---- alpha sweep, 6+ decades
    alphas = np.logspace(-5, 2, 57)
    W_flow, W_q, times = [], [], []
    for a in alphas:
        ts, W = flow(X, y, a)
        W_flow.append(W[-1])
        W_q.append(q_alpha_min(X, y, a)[0])
    W_flow, W_q = np.array(W_flow), np.array(W_q)
    print("sweep done", time.time() - t0, "max |flow - Q|", np.abs(W_flow - W_q).max())

    # ---- training trajectories for a few alphas (stem-plot movie over training time)
    traj = {}
    for a in [1e-4, 1e-2, 1e-1, 1.0, 10.0]:
        te = np.concatenate([[0], np.logspace(-3, 6, 721)])
        ts, W = flow(X, y, a, T=1e6, t_eval=te)
        traj[f"t_{a:g}"] = ts
        traj[f"W_{a:g}"] = W

    # ---- discrete GD step-size gap (constant step eta = c / (8 lam max|w_bp|)) for alpha in a small set
    gd_alphas = np.array([1e-2, 1e-1, 1.0])
    lam = np.linalg.eigvalsh(X.T @ X / n)[-1]
    gd = {}
    scale = 8 * lam * np.abs(bp).max()  # ~ largest Hessian eigenvalue at the sparse solution
    for c in [0.05, 0.2, 0.5, 0.8]:
        Wg, _, steps = diag_net_gd(X, y, gd_alphas, steps=1_500_000, tol=1e-24, fixed_eta=c / scale)
        gd[f"gd_c{c:g}"] = Wg
        print("gd c", c, steps, time.time() - t0, flush=True)

    # ---- sample-size sweep: recovery error of the flow limit vs n for several alpha (via Q_alpha closed form)
    ns = np.arange(10, 81, 5)
    a_list = np.array([1e-4, 1e-2, 1e-1, 1.0, 10.0])
    rec_err = np.zeros((len(a_list), len(ns), 10))
    bp_err = np.zeros((len(ns), 10))
    for s in range(10):
        r2 = np.random.default_rng(100 + s)
        Xb = r2.standard_normal((80, d))
        wb = np.zeros(d)
        wb[r2.choice(d, k, replace=False)] = r2.choice([-1, 1], k) * (1 + r2.random(k))
        for j, m in enumerate(ns):
            Xm, ym = Xb[:m], Xb[:m] @ wb
            for i, a in enumerate(a_list):
                wq = q_alpha_min(Xm, ym, a)[0]
                rec_err[i, j, s] = np.linalg.norm(wq - wb) / np.linalg.norm(wb)
            bp_err[j, s] = np.linalg.norm(basis_pursuit(Xm, ym) - wb) / np.linalg.norm(wb)
    print("n sweep done", time.time() - t0)

    np.savez("cache/diag_sparse.npz", X=X, y=y, w_star=w_star, bp=bp, l2=l2, alphas=alphas, W_flow=W_flow, W_q=W_q,
             gd_alphas=gd_alphas, ns=ns, a_list=a_list, rec_err=rec_err, bp_err=bp_err, **traj, **gd)

    # ---- 2D, n = 1 toy: x = (1, 0.4), y = 1.  Flow paths for a range of alphas, depth 2 and 3.
    x = np.array([[1.0, 0.4]])
    yy = np.array([1.0])
    toy = {}
    for D in (2, 3):
        for a in np.logspace(-3, 1, 17):
            te = np.concatenate([[0], np.logspace(-4, 7, 400)])
            ts, W = flow(x, yy, a, T=1e7, t_eval=te, depth=D)
            toy[f"D{D}_a{a:.4g}"] = W
    np.savez("cache/diag_toy.npz", x=x, y=yy, **toy)
    print("all done", time.time() - t0)
