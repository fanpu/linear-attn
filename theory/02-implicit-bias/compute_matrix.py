"""Deep matrix factorization (Arora, Cohen, Hu & Luo 2019) and the Razin & Cohen (2020) 2x2 example.

  python compute_matrix.py complete <depth> <m> [seed] [lr] [steps] [tag]  -> cache/mc_N<depth>_m<m>_s<seed><tag>.npz
  python compute_matrix.py nuclear <m> [seed]            -> cache/mc_nuc_m<m>_s<seed>.npz
  python compute_matrix.py razin                         -> cache/razin.npz

Completion: 100x100 rank-5 ground truth W* = U V^T (Gaussian factors, rescaled to ||W*||_F = 100), m observed entries
uniform without replacement.  Loss l(W) = 1/2 sum_Omega (W_ij - W*_ij)^2 (sum, as in the singular-value ODE),
full-batch GD on the factors W_N ... W_1 (hidden width 100), init N(0, 1e-3^2) per entry; depth 1 = W trained directly.
"""
import sys, time
import numpy as np

d, r = 100, 5


def ground_truth(seed):
    rng = np.random.default_rng(seed)
    U, V = rng.standard_normal((d, r)), rng.standard_normal((d, r))
    Ws = U @ V.T
    return Ws * (d / np.linalg.norm(Ws))


def mask_for(m, seed):
    rng = np.random.default_rng(1000 + seed)
    M = np.zeros(d * d)
    M[rng.choice(d * d, m, replace=False)] = 1
    return M.reshape(d, d)


def erank(s):
    p = s / s.sum()
    p = p[p > 0]
    return np.exp(-(p * np.log(p)).sum())


def complete(N, m, seed=0, init=1e-3, lr=None, steps=None, tag=""):
    Ws = ground_truth(seed)
    M = mask_for(m, seed)
    rng = np.random.default_rng(7 + seed)
    F = [init * rng.standard_normal((d, d)) for _ in range(N)]
    # step size: the loss is a sum over observed entries with unit curvature; factor curvature ~ N * ||W||^(2-2/N)
    lr = lr or {1: 0.5, 2: 5e-3, 3: 1.5e-3, 4: 8e-4}[N]
    steps = steps or {1: 20000, 2: 200000, 3: 200000, 4: 200000}[N]
    rec_steps = np.unique(np.round(np.logspace(0, np.log10(steps), 300)).astype(int))
    out = {k: [] for k in ["sv", "loss", "err", "nuc", "erank", "dsv_meas", "dsv_pred"]}
    k = 0
    t0 = time.time()
    Wprev_sv = None
    for t in range(1, steps + 1):
        # forward: prefix products P_j = W_j ... W_1
        P = [F[0]]
        for j in range(1, N):
            P.append(F[j] @ P[-1])
        W = P[-1]
        R = M * (W - Ws)
        # suffix products S_j = W_N ... W_{j+1}
        grads = [None] * N
        Sfx = None
        for j in range(N - 1, -1, -1):
            left = R if Sfx is None else Sfx.T @ R
            grads[j] = left if j == 0 else left @ P[j - 1].T
            Sfx = F[j] if Sfx is None else Sfx @ F[j]
        if t == rec_steps[k] or t == 1:
            Uu, s, Vt = np.linalg.svd(W)
            out["sv"].append(s[:30]); out["loss"].append(0.5 * (R ** 2).sum())
            out["err"].append(np.linalg.norm(W - Ws) / np.linalg.norm(Ws)); out["nuc"].append(s.sum()); out["erank"].append(erank(s))
            # Arora et al. Thm 3: dsigma_r/dt = -N (sigma_r^2)^(1-1/N) <grad l(W), u_r v_r^T>; one GD step ~ dt = lr
            pred = -N * (s[:10] ** 2) ** (1 - 1 / N) * np.einsum("ij,ir,jr->r", R, Uu[:, :10], Vt[:10].T)
            out["dsv_pred"].append(lr * pred)
            k += (t == rec_steps[k])
            need_next = True
        else:
            need_next = False
        for j in range(N):
            F[j] -= lr * grads[j]
        if need_next:
            Wn = F[0]
            for j in range(1, N):
                Wn = F[j] @ Wn
            s_next = np.linalg.svd(Wn, compute_uv=False)[:10]
            out["dsv_meas"].append(s_next - out["sv"][-1][:10])
        if k >= len(rec_steps):
            break
        if t % 20000 == 0:
            print(N, m, t, f"loss={out['loss'][-1]:.2e} err={out['err'][-1]:.3f} {time.time() - t0:.0f}s", flush=True)
    np.savez(f"cache/mc_N{N}_m{m}_s{seed}{tag}.npz", steps=rec_steps[: len(out["sv"])], lr=lr, Ws=Ws, M=M,
             W=W, **{k: np.array(v) for k, v in out.items()})


def nuclear(m, seed=0):
    import cvxpy as cp
    Ws, M = ground_truth(seed), mask_for(m, seed)
    X = cp.Variable((d, d))
    idx = np.nonzero(M)
    prob = cp.Problem(cp.Minimize(cp.normNuc(X)), [X[idx] == Ws[idx]])
    prob.solve(solver="SCS", eps=1e-7, max_iters=200000)
    Xv = X.value
    s = np.linalg.svd(Xv, compute_uv=False)
    np.savez(f"cache/mc_nuc_m{m}_s{seed}.npz", W=Xv, sv=s, err=np.linalg.norm(Xv - Ws) / np.linalg.norm(Ws), erank=erank(s))
    print("nuclear", m, np.linalg.norm(Xv - Ws) / np.linalg.norm(Ws))


def razin():
    """2x2 completion: observed (1,2)=1, (2,1)=1, (2,2)=0; W_11 unobserved.  Gradient flow in log-time on the factors
    of a depth-N product with balanced init.  det(W(0)) > 0 (W_j = a I) vs det(W(0)) < 0 (W_j = a diag(1, -1) on one factor)."""
    from scipy.integrate import solve_ivp
    Mask = np.array([[0, 1], [1, 1.0]])
    B = np.array([[0, 1], [1, 0.0]])
    res = {}

    def run(N, init_mats, tag, s_end=np.log(1e14)):
        shapes = [(2, 2)] * N

        def unpack(th):
            return [th[4 * j: 4 * j + 4].reshape(2, 2) for j in range(N)]

        def grad(th):
            F = unpack(th)
            P = [F[0]]
            for j in range(1, N):
                P.append(F[j] @ P[-1])
            R = Mask * (P[-1] - B)
            g, Sfx = [None] * N, None
            for j in range(N - 1, -1, -1):
                left = R if Sfx is None else Sfx.T @ R
                g[j] = left if j == 0 else left @ P[j - 1].T
                Sfx = F[j] if Sfx is None else Sfx @ F[j]
            return np.concatenate([x.ravel() for x in g]), P[-1], 0.5 * (R ** 2).sum()

        th0 = np.concatenate([x.ravel() for x in init_mats])
        # linear time to t=1, then log-time
        sol0 = solve_ivp(lambda t, th: -grad(th)[0], (0, 1), th0, rtol=1e-10, atol=1e-12, method="DOP853")
        s_eval = np.linspace(0, s_end, 600)
        sol = solve_ivp(lambda s, th: -np.exp(s) * grad(th)[0], (0, s_end), sol0.y[:, -1], t_eval=s_eval, rtol=1e-10,
                        atol=1e-13, method="LSODA")
        Wt, L = [], []
        for th in sol.y.T:
            _, W, l = grad(th)
            Wt.append(W); L.append(l)
        Wt = np.array(Wt)
        sv = np.linalg.svd(Wt, compute_uv=False)
        res[tag + "_t"] = np.exp(sol.t)
        res[tag + "_W"] = Wt
        res[tag + "_loss"] = np.array(L)
        res[tag + "_sv"] = sv
        print(tag, "final W", Wt[-1].round(4), "loss", L[-1], "sv", sv[-1])

    a = 0.1
    for N in (2, 3):
        run(N, [a ** (1 / N) * np.eye(2)] * N, f"N{N}_detpos")
        neg = [a ** (1 / N) * np.eye(2)] * (N - 1) + [a ** (1 / N) * np.diag([1.0, -1.0])]
        # balanced: diag(1,-1)^T diag(1,-1) = I, so still balanced
        run(N, neg, f"N{N}_detneg")
    # direct (depth 1) baseline: W_11 never receives gradient
    np.savez("cache/razin.npz", **res)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "complete":
        lr = float(sys.argv[5]) if len(sys.argv) > 5 else None
        steps = int(float(sys.argv[6])) if len(sys.argv) > 6 else None
        complete(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0, lr=lr, steps=steps, tag=sys.argv[7] if len(sys.argv) > 7 else "")
    elif what == "nuclear":
        nuclear(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif what == "razin":
        razin()
