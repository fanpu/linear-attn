"""Soudry et al. (2018) reproduction.  Writes cache/soudry_<part>.npz.

  part=hero   2D dataset: log-time gradient flow from t=1e-3 to t=1e100, plus the closed-form asymptote.
  part=gd     d=50, n=40 random-label Gaussian data, 8 datasets vectorized: GD / normalized GD / sign GD,
              float64, T steps, recorded at log-spaced steps; plus log-time flow to 1e100 per dataset.
"""
import sys, time
import numpy as np
from scipy.special import expit
from scipy.integrate import solve_ivp
from core import (hero_2d, gaussian_separable, svm, soudry_wtilde, flow_logtime, logistic_grad, log_checkpoints)

part = sys.argv[1]

if part == "hero":
    X, y = hero_2d()
    Z = y[:, None] * X
    wh, al = svm(X, y)
    wl_inf, _ = svm(X, y, "linf")
    wt, S = soudry_wtilde(X, y, wh, al)
    # early phase in linear time t in [0, 1] (dense), then log time s in [0, log 1e100]
    t_lin = np.concatenate([[0.0], np.logspace(-3, 0, 121)])
    sol0 = solve_ivp(lambda t, w: -logistic_grad(Z, w), (0, 1), np.zeros(2), t_eval=t_lin, rtol=1e-12, atol=1e-13, method="DOP853")
    s_eval = np.linspace(0, np.log(1e100), 4001)
    s, W = flow_logtime(Z, s_eval[-1], w0=sol0.y[:, -1], s_eval=s_eval)
    t_all = np.concatenate([t_lin[:-1], np.exp(s)])
    W_all = np.vstack([sol0.y.T[:-1], W])
    # discrete GD check (eta = 0.1) up to 1e6 steps: w_gd(k) should track the flow at time eta*k
    eta = 0.1
    ts, Wgd = [], []
    w = np.zeros(2)
    ck = set(log_checkpoints(10 ** 6, 20).tolist())
    for k in range(1, 10 ** 6 + 1):
        w = w - eta * logistic_grad(Z, w)
        if k in ck:
            ts.append(k); Wgd.append(w.copy())
    np.savez("cache/soudry_hero.npz", X=X, y=y, w_hat=wh, alpha=al, w_tilde=wt, S=S, t=t_all, W=W_all,
             gd_steps=np.array(ts), gd_W=np.array(Wgd), gd_eta=eta, w_linf=wl_inf)
    print("hero done", len(t_all))

elif part in ("gd", "ngd", "sign", "ngd_sqrt"):
    T = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10 ** 7
    seeds = range(8)
    data = [gaussian_separable(40, 50, s) for s in seeds]
    Z = np.stack([y[:, None] * X for X, y in data])  # S,n,d
    Sn = Z.shape[0]
    lam = np.array([np.linalg.norm(z, 2) ** 2 for z in Z])
    eta = {"gd": 1.0 / (0.25 * lam), "ngd": 0.1 * np.ones(Sn), "sign": 0.01 * np.ones(Sn), "ngd_sqrt": 0.5 * np.ones(Sn)}[part][:, None]
    ts = log_checkpoints(T, 40)
    rec = np.empty((len(ts), Sn, 50))
    w = np.zeros((Sn, 50))
    k = 0
    t0 = time.time()
    for t in range(1, T + 1):
        m = np.einsum("snd,sd->sn", Z, w)
        if part == "gd":
            g = -np.einsum("sn,snd->sd", expit(-m), Z)
        else:
            # scale-free methods only need the gradient's direction: use log-domain weights so that
            # sigma(-m) underflowing to 0 at huge margins cannot freeze the iterate
            lw = -np.logaddexp(0.0, m)
            p = np.exp(lw - lw.max(1, keepdims=True))
            g = -np.einsum("sn,snd->sd", p, Z)
        if part == "gd":
            w -= eta * g
        elif part == "ngd":
            w -= eta * g / np.linalg.norm(g, axis=1, keepdims=True)
        elif part == "ngd_sqrt":
            w -= eta / np.sqrt(t) * g / np.linalg.norm(g, axis=1, keepdims=True)
        else:
            w -= eta * np.sign(g)
        if t == ts[k]:
            rec[k] = w
            k += 1
            if k % 40 == 0:
                print(part, t, f"{time.time() - t0:.0f}s", flush=True)
    out = dict(steps=ts, W=rec, eta=eta[:, 0], Z=Z)
    if part == "gd":
        W_hat, A, W_tilde, nSV, next_margin = [], [], [], [], []
        flows = []
        s_eval = np.linspace(0, np.log(1e100), 801)
        for i, (X, y) in enumerate(data):
            wh, al = svm(X, y)
            wt, S = soudry_wtilde(X, y, wh, al, eta=eta[i, 0])
            W_hat.append(wh); A.append(al); W_tilde.append(wt); nSV.append(len(S))
            mm = np.sort(Z[i] @ wh)
            next_margin.append(mm[len(S)])
            s, Wf = flow_logtime(Z[i], s_eval[-1], s_eval=s_eval)
            flows.append(Wf)
        out.update(W_hat=np.array(W_hat), alpha=np.array(A), W_tilde_sv=np.array(W_tilde), nSV=np.array(nSV),
                   next_margin=np.array(next_margin), flow_s=s_eval, flow_W=np.array(flows))
    if part == "sign":
        out["W_linf"] = np.array([svm(X, y, "linf")[0] for X, y in data])
        out["W_hat"] = np.array([svm(X, y)[0] for X, y in data])
    np.savez(f"cache/soudry_{part}.npz", **out)
    print(part, "done", time.time() - t0)
