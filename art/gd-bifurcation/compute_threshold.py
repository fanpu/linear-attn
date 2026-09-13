"""Where does period 2 really start?  Sharpness adaptivity and the balanced-minimum rule.

For f = 1/2(prod_i x_i - 1)^2 the Hessian at any global minimum is g g^T with g_i = 1/x_i, so its
sharpness is s = sum_i 1/x_i^2.  Subject to prod x_i = 1, AM-GM gives s >= k with equality only at the
balanced minima |x_i| = 1.  Hence no global minimum is linearly stable for eta > 2/k  (rule: eta* = 2/s_min,
s_min = k).  Below 2/k GD may still converge to an *unbalanced* minimum whose sharpness adapts toward 2/eta.

Candidate rules compared per init x0:
  (i)   2 / lambda_max(Hessian at x0)                        -- "sharpness at init"
  (ii)  2 / s_GF(x0), s_GF = sharpness of the minimum reached by gradient flow from x0
        (gradient flow conserves x_i^2 - x_j^2)               -- "the minimum you would reach"
  (iii) 2 / s_min = 2 / k                                     -- "flattest minimum"
Measured: the smallest eta at which GD (from x0, after T steps) is not converged to a fixed point.

Output: cache/threshold_k{k}.npz
"""
import numpy as np
from scipy.optimize import brentq
import npmaps as nm

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"


def s_gf(x0):
    x0 = np.asarray(x0, float)
    k = len(x0)
    q = x0 ** 2
    # x_i^2 = q_i + c, prod sqrt(q_i + c) = 1
    f = lambda c: 0.5 * np.sum(np.log(q + c))
    c = brentq(f, -q.min() + 1e-14, 10.0 + q.max())
    xs = np.sqrt(q + c)
    return np.sum(1 / xs ** 2), xs


def lam_init(x0):
    J = nm.prod_jac(np.asarray(x0, float)[None], np.array([1.0]))[0]
    H = np.eye(len(x0)) - J
    return np.linalg.eigvalsh(H).max()


def run(k, inits, etas, T_list, tol=1e-9):
    N, M = len(inits), len(etas)
    x = np.repeat(inits, M, 0)
    e = np.tile(etas, N)
    out = {}
    t = 0
    for T in T_list:
        with np.errstate(all="ignore"):
            while t < T:
                x = nm.prod_step(x, e)
                t += 1
            p0 = np.prod(x, 1)
            x1 = nm.prod_step(x, e)
            p1 = np.prod(x1, 1)
            x2 = nm.prod_step(x1, e)
            p2 = np.prod(x2, 1)
        conv = np.abs(p1 - p0) < tol
        per2 = (~conv) & (np.abs(p2 - p0) < tol)
        alive = np.isfinite(p0) & (np.abs(x).max(1) < 1e6)
        s = np.sum(1 / x ** 2, 1)
        out[T] = dict(conv=conv.reshape(N, M), per2=per2.reshape(N, M), alive=alive.reshape(N, M),
                      sharp=s.reshape(N, M), imb=(np.abs(x).max(1) - np.abs(x).min(1)).reshape(N, M))
    return out


def main():
    rng = np.random.default_rng(1)
    for k in [2, 4]:
        inits = np.exp(0.35 * rng.standard_normal((48, k)))
        # also the declared fixed init first
        fixed = {2: (1.1, 0.9), 4: (1.1, 0.9, 1.05, 0.95)}[k]
        inits = np.vstack([fixed, inits])
        etas = np.linspace(0.3 * 2 / k, 1.15 * 2 / k, 1700)
        T_list = [100, 1000, 10000, 100000]
        o = run(k, inits, etas, T_list)
        rule_init = np.array([2 / lam_init(x0) for x0 in inits])
        rule_gf = np.array([2 / s_gf(x0)[0] for x0 in inits])
        save = dict(k=k, inits=inits, etas=etas, T_list=np.array(T_list), rule_init=rule_init, rule_gf=rule_gf,
                    rule_flat=2 / k)
        for T in T_list:
            for key, v in o[T].items():
                save[f"{key}_T{T}"] = v
            # measured onset: first eta (per init) where GD is alive but not converged
            onset = np.full(len(inits), np.nan)
            for i in range(len(inits)):
                bad = o[T]["alive"][i] & ~o[T]["conv"][i]
                if bad.any():
                    onset[i] = etas[np.argmax(bad)]
            save[f"onset_T{T}"] = onset
            print(f"k={k} T={T}: measured onset / (2/k): median {np.nanmedian(onset) * k / 2:.4f} "
                  f"min {np.nanmin(onset) * k / 2:.4f} max {np.nanmax(onset) * k / 2:.4f}; "
                  f"rule_gf/(2/k) median {np.median(rule_gf) * k / 2:.3f}; rule_init/(2/k) median {np.median(rule_init) * k / 2:.3f}")
        np.savez(f"{CACHE}/threshold_k{k}.npz", **save)


if __name__ == "__main__":
    main()
