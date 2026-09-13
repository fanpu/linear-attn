"""Universality overlay data: the logistic map and GD on 1/2(x1x2x3x4-1)^2 on a common logarithmic axis.

For each system, eta is sampled as eta = eta_inf - A * 10^(-s), s in [s0, s1], where eta_inf and A come from
the Newton-Floquet bifurcation points (A = (eta_inf - eta_n) delta^n at the deepest measured n, so that
the n-th doubling sits at s = n log10(delta) in both systems).  Full dynamics (all four GD coordinates),
float64, 120000 burn-in steps, 2048 recorded iterates per eta.  Output: cache/universality.npz
"""
import json
import numpy as np
import npmaps as nm

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"
DELTA = 4.669201609102990


def main():
    A = json.load(open(f"{CACHE}/feigenbaum_A.json"))
    out = {}
    N, burn, rec = 8000, 120000, 2048
    s = np.linspace(-0.45, 6.2, N)
    for name, key in [("logistic", "A_logistic"), ("prod4", "A_bal4")]:
        eta = np.array(A[key]["eta"])
        einf = float(eta[-1] + (eta[-1] - eta[-2]) / (DELTA - 1))  # geometric extrapolation
        n = len(eta)
        Acoef = (einf - eta[-3]) * DELTA ** (n - 2)
        e = einf - Acoef * 10 ** (-s)
        with np.errstate(all="ignore"):
            if name == "logistic":
                x = np.full(N, 0.3)
                for _ in range(burn):
                    x = e * x * (1 - x)
                R = np.empty((N, rec), np.float32)
                for i in range(rec):
                    x = e * x * (1 - x)
                    R[:, i] = x
            else:
                x = np.broadcast_to(np.array([1.1, 0.9, 1.05, 0.95]), (N, 4)).copy()
                for _ in range(burn):
                    x = nm.prod_step(x, e)
                R = np.empty((N, rec), np.float32)
                for i in range(rec):
                    x = nm.prod_step(x, e)
                    R[:, i] = np.prod(x, 1)
        out[f"{name}_vals"] = R
        out[f"{name}_eta"] = e
        out[f"{name}_s_bif"] = np.log10(Acoef / (einf - eta))
        out[f"{name}_einf"] = einf
        out[f"{name}_A"] = Acoef
        print(name, "eta_inf", einf, "A", Acoef, "s of bifurcations", np.round(out[f'{name}_s_bif'], 3), flush=True)
    out["s"] = s
    np.savez(f"{CACHE}/universality.npz", **out)


if __name__ == "__main__":
    main()
