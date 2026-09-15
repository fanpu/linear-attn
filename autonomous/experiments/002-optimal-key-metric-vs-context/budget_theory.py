"""Finite-context budget model for the best key-metric exponent (no drift, noiseless, from scratch).

Direction i (input variance lam_i) is refreshed at rate r_i(s) = lam_i^(1-s) / sum_j lam_j^(1-s) per token
(beta = 1 projection; a metric Sigma^-s tilts refreshes toward lam^(1-s)). Its risk decays as lam_i exp(-r_i t),
so the context-averaged risk is  R(s, T) = sum_i lam_i (1 - exp(-r_i T)) / (r_i T).
T -> infinity: R ~ sum_i lam_i / r_i -> Cauchy-Schwarz -> s* = 1/2.   T -> 0: R ~ sum lam_i (1 - r_i T/2) -> greedy, s* -> 0.
Prints s*(T) for the experiment's grid; compare with cache/metric.json (qd = 0, s2 = 0)."""
import json
import numpy as np


d, kappa = 32, 100
lam = np.logspace(0, np.log10(kappa), d); lam /= lam.mean()


def R(s, T):
    r = lam ** (1 - s); r /= r.sum()
    return np.sum(lam * -np.expm1(-r * T) / (r * T))


Ts = [d // 2, d, 2 * d, 4 * d, 16 * d, 64 * d]
grid = np.linspace(-0.5, 1.5, 4001)
theory = [grid[np.argmin([R(s, T) for s in grid])] for T in Ts]
sim = json.load(open("cache/metric.json"))["qd=0.0,s2=0.0"]
print("T/d     ", [T / d for T in Ts])
print("theory s*", np.round(theory, 3).tolist())
print("sim s*   ", np.round(sim["s_star"], 3).tolist(), "+-", np.round(sim["s_star_sd"], 2).tolist())
