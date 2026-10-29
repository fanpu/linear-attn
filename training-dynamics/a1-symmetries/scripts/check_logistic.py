"""Check for Problem (logistic): balanced learning of one mode is a logistic curve.  [AI]

Uses YOUR balanced_product and time_to_learn (symmetries/logistic.py). Part 1: for targets
s = 0.5, 1, 2, 4 and 15 initial products u0 from 1e-8 to 1e-1 it integrates the FULL 2-D gradient
flow on ½(s - ab)² from a = b = √u0 and measures the time at which ab first reaches s/2 (a root
find on the dense ODE solution), then compares with your time_to_learn and fits the slope of the
half-time against ln(1/u0). Part 2: the whole curve u(t) against your balanced_product.
Part 3 (Problem logistic (e), predict first): the half-time at fixed u0 = ab = 1e-4, s = 1, when
the start is unbalanced, c = a0² - b0² from 0 to 1.

    uv run python a1-symmetries/scripts/check_logistic.py [--out DIR] [--solutions]
Writes logistic.png and the "logistic" section of results.json.
"""
import numpy as np
from scipy.optimize import brentq

from _common import finish, mine, parse

from tdyn.integrate import solve_flow
from tdyn.plotting import PALETTE, savefig

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
logi = mine("logistic")


def field(s):
    def f(th):
        a, b = th[..., 0], th[..., 1]
        r = a * b - s
        return np.stack([-b * r, -a * r], axis=-1)
    return f


def half_time(a0, b0, s, T):
    tr = solve_flow(field(s), [a0, b0], T, n_points=2001)
    u = tr.y[:, 0] * tr.y[:, 1]
    k = int(np.argmax(u >= s / 2))
    g = lambda t: float(np.prod(tr.sol(t))) - s / 2
    return brentq(g, tr.t[k - 1], tr.t[k], xtol=1e-12), tr


S_LIST = [0.5, 1.0, 2.0, 4.0]
U0 = np.logspace(-8, -1, 15)
meas = np.zeros((len(S_LIST), len(U0)))
for i, s in enumerate(S_LIST):
    for j, u0 in enumerate(U0):
        meas[i, j], _ = half_time(np.sqrt(u0), np.sqrt(u0), s, 1.3 * np.log(s / u0) / (2 * s) + 2.0 / s)
pred = np.array([[float(logi.time_to_learn(u0, s, 0.5)) for u0 in U0] for s in S_LIST])
rel = np.abs(pred - meas) / meas
slopes = [float(np.polyfit(np.log(1 / U0), meas[i], 1)[0]) for i in range(len(S_LIST))]
print(f"max relative error of your half-time over {meas.size} (s, u0) pairs: {rel.max():.2e}")
for s, sl in zip(S_LIST, slopes):
    print(f"s = {s:<4g} fitted d t_half / d ln(1/u0) = {sl:.4f}")

curve_err = 0.0
curves = {}
for s in S_LIST:
    tr = solve_flow(field(s), [1e-3, 1e-3], 18.0, n_points=900)     # u0 = 1e-6
    u = tr.y[:, 0] * tr.y[:, 1]
    mine_u = np.asarray(logi.balanced_product(1e-6, s, tr.t), dtype=float)
    curve_err = max(curve_err, float(np.max(np.abs(mine_u - u)) / s))
    curves[s] = (tr.t, u, mine_u)
print(f"max |your u(t) - integrated u(t)| / s over four curves: {curve_err:.2e}")

p0 = 1e-4
CS = np.array([0.0, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0])
unbal = []
for c in CS:
    a0 = np.sqrt((c + np.sqrt(c * c + 4 * p0 * p0)) / 2)
    unbal.append(half_time(a0, p0 / a0, 1.0, 12.0)[0])
unbal = np.array(unbal)
for c, t in zip(CS, unbal):
    print(f"unbalanced start, u0 = ab = 1e-4, s = 1, c = {c:<7g} half-time {t:.3f}")

fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0), layout="constrained")
ax = axs[0]
for i, s in enumerate(S_LIST):
    ax.plot(np.log(1 / U0), meas[i], "o", ms=3.5, color=PALETTE[i], label=f"$s={s:g}$")
    ax.plot(np.log(1 / U0), pred[i], "-", lw=0.8, color=PALETTE[i])
ax.set_xlabel(r"$\ln(1/u_0)$"); ax.set_ylabel("time to reach $s/2$")
ax.set_title("half-time: flow (dots), yours (lines)"); ax.legend(frameon=False, fontsize=7)
ax = axs[1]
for i, s in enumerate(S_LIST):
    t, u, mu = curves[s]
    ax.plot(t, u / s, color=PALETTE[i], lw=2.2, alpha=0.5, label=f"$s={s:g}$")
    ax.plot(t, mu / s, "k--", lw=0.7)
ax.set_xlabel("time $t$"); ax.set_ylabel("$u(t)/s$"); ax.set_xlim(0, 18)
ax.set_title(r"$u_0=10^{-6}$: a plateau, then a sharp rise"); ax.legend(frameon=False, fontsize=7)
ax = axs[2]
ax.semilogx(np.maximum(CS, 3e-5), unbal, "o-", color=PALETTE[3], ms=4)
ax.set_xticks([3e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(["0", "1e-4", "1e-3", "1e-2", "0.1", "1"])
ax.set_xlabel(r"imbalance $c=a_0^2-b_0^2$ (at fixed $a_0b_0=10^{-4}$)"); ax.set_ylabel("half-time, $s=1$")
ax.set_title("unbalanced starts")
savefig(fig, args.out / "logistic.png")

finish(args, "logistic", {
    "targets": S_LIST, "u0": U0, "half_time_measured": meas, "half_time_yours": pred,
    "max_rel_error": float(rel.max()), "slope_vs_log_1_over_u0": slopes, "slope_formula_1_over_2s": [1 / (2 * s) for s in S_LIST],
    "max_curve_error_over_s": curve_err, "unbalanced_c": CS, "unbalanced_half_time": unbal,
})
