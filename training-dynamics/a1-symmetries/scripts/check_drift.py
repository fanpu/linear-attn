"""Check for Problem (broken): finite steps, momentum, minibatches and weight decay break c.  [AI]

Uses YOUR imbalance_after_step, drift_integral and predicted_log_drift (symmetries/drift.py) and
the two formulas in the FORMULAS block below. Five parts, all on the anchor:
  A  exact one-step update of c along GD runs (η = 0.1 and 0.3, with and without weight decay);
  B  gradient flow with weight decay γ = 0.01, 0.05 (convention L + γ(a² + b²)): c(t) vs your formula;
  C  GD from (a0, b0) = (0.3, 0.1) (c0 = 0.08, p0 = 0.03) run to convergence for six step sizes:
     the one-step change of ln c (slope 2 in η) and the total change ln(c∞/c0) (slope 1), against
     your first-order prediction -η I(p0, c0);
  D  heavy-ball momentum β = 0.3, 0.5 from rest, same start: ln(c∞/c0) against your prediction;
  E  minibatch SGD on 64 examples x_i (normalized so mean x_i² = 1, y_i = x_i), batch size
     B = 1, 4, 16 with replacement, η = 0.004, 32 seeds: measured drift / (-η I) against your factor.

    uv run python a1-symmetries/scripts/check_drift.py [--out DIR] [--solutions]
Writes drift.png and the "drift" section of results.json.
"""
import numpy as np

from _common import finish, mine, parse

from tdyn.integrate import solve_flow
from tdyn.plotting import PALETTE, savefig
from tdyn.swap import use_solutions

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
dr = mine("drift")


# ---- FORMULAS: replace the two bodies with your Derive deliverables (Problem broken (a), (e)) ----
def wd_imbalance(c0, gamma, t):
    """c(t) under gradient flow on L + γ(a² + b²) started with imbalance c0."""
    raise NotImplementedError("type your formula here")


def sgd_drift_factor(var_x2, B):
    """E[m_B²] where m_B is the mean of x_i² over a batch of B examples drawn with replacement,
    given mean x_i² = 1 and variance var_x2 of x_i² over the dataset: the factor by which SGD's
    first-order drift exceeds GD's."""
    raise NotImplementedError("type your formula here")
# -------------------------------------------------------------------------------------------------


if use_solutions():
    def wd_imbalance(c0, gamma, t):
        return c0 * np.exp(-4.0 * gamma * t)

    def sgd_drift_factor(var_x2, B):
        return 1.0 + var_x2 / B


def heavy_ball(a0, b0, eta, steps, beta=0.0, gamma=0.0):
    th, prev = np.array([a0, b0], float), np.array([a0, b0], float)
    out = np.empty((steps + 1, 2)); out[0] = th
    for k in range(steps):
        a, b = th
        r = a * b - 1.0
        g = np.array([b * r + 2 * gamma * a, a * r + 2 * gamma * b])
        th, prev = th - eta * g + beta * (th - prev), th
        out[k + 1] = th
    return out


# A: exact one-step update ------------------------------------------------------------------------
stepA = {}
for eta, gamma in [(0.1, 0.0), (0.3, 0.0), (0.1, 0.05), (0.3, 0.2)]:
    tr = heavy_ball(1.6, 0.3, eta, 200, gamma=gamma)
    c = tr[:, 0] ** 2 - tr[:, 1] ** 2
    pred = np.asarray(dr.imbalance_after_step(tr[:-1, 0], tr[:-1, 1], eta, gamma), dtype=float)
    stepA[f"eta={eta:g},gamma={gamma:g}"] = float(np.max(np.abs(pred - c[1:])))
print("A  max |c_{k+1} - your formula| over 200 GD steps:", {k: f"{v:.1e}" for k, v in stepA.items()})

# B: weight decay under the flow ------------------------------------------------------------------
wd_err, wd_curves = {}, {}
for gamma in (0.01, 0.05):
    def f(th, g=gamma):
        a, b = th[..., 0], th[..., 1]
        r = a * b - 1
        return np.stack([-(b * r + 2 * g * a), -(a * r + 2 * g * b)], -1)
    tr = solve_flow(f, [1.5, 0.5], 40.0, n_points=401)
    c = tr.y[:, 0] ** 2 - tr.y[:, 1] ** 2
    form = np.array([wd_imbalance(2.0, gamma, t) for t in tr.t])
    wd_err[f"gamma={gamma:g}"] = float(np.max(np.abs(c - form) / np.abs(form)))
    wd_curves[gamma] = (tr.t, c, form)
print("B  max relative |c(t) - your formula| on [0, 40]:", {k: f"{v:.1e}" for k, v in wd_err.items()})

# C, D: GD and momentum, drift vs step size -------------------------------------------------------
a0, b0 = 0.3, 0.1
c0, p0 = a0 ** 2 - b0 ** 2, a0 * b0
I = float(dr.drift_integral(p0, c0))
etas = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
T = 40.0
total, preds = {}, {}
for beta in (0.0, 0.3, 0.5):
    meas, pr = [], []
    for eta in etas:
        tr = heavy_ball(a0, b0, eta, int(T / (eta / (1 - beta))), beta=beta)
        cc = tr[:, 0] ** 2 - tr[:, 1] ** 2
        meas.append(np.log(cc[-1] / c0))
        pr.append(float(dr.predicted_log_drift(a0, b0, eta, beta)))
    total[beta], preds[beta] = np.array(meas), np.array(pr)
first = np.array([abs(np.log((lambda t: t[1, 0] ** 2 - t[1, 1] ** 2)(heavy_ball(a0, b0, eta, 1)) / c0))
                  for eta in etas])                                  # |ln(c_1/c_0)| after one step
slope_total = float(np.polyfit(np.log(etas), np.log(np.abs(total[0.0])), 1)[0])
slope_first = float(np.polyfit(np.log(etas), np.log(first), 1)[0])
print(f"C  I(p0, c0) = {I:.5f};  log-log slope in eta: one step {slope_first:.3f}, whole run {slope_total:.3f}")
for beta in (0.0, 0.3, 0.5):
    ratios = total[beta] / preds[beta]
    print(f"{'C' if beta == 0 else 'D'}  beta = {beta:<4g} measured / predicted ln(c_inf/c0) for eta = "
          + ", ".join(f"{e:g}: {r:.3f}" for e, r in zip(etas, ratios)))

# E: minibatch SGD --------------------------------------------------------------------------------
x = np.random.default_rng(3).standard_normal(64)
x /= np.sqrt(np.mean(x ** 2))
x2 = x ** 2
var_x2 = float(np.var(x2))
eta_sgd, n_seeds, steps = 0.004, 32, int(T / 0.004)
sgd = {}
for B in (1, 4, 16):
    rng = np.random.default_rng(100 + B)
    a, b = np.full(n_seeds, a0), np.full(n_seeds, b0)
    for k in range(steps):
        mB = x2[rng.integers(0, 64, size=(n_seeds, B))].mean(axis=1)
        r = a * b - 1.0
        a, b = a - eta_sgd * b * r * mB, b - eta_sgd * a * r * mB
    logd = np.log((a * a - b * b) / c0)
    ratio = logd / (-eta_sgd * I)
    sgd[B] = (float(ratio.mean()), float(ratio.std() / np.sqrt(n_seeds)), float(sgd_drift_factor(var_x2, B)))
    print(f"E  B = {B:<3d} measured drift / (-eta I) = {sgd[B][0]:.3f} +- {sgd[B][1]:.3f}   your factor {sgd[B][2]:.3f}")

fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.1), layout="constrained")
ax = axs[0]
for i, beta in enumerate((0.0, 0.3, 0.5)):
    lab = "GD" if beta == 0 else rf"momentum $\beta={beta:g}$"
    ax.loglog(etas, -total[beta], "o", color=PALETTE[i], ms=4, label=lab + r", $\ln(c_\infty/c_0)$")
    ax.loglog(etas, -preds[beta], "-", color=PALETTE[i], lw=0.8)
ax.loglog(etas, first, "s", color=PALETTE[3], ms=3.5, label=r"GD, one step $|\ln(c_1/c_0)|$")
ax.set_xlabel(r"step size $\eta$"); ax.set_ylabel(r"$|\Delta \ln c|$")
ax.set_title("drift vs step (lines: your prediction)"); ax.legend(frameon=False, fontsize=6.3)
ax = axs[1]
for i, gamma in enumerate((0.01, 0.05)):
    t, c, form = wd_curves[gamma]
    ax.plot(t, c, color=PALETTE[i], lw=2.4, alpha=0.5, label=rf"flow, $\gamma={gamma:g}$")
    ax.plot(t, form, "k--", lw=0.7)
ax.set_xlabel("time $t$"); ax.set_ylabel("$c(t)$"); ax.set_title("weight decay (dashed: yours)")
ax.legend(frameon=False, fontsize=7)
ax = axs[2]
Bs = np.array([1, 4, 16])
ax.errorbar(Bs, [sgd[B][0] for B in Bs], yerr=[2 * sgd[B][1] for B in Bs], fmt="o", color=PALETTE[3], ms=4,
            label="SGD, measured (±2 s.e.)")
bb = np.linspace(1, 16, 100)
ax.plot(bb, [sgd_drift_factor(var_x2, v) for v in bb], "k-", lw=0.8, label="your factor")
ax.axhline(1.0, color="0.5", lw=0.6, ls=":")
ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16]); ax.set_xticklabels(["1", "4", "16"])
ax.set_xlabel("batch size $B$"); ax.set_ylabel(r"drift $/(-\eta I)$"); ax.set_title(r"minibatches, $\eta=0.004$")
ax.legend(frameon=False, fontsize=7)
savefig(fig, args.out / "drift.png")

finish(args, "drift", {
    "one_step_max_error": stepA, "weight_decay_max_rel_error": wd_err, "drift_integral_I": I,
    "etas": etas, "gd_log_drift": total[0.0], "gd_log_drift_predicted": preds[0.0],
    "momentum_0.3_log_drift": total[0.3], "momentum_0.3_predicted": preds[0.3],
    "momentum_0.5_log_drift": total[0.5], "momentum_0.5_predicted": preds[0.5],
    "gd_one_step_abs_log_change": first, "slope_one_step": slope_first, "slope_whole_run": slope_total,
    "sgd_var_x2": var_x2, "sgd_ratio_mean": {str(B): sgd[B][0] for B in sgd},
    "sgd_ratio_stderr": {str(B): sgd[B][1] for B in sgd}, "sgd_factor_formula": {str(B): sgd[B][2] for B in sgd},
})
