"""Check for Problem (slowing): critical slowing down near γ = ½.  [AI]

Uses YOUR relaxation_time and time_to_within (wdecay/slowing.py) and the formula in the FORMULAS
block below. Three parts:
  A  from the unbalanced start (0.5, 0.3), for 9 values of |γ - ½| from 1e-3 to 1e-1 on each side:
     the time the flow needs to bring its distance to the fixed point from 1e-4 down to 1e-8, divided
     by ln(1e4) (the measured relaxation time), against your τ(γ); log-log slopes on each side;
  B  from the balanced start (0.5, 0.5): the first time within ε = 1e-4 of the fixed point, against
     your time_to_within, and its log-log slope (not exactly -1: see the handout);
  C  at γ = ½ exactly, from (0.5, 0.5): ab(t) against your formula, and the late-time exponent of
     the distance to the origin.

    uv run python a2-weight-decay/scripts/check_slowing.py [--out DIR] [--solutions]
Writes slowing.png and the "slowing" section of results.json.
"""
import numpy as np
from scipy.integrate import solve_ivp

from _common import finish, mine, parse

from tdyn.integrate import solve_flow
from tdyn.plotting import PALETTE, savefig
from tdyn.swap import use_solutions

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
sl = mine("slowing")


# ---- FORMULAS: replace the body with your Derive deliverable (Problem slowing (c)) ---------------
def critical_product(p0, t):
    """p(t) = a(t)b(t) under gradient flow at γ = ½ exactly, from a balanced start with p(0) = p0."""
    raise NotImplementedError("type your formula here")
# -------------------------------------------------------------------------------------------------


if use_solutions():
    def critical_product(p0, t):
        return p0 / (1.0 + 2.0 * p0 * t)


def grad(y, gamma):
    a, b = y
    r = a * b - 1
    return np.array([b * r + 2 * gamma * a, a * r + 2 * gamma * b])


def fixed_point(gamma):
    u = np.sqrt(1 - 2 * gamma) if gamma < 0.5 else 0.0
    return np.array([u, u])


def crossing_times(theta0, gamma, epss, T):
    """First times the flow from theta0 comes within each eps of the fixed point (event location)."""
    target = fixed_point(gamma)
    events = []
    for e in epss:
        def ev(t, y, e=e):
            return np.linalg.norm(y - target) - e
        ev.direction = -1
        events.append(ev)
    res = solve_ivp(lambda t, y: -grad(y, gamma), (0.0, T), np.asarray(theta0, float), method="DOP853",
                    rtol=1e-11, atol=1e-13, events=events)
    return [float(te[0]) for te in res.t_events]


# A: relaxation time ------------------------------------------------------------------------------
dg = np.logspace(-3, -1, 9)
tauA = {}
for side, sgn in (("below", -1), ("above", 1)):
    meas, pred = [], []
    for d in dg:
        g = 0.5 + sgn * d
        tau = float(sl.relaxation_time(g))
        t1, t2 = crossing_times([0.5, 0.3], g, [1e-4, 1e-8], 40.0 / d)
        meas.append((t2 - t1) / np.log(1e4))
        pred.append(tau)
    meas, pred = np.array(meas), np.array(pred)
    slope = float(np.polyfit(np.log(dg), np.log(meas), 1)[0])
    tauA[side] = {"measured": meas, "yours": pred, "slope": slope, "amplitude": float(np.mean(meas * dg))}
    print(f"A  {side} ½: max |measured/your τ - 1| = {np.max(np.abs(meas / pred - 1)):.1e}, "
          f"log-log slope {slope:.4f}, τ·|γ-½| = {tauA[side]['amplitude']:.4f}")
print(f"A  amplitude ratio above/below = {tauA['above']['amplitude'] / tauA['below']['amplitude']:.4f}")

# B: time to within eps from a balanced start -----------------------------------------------------
eps = 1e-4
tB = {}
for side, sgn in (("below", -1), ("above", 1)):
    meas = np.array([crossing_times([0.5, 0.5], 0.5 + sgn * d, [eps], 40.0 / d)[0] for d in dg])
    yours = np.array([float(sl.time_to_within(0.5 + sgn * d, 0.5, eps)) for d in dg])
    slope = float(np.polyfit(np.log(dg), np.log(meas), 1)[0])
    tB[side] = {"measured": meas, "yours": yours, "max_rel_error": float(np.max(np.abs(yours / meas - 1))),
                "slope": slope}
    print(f"B  {side} ½: max |your t_eps / measured - 1| = {tB[side]['max_rel_error']:.1e}, log-log slope {slope:.4f}")

# C: γ = ½ exactly -------------------------------------------------------------------------------
T = 1e5
tr = solve_flow(lambda y: -grad(y, 0.5), [0.5, 0.5], T, n_points=2)
ts = np.logspace(-1, 5, 601)
ys = tr.sol(ts)
p = ys[:, 0] * ys[:, 1]
errC = float(np.max(np.abs(p / np.array([critical_product(0.25, t) for t in ts]) - 1)))
late = ts > 1e3
dist = np.linalg.norm(ys, axis=1)
expC = float(np.polyfit(np.log(ts[late]), np.log(dist[late]), 1)[0])
print(f"C  γ = ½: max relative |ab(t) - your formula| = {errC:.1e}; late-time exponent of |θ(t)|: {expC:.4f}")

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1), layout="constrained")
ax = axs[0]
for i, side in enumerate(("below", "above")):
    ax.loglog(dg, tauA[side]["measured"], "o", ms=4, color=PALETTE[i], label=rf"{side} $\frac{{1}}{{2}}$, measured")
    ax.loglog(dg, tauA[side]["yours"], "-", lw=0.8, color=PALETTE[i])
ax.set_xlabel(r"$|\gamma-\frac{1}{2}|$"); ax.set_ylabel(r"relaxation time $\tau$")
ax.set_title("late-time decay (lines: yours)"); ax.legend(frameon=False, fontsize=7)
ax = axs[1]
for i, side in enumerate(("below", "above")):
    ax.loglog(dg, tB[side]["measured"], "o", ms=4, color=PALETTE[i], label=rf"{side} $\frac{{1}}{{2}}$, slope {tB[side]['slope']:.2f}")
    ax.loglog(dg, tB[side]["yours"], "-", lw=0.8, color=PALETTE[i])
ax.set_xlabel(r"$|\gamma-\frac{1}{2}|$"); ax.set_ylabel(r"$t_\varepsilon$, $\varepsilon=10^{-4}$")
ax.set_title(r"time to within $\varepsilon$ from $(0.5,0.5)$"); ax.legend(frameon=False, fontsize=7)
ax = axs[2]
ax.loglog(ts, dist, color=PALETTE[2], lw=2.4, alpha=0.5, label=r"$|\theta(t)|$ at $\gamma=\frac{1}{2}$")
ax.loglog(ts, np.sqrt(2 * np.array([critical_product(0.25, t) for t in ts])), "k--", lw=0.7, label="your formula")
for i, d in enumerate((1e-2, 1e-3)):
    trd = solve_flow(lambda y, d=d: -grad(y, 0.5 + d), [0.5, 0.5], T, n_points=2)
    ax.loglog(ts, np.linalg.norm(trd.sol(ts), axis=1) + 1e-300, color=PALETTE[i], lw=0.9,
              label=rf"$\gamma=\frac{{1}}{{2}}+{d:g}$")
ax.set_ylim(1e-6, 1); ax.set_xlabel("time $t$"); ax.set_ylabel("distance to the origin")
ax.set_title(f"at the critical point (exponent {expC:.3f})"); ax.legend(frameon=False, fontsize=7)
savefig(fig, args.out / "slowing.png")

finish(args, "slowing", {
    "abs_gamma_minus_half": dg,
    "tau_below_measured": tauA["below"]["measured"], "tau_above_measured": tauA["above"]["measured"],
    "tau_slope_below": tauA["below"]["slope"], "tau_slope_above": tauA["above"]["slope"],
    "tau_amplitude_below": tauA["below"]["amplitude"], "tau_amplitude_above": tauA["above"]["amplitude"],
    "t_eps_below_measured": tB["below"]["measured"], "t_eps_above_measured": tB["above"]["measured"],
    "t_eps_max_rel_error": {k: v["max_rel_error"] for k, v in tB.items()},
    "t_eps_slope": {k: v["slope"] for k, v in tB.items()},
    "critical_product_max_rel_error": errC, "critical_exponent": expC,
})
