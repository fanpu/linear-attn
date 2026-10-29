"""Check for Problem (flow-limit): GD is forward Euler on gradient flow.  [AI]

GD on the anchor model from (a, b) = (1.5, 0.5) at η in {0.1, 0.03, 0.01, 0.003}, compared with a
high-accuracy solution of the flow at the same times t_k = kη, t in [0, 10]. Fits the log-log slope
of (i) the one-step error |θ_1 - θ(η)| and (ii) the largest error over [0, 10] against η.
Then runs η = 0.9 from the same start, where the small-step picture breaks.

    uv run python a0-prereqs/scripts/check_flow_limit.py [--out DIR]
Writes flow_limit_error.png, flow_limit_eta09.png and the "flow_limit" section of results.json.
"""
import numpy as np

from _common import finish, parse

from tdyn.integrate import solve_flow
from tdyn.models import anchor_flow, anchor_grad, anchor_loss
from tdyn.optim import gd_trajectory
from tdyn.plotting import PALETTE, savefig

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
theta0 = np.array([1.5, 0.5])
T = 10.0
etas = np.array([0.1, 0.03, 0.01, 0.003])
flow = solve_flow(anchor_flow, theta0, T, n_points=2001)

local_err, global_err = [], []
for eta in etas:
    K = int(round(T / eta))
    gd = gd_trajectory(anchor_grad, theta0, eta, K)
    ref = flow.sol(np.arange(K + 1) * eta)
    err = np.linalg.norm(gd - ref, axis=1)
    local_err.append(err[1])
    global_err.append(err.max())
local_err, global_err = np.array(local_err), np.array(global_err)
slope_local = np.polyfit(np.log(etas), np.log(local_err), 1)[0]
slope_global = np.polyfit(np.log(etas), np.log(global_err), 1)[0]
print("eta      one-step error   max error on [0,10]")
for e, l, g in zip(etas, local_err, global_err):
    print(f"{e:<8} {l:.3e}        {g:.3e}")
print(f"fitted slopes: one-step {slope_local:.3f} (theory 2), global {slope_global:.3f} (theory 1)")

fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.loglog(etas, global_err, "o-", color=PALETTE[0], label=f"max error on $[0,10]$ (slope {slope_global:.2f})")
ax.loglog(etas, local_err, "s-", color=PALETTE[1], label=f"one-step error (slope {slope_local:.2f})")
ax.loglog(etas, global_err[0] * (etas / etas[0]), ":", color="0.4", label=r"$\propto\eta$")
ax.loglog(etas, local_err[0] * (etas / etas[0]) ** 2, "--", color="0.4", label=r"$\propto\eta^2$")
ax.set_xlabel(r"step size $\eta$")
ax.set_ylabel(r"$\|\theta_k - \theta(k\eta)\|$")
ax.set_title("GD vs gradient flow, anchor model from (1.5, 0.5)")
ax.legend(frameon=False)
savefig(fig, args.out / "flow_limit_error.png")

# --- where the limit fails: eta = 0.9 ---------------------------------------------------
eta_big, K_big = 0.9, 60
gd_big = gd_trajectory(anchor_grad, theta0, eta_big, K_big)
t_big = np.arange(K_big + 1) * eta_big
flow_big = solve_flow(anchor_flow, theta0, t_big[-1], n_points=1201)
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
ax = axs[0]
aa = np.linspace(0.75, 2.6, 300)
ax.plot(aa, 1 / aa, color="k", lw=1.0, label="$ab=1$")
ax.plot(flow_big.y[:, 0], flow_big.y[:, 1], color=PALETTE[0], label="gradient flow")
ax.plot(gd_big[:, 0], gd_big[:, 1], ".-", color=PALETTE[3], ms=3, lw=0.6, label=r"GD, $\eta=0.9$")
ax.plot(*theta0, "ko", ms=4)
ax.plot(*flow_big.y[-1], "*", color=PALETTE[0], ms=11, mec="k", mew=0.5, zorder=5, label="flow endpoint")
ax.plot(*gd_big[-1], "*", color=PALETTE[3], ms=11, mec="k", mew=0.5, zorder=5, label="GD endpoint")
ax.set_xlim(0.9, 2.0); ax.set_ylim(0.2, 1.3)
ax.set_xlabel("$a$"); ax.set_ylabel("$b$"); ax.legend(frameon=False, loc="upper right", fontsize=7)
ax.set_title("trajectories")
ax = axs[1]
ax.plot(flow_big.t, flow_big.y[:, 0] * flow_big.y[:, 1], color=PALETTE[0], label="gradient flow")
ax.plot(t_big, gd_big[:, 0] * gd_big[:, 1], ".-", color=PALETTE[3], ms=3, lw=0.6, label=r"GD, $\eta=0.9$")
ax.axhline(1.0, color="k", lw=0.6)
ax.set_xlim(0, 20)
ax.set_xlabel(r"time $t = k\eta$"); ax.set_ylabel("product $ab$"); ax.legend(frameon=False)
ax.set_title("the product $ab$ over time")
savefig(fig, args.out / "flow_limit_eta09.png")

end_gd, end_flow = gd_big[-1], flow_big.y[-1]
ab_gd = gd_big[:, 0] * gd_big[:, 1]
r_gd = ab_gd - 1.0
live = np.abs(r_gd[:-1]) > 1e-6                     # ignore sign flips of round-off-sized residuals
overshoots = int(np.sum((np.sign(r_gd[1:]) != np.sign(r_gd[:-1])) & live))
print(f"eta=0.9: GD ends at ({end_gd[0]:.4f}, {end_gd[1]:.4f}), flow at ({end_flow[0]:.4f}, {end_flow[1]:.4f})")
print(f"         ab crosses 1 {overshoots} times under GD (flow: never); a^2-b^2 goes 2.000 -> {end_gd[0]**2 - end_gd[1]**2:.4f}")
finish(args, "flow_limit", {
    "etas": etas, "one_step_error": local_err, "max_error": global_err,
    "slope_one_step": slope_local, "slope_max_error": slope_global,
    "eta09_gd_end": end_gd, "eta09_flow_end": end_flow,
    "eta09_gd_imbalance_end": end_gd[0] ** 2 - end_gd[1] ** 2,
    "eta09_ab_crossings": overshoots, "eta09_gd_final_loss": float(anchor_loss(end_gd)),
})
