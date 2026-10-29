"""Check for Problem (hessian): sharpness and trace along the anchor's solution manifold.  [AI]

Uses YOUR hvp, power_iteration and hutchinson_trace (prereqs/hessian.py), all through
Hessian-vector products only. (1) Along ab = 1, for 41 values of a in [0.4, 2.5], estimates the
sharpness λ_max(∇²L) and the trace, and compares both with a² + b². (2) Marks 2/η for η = 0.5 and
η = 1 (the GD stability threshold of Problem gd-stability; A3 builds on this picture). (3) Revisits
Problem (flow-limit): the sharpness of the flow's endpoint and of GD's endpoint at η = 0.9.
(4) On a 193-parameter MLP, compares with the exact Hessian from torch.autograd.functional.hessian.

    uv run python a0-prereqs/scripts/check_sharpness.py [--out DIR] [--solutions]
Writes sharpness.png and the "sharpness" section of results.json.
"""
import time

import numpy as np
import torch

from _common import finish, mine, parse

from tdyn.hessian import flat_loss
from tdyn.integrate import solve_flow
from tdyn.models import anchor_flow, anchor_grad, anchor_loss_torch, mlp, regression_batch
from tdyn.optim import gd_trajectory
from tdyn.plotting import PALETTE, savefig

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
H_ = mine("hessian")


def anchor_hvp_fn(point):
    p = torch.tensor(point, dtype=torch.float64)
    return lambda v: H_.hvp(anchor_loss_torch, p, v)


# (1) along the manifold ---------------------------------------------------------------------
a_grid = np.geomspace(0.4, 2.5, 41)
sharp, trace = [], []
for a in a_grid:
    f = anchor_hvp_fn((a, 1.0 / a))
    lam, _ = H_.power_iteration(f, 2, 30, seed=0)
    sharp.append(lam)
    trace.append(H_.hutchinson_trace(f, 2, 400, seed=0))
sharp, trace = np.array(sharp), np.array(trace)
exact = a_grid**2 + 1 / a_grid**2
rel_sharp = np.max(np.abs(sharp - exact) / exact)
trace_err_sd = np.max(np.abs(trace - exact)) / (2 / np.sqrt(400))     # per-sample std is 2 here
print(f"along ab=1: max relative error of sharpness vs a^2+b^2: {rel_sharp:.1e}; "
      f"max trace error {np.max(np.abs(trace - exact)):.3f} ({trace_err_sd:.1f} Hutchinson std)")
stable = a_grid[sharp < 2 / 0.5]
print(f"eta = 0.5: minima with sharpness < 4 measured for a in [{stable.min():.3f}, {stable.max():.3f}] "
      f"(grid spacing {a_grid[1] / a_grid[0] - 1:.1%})")

# (3) the two endpoints of Problem (flow-limit) -------------------------------------------------
theta0 = np.array([1.5, 0.5])
flow_end = solve_flow(anchor_flow, theta0, 60.0, n_points=3).y[-1]
gd_end = gd_trajectory(anchor_grad, theta0, 0.9, 400)[-1]
s_flow = H_.power_iteration(anchor_hvp_fn(tuple(flow_end)), 2, 50)[0]
s_gd = H_.power_iteration(anchor_hvp_fn(tuple(gd_end)), 2, 50)[0]
print(f"flow endpoint {np.round(flow_end, 4)}: sharpness {s_flow:.4f}; GD (eta=0.9) endpoint "
      f"{np.round(gd_end, 4)}: sharpness {s_gd:.4f}; 2/0.9 = {2 / 0.9:.4f}")

# (4) a 193-parameter MLP ---------------------------------------------------------------------
net = mlp(4, 32, 1, seed=0)
x, y = regression_batch(64, 4, 1, seed=0)
loss_fn, theta = flat_loss(net, lambda out: 0.5 * ((out - y) ** 2).mean(), x)
t0 = time.perf_counter()
H = torch.autograd.functional.hessian(loss_fn, theta)
t_exact = time.perf_counter() - t0
ev = torch.linalg.eigvalsh(H)
f = lambda v: H_.hvp(loss_fn, theta, v)
t0 = time.perf_counter()
lam_mlp, _ = H_.power_iteration(f, theta.numel(), 100, seed=0)
tr_mlp = H_.hutchinson_trace(f, theta.numel(), 200, seed=0)
t_hvp = time.perf_counter() - t0
off = float((H**2).sum() - (H.diagonal() ** 2).sum())
print(f"MLP ({theta.numel()} params): power iteration {lam_mlp:.6f} vs exact {float(ev[-1]):.6f}; "
      f"Hutchinson (200) {tr_mlp:.3f} vs exact trace {float(H.trace()):.3f} "
      f"(std {np.sqrt(2 * off / 200):.3f}); 301 hvps {t_hvp:.2f} s, exact Hessian {t_exact:.2f} s")

# figure ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.4))
fine = np.geomspace(0.35, 2.8, 400)
ax.plot(fine, fine**2 + fine**-2, color="0.3", lw=1.0, label=r"$a^2+b^2$ with $b=1/a$")
ax.plot(a_grid, sharp, "o", ms=3.5, color=PALETTE[0], label=r"power iteration (sharpness)")
ax.plot(a_grid, trace, "x", ms=4, color=PALETTE[1], label=r"Hutchinson, 400 probes (trace)")
for eta, col, ls in [(0.5, PALETTE[2], "--"), (0.9, "0.5", ":"), (1.0, PALETTE[3], "--")]:
    ax.axhline(2 / eta, color=col, ls=ls, lw=1.0, label=rf"$2/\eta$, $\eta={eta:g}$")
ax.fill_between(fine, 0, 2 / 0.5, where=fine**2 + fine**-2 < 4, color=PALETTE[2], alpha=0.12, lw=0)
ax.plot(flow_end[0], s_flow, "*", ms=10, color=PALETTE[0], mec="k", mew=0.5, label="flow endpoint, Problem (flow-limit)")
ax.plot(gd_end[0], s_gd, "*", ms=10, color=PALETTE[3], mec="k", mew=0.5, label=r"GD endpoint, $\eta=0.9$")
ax.set_xscale("log")
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xticks([0.4, 0.5, 0.7, 1, 1.5, 2, 2.5]); ax.set_xticklabels(["0.4", "0.5", "0.7", "1", "1.5", "2", "2.5"])
ax.set_ylim(0, 8); ax.set_xlim(0.35, 2.8)
ax.set_xlabel(r"position $a$ on the manifold $ab=1$")
ax.set_ylabel(r"$\lambda_{\max}(\nabla^2 L)$ and $\mathrm{tr}\,\nabla^2 L$")
ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2)
savefig(fig, args.out / "sharpness.png")

finish(args, "sharpness", {
    "manifold_max_rel_err_sharpness": rel_sharp, "manifold_max_trace_err": float(np.max(np.abs(trace - exact))),
    "eta05_stable_a_min": float(stable.min()), "eta05_stable_a_max": float(stable.max()),
    "flow_end": flow_end, "flow_end_sharpness": s_flow, "gd09_end": gd_end, "gd09_end_sharpness": s_gd,
    "mlp_params": theta.numel(), "mlp_power_iteration": lam_mlp, "mlp_exact_top": float(ev[-1]),
    "mlp_hutchinson_200": tr_mlp, "mlp_exact_trace": float(H.trace()),
})
