"""Check for Problem (gd-stability): GD on a 2-D quadratic with eigenvalues 1 and 10.  [AI]

Uses YOUR gd (prereqs/gd.py). L(θ) = ½ θᵀHθ with H = R diag(1, 10) Rᵀ, R a 30-degree rotation.
For each η on a fine grid it runs 200 steps and measures the contraction factor per step,
ρ̂(η) = (|θ_200| / |θ_100|)^(1/100), then reads off the critical step size (first η with ρ̂ > 1),
the step size with the smallest ρ̂, and that smallest ρ̂. Compare them with your deliverables.

    uv run python a0-prereqs/scripts/check_gd_stability.py [--out DIR] [--solutions]
Writes gd_stability.png and the "gd_stability" section of results.json.
"""
import numpy as np

from _common import finish, mine, parse

from tdyn.plotting import PALETTE, savefig

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
gd = mine("gd").gd
lam = np.array([1.0, 10.0])
c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
R = np.array([[c, -s], [s, c]])
H = R @ np.diag(lam) @ R.T
theta0 = np.array([1.0, 1.0])
grad = lambda th: H @ th

etas = np.round(np.arange(0.0005, 0.2405, 0.0005), 6)
K0, K = 100, 200
rho = []
with np.errstate(over="ignore", invalid="ignore"):
    for eta in etas:
        tr = np.asarray(gd(grad, theta0, eta, K))
        rho.append((np.linalg.norm(tr[K]) / np.linalg.norm(tr[K0])) ** (1.0 / (K - K0)))
rho = np.array(rho)
eta_crit = float(etas[np.argmax(rho > 1.0)])
i_best = int(np.argmin(rho))
print(f"measured: first eta with rho > 1: {eta_crit:.4f}; best eta {etas[i_best]:.4f} with rho {rho[i_best]:.4f}")

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
ax = axs[0]
ax.plot(etas, np.abs(1 - etas * lam[0]), ":", color=PALETTE[2], label=r"$|1-\eta\lambda_{\min}|$")
ax.plot(etas, np.abs(1 - etas * lam[1]), "--", color=PALETTE[1], label=r"$|1-\eta\lambda_{\max}|$")
ax.plot(etas, rho, color=PALETTE[0], lw=2.2, alpha=0.6, label=r"measured $\hat\rho(\eta)$")
ax.axhline(1.0, color="k", lw=0.6)
for x, txt in [(0.1, r"$1/\lambda_{\max}$"), (0.2, r"$2/\lambda_{\max}$")]:
    ax.axvline(x, color="0.5", lw=0.6)
    ax.text(x, 1.42, txt, ha="center", va="bottom", fontsize=7)
ax.set_ylim(0, 1.5); ax.set_xlim(0, 0.24)
ax.set_xlabel(r"step size $\eta$"); ax.set_ylabel("contraction factor per step")
ax.legend(frameon=False, loc="lower left", fontsize=7)
ax.set_title(r"$\lambda(H) = \{1, 10\}$")
ax = axs[1]
for eta, col in [(0.04, PALETTE[2]), (0.17, PALETTE[1]), (0.21, PALETTE[3])]:
    tr = np.asarray(gd(grad, theta0, eta, 60))
    z = tr @ R                                  # coordinates in the eigenbasis of H
    ax.semilogy(np.abs(z[:, 1]) + 1e-300, "-", color=col, label=rf"$\eta={eta}$, stiff mode")
    ax.semilogy(np.abs(z[:, 0]) + 1e-300, ":", color=col, label=rf"$\eta={eta}$, slow mode")
ax.set_xlabel("step $k$"); ax.set_ylabel(r"$|$mode coordinate$|$")
ax.set_ylim(1e-8, 1e3)
ax.legend(frameon=False, fontsize=6.5, ncol=1, loc="lower left")
ax.set_title("the two eigen-directions separately")
savefig(fig, args.out / "gd_stability.png")

finish(args, "gd_stability", {
    "eta_crit_measured": eta_crit, "eta_best_measured": float(etas[i_best]),
    "rho_best_measured": float(rho[i_best]),
    "rho_at": {f"{e}": float(rho[np.argmin(np.abs(etas - e))]) for e in (0.01, 0.05, 0.1, 0.15, 0.19)},
})
