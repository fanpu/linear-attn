"""Check for Problem (stationary): the bifurcation at γ = ½.  [AI]

Uses YOUR stationary_points and learned_product (wdecay/stationary.py) and the two eigenvalue formulas
in the FORMULAS block below. Three parts:
  A  gradient flow on L_γ from the small balanced start (0.1, 0.1) for 79 values of γ in (0, 1), run to
     t = 3000: the learned product ab against your learned_product, and the one-layer ridge model
     (worked example) for contrast;
  B  your stationary points for γ = 0.1, 0.3, 0.45, 0.7, 0.9: gradient norm there, and the Hessian
     eigenvalues computed by autograd (tdyn.hessian.exact_hessian) against your formulas and types;
  C  near γ = ½ the learned ab against |γ - ½| on log-log axes: the order parameter vanishes linearly.

    uv run python a2-weight-decay/scripts/check_bifurcation.py [--out DIR] [--solutions]
Writes bifurcation.png and the "bifurcation" section of results.json.
"""
import numpy as np
import torch

from _common import finish, mine, parse

from tdyn.hessian import exact_hessian
from tdyn.integrate import solve_flow
from tdyn.plotting import PALETTE, savefig
from tdyn.swap import use_solutions

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
st = mine("stationary")


# ---- FORMULAS: replace the two bodies with your Derive deliverables (Problem stationary (a)) -----
def origin_eigenvalues(gamma):
    """The two Hessian eigenvalues of L_γ at the origin, ascending."""
    raise NotImplementedError("type your formula here")


def minimum_eigenvalues(gamma):
    """The two Hessian eigenvalues of L_γ at the nonzero minima (γ < ½), ascending."""
    raise NotImplementedError("type your formula here")
# -------------------------------------------------------------------------------------------------


if use_solutions():
    def origin_eigenvalues(gamma):
        return np.array([2 * gamma - 1, 2 * gamma + 1])

    def minimum_eigenvalues(gamma):
        return np.sort([4 * gamma, 2 - 4 * gamma])


def field(gamma):
    def f(th):
        a, b = th
        r = a * b - 1
        return -np.array([b * r + 2 * gamma * a, a * r + 2 * gamma * b])
    return f


# A: learned product -----------------------------------------------------------------------------
gammas = np.concatenate([np.linspace(0.0125, 0.4875, 39), np.linspace(0.5125, 0.9875, 39), [0.5 - 1e-3]])
gammas = np.sort(gammas)
learned = np.array([np.prod(solve_flow(field(g), [0.1, 0.1], 3000.0, n_points=2).y[-1]) for g in gammas])
pred = np.asarray(st.learned_product(gammas), dtype=float)
errA = float(np.max(np.abs(learned - pred)))
print(f"A  max |ab(t=3000) - your learned_product| over {len(gammas)} values of γ: {errA:.1e}")

# B: stationary points and Hessians ---------------------------------------------------------------
def L_torch(gamma):
    return lambda th: 0.5 * (th[0] * th[1] - 1) ** 2 + gamma * (th[0] ** 2 + th[1] ** 2)


B = {}
for gamma in (0.1, 0.3, 0.45, 0.7, 0.9):
    rows = []
    for p, kind in st.stationary_points(gamma):
        p = np.asarray(p, dtype=float)
        H = exact_hessian(L_torch(gamma), torch.tensor(p, dtype=torch.float64)).numpy()
        lam = np.linalg.eigvalsh(H)
        mine_lam = origin_eigenvalues(gamma) if np.allclose(p, 0) else minimum_eigenvalues(gamma)
        gnorm = float(np.linalg.norm(field(gamma)(p)))
        rows.append({"point": p, "kind": kind, "autograd_eigs": lam, "your_eigs": np.asarray(mine_lam, float),
                     "grad_norm": gnorm})
    B[f"gamma={gamma:g}"] = rows
    print(f"B  γ = {gamma:<4g}: " + "; ".join(
        f"({r['point'][0]:+.4f}, {r['point'][1]:+.4f}) {r['kind']}, eigs {np.round(r['autograd_eigs'], 6)} "
        f"vs yours {np.round(r['your_eigs'], 6)}, |∇L| = {r['grad_norm']:.0e}" for r in rows))
eig_err = max(float(np.max(np.abs(r["autograd_eigs"] - r["your_eigs"]))) for rows in B.values() for r in rows)
print(f"B  max |autograd eigenvalue - yours| = {eig_err:.1e}")

# C: order parameter near ½ ------------------------------------------------------------------------
dg = np.logspace(-3, -1, 7)
pc = np.array([np.prod(solve_flow(field(0.5 - d), [0.1, 0.1], 20.0 / d, n_points=2).y[-1]) for d in dg])
slopeC = float(np.polyfit(np.log(dg), np.log(pc), 1)[0])
print(f"C  log-log slope of the learned ab against ½ - γ: {slopeC:.4f}")

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1), layout="constrained")
ax = axs[0]
gg = np.linspace(0.0, 1.0, 401)
ax.plot(gg, 1 / (1 + 2 * gg), color="0.55", ls="--", lw=1.0, label=r"one layer (ridge): $1/(1+2\gamma)$")
ax.plot(gg[gg != 0.5], np.asarray(st.learned_product(gg[gg != 0.5]), float), "k-", lw=0.9, label="your learned_product")
ax.plot(gammas, learned, "o", ms=3, color=PALETTE[0], label="gradient flow, $t=3000$")
ax.axvline(0.5, color=PALETTE[3], lw=0.6, ls=":")
ax.set_xlabel(r"weight decay $\gamma$"); ax.set_ylabel(r"learned $ab$"); ax.set_title("two layers: continuous transition")
ax.legend(frameon=False, fontsize=6.8)
ax = axs[1]
g1, g2 = np.linspace(0, 0.5, 200, endpoint=False), np.linspace(0, 1.0, 200)
o = np.array([origin_eigenvalues(g) for g in g2]); mn = np.array([minimum_eigenvalues(g) for g in g1])
ax.plot(g2, o[:, 0], color=PALETTE[3], label=r"origin, $\lambda_{\min}$")
ax.plot(g2, o[:, 1], color=PALETTE[3], ls="--", label=r"origin, $\lambda_{\max}$")
ax.plot(g1, mn[:, 0], color=PALETTE[0], label=r"nonzero min, $\lambda_{\min}$")
ax.plot(g1, mn[:, 1], color=PALETTE[0], ls="--", label=r"nonzero min, $\lambda_{\max}$")
for key, rows in B.items():
    g = float(key.split("=")[1])
    for r in rows:
        ax.plot([g, g], r["autograd_eigs"], "k.", ms=4)
ax.axhline(0, color="0.5", lw=0.6); ax.axvline(0.5, color=PALETTE[3], lw=0.6, ls=":")
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("Hessian eigenvalue"); ax.set_title("your eigenvalues (dots: autograd)")
ax.legend(frameon=False, fontsize=6.3)
ax = axs[2]
ax.loglog(dg, pc, "o", ms=4, color=PALETTE[0], label="gradient flow")
ax.loglog(dg, np.asarray(st.learned_product(0.5 - dg), float), "k-", lw=0.8, label="your learned_product")
ax.set_xlabel(r"$\frac{1}{2}-\gamma$"); ax.set_ylabel(r"learned $ab$"); ax.set_title(f"near the transition (slope {slopeC:.3f})")
ax.legend(frameon=False, fontsize=7)
savefig(fig, args.out / "bifurcation.png")

finish(args, "bifurcation", {
    "gammas": gammas, "learned_product_flow": learned, "learned_product_max_error": errA,
    "hessian_eig_max_error": eig_err,
    "stationary": {k: [{"point": r["point"], "kind": r["kind"], "eigs": r["autograd_eigs"]} for r in v]
                   for k, v in B.items()},
    "order_parameter_slope": slopeC,
})
