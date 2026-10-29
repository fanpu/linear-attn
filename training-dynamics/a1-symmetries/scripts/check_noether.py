"""Check for Problem (noether): the rescaling symmetry conserves the imbalance.  [AI]

Uses YOUR rescale, generator and conserved (symmetries/noether.py). Part 1: at 2000 random points,
the Noether constraint <∇L, X> = 0 for three losses that depend on (a, b) only through ab
(squared loss with target 2, logistic loss, a 16-example regression), and for one that does not
(the anchor with a skip connection, f = (ab + 0.3a)x). Part 2: your Q along accurate gradient-flow
solutions of all four, and along GD with η = 0.1 on the anchor. Part 3: the Hessian identity
H X + (∂X/∂θ) ∇L = 0 and H X = 0 on the minima. Part 4: small initialization gives balanced layers:
the size of c for i.i.d. N(0, σ²) initial weights and the gap a∞ - b∞ at the end of the flow,
against your formulas (FORMULAS block below).

    uv run python a1-symmetries/scripts/check_noether.py [--out DIR] [--solutions]
Writes noether.png and the "noether" section of results.json.
"""
import numpy as np

from _common import finish, mine, parse

from tdyn.integrate import solve_flow
from tdyn.models import anchor_flow, anchor_hessian
from tdyn.optim import gd_trajectory
from tdyn.plotting import PALETTE, savefig
from tdyn.swap import use_solutions

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
noether = mine("noether")


# ---- FORMULAS: replace the two bodies with your Derive deliverables (Problem noether (c)) -------
def rms_imbalance(sigma):
    """sqrt(E[c²]) for a0, b0 i.i.d. N(0, σ²)."""
    raise NotImplementedError("type your formula here")


def endpoint_gap(c):
    """a∞ - b∞ to leading order in small c, for the anchor (s = 1), on the positive branch."""
    raise NotImplementedError("type your formula here")
# -------------------------------------------------------------------------------------------------


if use_solutions():
    def rms_imbalance(sigma):
        return 2.0 * sigma ** 2

    def endpoint_gap(c):
        return c / 2.0

rng = np.random.default_rng(0)
x_d = np.random.default_rng(1).standard_normal(16)
y_d = 0.7 * x_d + 0.3 * np.random.default_rng(2).standard_normal(16)


def grad_of(dl):
    def g(th):
        a, b = th[..., 0], th[..., 1]
        d = dl(a * b)
        return np.stack([d * b, d * a], axis=-1)
    return g


def grad_skip(th):                                  # L = ½(ab + 0.3a - 1)²: not a function of ab
    a, b = th[..., 0], th[..., 1]
    r = a * b + 0.3 * a - 1.0
    return np.stack([(b + 0.3) * r, a * r], axis=-1)


losses = {
    "squared, target 2": grad_of(lambda p: p - 2.0),
    "logistic": grad_of(lambda p: -1.0 / (1.0 + np.exp(p))),
    "16-example regression": grad_of(lambda p: np.mean(x_d * (p[..., None] * x_d - y_d), axis=-1)),
    "skip connection (breaks symmetry)": grad_skip,
}

# Part 1: Noether constraint ---------------------------------------------------------------------
pts = rng.uniform(-2, 2, size=(2000, 2))
X = np.asarray(noether.generator(pts))
ortho = {}
for name, g in losses.items():
    G = g(pts)
    ortho[name] = float(np.max(np.abs(np.sum(G * X, -1)) / (np.linalg.norm(G, axis=-1) * np.linalg.norm(X, axis=-1))))
    print(f"max |<grad L, X>| / (|grad L||X|)  {name:<36} {ortho[name]:.2e}")

# Part 2: Q along the flow, and along GD ----------------------------------------------------------
T, th0 = 10.0, np.array([1.2, 0.3])
drift, curves = {}, {}
for name, g in losses.items():
    tr = solve_flow(lambda th: -g(th), th0, T, n_points=401)
    Q = np.asarray(noether.conserved(tr.y), dtype=float)
    drift[name] = float(np.max(np.abs(Q - Q[0])))
    curves[name] = (tr.t, np.abs(Q - Q[0]))
eta = 0.1
gd = gd_trajectory(losses["squared, target 2"], th0, eta, int(T / eta))
Qgd = np.asarray(noether.conserved(gd), dtype=float)
drift["GD eta=0.1, squared target 2"] = float(np.max(np.abs(Qgd - Qgd[0])))
for k, v in drift.items():
    print(f"max |Q(t) - Q(0)| on t in [0, {T:g}]  {k:<36} {v:.2e}")

# Part 3: Hessian identity on the anchor ---------------------------------------------------------
h = 1e-6
P = rng.uniform(-2, 2, size=(200, 2))
res = []
for th in P:
    Jx = np.stack([(np.asarray(noether.generator(th + h * e)) - np.asarray(noether.generator(th - h * e))) / (2 * h)
                   for e in np.eye(2)], axis=1)          # ∂X/∂θ, column j = ∂X/∂θ_j
    a, b = th
    gL = np.array([b * (a * b - 1), a * (a * b - 1)])
    res.append(np.linalg.norm(anchor_hessian(th) @ np.asarray(noether.generator(th)) + Jx @ gL))
hess_identity = float(np.max(res))
on_min = float(max(np.linalg.norm(anchor_hessian([a, 1 / a]) @ np.asarray(noether.generator([a, 1 / a])))
                   for a in np.linspace(0.2, 3, 30)))
print(f"max |H X + (dX/dtheta) grad L| at 200 random points: {hess_identity:.2e};  max |H X| on ab = 1: {on_min:.2e}")

# Part 4: small init gives balanced layers --------------------------------------------------------
sigmas = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
rms_c, rms_gap, rms_c_flow = [], [], []
for sg in sigmas:
    ab = sg * rng.standard_normal((20000, 2))
    rms_c.append(np.sqrt(np.mean((ab[:, 0] ** 2 - ab[:, 1] ** 2) ** 2)))
    gaps, cs = [], []
    for a0, b0 in ab[:150]:
        if a0 + b0 < 0:                              # mirror to the positive branch (same law)
            a0, b0 = -a0, -b0
        end = solve_flow(anchor_flow, [a0, b0], 60.0, n_points=2).y[-1]          # the anchor, s = 1
        gaps.append(end[0] - end[1])
        cs.append(a0 * a0 - b0 * b0)
    gaps, cs = np.array(gaps), np.array(cs)
    rms_gap.append(np.sqrt(np.mean(gaps ** 2)))
    rms_c_flow.append(np.sqrt(np.mean(cs ** 2)))
rms_c, rms_gap, rms_c_flow = map(np.array, (rms_c, rms_gap, rms_c_flow))
pred_c = np.array([rms_imbalance(s) for s in sigmas])
pred_gap = np.abs(np.array([endpoint_gap(c) for c in rms_c_flow]))
print("sigma   rms c (20000 draws)  formula   rms(a_inf - b_inf) (150 flows)  formula at their rms c")
for i, sg in enumerate(sigmas):
    print(f"{sg:<7} {rms_c[i]:<20.4e} {pred_c[i]:<9.4e} {rms_gap[i]:<31.4e} {pred_gap[i]:.4e}")

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), layout="constrained")
ax = axs[0]
for i, (name, (t, d)) in enumerate(curves.items()):
    ax.semilogy(t, np.maximum(d, 1e-17), color=PALETTE[i], label=f"flow, {name}")
ax.semilogy(np.arange(len(Qgd)) * eta, np.maximum(np.abs(Qgd - Qgd[0]), 1e-17), color=PALETTE[4], ls="--",
            label=r"GD $\eta=0.1$, squared target 2")
ax.set_ylim(1e-17, 10); ax.set_xlabel("time $t$ (GD: $k\\eta$)"); ax.set_ylabel(r"$|Q(t)-Q(0)|$")
ax.set_title("your $Q$ along four flows and one GD run")
ax.legend(frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.64))
ax = axs[1]
ax.loglog(sigmas, rms_c, "o", color=PALETTE[0], label=r"rms $c$, measured")
ax.loglog(sigmas, pred_c, "-", color=PALETTE[0], lw=0.8, label="your formula")
ax.loglog(sigmas, rms_gap, "s", color=PALETTE[1], label=r"rms $(a_\infty-b_\infty)$, flow")
ax.loglog(sigmas, pred_gap, "-", color=PALETTE[1], lw=0.8, label="your formula")
ax.set_xlabel(r"init scale $\sigma$"); ax.set_title("small init, balanced layers")
ax.legend(frameon=False, fontsize=7)
savefig(fig, args.out / "noether.png")

finish(args, "noether", {
    "orthogonality": ortho, "flow_drift": drift, "hessian_identity_residual": hess_identity,
    "hessian_null_on_minima": on_min, "sigmas": sigmas, "rms_c": rms_c, "rms_c_formula": pred_c,
    "rms_endpoint_gap": rms_gap, "endpoint_gap_formula": pred_gap,
})
