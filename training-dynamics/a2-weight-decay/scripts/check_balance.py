"""Check for Problem (balance): weight decay forces balance.  [AI]

Uses the formula in the FORMULAS block below (Problem balance (b)). Three parts:
  A  a two-layer linear network W2 W1 (d_in = 4, m = 3, d_out = 2, correlated inputs) under gradient
     flow on ½E|W2W1x - y|² + γ(|W1|² + |W2|²), γ = 0.05 and 0.2: the imbalance matrix
     D = W1W1ᵀ - W2ᵀW2 along the flow against your formula;
  B  the depth-3 scalar net abc with weight decay γ = 0.1: a² - b² and b² - c² against the same law;
  C  where the flow ends: 200 random starts of the two-layer anchor with γ = 0 and γ = 0.1 (|a∞| vs
     |b∞|), and the per-neuron balance of the matrix network at t = 200.

    uv run python a2-weight-decay/scripts/check_balance.py [--out DIR] [--solutions]
Writes balance.png and the "balance" section of results.json.
"""
import numpy as np

from _common import finish, parse

from tdyn.integrate import solve_flow
from tdyn.plotting import PALETTE, savefig
from tdyn.swap import use_solutions

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])


# ---- FORMULAS: replace the body with your Derive deliverable (Problem balance (b)) ---------------
def imbalance_at(D0, gamma, t):
    """D(t) under gradient flow on L(W2 W1) + γ(|W1|² + |W2|²), given D(0) = D0 (an array, or a
    scalar such as c0 = a0² - b0² for the anchor)."""
    raise NotImplementedError("type your formula here")
# -------------------------------------------------------------------------------------------------


if use_solutions():
    def imbalance_at(D0, gamma, t):
        return np.asarray(D0) * np.exp(-4.0 * gamma * t)

# A: matrix network ------------------------------------------------------------------------------
rng = np.random.default_rng(0)
d_in, m, d_out = 4, 3, 2
Q, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
Sxx = Q @ np.diag(np.linspace(0.5, 2.0, d_in)) @ Q.T
Syx = rng.standard_normal((d_out, d_in))
W1_0 = 0.8 * rng.standard_normal((m, d_in))
W2_0 = 0.3 * rng.standard_normal((d_out, m))
n1 = m * d_in


def unpack(th):
    return th[:n1].reshape(m, d_in), th[n1:].reshape(d_out, m)


def matrix_field(gamma):
    def f(th):
        W1, W2 = unpack(th)
        G = W2 @ W1 @ Sxx - Syx
        return -np.concatenate([(W2.T @ G + 2 * gamma * W1).ravel(), (G @ W1.T + 2 * gamma * W2).ravel()])
    return f


th0 = np.concatenate([W1_0.ravel(), W2_0.ravel()])
D0 = W1_0 @ W1_0.T - W2_0.T @ W2_0
matA, curvesA = {}, {}
for gamma in (0.05, 0.2):
    tr = solve_flow(matrix_field(gamma), th0, 20.0, n_points=201)
    errs, norms = [], []
    for t, th in zip(tr.t, tr.y):
        W1, W2 = unpack(th)
        D = W1 @ W1.T - W2.T @ W2
        errs.append(np.linalg.norm(D - np.asarray(imbalance_at(D0, gamma, t))) / np.linalg.norm(D0))
        norms.append(np.linalg.norm(D))
    matA[f"gamma={gamma:g}"] = float(max(errs))
    curvesA[gamma] = (tr.t, np.array(norms))
print("A  matrix net: max |D(t) - your D(t)|_F / |D0|_F on [0, 20]:", {k: f"{v:.1e}" for k, v in matA.items()})

# B: depth 3 -------------------------------------------------------------------------------------
gB = 0.1


def f3(th):
    a, b, c = th
    r = a * b * c - 1.0
    return -np.array([b * c * r + 2 * gB * a, a * c * r + 2 * gB * b, a * b * r + 2 * gB * c])


tr = solve_flow(f3, [1.3, 0.4, -0.9], 20.0, n_points=201)
a, b, c = tr.y.T
pairs = {"a2-b2": a * a - b * b, "b2-c2": b * b - c * c}
depth3_err = {k: float(np.max(np.abs(v - np.array([imbalance_at(v[0], gB, t) for t in tr.t])))) / abs(v[0])
              for k, v in pairs.items()}
print("B  depth 3: max relative error of a²-b², b²-c² against your formula:", {k: f"{v:.1e}" for k, v in depth3_err.items()})

# C: endpoints -----------------------------------------------------------------------------------
starts = np.random.default_rng(1).uniform(-2, 2, size=(200, 2))
ends = {}
for gamma in (0.0, 0.1):
    def f2(th, g=gamma):
        a, b = th
        r = a * b - 1
        return -np.array([b * r + 2 * g * a, a * r + 2 * g * b])
    ends[gamma] = np.array([solve_flow(f2, s0, 200.0, n_points=2).y[-1] for s0 in starts])
gap = {f"gamma={g:g}": float(np.max(np.abs(np.abs(e[:, 0]) - np.abs(e[:, 1])))) for g, e in ends.items()}
print("C  max ||a∞| - |b∞|| over 200 starts:", {k: f"{v:.3g}" for k, v in gap.items()})
W1, W2 = unpack(solve_flow(matrix_field(0.05), th0, 200.0, n_points=2).y[-1])
neuron_gap = float(np.max(np.abs(np.linalg.norm(W1, axis=1) ** 2 - np.linalg.norm(W2, axis=0) ** 2)))
D_end = float(np.linalg.norm(W1 @ W1.T - W2.T @ W2))
print(f"C  matrix net at t = 200, γ = 0.05: |D|_F = {D_end:.2e}, max_j | |row j of W1|² - |col j of W2|² | = {neuron_gap:.2e}")

fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2), layout="constrained")
ax = axs[0]
for i, gamma in enumerate((0.05, 0.2)):
    t, nrm = curvesA[gamma]
    ax.semilogy(t, nrm, color=PALETTE[i], lw=2.4, alpha=0.5, label=rf"matrix net $|D(t)|_F$, $\gamma={gamma:g}$")
    ax.semilogy(t, [np.linalg.norm(imbalance_at(D0, gamma, s)) for s in t], "k--", lw=0.7)
for j, (k, v) in enumerate(pairs.items()):
    ax.semilogy(tr.t, np.abs(v), color=PALETTE[2 + j], lw=2.4, alpha=0.5,
                label=rf"depth 3 $|{k[0]}^2-{k[3]}^2|$, $\gamma={gB:g}$")
    ax.semilogy(tr.t, np.abs([imbalance_at(v[0], gB, t) for t in tr.t]), "k--", lw=0.7)
ax.set_xlabel("time $t$"); ax.set_ylabel("imbalance"); ax.set_title("imbalance decays (dashed: yours)")
ax.legend(frameon=False, fontsize=6.5)
ax = axs[1]
for i, gamma in enumerate((0.0, 0.1)):
    e = ends[gamma]
    ax.plot(np.abs(e[:, 0]), np.abs(e[:, 1]), "o", ms=3, color=PALETTE[i], alpha=0.7,
            label=rf"$\gamma={gamma:g}$")
ax.plot([0, 3], [0, 3], color="0.6", lw=0.6, ls=":")
ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect("equal")
ax.set_xlabel(r"$|a_\infty|$"); ax.set_ylabel(r"$|b_\infty|$"); ax.set_title("where 200 flows end")
ax.legend(frameon=False, fontsize=7)
savefig(fig, args.out / "balance.png")

finish(args, "balance", {
    "matrix_max_rel_error": matA, "depth3_max_rel_error": depth3_err, "endpoint_abs_gap": gap,
    "matrix_D_end": D_end, "matrix_neuron_gap": neuron_gap,
})
