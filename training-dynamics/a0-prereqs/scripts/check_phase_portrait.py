"""Check for Problem (phase-portrait): the anchor model's phase portrait.  [AI]

Uses YOUR classify_fixed_point (prereqs/fixed_points.py). Reproduces the phase portrait of
James Ang's post (papers/ang2026-training-dynamics-post.md, end of §2): gradient flow on
L = ½(ab-1)² from a grid of initializations, coloured by the branch of ab = 1 each reaches,
with the separatrix a = -b and the conserved curves a² - b² = c. Also measures the flow's
Jacobian at fixed points by finite differences (not from the Hessian formula) and prints its
eigenvalues, to compare with your Derive deliverable.

    uv run python a0-prereqs/scripts/check_phase_portrait.py [--out DIR] [--solutions]
Writes phase_portrait.png and the "phase_portrait" section of results.json.
"""
import numpy as np

from _common import finish, mine, parse

from tdyn.integrate import rk4
from tdyn.models import anchor_flow
from tdyn.plotting import PALETTE, phase_portrait, savefig

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
classify = mine("fixed_points").classify_fixed_point


def fd_jacobian(F, x, h=1e-6):
    """Central finite differences: column j is (F(x + h e_j) - F(x - h e_j)) / 2h."""
    x = np.asarray(x, dtype=float)
    cols = [(F(x + h * e) - F(x - h * e)) / (2 * h) for e in np.eye(len(x))]
    return np.stack(cols, axis=1)


# --- fixed points: linearization ------------------------------------------------------------
fixed = {"origin": (0.0, 0.0)}
for a in (0.5, 1.0, 2.0, -1.0):
    fixed[f"a={a:g} on ab=1"] = (a, 1.0 / a)
fp_rows = {}
print(f"{'fixed point':<18} {'eigenvalues of J (finite diff.)':<34} {'a^2+b^2':>8}  your label")
for name, p in fixed.items():
    J = fd_jacobian(anchor_flow, p)
    ev = np.sort(np.linalg.eigvals(J).real)
    lab = classify(J)
    fp_rows[name] = {"point": p, "eigenvalues": ev, "a2_plus_b2": p[0] ** 2 + p[1] ** 2, "label": lab}
    print(f"{name:<18} {np.array2string(ev, precision=6):<34} {p[0]**2 + p[1]**2:8.4f}  {lab}")

# --- trajectories from a grid --------------------------------------------------------------
g = np.linspace(-2.4, 2.4, 13)
A0, B0 = np.meshgrid(g, g)
inits = np.stack([A0.ravel(), B0.ravel()], axis=1)
inits = inits[np.abs(inits[:, 0] + inits[:, 1]) > 1e-9]          # the separatrix is drawn separately
dt, steps = 0.005, 4000                                            # t in [0, 20]
traj = rk4(anchor_flow, inits, dt, steps)                          # (steps+1, N, 2), vectorized
final = traj[-1]
branch = np.sign(final[:, 0])                                      # +1: a, b > 0;  -1: a, b < 0
predicted = np.sign(inits[:, 0] + inits[:, 1])
agree = float(np.mean(branch == predicted))
final_loss = 0.5 * (final[:, 0] * final[:, 1] - 1) ** 2
c_drift = np.max(np.abs((traj[..., 0] ** 2 - traj[..., 1] ** 2) - (inits[:, 0] ** 2 - inits[:, 1] ** 2)))
sep = rk4(anchor_flow, np.array([[2.0, -2.0], [-2.0, 2.0]]), dt, steps)[-1]
print(f"{len(inits)} inits: branch = sign(a0 + b0) for {agree:.0%}; max final loss {final_loss.max():.2e}; "
      f"max |c(t) - c(0)| {c_drift:.1e}; start on a = -b ends at {np.abs(sep).max():.1e} from the origin")

# --- figure ----------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.9))
lim = (-2.5, 2.5)
X, Y = np.meshgrid(np.linspace(*lim, 400), np.linspace(*lim, 400))
aa = np.linspace(0.38, 2.5, 200)
for ax in axs:
    ax.contour(X, Y, X**2 - Y**2, levels=[-4, -2, -1, -0.3, 0.3, 1, 2, 4], colors="0.55",
               linewidths=0.5, linestyles="--")
    ax.plot(aa, 1 / aa, "k", lw=1.6); ax.plot(-aa, -1 / aa, "k", lw=1.6)
    ax.plot([-2.5, 2.5], [2.5, -2.5], color="#D62728", lw=1.6)
phase_portrait(anchor_flow, xlim=lim, ylim=lim, ax=axs[0], density=1.0, stream_color="0.35")
axs[0].plot(0, 0, "o", ms=6, mfc="white", mec="k", zorder=5)
axs[0].annotate(f"origin: {fp_rows['origin']['label']}", (0, 0), (0.25, -1.2), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.5), bbox=dict(fc="white", ec="none", pad=1), zorder=6)
axs[0].annotate(f"ab = 1: {fp_rows['a=1 on ab=1']['label']}", (1.6, 1 / 1.6), (0.6, 2.05), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.5), bbox=dict(fc="white", ec="none", pad=1), zorder=6)
axs[0].set_title("vector field $-\\nabla L$ and conserved curves")
ax = axs[1]
ax.contourf(X, Y, X + Y, levels=[-10, 0, 10], colors=["#FBE3C8", "#D6E6F4"], zorder=0)
for k in range(len(inits)):
    col = PALETTE[0] if branch[k] > 0 else PALETTE[1]
    ax.plot(traj[:, k, 0], traj[:, k, 1], color=col, lw=0.8, zorder=2)
    ax.plot(*inits[k], "o", ms=1.8, color=col, zorder=3)
ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
ax.set_xlabel("$a$"); ax.set_ylabel("$b$")
ax.set_title("where each initialization ends")
savefig(fig, args.out / "phase_portrait.png")

finish(args, "phase_portrait", {
    "fixed_points": fp_rows, "n_inits": len(inits), "branch_matches_sign_a_plus_b": agree,
    "max_final_loss": float(final_loss.max()), "max_conservation_drift": float(c_drift),
    "separatrix_start_final_norm": float(np.abs(sep).max()),
})
