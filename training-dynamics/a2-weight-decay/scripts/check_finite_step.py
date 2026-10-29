"""Check for Problem (finite-step): oscillation and divergence regions in the (γ, η) plane.  [AI]

Uses YOUR max_stable_step and predicted_phase (wdecay/finite_step.py) and the two formulas in the
FORMULAS block below. GD θ_{k+1} = θ_k - η∇L_γ(θ_k) on the two-layer anchor, 10,000 steps per cell:
  A  from the unbalanced start (0.6, 0.4), a 100 x 100 grid of η in (0, 2.5] and γ in (0, 1): each cell
     is classified as converging to the nonzero minimum, converging to the origin, oscillating (bounded
     but not converging) or diverging; agreement with your predicted_phase on cells more than 3% away
     from your boundary η_max(γ);
  B  from the balanced start (0.45, 0.45) (the run never leaves the line a = b), γ in (0, ½) and η in (0, 5]:
     the measured onset of oscillation and of divergence for each γ against your two formulas.
"Converging" means the distance to the nearest fixed point, maximized over the last 200 steps, is below
0.05 and less than half its maximum over steps 4800-5000 (or below 1e-9).

    uv run python a2-weight-decay/scripts/check_finite_step.py [--out DIR] [--solutions]
Writes finite_step.png and the "finite_step" section of results.json.
"""
import numpy as np

from _common import finish, mine, parse

from tdyn.plotting import PALETTE, savefig
from tdyn.swap import use_solutions

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

args = parse(__doc__.splitlines()[0])
fs = mine("finite_step")


# ---- FORMULAS: replace the two bodies with your Derive deliverables (Problem finite-step (d)) ----
def balanced_oscillation_onset(gamma):
    """For γ < ½ and a balanced start: the η above which GD stops converging to (u*, u*)."""
    raise NotImplementedError("type your formula here")


def balanced_divergence_onset(gamma):
    """For γ < ½ and a balanced start: the η above which GD on the line a = b diverges."""
    raise NotImplementedError("type your formula here")
# -------------------------------------------------------------------------------------------------


if use_solutions():
    def balanced_oscillation_onset(gamma):
        return 1.0 / (1.0 - 2.0 * gamma)

    def balanced_divergence_onset(gamma):
        return 2.0 / (1.0 - 2.0 * gamma)

NONZERO, ZERO, OSC, DIV = 1, 2, 3, 4


def classify(E, G, a0, b0, N=10000, W=200):
    """Run GD on every cell of the grids E (η) and G (γ) at once; return labels."""
    us = np.sqrt(np.clip(1 - 2 * G, 0, None))

    def dists(a, b):
        dz = np.hypot(a, b)
        dn = np.where(np.sign(a) == np.sign(b), np.hypot(np.abs(a) - us, np.abs(b) - us), np.inf)
        return dz, np.where(G < 0.5, dn, np.inf)

    a, b = np.full(E.shape, a0), np.full(E.shape, b0)
    m1, m2 = np.zeros(E.shape), np.zeros(E.shape)
    with np.errstate(all="ignore"):
        for k in range(1, N + 1):
            r = a * b - 1
            a, b = a - E * (b * r + 2 * G * a), b - E * (a * r + 2 * G * b)
            big = ~(np.abs(a) < 1e6) | ~(np.abs(b) < 1e6)
            a[big], b[big] = np.inf, np.inf
            if N // 2 - W < k <= N // 2 or k > N - W:
                d = np.minimum(*dists(a, b))
                if k <= N // 2:
                    m1 = np.maximum(m1, d)
                else:
                    m2 = np.maximum(m2, d)
        div = ~np.isfinite(a)
        dz, dn = dists(a, b)
        conv = (~div) & ((m2 < 1e-9) | ((m2 < 0.5 * m1) & (m2 < 0.05)))
    lab = np.full(E.shape, OSC)
    lab[conv & (dz < dn)] = ZERO
    lab[conv & ~(dz < dn)] = NONZERO
    lab[div] = DIV
    return lab


# A: unbalanced start ---------------------------------------------------------------------------
etas = np.linspace(0.025, 2.5, 100)
gams = np.linspace(0.005, 0.995, 100)
E, G = np.meshgrid(etas, gams, indexing="ij")
labA = classify(E, G, 0.6, 0.4)
em = np.asarray(fs.max_stable_step(G), dtype=float)
code = {"nonzero": NONZERO, "zero": ZERO, "unstable": OSC}
pred = np.vectorize(lambda e, g: code[fs.predicted_phase(e, g)])(E, G)
meas = np.where(labA >= OSC, OSC, labA)
far = np.abs(E / em - 1) > 0.03
agreeA = float(np.mean((pred == meas)[far]))
agree_all = float(np.mean(pred == meas))
counts = {k: int(np.sum(labA == v)) for k, v in (("nonzero", NONZERO), ("zero", ZERO), ("oscillate", OSC), ("diverge", DIV))}
osc_gammas = G[labA == OSC]
print(f"A  cells: {counts}")
print(f"A  agreement with your predicted_phase: {agreeA:.4f} of {far.sum()} cells away from the boundary, {agree_all:.4f} of all")
print(f"A  oscillating cells occur for γ in [{osc_gammas.min():.3f}, {osc_gammas.max():.3f}]")

# B: balanced start -----------------------------------------------------------------------------
etasB = np.linspace(0.01, 5.0, 500)
gamsB = np.linspace(0.02, 0.38, 19)
EB, GB = np.meshgrid(etasB, gamsB, indexing="ij")
labB = classify(EB, GB, 0.45, 0.45)
osc_on, div_on = [], []
for j, g in enumerate(gamsB):
    col = labB[:, j]
    osc_on.append(etasB[np.argmax(col != NONZERO)] if np.any(col != NONZERO) else np.nan)
    div_on.append(etasB[np.argmax(col == DIV)] if np.any(col == DIV) else np.nan)
osc_on, div_on = np.array(osc_on), np.array(div_on)
po = np.array([balanced_oscillation_onset(g) for g in gamsB])
pdv = np.array([balanced_divergence_onset(g) for g in gamsB])
ok = np.isfinite(div_on)
err_osc = float(np.nanmax(np.abs(osc_on / po - 1)))
err_div = float(np.max(np.abs(div_on[ok] / pdv[ok] - 1)))
print(f"B  balanced start: max relative error of the oscillation onset {err_osc:.4f}, of the divergence onset "
      f"{err_div:.4f} ({ok.sum()} values of γ with η_div ≤ 5; grid step in η is 0.01)")

colors = ["#ffffff", "#9ecae1", "#c7e9c0", "#fdd0a2", "#d9d9d9"]
cmap = ListedColormap(colors)
legend = [Patch(color=colors[NONZERO], label="converges to nonzero"), Patch(color=colors[ZERO], label="converges to 0"),
          Patch(color=colors[OSC], label="oscillates"), Patch(color=colors[DIV], label="diverges")]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
ax = axs[0]
ax.pcolormesh(G, E, labA, cmap=cmap, vmin=0, vmax=4, shading="nearest")
gg = np.linspace(0.001, 0.999, 999)
gg = gg[np.abs(gg - 0.5) > 1e-9]
ax.plot(gg, np.asarray(fs.max_stable_step(gg), float), "k-", lw=1.0, label=r"your $\eta_{\max}(\gamma)$")
ax.axvline(0.5, color="k", ls=":", lw=0.8)
ax.set_xlim(0, 1); ax.set_ylim(0, 2.5)
ax.set_xlabel(r"weight decay $\gamma$"); ax.set_ylabel(r"step size $\eta$"); ax.set_title("GD from $(0.6, 0.4)$")
ax.legend(handles=legend + [plt.Line2D([], [], color="k", label=r"your $\eta_{\max}(\gamma)$")], frameon=True,
          fontsize=6.5, loc="upper right")
ax = axs[1]
ax.pcolormesh(GB, EB, labB, cmap=cmap, vmin=0, vmax=4, shading="nearest")
g2 = np.linspace(0.01, 0.39, 200)
ax.plot(g2, [balanced_oscillation_onset(g) for g in g2], "k-", lw=1.0, label="your oscillation onset")
ax.plot(g2, [balanced_divergence_onset(g) for g in g2], "k--", lw=1.0, label="your divergence onset")
ax.set_xlim(0.01, 0.39); ax.set_ylim(0, 5)
ax.set_xlabel(r"weight decay $\gamma$"); ax.set_ylabel(r"step size $\eta$"); ax.set_title("GD from $(0.45, 0.45)$: the line $a=b$")
ax.legend(frameon=True, fontsize=6.5, loc="upper left")
savefig(fig, args.out / "finite_step.png")

finish(args, "finite_step", {
    "unbalanced_counts": counts, "unbalanced_agreement_away_from_boundary": agreeA,
    "unbalanced_agreement_all": agree_all, "oscillation_gamma_range": [float(osc_gammas.min()), float(osc_gammas.max())],
    "balanced_gammas": gamsB, "balanced_oscillation_onset": osc_on, "balanced_divergence_onset": div_on,
    "balanced_oscillation_max_rel_error": err_osc, "balanced_divergence_max_rel_error": err_div,
})
