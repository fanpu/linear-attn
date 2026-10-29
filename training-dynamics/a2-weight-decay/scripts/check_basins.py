"""Check for Problem (basins): which starts fall into the zero solution at depth 3.  [AI]

Uses YOUR balanced_endpoint and zero_basin_radius (wdecay/basins.py) and the formula in the FORMULAS
block below. Four parts, on L_γ = ½(abc - s)² + γ(a² + b² + c²):
  A  the balanced ray, γ = 0.1, s = 1: 201 starts u0 in [-0.5, 1.5], each integrated in 3-D, against your
     balanced_endpoint; the measured edge of the zero basin against your zero_basin_radius;
  B  the plane a = b (it is invariant: the flow never leaves it), starts (x, x, y) on an 81 x 81 grid,
     γ = 0.1: which end at zero; the saddle's stable manifold (backward integration) as the boundary;
  C  random starts a0, b0, c0 i.i.d. N(0, σ²), 400 per σ, γ = 0.03, targets s = 1 and s = 0.4: the
     fraction that learns (|abc| ends above 1e-3 s), against σ;
  D  your small-γ approximation of the basin radius against your exact one.

    uv run python a2-weight-decay/scripts/check_basins.py [--out DIR] [--solutions]
Writes basins.png and the "basins" section of results.json.
"""
import numpy as np
from scipy.integrate import solve_ivp

from _common import finish, mine, parse

from tdyn.plotting import PALETTE, savefig
from tdyn.swap import use_solutions

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

args = parse(__doc__.splitlines()[0])
bs = mine("basins")


# ---- FORMULAS: replace the body with your Derive deliverable (Problem basins (a)) ----------------
def small_gamma_radius(gamma, s):
    """The radius of the zero basin on the balanced ray to leading order in small γ."""
    raise NotImplementedError("type your formula here")
# -------------------------------------------------------------------------------------------------


if use_solutions():
    def small_gamma_radius(gamma, s):
        return 2.0 * gamma / s


def grad3(th, gamma, s):
    a, b, c = th[..., 0], th[..., 1], th[..., 2]
    r = a * b * c - s
    return np.stack([b * c * r + 2 * gamma * a, a * c * r + 2 * gamma * b, a * b * r + 2 * gamma * c], -1)


def batch_end(starts, gamma, s, T, sign=-1.0):
    """Integrate many independent flows at once (one stacked ODE); sign=+1 runs time backwards."""
    starts = np.asarray(starts, float)
    shape = starts.shape
    res = solve_ivp(lambda t, y: sign * grad3(y.reshape(shape), gamma, s).ravel(), (0.0, T), starts.ravel(),
                    method="DOP853", rtol=1e-9, atol=1e-11)
    return res.y[:, -1].reshape(shape)


# A: the balanced ray ---------------------------------------------------------------------------
gA, sA = 0.1, 1.0
u0 = np.linspace(-0.5, 1.5, 201)
endsA = batch_end(np.repeat(u0[:, None], 3, 1), gA, sA, 2000.0)[:, 0]
predA = np.asarray(bs.balanced_endpoint(u0, gA, sA), dtype=float)
rA = float(bs.zero_basin_radius(gA, sA))
away = np.abs(u0 - rA) > 0.01
errA = float(np.max(np.abs(endsA - predA)[away]))
pos = u0 > 0
last_zero = float(u0[pos & (np.abs(endsA) < 1e-6)].max())
first_one = float(u0[pos & (endsA > 0.1)].min())
print(f"A  max |u∞ - your balanced_endpoint| (|u0 - radius| > 0.01): {errA:.1e}")
print(f"A  measured edge of the zero basin in ({last_zero:.3f}, {first_one:.3f}); your radius {rA:.5f}")

# B: the invariant plane a = b ------------------------------------------------------------------
xs = np.linspace(-1.5, 1.5, 81)
X, Y = np.meshgrid(xs, xs)
startsB = np.stack([X.ravel(), X.ravel(), Y.ravel()], -1)
endsB = batch_end(startsB, gA, sA, 300.0)
learnB = (np.abs(np.prod(endsB, -1)) > 1e-3).reshape(X.shape)
saddle = np.full(3, rA)
v = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
manifold = []
for sg in (1, -1):
    res = solve_ivp(lambda t, y: grad3(y, gA, sA), (0, 40.0), saddle + sg * 1e-6 * v, method="DOP853",
                    rtol=1e-10, atol=1e-12, dense_output=True,
                    events=lambda t, y: np.max(np.abs(y)) - 3.0)
    tt = np.linspace(0, res.t[-1], 2000)
    manifold.append(res.sol(tt).T)
print(f"B  fraction of the plane's grid that learns: {learnB.mean():.3f}")

# C: random Gaussian starts ---------------------------------------------------------------------
gC = 0.03
sigmas = np.logspace(-1.5, 0.0, 13)
rng = np.random.default_rng(0)
fracC, radC = {}, {}
for sC in (1.0, 0.4):
    fr = []
    for sig in sigmas:
        st = sig * rng.standard_normal((400, 3))
        e = batch_end(st, gC, sC, 400.0)
        fr.append(float(np.mean(np.abs(np.prod(e, -1)) > 1e-3 * sC)))
    fracC[sC] = np.array(fr)
    radC[sC] = float(bs.zero_basin_radius(gC, sC))
    print(f"C  s = {sC:<3g}: fraction learning for σ = " + ", ".join(f"{sg:.3f}: {f:.2f}" for sg, f in zip(sigmas, fr))
          + f"   (your radius {radC[sC]:.4f})")

# D: small-γ radius ------------------------------------------------------------------------------
gD = np.array([1e-3, 3e-3, 1e-2, 3e-2, 0.1])
exactD = np.array([float(bs.zero_basin_radius(g, 1.0)) for g in gD])
approxD = np.array([small_gamma_radius(g, 1.0) for g in gD])
print("D  exact radius / your small-γ formula, s = 1: " + ", ".join(f"γ={g:g}: {e / a:.4f}" for g, e, a in zip(gD, exactD, approxD)))

cmap = ListedColormap(["#dddddd", "#9ecae1"])
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
ax = axs[0]
ax.plot(u0, endsA, "o", ms=2.5, color=PALETTE[0], label="flow in 3-D")
ax.plot(u0, predA, "k-", lw=0.8, label="your balanced_endpoint")
ax.axvline(rA, color=PALETTE[3], ls=":", lw=0.8, label=f"your radius {rA:.3f}")
ax.set_xlabel(r"start $u_0$ on $a=b=c$"); ax.set_ylabel(r"$u_\infty$"); ax.set_title(rf"balanced ray, $\gamma={gA:g}$")
ax.legend(frameon=False, fontsize=7)
ax = axs[1]
ax.pcolormesh(X, Y, learnB.astype(int), cmap=cmap, shading="nearest", vmin=0, vmax=1)
for m in manifold:                      # (a, b, c) -> (-a, -b, c) is a symmetry: mirror the boundary
    ax.plot(m[:, 0], m[:, 2], color=PALETTE[3], lw=1.0)
    ax.plot(-m[:, 0], m[:, 2], color=PALETTE[3], lw=1.0)
ax.plot([rA, -rA], [rA, rA], "o", color=PALETTE[3], ms=4)
ax.plot(xs, xs, color="0.4", lw=0.5, ls=":")
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
ax.set_xlabel(r"$a_0=b_0$"); ax.set_ylabel(r"$c_0$"); ax.set_title("plane $a=b$: blue learns, grey ends at 0")
ax = axs[2]
for i, sC in enumerate((1.0, 0.4)):
    ax.semilogx(sigmas, fracC[sC], "o-", ms=3.5, color=PALETTE[i], label=f"$s={sC:g}$")
    ax.axvline(radC[sC], color=PALETTE[i], ls=":", lw=0.8)
ax.set_xlabel(r"init scale $\sigma$"); ax.set_ylabel("fraction that learns")
ax.set_title(rf"Gaussian starts, $\gamma={gC:g}$ (dotted: your radius)"); ax.legend(frameon=False, fontsize=7)
savefig(fig, args.out / "basins.png")

finish(args, "basins", {
    "balanced_max_error": errA, "balanced_edge_measured": [last_zero, first_one], "balanced_radius_yours": rA,
    "plane_fraction_learning": float(learnB.mean()), "sigmas": sigmas,
    "fraction_learning_s1": fracC[1.0], "fraction_learning_s0.4": fracC[0.4],
    "radius_ratio_exact_over_small_gamma": exactD / approxD,
})
