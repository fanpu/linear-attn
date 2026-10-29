"""Check for Problem (depth3): depth 3 makes the transition discontinuous.  [AI]

Uses YOUR balanced_roots, critical_gamma and coexistence_gamma (wdecay/depth3.py) and the formula in
the FORMULAS block below. Four parts, on L_γ = ½(abc - s)² + γ(a² + b² + c²):
  A  the Hessian at the origin by autograd for γ = 0.01, 0.1, 0.5 (compare with your (a));
  B  gradient flow from (1, 1.01, 0.99) for 80 values of γ in (0, 0.4), s = 1: the learned product
     abc against the larger balanced root of your balanced_roots, and the two-layer anchor for contrast;
  C  the measured γ*: bisection on γ for the largest value at which the flow from (1, 1, 1) still
     learns, for s = 0.25, 0.5, 1, 2, against your critical_gamma(s); the product just before the jump;
  D  an adiabatic sweep (each γ warm-started from the previous end point plus a 1e-3 random kick):
     γ up from 0 to 0.6, then back down, at depth 2 and depth 3 (hysteresis).

    uv run python a2-weight-decay/scripts/check_depth3.py [--out DIR] [--solutions]
Writes depth3.png and the "depth3" section of results.json.
"""
import numpy as np
import torch

from _common import finish, mine, parse

from tdyn.hessian import exact_hessian
from tdyn.plotting import PALETTE, savefig
from tdyn.swap import use_solutions

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
d3 = mine("depth3")


# ---- FORMULAS: replace the body with your Derive deliverable (Problem depth3 (b)) ----------------
def product_at_jump(s):
    """The product abc at the nonzero minimum as γ → γ*(s) from below."""
    raise NotImplementedError("type your formula here")
# -------------------------------------------------------------------------------------------------


if use_solutions():
    def product_at_jump(s):
        return s / 4.0


def field3(gamma, s=1.0):
    def f(th):
        a, b, c = th
        r = a * b * c - s
        return -np.array([b * c * r + 2 * gamma * a, a * c * r + 2 * gamma * b, a * b * r + 2 * gamma * c])
    return f


def field2(gamma):
    def f(th):
        a, b = th
        r = a * b - 1
        return -np.array([b * r + 2 * gamma * a, a * r + 2 * gamma * b])
    return f


def end(f, th0, T):
    """End point of the flow at time T, or earlier once the speed |θ̇| has dropped below 1e-11."""
    def settled(t, y):
        return np.linalg.norm(f(y)) - 1e-11
    settled.terminal, settled.direction = True, -1
    from scipy.integrate import solve_ivp
    res = solve_ivp(lambda t, y: f(y), (0.0, T), np.asarray(th0, float), method="DOP853",
                    rtol=1e-10, atol=1e-12, events=settled)
    return res.y[:, -1]


# A: Hessian at the origin -----------------------------------------------------------------------
hessA = {}
for g in (0.01, 0.1, 0.5):
    H = exact_hessian(lambda th: 0.5 * (th[0] * th[1] * th[2] - 1) ** 2 + g * (th ** 2).sum(),
                      torch.zeros(3, dtype=torch.float64)).numpy()
    hessA[f"gamma={g:g}"] = H
    print(f"A  γ = {g:<4g} Hessian at the origin by autograd: diag {np.diag(H)}, max |off-diagonal| "
          f"{np.max(np.abs(H - np.diag(np.diag(H)))):.1e}")

# B: learned product vs γ ------------------------------------------------------------------------
gstar = float(d3.critical_gamma(1.0))
gammas = np.linspace(0.005, 0.4, 80)
p3 = np.array([np.prod(end(field3(g), [1.0, 1.01, 0.99], 4000.0)) for g in gammas])
p2 = np.array([np.prod(end(field2(g), [1.0, 1.01], 4000.0)) for g in gammas])


def predicted_p3(g, s=1.0):
    r = np.asarray(d3.balanced_roots(g, s), dtype=float)
    return float(r[-1] ** 3) if len(r) else 0.0


pred3 = np.array([predicted_p3(g) for g in gammas])
away = np.abs(gammas - gstar) > 0.005
errB = float(np.max(np.abs(p3 - pred3)[away]))
print(f"B  max |abc(t=4000) - your larger root cubed| (|γ - γ*| > 0.005): {errB:.1e}")

# C: γ* by bisection, for several targets ---------------------------------------------------------
svals = np.array([0.25, 0.5, 1.0, 2.0])
gs_meas, p_jump = [], []
for s in svals:
    lo, hi = 0.0, s
    p_lo = s
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        pm = np.prod(end(field3(mid, s), [s ** (1 / 3)] * 3, 2e4))
        if pm > 1e-3 * s:
            lo, p_lo = mid, pm
        else:
            hi = mid
    gs_meas.append(0.5 * (lo + hi))
    p_jump.append(p_lo)
gs_meas, p_jump = np.array(gs_meas), np.array(p_jump)
gs_yours = np.array([float(d3.critical_gamma(s)) for s in svals])
slopeC = float(np.polyfit(np.log(svals), np.log(gs_meas), 1)[0])
for s, gm, gy, pj in zip(svals, gs_meas, gs_yours, p_jump):
    print(f"C  s = {s:<4g} measured γ* = {gm:.7f}, yours {gy:.7f} (rel. diff {abs(gm / gy - 1):.1e}); "
          f"abc just before the jump {pj:.4f}, your formula {product_at_jump(s):.4f}")
print(f"C  log-log slope of γ* against s: {slopeC:.4f};  your γ_c(1) = {float(d3.coexistence_gamma(1.0)):.5f}")

# D: adiabatic sweeps ----------------------------------------------------------------------------
rng = np.random.default_rng(0)
gsweep = np.linspace(0.0, 0.6, 61)[1:]
sweeps = {}
for depth, field, th in ((2, field2, np.array([1.0, 1.0])), (3, field3, np.array([1.0, 1.0, 1.0]))):
    up, down = [], []
    for g in gsweep:
        th = end(field(g), th + 1e-3 * rng.standard_normal(th.size), 400.0)
        up.append(np.prod(th))
    for g in gsweep[::-1]:
        th = end(field(g), th + 1e-3 * rng.standard_normal(th.size), 400.0)
        down.append(np.prod(th))
    sweeps[depth] = (np.array(up), np.array(down[::-1]))
    print(f"D  depth {depth}: product on the way up at γ = 0.1, 0.3: {sweeps[depth][0][9]:.4f}, {sweeps[depth][0][29]:.4f}; "
          f"on the way down: {sweeps[depth][1][9]:.4f}, {sweeps[depth][1][29]:.4f}")

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1), layout="constrained")
ax = axs[0]
ax.plot(gammas, p2, "o", ms=2.5, color=PALETTE[0], label="depth 2, flow")
ax.plot(gammas, p3, "o", ms=2.5, color=PALETTE[3], label="depth 3, flow")
ax.plot(gammas, pred3, "k-", lw=0.8, label="depth 3, your root$^3$")
ax.axvline(gstar, color=PALETTE[3], ls=":", lw=0.8)
ax.axvline(float(d3.coexistence_gamma(1.0)), color="0.5", ls="--", lw=0.6)
ax.text(gstar + 0.005, 0.9, r"$\gamma^*$", fontsize=8); ax.text(float(d3.coexistence_gamma(1.0)) - 0.035, 0.9, r"$\gamma_c$", fontsize=8)
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("learned product"); ax.set_title(r"from $(1,1,1)$: a jump at depth 3")
ax.legend(frameon=False, fontsize=7)
ax = axs[1]
for i, depth in enumerate((2, 3)):
    up, down = sweeps[depth]
    col = PALETTE[0] if depth == 2 else PALETTE[3]
    ax.plot(gsweep, up, "-", color=col, lw=2.6, alpha=0.45, label=rf"depth {depth}, $\gamma$ up")
    ax.plot(gsweep, down, "--", color=col, lw=1.0, label=rf"depth {depth}, $\gamma$ down")
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("product"); ax.set_title("adiabatic sweep: hysteresis")
ax.legend(frameon=False, fontsize=7)
ax = axs[2]
ss = np.logspace(np.log10(0.2), np.log10(2.5), 50)
ax.loglog(ss, [float(d3.critical_gamma(s)) for s in ss], "k-", lw=0.8, label="your $\\gamma^*(s)$")
ax.loglog(svals, gs_meas, "o", ms=5, color=PALETTE[3], label=f"bisection (slope {slopeC:.3f})")
ax.loglog(ss, ss / 2, color=PALETTE[0], ls="--", lw=0.8, label=r"depth 2: $s/2$")
ax.set_xticks([0.25, 0.5, 1, 2]); ax.set_xticklabels(["0.25", "0.5", "1", "2"]); ax.minorticks_off()
ax.set_xlabel("target $s$"); ax.set_ylabel(r"critical $\gamma$"); ax.set_title("weak signals collapse first")
ax.legend(frameon=False, fontsize=7)
savefig(fig, args.out / "depth3.png")

finish(args, "depth3", {
    "hessian_origin": hessA, "gammas": gammas, "product_depth3_flow": p3, "product_depth2_flow": p2,
    "product_depth3_max_error": errB, "s_values": svals, "gamma_star_measured": gs_meas,
    "gamma_star_yours": gs_yours, "product_before_jump": p_jump, "gamma_star_slope_in_s": slopeC,
    "sweep_gammas": gsweep, "sweep_depth2_up": sweeps[2][0], "sweep_depth2_down": sweeps[2][1],
    "sweep_depth3_up": sweeps[3][0], "sweep_depth3_down": sweeps[3][1],
})
