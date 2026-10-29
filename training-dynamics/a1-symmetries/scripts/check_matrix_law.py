"""Check for Problem (matrix-law): D = W1 W1ᵀ - W2ᵀ W2 is conserved by the flow, not by GD.  [AI]

Uses YOUR two_layer_grads, imbalance and gd_imbalance_change (symmetries/two_layer.py). A
two-layer linear net with d_in = 4, hidden m = 3, d_out = 2 on correlated (NOT whitened) inputs,
Σxx with eigenvalues 0.5 ... 2, random Σyx, initial weights N(0, 0.5²). Gradient flow by RK4 with
step 1e-3 on t in [0, 10]; GD with η = 0.2, 0.1, 0.05, 0.02, 0.01, 0.005 for the same time 10.
It reports the Frobenius drift |D(t) - D(0)|_F, fits its log-log slope against η at t = 10, and
compares every GD step's change of D with your exact formula.

    uv run python a1-symmetries/scripts/check_matrix_law.py [--out DIR] [--solutions]
Writes matrix_law.png and the "matrix_law" section of results.json.
"""
import numpy as np

from _common import finish, mine, parse

from tdyn.integrate import rk4
from tdyn.plotting import PALETTE, savefig

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
tl = mine("two_layer")

rng = np.random.default_rng(0)
d_in, m, d_out = 4, 3, 2
Q, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
Sxx = Q @ np.diag(np.linspace(0.5, 2.0, d_in)) @ Q.T
Syx = rng.standard_normal((d_out, d_in))
W1_0 = 0.5 * rng.standard_normal((m, d_in))
W2_0 = 0.5 * rng.standard_normal((d_out, m))
n1 = W1_0.size
unpack = lambda th: (th[:n1].reshape(m, d_in), th[n1:].reshape(d_out, m))
D0 = np.asarray(tl.imbalance(W1_0, W2_0))


def flow(th):
    g1, g2 = tl.two_layer_grads(*unpack(th), Sxx, Syx)
    return -np.concatenate([np.ravel(g1), np.ravel(g2)])


T, dt = 10.0, 1e-3
traj = rk4(flow, np.concatenate([W1_0.ravel(), W2_0.ravel()]), dt, int(T / dt))
t_flow = np.arange(len(traj))[::50] * dt
d_flow = np.array([np.linalg.norm(np.asarray(tl.imbalance(*unpack(th))) - D0) for th in traj[::50]])
print(f"gradient flow (RK4, dt = {dt:g}): max |D(t) - D(0)|_F on [0, {T:g}] = {d_flow.max():.2e}")

etas = np.array([0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
drift_T, step_err, gd_curves = [], 0.0, {}
for eta in etas:
    W1, W2 = W1_0.copy(), W2_0.copy()
    ds = [0.0]
    for k in range(int(round(T / eta))):
        pred = np.asarray(tl.gd_imbalance_change(W1, W2, Sxx, Syx, eta))
        g1, g2 = tl.two_layer_grads(W1, W2, Sxx, Syx)
        D_before = W1 @ W1.T - W2.T @ W2
        W1, W2 = W1 - eta * np.asarray(g1), W2 - eta * np.asarray(g2)
        step_err = max(step_err, float(np.linalg.norm((W1 @ W1.T - W2.T @ W2) - D_before - pred)))
        ds.append(np.linalg.norm(W1 @ W1.T - W2.T @ W2 - D0))
    drift_T.append(ds[-1])
    gd_curves[eta] = (np.arange(len(ds)) * eta, np.array(ds))
    print(f"GD eta = {eta:<6g} |D(T) - D(0)|_F = {ds[-1]:.3e}")
drift_T = np.array(drift_T)
slope = float(np.polyfit(np.log(etas), np.log(drift_T), 1)[0])
print(f"log-log slope of the drift at t = {T:g} against eta: {slope:.3f}")
print(f"max over all GD steps of |actual change of D - your formula|_F: {step_err:.2e}")

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.1), layout="constrained")
ax = axs[0]
ax.semilogy(t_flow, np.maximum(d_flow, 1e-17), color="k", lw=1.2, label="gradient flow (RK4)")
for i, eta in enumerate(etas[::-1]):
    t, d = gd_curves[eta]
    ax.semilogy(t[1:], d[1:], color=PALETTE[i], label=rf"GD $\eta={eta:g}$")
ax.set_xlabel("time $t$ (GD: $k\\eta$)"); ax.set_ylabel(r"$\|D(t)-D(0)\|_F$"); ax.set_ylim(1e-16, 10)
ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="center right"); ax.set_title("drift of the conserved matrix")
ax = axs[1]
ax.loglog(etas, drift_T, "o", color=PALETTE[3], label=r"GD, $t=10$")
ax.loglog(etas, drift_T[-1] * etas / etas[-1], "k--", lw=0.7, label="slope 1")
ax.set_xlabel(r"step size $\eta$"); ax.set_ylabel(r"$\|D(10)-D(0)\|_F$")
ax.legend(frameon=False, fontsize=7); ax.set_title(f"fitted slope {slope:.2f}")
savefig(fig, args.out / "matrix_law.png")

finish(args, "matrix_law", {
    "flow_max_drift": float(d_flow.max()), "etas": etas, "gd_drift_at_T": drift_T, "loglog_slope": slope,
    "max_step_formula_error": step_err,
})
