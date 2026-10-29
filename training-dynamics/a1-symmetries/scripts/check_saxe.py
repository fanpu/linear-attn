"""Check for Problem (saxe): a two-layer linear net learns its singular modes one at a time.  [AI]

Uses YOUR aligned_init, saxe_prediction and effective_u0 (symmetries/saxe.py). Whitened inputs,
d_in = 8, d_out = 6, hidden width 6, Σyx = U S Vᵀ with singular values 4, 2, 1, 0.5 (rank 4).
Run 1: gradient flow of the full matrix ODE from your aligned balanced init with u0 = 1e-6 per
mode, against your prediction (the mode strengths u_α(t) = u_αᵀ W(t) v_α and |W - W_pred|).
Run 2: the same flow from a SMALL RANDOM init (entries N(0, 1e-6)), against your prediction
with u0 replaced by your effective_u0. The half-times t_α at which u_α first reaches s_α/2 are
compared; the remaining delay of the weaker modes is the competition between modes that the
aligned init removes (Saxe et al. 2014, Fig. 3 right).

    uv run python a1-symmetries/scripts/check_saxe.py [--out DIR] [--solutions]
Writes saxe.png and the "saxe" section of results.json.
"""
import numpy as np

from _common import finish, mine, parse

from tdyn.integrate import solve_flow
from tdyn.plotting import PALETTE, savefig

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
sx = mine("saxe")

rng = np.random.default_rng(0)
d_in, d_out, hidden = 8, 6, 6
svals = np.array([4.0, 2.0, 1.0, 0.5])
U, _ = np.linalg.qr(rng.standard_normal((d_out, d_out)))
V, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
U, V = U[:, :4], V[:, :4]
Syx = U @ np.diag(svals) @ V.T
n1 = hidden * d_in


def field(th):
    W1, W2 = th[:n1].reshape(hidden, d_in), th[n1:].reshape(d_out, hidden)
    G = Syx - W2 @ W1
    return np.concatenate([(W2.T @ G).ravel(), (G @ W1.T).ravel()])


def product(th):
    return th[n1:].reshape(d_out, hidden) @ th[:n1].reshape(hidden, d_in)


def strengths(W):
    return np.einsum("ia,ij,ja->a", U, W, V)       # u_αᵀ W v_α


def half_times(t, u):
    return np.array([np.interp(s / 2, np.maximum.accumulate(u[:, a]), t) for a, s in enumerate(svals)])


T, n = 24.0, 4801
u0 = 1e-6
W1, W2 = (np.asarray(w, dtype=float) for w in sx.aligned_init(Syx, u0, hidden))
tr = solve_flow(field, np.concatenate([W1.ravel(), W2.ravel()]), T, n_points=n)
W_sim = np.array([product(th) for th in tr.y])
W_pred = np.array([np.asarray(sx.saxe_prediction(Syx, u0, t), dtype=float) for t in tr.t])
aligned_err = float(np.max(np.abs(W_sim - W_pred)))
u_sim_al = np.array([strengths(W) for W in W_sim])
th_al = half_times(tr.t, u_sim_al)
th_formula = np.log((svals - u0) / u0) / (2 * svals)
print(f"aligned init: max |W_flow(t) - W_yours(t)| on [0, {T:g}] = {aligned_err:.2e}")

W1r = 1e-3 * rng.standard_normal((hidden, d_in))
W2r = 1e-3 * rng.standard_normal((d_out, hidden))
u_eff = np.asarray(sx.effective_u0(W1r, W2r, Syx), dtype=float)
trr = solve_flow(field, np.concatenate([W1r.ravel(), W2r.ravel()]), T, n_points=n)
u_sim_r = np.array([strengths(product(th)) for th in trr.y])
u_pred_r = np.array([strengths(np.asarray(sx.saxe_prediction(Syx, u_eff, t), dtype=float)) for t in trr.t])
th_r = half_times(trr.t, u_sim_r)
th_r_pred = np.log((svals - u_eff) / u_eff) / (2 * svals)
delay = (th_r - th_r_pred) / th_r_pred
print("mode  s_a   t_half aligned (flow / formula)   random init: u0_eff    t_half flow / yours   rel. delay   s_a * t_half")
for a, s in enumerate(svals):
    print(f"{a:<5} {s:<5g} {th_al[a]:8.4f} / {th_formula[a]:8.4f}          {u_eff[a]:.3e}   {th_r[a]:8.4f} / {th_r_pred[a]:8.4f}   "
          f"{delay[a]:+.3f}       {s * th_r[a]:.3f}")
loss_r = 0.5 * np.array([np.sum((Syx - product(th)) ** 2) for th in trr.y])

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), layout="constrained")
ax = axs[0]
for a, s in enumerate(svals):
    ax.plot(trr.t, u_sim_r[:, a], color=PALETTE[a], lw=2.4, alpha=0.5, label=f"$s_{a + 1}={s:g}$, random init")
    ax.plot(trr.t, u_pred_r[:, a], "k--", lw=0.7)
ax.set_xlabel("time $t$"); ax.set_ylabel(r"mode strength $u_\alpha(t)=u_\alpha^\top W(t)\,v_\alpha$")
ax.set_title("flow (colour) vs yours with $u_0^{\\rm eff}$ (dashed)")
ax.legend(frameon=False, fontsize=7, loc="center right")
ax = axs[1]
ax.semilogy(trr.t, loss_r, color=PALETTE[3]); ax.set_ylim(1e-3, 30)
for a in range(4):
    ax.axvline(th_r_pred[a], color=PALETTE[a], lw=0.7, ls=":")
ax.set_xlabel("time $t$"); ax.set_ylabel(r"$\frac{1}{2}\|\Sigma_{yx}-W(t)\|_F^2$")
ax.set_title("loss: plateaus and drops (dotted: your half-times)")
savefig(fig, args.out / "saxe.png")

finish(args, "saxe", {
    "singular_values": svals, "aligned_max_abs_error": aligned_err, "aligned_half_times_flow": th_al,
    "half_time_formula_u0_1e-6": th_formula, "random_u0_eff": u_eff, "random_half_times_flow": th_r,
    "random_half_times_yours": th_r_pred, "random_relative_delay": delay, "s_times_half_time_random": svals * th_r,
})
