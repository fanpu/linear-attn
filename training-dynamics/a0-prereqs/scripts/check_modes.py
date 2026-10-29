"""Check for Problem (modes): linear regression relaxes mode by mode.  [AI]

Uses YOUR mode_dynamics (prereqs/modes.py). A 5-input, 3-output regression whose input covariance
Σ_xx has eigenvalues 0.1, 0.316, 1, 3.16, 10 (two decades) in a random orthonormal basis U. The
per-mode error e_i(t) = |(W(t) - W*) u_i| is computed from your closed form and from an RK4
integration of dW/dt = Σ_yx - W Σ_xx; the decay rate of each mode is fitted from the log of e_i(t)
and compared with the eigenvalue λ_i. This is the one-layer case of Saxe et al. (2014), §1.

    uv run python a0-prereqs/scripts/check_modes.py [--out DIR] [--solutions]
Writes modes.png and the "modes" section of results.json.
"""
import numpy as np

from _common import finish, mine, parse

from tdyn.integrate import rk4
from tdyn.plotting import PALETTE, savefig

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
mode_dynamics = mine("modes").mode_dynamics

rng = np.random.default_rng(0)
d_in, d_out = 5, 3
U, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
lam = np.logspace(-1, 1, d_in)
Sxx = U @ np.diag(lam) @ U.T
W_true = rng.standard_normal((d_out, d_in))
Syx = W_true @ Sxx                     # population statistics of y = W_true x (noise-free)
W0 = 0.1 * rng.standard_normal((d_out, d_in))
W_star = Syx @ np.linalg.inv(Sxx)

T, dt = 40.0, 0.005
n = int(T / dt)
f = lambda w: (Syx - w.reshape(d_out, d_in) @ Sxx).ravel()
W_rk4 = rk4(f, W0.ravel(), dt, n).reshape(n + 1, d_out, d_in)
ts = np.arange(n + 1) * dt
every = 20
ts_s = ts[::every]
W_cf = np.array([mode_dynamics(W0, Sxx, Syx, t) for t in ts_s])
max_dev = float(np.max(np.abs(W_cf - W_rk4[::every])))
err_cf = np.linalg.norm((W_cf - W_star) @ U, axis=1)        # (len(ts_s), d_in): e_i(t)
rates = []
for i in range(d_in):
    ok = err_cf[:, i] > 1e-10 * err_cf[0, i]                 # stay above round-off
    rates.append(-np.polyfit(ts_s[ok], np.log(err_cf[ok, i]), 1)[0])
rates = np.array(rates)
print(f"max |closed form - RK4| over t in [0, {T:g}]: {max_dev:.2e}")
print("mode  eigenvalue  fitted decay rate")
for i in range(d_in):
    print(f"{i:<5} {lam[i]:<11.4f} {rates[i]:.4f}")

fig, ax = plt.subplots(figsize=(5.0, 3.3))
for i in range(d_in):
    ax.semilogy(ts_s, err_cf[:, i], color=PALETTE[i], lw=2.0, alpha=0.55,
                label=rf"mode {i + 1}: $\lambda={lam[i]:.3g}$, fit {rates[i]:.3g}")
    ax.semilogy(ts_s, err_cf[0, i] * np.exp(-lam[i] * ts_s), "--", color="k", lw=0.6)
ax.set_ylim(1e-12, 10); ax.set_xlim(0, T)
ax.set_xlabel("time $t$"); ax.set_ylabel(r"$\|(W(t)-W^\star)\,u_i\|$")
ax.set_title(r"per-mode error; dashed: $e_i(0)\,e^{-\lambda_i t}$")
ax.legend(frameon=False, fontsize=7, loc="lower right")
savefig(fig, args.out / "modes.png")

finish(args, "modes", {"eigenvalues": lam, "fitted_rates": rates,
                       "max_rel_rate_error": float(np.max(np.abs(rates - lam) / lam)),
                       "max_abs_dev_closed_form_vs_rk4": max_dev})
