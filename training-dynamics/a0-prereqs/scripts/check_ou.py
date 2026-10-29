"""Check for Problem (ou): the Ornstein-Uhlenbeck process by Euler-Maruyama.  [AI]

Uses YOUR ou_simulate (prereqs/sde.py). Part 1: 5000 paths of dX = -θX dt + σ dW with θ = 1,
σ = 0.5, X_0 = 2, Δt = 0.01 up to t = 8; compares the sample mean and variance over time and the
histogram at t = 8 with your formulas (typed into the FORMULAS below by you; the reference
solution's formulas are used under --solutions). Part 2: the stationary variance of Euler-Maruyama
for growing step Δt, against the continuous-time value; this is SGD near a minimum (θ = curvature,
Δt = step size), the picture A4 builds on; compared with your discrete-time formula (part e).

    uv run python a0-prereqs/scripts/check_ou.py [--out DIR] [--solutions]
Writes ou.png and the "ou" section of results.json.
"""
import numpy as np

from _common import finish, mine, parse

from tdyn.plotting import PALETTE, savefig
from tdyn.swap import use_solutions

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
ou_simulate = mine("sde").ou_simulate


# ---- FORMULAS: replace the three bodies with your Derive deliverables (Problem ou) --------------
def mean_formula(x0, theta, sigma, t):
    raise NotImplementedError("type your E[X_t] here")


def var_formula(x0, theta, sigma, t):
    raise NotImplementedError("type your Var[X_t] here")


def stationary_var_formula(theta, sigma):
    raise NotImplementedError("type your stationary variance here")


def em_stationary_var_formula(theta, sigma, dt):
    raise NotImplementedError("type your Euler-Maruyama stationary variance here (part e)")
# -------------------------------------------------------------------------------------------------


if use_solutions():
    def mean_formula(x0, theta, sigma, t):
        return x0 * np.exp(-theta * t)

    def var_formula(x0, theta, sigma, t):
        return sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * t))

    def stationary_var_formula(theta, sigma):
        return sigma**2 / (2 * theta)

    def em_stationary_var_formula(theta, sigma, dt):
        return sigma**2 / (theta * (2 - theta * dt))

x0, theta, sigma, dt, steps, n = 2.0, 1.0, 0.5, 0.01, 800, 5000
X = np.asarray(ou_simulate(x0, theta, sigma, dt, steps, n, np.random.default_rng(0)))
t = np.arange(steps + 1) * dt
m_hat, v_hat = X.mean(axis=1), X.var(axis=1)
m_th, v_th = mean_formula(x0, theta, sigma, t), var_formula(x0, theta, sigma, t)
v_inf = stationary_var_formula(theta, sigma)
i1 = int(round(1.0 / dt))
print(f"t=1: mean {m_hat[i1]:.4f} (formula {m_th[i1]:.4f}), variance {v_hat[i1]:.4f} (formula {v_th[i1]:.4f})")
print(f"t=8: variance {v_hat[-1]:.4f}; stationary formula {v_inf:.4f}; "
      f"sampling std of the estimate {v_inf * np.sqrt(2 / n):.4f}")

# Part 2: stationary variance of Euler-Maruyama vs step size -------------------------------------
dts = np.array([0.01, 0.1, 0.3, 0.6, 1.0, 1.4, 1.7, 1.9])
v_em = []
for h in dts:
    burn = int(np.ceil(20.0 / (theta * h)))                      # many relaxation times
    Y = np.asarray(ou_simulate(0.0, theta, sigma, h, burn + 200, 4000, np.random.default_rng(1)))
    v_em.append(Y[burn:].var())                                  # pooled over the last 200 steps
v_em = np.array(v_em)
v_em_th = np.array([em_stationary_var_formula(theta, sigma, h) for h in dts])
for h, v, vt in zip(dts, v_em, v_em_th):
    print(f"dt={h:<5} theta*dt={theta * h:<4} stationary variance {v:.4f} "
          f"(your discrete formula {vt:.4f}, continuous time {v_inf:.4f})")

fig, axs = plt.subplots(1, 3, figsize=(8.8, 2.9), layout="constrained")
ax = axs[0]
ax.plot(t, m_hat, color=PALETTE[0], lw=2.2, alpha=0.5, label="sample mean")
ax.plot(t, m_th, "k--", lw=0.8, label="formula")
ax.plot(t, v_hat, color=PALETTE[1], lw=2.2, alpha=0.6, label="sample variance")
ax.plot(t, v_th, "k:", lw=0.9, label="formula")
for k in range(5):
    ax.plot(t, X[:, k], color="0.6", lw=0.4, zorder=0)
ax.set_xlabel("time $t$"); ax.set_ylim(-1.2, 2.1); ax.legend(frameon=False, fontsize=6.5)
ax.set_title("mean and variance (5 paths in grey)")
ax = axs[1]
ax.hist(X[-1], bins=60, density=True, color=PALETTE[0], alpha=0.5, label="paths at $t=8$")
xs = np.linspace(-1.5, 1.5, 300)
ax.plot(xs, np.exp(-xs**2 / (2 * v_inf)) / np.sqrt(2 * np.pi * v_inf), "k", lw=1.0,
        label=r"$\mathcal{N}(0, v_\infty)$")
ax.set_xlabel("$X_8$"); ax.set_ylim(0, 1.5)
ax.legend(frameon=False, fontsize=6.5, loc="upper left"); ax.set_title("histogram vs Gaussian")
ax = axs[2]
ax.semilogy(theta * dts, v_em, "o", color=PALETTE[3], ms=4, label="Euler–Maruyama, measured")
ax.axhline(v_inf, color="k", lw=0.8, ls="--", label=r"continuous time $v_\infty$")
hh = np.linspace(0.005, 1.95, 300)
ax.semilogy(theta * hh, [em_stationary_var_formula(theta, sigma, h) for h in hh], color="k", lw=0.8,
            label="your discrete-time formula")
ax.axvline(2.0, color="0.5", lw=0.6)
ax.text(1.97, v_inf * 1.3, r"$\theta\Delta t=2$", ha="right", fontsize=7)
ax.set_xlabel(r"$\theta\,\Delta t$"); ax.set_ylabel("stationary variance")
ax.set_xlim(0, 2.1); ax.set_ylim(0.08, 8); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.set_yticks([0.1, 0.3, 1, 3]); ax.set_yticklabels(["0.1", "0.3", "1", "3"])
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_title("larger steps, larger spread")
savefig(fig, args.out / "ou.png")

finish(args, "ou", {
    "mean_t1": m_hat[i1], "mean_t1_formula": m_th[i1], "var_t1": v_hat[i1], "var_t1_formula": v_th[i1],
    "var_t8": v_hat[-1], "stationary_var_formula": v_inf,
    "em_dts": dts, "em_stationary_var": v_em, "em_stationary_var_formula": v_em_th,
})
