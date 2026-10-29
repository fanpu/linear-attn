"""Check for Problem (endpoint): where gradient flow lands, predicted from c alone.  [AI]

Uses YOUR flow_endpoint (symmetries/endpoint.py). For targets s = 0.5, 1, 2 it draws 150
initializations uniformly from [-2, 2]² (skipping |a0 + b0| < 0.1, where the flow takes very long
to leave the saddle), integrates the gradient flow on ½(s - ab)² accurately to t = 60, and compares
the end point with your prediction. It also integrates from three starts on or near the line
a0 = -b0, where the time to leave the origin's neighbourhood grows like ln(1/|a0 + b0|).

    uv run python a1-symmetries/scripts/check_endpoint.py [--out DIR] [--solutions]
Writes endpoint.png and the "endpoint" section of results.json.
"""
import numpy as np

from _common import finish, mine, parse

from tdyn.integrate import solve_flow
from tdyn.plotting import PALETTE, phase_portrait, savefig

import matplotlib.pyplot as plt

args = parse(__doc__.splitlines()[0])
flow_endpoint = mine("endpoint").flow_endpoint


def field(s):
    def f(th):
        a, b = th[..., 0], th[..., 1]
        r = a * b - s
        return np.stack([-b * r, -a * r], axis=-1)
    return f


rng = np.random.default_rng(0)
errors, examples = {}, {}
for s in (0.5, 1.0, 2.0):
    starts = rng.uniform(-2, 2, size=(400, 2))
    starts = starts[np.abs(starts.sum(1)) >= 0.1][:150]
    ends = np.array([solve_flow(field(s), th0, 60.0, n_points=2).y[-1] for th0 in starts])
    pred = np.array([np.asarray(flow_endpoint(a0, b0, s), dtype=float) for a0, b0 in starts])
    errors[f"s={s:g}"] = float(np.max(np.abs(pred - ends)))
    examples[s] = (starts, ends, pred)
    print(f"s = {s:<4g} max |predicted - integrated endpoint| over {len(starts)} starts: {errors[f's={s:g}']:.2e}")

near = {}
for eps in (1e-2, 1e-4, 1e-6):
    a0, b0 = 0.7, -0.7 + eps
    tr = solve_flow(field(1.0), [a0, b0], 60.0, n_points=6001)
    loss = 0.5 * (tr.y[:, 0] * tr.y[:, 1] - 1) ** 2
    t_esc = float(tr.t[np.argmax(loss < 0.25)]) if np.any(loss < 0.25) else float("inf")
    end_err = float(np.max(np.abs(tr.y[-1] - np.asarray(flow_endpoint(a0, b0, 1.0)))))
    near[f"eps={eps:g}"] = {"time_to_half_loss": t_esc, "endpoint_error": end_err}
    print(f"start (0.7, -0.7 + {eps:g}): loss falls below 1/4 at t = {t_esc:.2f}; endpoint error {end_err:.1e}")
on_line = np.asarray(flow_endpoint(0.7, -0.7, 1.0), dtype=float)
print(f"start exactly on a0 = -b0: your endpoint {on_line}")

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.9), layout="constrained")
starts, ends, pred = examples[1.0]
cols = [PALETTE[0] if a + b > 0 else PALETTE[1] for a, b in starts]
trajs = [solve_flow(field(1.0), th0, 8.0, n_points=200).y for th0 in starts[:40]]
phase_portrait(field(1.0), trajs, ax=axs[0], traj_colors=cols[:40], xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
axs[0].plot(pred[:, 0], pred[:, 1], "kx", ms=5, mew=1.0, zorder=5, label="your predicted endpoints")
xs = np.linspace(0.38, 2.5, 100)
for sg in (1, -1):
    axs[0].plot(sg * xs, sg / xs, "k-", lw=0.8)
axs[0].plot([-2.5, 2.5], [2.5, -2.5], color=PALETTE[3], lw=0.8)
axs[0].set_title("$s=1$: 40 flows and your endpoints"); axs[0].legend(frameon=False, fontsize=7, loc="lower left")
ax = axs[1]
for i, s in enumerate((0.5, 1.0, 2.0)):
    st, en, pr = examples[s]
    ax.plot(en[:, 0], pr[:, 0], "o", ms=3, color=PALETTE[i], label=f"$a_\\infty$, $s={s:g}$")
lim = 3.2
ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.6)
ax.set_xlabel(r"$a_\infty$ from the integrated flow"); ax.set_ylabel(r"$a_\infty$ predicted")
ax.set_aspect("equal"); ax.legend(frameon=False, fontsize=7); ax.set_title("prediction vs integration")
savefig(fig, args.out / "endpoint.png")

finish(args, "endpoint", {"max_abs_error": errors, "near_saddle_line": near, "on_line_endpoint": on_line})
