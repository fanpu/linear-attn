"""Static figures for the NTK width sweep (reads cache/ntk_widths_summary.npz)."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import style as S
from analyze_ntk import slope

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = f"{HERE}/figures"
os.makedirs(FIG, exist_ok=True)
d = np.load(f"{HERE}/cache/ntk_widths_summary.npz")
M = d["m"]
WS = np.unique(M)
RAMP = LinearSegmentedColormap.from_list("w", ["#c7b5ec", "#8561cf", "#361a7d", "#1a0b45"])


def loglog(ax):
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(WS[::2]); ax.set_xticklabels([f"{w:,}" for w in WS[::2]])
    ax.minorticks_off()
    ax.set_xlabel("hidden width  m")


def series(ax, key, color, label, lab_xy=None, marker="o", guide=None, guide_label=None, fit_from=512):
    y = d[key]
    ax.scatter(M, y, s=14, color=color, alpha=0.35, lw=0, zorder=3)
    means = np.array([np.exp(np.log(y[M == w]).mean()) for w in WS])
    ax.plot(WS, means, marker, color=color, ms=6.5, mec=S.PAPER, mew=1.2, lw=0, zorder=4)
    sel = M >= fit_from
    s, se, b = slope(M[sel], y[sel])
    if guide is not None:  # theory guide anchored at the widest mean
        xs = np.array([WS[0] / 1.4, WS[-1] * 1.4])
        ax.plot(xs, means[-1] * (xs / WS[-1]) ** guide, color=S.INK, lw=1.0, ls=(0, (4, 3)), zorder=2, alpha=.8)
    if lab_xy is not None:
        ax.annotate(f"{label}\nslope {s:+.2f} ± {se:.2f}", lab_xy, color=color, fontsize=10, fontweight="bold",
                    ha="left", va="center", linespacing=1.3)
    return s, se


S.use("light")
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(wspace=0.32))

ax = axs[0]
series(ax, "da", S.RICH, "readout weights a", (1500, 0.75), guide=-0.5)
series(ax, "dW", S.LAZY, "first-layer weights W", (70, 0.03), guide=-0.5)
loglog(ax)
ax.set_ylabel(r"relative change  $\|\theta_T-\theta_0\|\,/\,\|\theta_0\|$")
ax.set_title("Weights barely move")
ax.text(20000, 0.09, r"$\propto m^{-1/2}$", fontsize=10.5, color=S.INK)
ax.set_ylim(0.008, 4)

ax = axs[1]
series(ax, "dKinit", S.MUTED, "random init vs. $m=\\infty$ kernel", (70, 0.025), guide=-0.5, marker="s")
series(ax, "dK", S.LAZY, "change during training", (1000, 1.4), guide=-1.0)
loglog(ax)
ax.set_ylabel(r"$\|\Theta_T-\Theta_0\|_F\,/\,\|\Theta_0\|_F$")
ax.set_title("The tangent kernel freezes")
ax.text(420, 3.0, r"$\propto m^{-1}$", fontsize=10.5, color=S.INK)
ax.text(20000, 0.012, r"$\propto m^{-1/2}$", fontsize=10.5, color=S.INK)

ax = axs[2]
series(ax, "dfun", S.LAZY, "network vs. its linearization\n(test-set outputs)", (70, 0.035), guide=-0.5)
loglog(ax)
ax.set_ylabel(r"$\|f_T-f^{\,\mathrm{lin}}_T\|\,/\,\|f^{\,\mathrm{lin}}_T-f_0\|$")
ax.set_title("…so the network is its own linear model")
ax.text(20000, 0.013, r"$\propto m^{-1/2}$", fontsize=10.5, color=S.INK)
S.save(fig, f"{FIG}/ntk_width_scaling.png")

# ---------------------------------------------------------------- kernel movement over training + scatter
fig = plt.figure(figsize=(15, 4.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1, 1, 0.9], wspace=0.35)
ax = fig.add_subplot(gs[0])
steps = d["steps"]
for i, w in enumerate(WS):
    curves = d["ksub_rel"][M == w]
    c = RAMP(i / (len(WS) - 1))
    ax.plot(np.maximum(steps, 1), np.exp(np.log(np.maximum(curves, 1e-6)).mean(0)), color=c, lw=1.8)
    ax.text(3300, np.exp(np.log(np.maximum(curves[:, -1], 1e-6)).mean()), f"{w:,}", color=c, fontsize=8.5, va="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1, 5200); ax.set_ylim(1e-4, 3)
ax.set_xlabel("gradient-descent step"); ax.set_ylabel(r"$\|\Theta_t-\Theta_0\|_F / \|\Theta_0\|_F$")
ax.set_title("Kernel drift during training, by width")
ax.text(1.3, 1.5, "narrow nets keep reshaping their kernel;\nwide ones lock it in after the first steps",
        fontsize=9.5, color=S.MUTED, va="top")

yte = d["yte"]
for j, w in enumerate([64, 16384]):
    ax = fig.add_subplot(gs[1 + j])
    k = np.where(M == w)[0][0]
    fl, fn = d["flin_test"][k], d["fnet_test"][k]
    ax.axline((0, 0), slope=1, color=S.RULE, lw=1.2, zorder=1)
    ax.scatter(fl, fn, s=5, lw=0, alpha=.55, color=np.where(yte > 0, S.RICH, S.LAZY), zorder=2)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.set_xlabel("linearized model output"); ax.set_ylabel("network output" if j == 0 else "")
    ax.set_title(f"width {w:,}")
    r = np.corrcoef(fl, fn)[0, 1]
    ax.text(-2.0, 1.9, f"relative gap {d['dfun'][k]:.2f}", fontsize=9, color=S.MUTED, va="top")

ax = fig.add_subplot(gs[3])
en = np.array([d["err_net"][M == w].mean() for w in WS]); el = np.array([d["err_lin"][M == w].mean() for w in WS])
ax.plot(WS, 100 * en, "o-", color=S.LAZY, ms=5, lw=1.6)
ax.plot(WS, 100 * el, "s--", color=S.MUTED, ms=4.5, lw=1.2)
ax.axhline(100 * float(d["err_inf"]), color=S.INK, lw=1, ls=(0, (4, 3)))
ax.text(20000, 100 * float(d["err_inf"]), "  $m=\\infty$ NTK\n  regression", fontsize=8.5, va="center", color=S.INK)
ax.text(20000, 100 * en[-1] - 0.15, "  network", color=S.LAZY, fontsize=9, fontweight="bold", va="center")
ax.text(20000, 100 * el[-1] + 0.25, "  linearized", color=S.MUTED, fontsize=9, fontweight="bold", va="center")
ax.set_xscale("log", base=2); ax.minorticks_off()
ax.set_xticks([64, 1024, 16384]); ax.set_xticklabels(["64", "1,024", "16,384"])
ax.set_xlabel("width m"); ax.set_ylabel("test error (%)")
ax.set_title("Same errors, too")
S.save(fig, f"{FIG}/ntk_linearization.png")
