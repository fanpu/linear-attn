"""Figures for the math primer (m0-math-primer/handout.typ). CPU, about a second.

    uv run python m0-math-primer/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"


def bifurcations(ax_pf, ax_sn):
    mu = np.linspace(-1, 1, 401)
    # pitchfork  x' = mu x - x^3
    ax_pf.plot(mu[mu <= 0], 0 * mu[mu <= 0], "k-", lw=2, label="stable")
    ax_pf.plot(mu[mu >= 0], 0 * mu[mu >= 0], "k--", lw=1.5, label="unstable")
    mp = mu[mu >= 0]
    ax_pf.plot(mp, np.sqrt(mp), "k-", lw=2)
    ax_pf.plot(mp, -np.sqrt(mp), "k-", lw=2)
    ax_pf.set_title(r"pitchfork: $\dot x=\mu x-x^3$ (continuous)", fontsize=9)
    ax_pf.set_xlabel(r"control parameter $\mu$")
    ax_pf.set_ylabel(r"fixed points $x^*$")
    ax_pf.legend(fontsize=8, loc="upper left")
    # saddle-node  x' = mu - x^2
    mp = mu[mu >= 0]
    ax_sn.plot(mp, np.sqrt(mp), "k-", lw=2, label="stable")
    ax_sn.plot(mp, -np.sqrt(mp), "k--", lw=1.5, label="unstable")
    ax_sn.annotate("", xy=(0, -0.9), xytext=(0.0, -0.05),
                   arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax_sn.text(0.04, -0.6, "below 0 no fixed point:\nthe state runs away", fontsize=8, color="tab:red")
    ax_sn.set_title(r"saddle-node: $\dot x=\mu-x^2$ (discontinuous)", fontsize=9)
    ax_sn.set_xlabel(r"control parameter $\mu$")
    ax_sn.set_xlim(-1, 1)
    ax_sn.set_ylim(-1.05, 1.05)
    ax_sn.legend(fontsize=8, loc="upper left")


def anchor_portrait(ax, s=1.0):
    g = np.linspace(-2.2, 2.2, 301)
    A, B = np.meshgrid(g, g)
    R = A * B - s
    U, V = -B * R, -A * R
    speed = np.sqrt(U**2 + V**2)
    ax.streamplot(A, B, U, V, density=1.1, color=np.log10(speed + 1e-3), cmap="Greys",
                  linewidth=0.7, arrowsize=0.7)
    for c in (-2, -0.75, 0.75, 2):  # conserved curves a^2 - b^2 = c
        ax.contour(A, B, A**2 - B**2, levels=[c], colors="tab:blue", linestyles="dashed", linewidths=0.8)
    a = np.linspace(0.42, 2.2, 200)
    ax.plot(a, s / a, "k-", lw=2.2)
    ax.plot(-a, -s / a, "k-", lw=2.2, label=r"$ab=s$ (fixed points)")
    ax.plot(g, -g, color="tab:red", lw=1.5, label=r"$a=-b$ (into the saddle)")
    ax.plot([0], [0], "o", color="tab:red", ms=6)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_xlabel("$a$")
    ax.set_ylabel("$b$")
    ax.set_title(r"anchor flow, $L=\frac{1}{2}(ab-1)^2$; dashed: $a^2-b^2=$ const", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)


def main():
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    bifurcations(axes[0], axes[1])
    anchor_portrait(axes[2])
    fig.tight_layout()
    fig.savefig(OUT / "fixed_points.png", dpi=170)
    print(OUT / "fixed_points.png")


if __name__ == "__main__":
    main()
