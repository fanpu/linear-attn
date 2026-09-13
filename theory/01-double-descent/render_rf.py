"""Random-features regression vs Mei-Montanari: python render_rf.py"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

import style as S

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"


def psi_axis(ax):
    ax.set_xscale("log"); ax.set_xlim(0.1, 30)
    t = [0.1, 0.3, 1, 3, 10, 30]
    ax.xaxis.set_major_locator(FixedLocator(t)); ax.set_xticklabels([f"{v:g}" for v in t])
    ax.xaxis.set_minor_formatter(NullFormatter())


def main():
    z = np.load(HERE / "cache" / "rf.npz")
    psi1, th, lams, th_opt = z["th_psi1"], z["th"], z["th_lams"], z["th_opt"]
    psi2 = float(z["psi2"])
    S.use("light")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.9), sharey=True)

    # (a) ridgeless, three d
    ax = axs[0]
    m = psi1 < psi2
    ax.plot(psi1[m], th[0, m, 0], color=S.INK, lw=1.6, zorder=5)
    ax.plot(psi1[~m], th[0, ~m, 0], color=S.INK, lw=1.6, zorder=5)
    cols = S.ramp(S.INDIGO, 3, lo=0.35, hi=1.0)
    for col, d, mk in zip(cols, [50, 100, 200], ["o", "o", "o"]):
        Ns = z[f"sim_d{d}_N"]; R = z[f"sim_d{d}"][:, :, 0]
        mean = R.mean(1); se = R.std(1) / np.sqrt(R.shape[1])
        ax.errorbar(Ns / d, mean, yerr=2 * se, fmt=mk, ms=4.3, color=col, mec=S.PAPER, mew=0.8, elinewidth=1, capsize=0,
                    zorder=4, label=f"d = {d}, n = {int(psi2 * d)}")
    ax.axvline(psi2, color=S.GOLD, lw=1.1, zorder=0)
    ax.text(psi2 * 1.07, 0.06, "N = n", fontsize=10, color=S.INK, va="bottom")
    ax.axhline(1.0, color=S.MUTED, lw=1, ls=(0, (1, 3)))
    ax.text(0.11, 1.03, "null risk", fontsize=9.3, color=S.MUTED, va="bottom")
    psi_axis(ax); ax.set_ylim(0, 3.0)
    ax.set_xlabel(r"$\psi_1 = N/d$  (random features per input dimension)")
    ax.set_ylabel("test error (excluding noise)")
    ax.set_title("Ridgeless (λ → 0)", pad=8)
    ax.legend(loc="upper right", fontsize=9.3, handletextpad=0.3, borderaxespad=0.2)
    ax.text(0.12, 2.6, "line: Mei–Montanari, d → ∞", fontsize=9.5, color=S.INK)

    # (b) ridge family + optimal, d = 200
    ax = axs[1]
    d = 200
    Ns = z[f"sim_d{d}_N"]; R = z[f"sim_d{d}"]
    cols = S.ramp(S.TEAL, 3, lo=0.4, hi=1.0)
    ax.plot(psi1[m], th[0, m, 0], color=S.MUTED, lw=1, ls=(0, (3, 2)))
    ax.plot(psi1[~m], th[0, ~m, 0], color=S.MUTED, lw=1, ls=(0, (3, 2)))
    pos = {1: (2.0, (-12, 10)), 2: (1.6, (40, 14)), 3: (20, (10, 10))}
    for k, col in zip([1, 2, 3], cols):
        ax.plot(psi1, th[k, :, 0], color=col, lw=1.8)
        ax.plot(Ns / d, R[:, :, k].mean(1), "o", ms=4, color=col, mec=S.PAPER, mew=0.7)
        gx, off = pos[k]; j = np.argmin(np.abs(psi1 - gx))
        ax.annotate(f"λ = {lams[k]:g}", xy=(psi1[j], th[k, j, 0]), xytext=off, textcoords="offset points",
                    fontsize=9.5, color=S.INK, ha="right")
    ax.plot(psi1, th_opt[:, 0], color=S.INK, lw=2.6)
    ax.plot(Ns / d, R[:, :, 4].mean(1), "o", ms=4.6, color=S.INK, mec=S.PAPER, mew=0.9)
    ax.text(0.45, 0.62, "optimal λ", fontsize=10.5, fontweight="bold", ha="center", va="top")
    ax.axvline(psi2, color=S.GOLD, lw=1.1, zorder=0)
    psi_axis(ax)
    ax.set_xlabel(r"$\psi_1 = N/d$")
    ax.set_title("With ridge (d = 200, n = 600)", pad=8)
    fig.text(0.07, 1.0, "Random ReLU features on the sphere: the same double descent, the same cure",
             fontsize=12.5, fontweight="bold", ha="left")
    fig.text(0.07, 0.955, f"linear target $\\|\\beta\\|^2=1$, noise τ² = {float(z['tau2']):g}, n/d = {psi2:g}; dots: mean of 20 draws (±2 s.e.); lines: asymptotic formula",
             fontsize=9.5, color=S.MUTED, ha="left")
    S.save(fig, FIG / "rf_mm.png")

    # numbers
    for d in [50, 100, 200]:
        Ns = z[f"sim_d{d}_N"]; Rr = z[f"sim_d{d}"]
        far = np.abs(Ns / d / psi2 - 1) > 0.3
        thN = np.interp(np.log(Ns / d), np.log(psi1), th[0, :, 0])
        rel = np.abs(Rr[far, :, 0].mean(1) - thN[far]) / thN[far]
        thO = np.interp(np.log(Ns / d), np.log(psi1), th_opt[:, 0])
        relo = np.abs(Rr[:, :, 4].mean(1) - thO) / thO
        near = np.argmin(np.abs(Ns / d - psi2 * 0.95))
        print(f"d={d}: ridgeless rel err (|psi1/psi2-1|>0.3) median {np.median(rel):.3f} max {rel.max():.3f};"
              f" optimal-ridge rel err median {np.median(relo):.3f} max {relo.max():.3f}; at N=0.95n sim {Rr[near,:,0].mean():.2f} theory {thN[near]:.2f}")
    print("optimal ridge theory monotone decreasing:", bool(np.all(np.diff(th_opt[:, 0]) < 1e-9)))


if __name__ == "__main__":
    main()
