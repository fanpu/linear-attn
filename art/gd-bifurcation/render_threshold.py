"""Which rule gives the first period doubling?  Plate from cache/threshold_k{2,4}.npz.

Left: final sharpness of the point GD reaches (converged runs, T = 100000) vs eta, 49 inits, with the curve
2/eta and the floor s_min = k.  Right: measured onset (first eta where GD from x0 is not converged after T
steps) / (2/k) for T = 1e2 ... 1e5, one dot per init, against the candidate rules.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render_lib import CACHE, GAL

INK, RED, BLUE, PAPER = "#1b1b22", "#b3261e", "#1f5f99", "#f3eee2"


def main():
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5), dpi=220, facecolor=PAPER,
                            gridspec_kw=dict(width_ratios=[1.35, 1], hspace=0.35, wspace=0.22))
    for row, k in enumerate([2, 4]):
        d = np.load(f"{CACHE}/threshold_k{k}.npz")
        e = d["etas"]
        T = 100000
        conv = d[f"conv_T{T}"]
        sh = d[f"sharp_T{T}"]
        ax = axs[row, 0]
        ax.set_facecolor(PAPER)
        for i in range(conv.shape[0]):
            m = conv[i]
            ax.plot(e[m], sh[i][m], ".", ms=0.8, color=INK, alpha=0.5)
        ee = np.linspace(e[0], e[-1], 400)
        ax.plot(ee, 2 / ee, color=RED, lw=0.8, label=r"$2/\eta$ (edge of stability)")
        ax.axhline(k, color=BLUE, lw=0.8, label=r"$s_{\min}=k$ (balanced minimum)")
        ax.axvline(2 / k, color=BLUE, lw=0.6, ls="--")
        ax.set_ylim(k * 0.8, min(2 / e[0], 4 * k))
        ax.set_xlabel(r"step size $\eta$", family="serif")
        ax.set_ylabel("sharpness of the minimum reached", family="serif")
        ax.set_title(f"k = {k}: GD from 49 inits, converged after 10⁵ steps", family="serif", fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8)
        ax = axs[row, 1]
        ax.set_facecolor(PAPER)
        Ts = d["T_list"]
        for j, TT in enumerate(Ts):
            on = d[f"onset_T{TT}"] * k / 2
            ax.plot(np.full(len(on), j) + np.linspace(-0.25, 0.25, len(on)), on, ".", ms=3, color=INK)
        ax.axhline(1.0, color=BLUE, lw=0.9, label=r"$2/s_{\min}$")
        gf = d["rule_gf"] * k / 2
        ini = d["rule_init"] * k / 2
        xr = len(Ts) - 0.35
        ax.plot(np.full(len(gf), xr), gf, "_", ms=9, color=RED, label=r"$2/s_{GF}(x_0)$ per init (flow minimum)")
        ax.plot(np.full(len(ini), xr + 0.3), ini, "_", ms=9, color="#777", label=r"$2/\lambda_{\max}(x_0)$ per init")
        ax.set_xlim(-0.5, len(Ts) + 0.2)
        ax.set_xticks(range(len(Ts)))
        ax.set_xticklabels([f"T = 10^{int(np.log10(t))}" for t in Ts], family="serif", fontsize=8)
        ax.set_ylabel(r"measured onset  $\eta_1 / (2/k)$", family="serif")
        ax.set_ylim(0.3 if k == 4 else 0.7, 1.75 if k == 4 else 1.12)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        med = [np.nanmedian(d[f"onset_T{TT}"]) * k / 2 for TT in Ts]
        ax.set_title("finite-time onset is init-dependent; it converges to 2/s_min\nmedian: " + ", ".join(f"{m:.4f}" for m in med),
                     family="serif", fontsize=9, loc="left")
    fig.suptitle("The first doubling is at η = 2/s_min, where s_min = k is the sharpness of the balanced (flattest) global minimum of ½(x₁x₂…x_k − 1)²",
                 family="serif", fontsize=11, color=INK, x=0.06, ha="left")
    fig.savefig(f"{GAL}/plate_threshold_rule.png", facecolor=PAPER)
    print("wrote plate_threshold_rule.png")


if __name__ == "__main__":
    main()
