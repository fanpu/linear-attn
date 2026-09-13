"""Frozen random features vs. a fully trained two-layer ReLU net: python render_twolayer.py"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

import style as S

HERE = pathlib.Path(__file__).resolve().parent
RF_C, FULL_C = "#5b4bc4", "#e0663a"      # validated pair (all-pairs CVD dE 27)


def main():
    z = np.load(HERE / "cache" / "twolayer.npz")
    Ns, res, d, n = z["widths"], z["res"], int(z["D"]), int(z["n"])
    S.use("light")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[2.1, 1], hspace=0.07))
    psi1, th = z["th_psi1"], z["th"]
    m = psi1 < n / d
    a1.plot(psi1[m] * d, th[m], color=RF_C, lw=1.4, alpha=0.9); a1.plot(psi1[~m] * d, th[~m], color=RF_C, lw=1.4, alpha=0.9)
    for k, col, lab in [(0, RF_C, "first layer frozen (random features, min-norm)"), (1, FULL_C, "both layers trained (Adam, no regularization)")]:
        med = np.median(res[:, :, k], 1)
        lo, hi = res[:, :, k].min(1), res[:, :, k].max(1)
        a1.vlines(Ns, lo, hi, color=col, lw=1.2, alpha=0.45)
        a1.plot(Ns, med, "o", color=col, ms=5, mec=S.PAPER, mew=0.8, label=lab, zorder=5)
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_ylim(0.02, 1000)
    for ax in (a1, a2):
        ax.axvline(n, color=S.GOLD, lw=1.2, zorder=0)
        ax.axvline(n / d, color=S.GOLD, lw=1.2, zorder=0, ls=(0, (4, 2)))
    a1.text(n * 1.08, 400, "N = n\n(features = samples)", fontsize=9.5, color=S.INK, va="top")
    a1.text(n / d * 1.08, 400, "N·d = n\n(trainable weights ≈ samples)", fontsize=9.5, color=S.INK, va="top")
    a1.axhline(1.0, color=S.MUTED, lw=1, ls=(0, (1, 3)))
    a1.text(2.1, 1.12, "null risk", fontsize=9, color=S.MUTED)
    a1.legend(loc="lower center", bbox_to_anchor=(0.6, 0.0), fontsize=9.5, handletextpad=0.2)
    a1.set_ylabel("test error (log)")
    a1.set_title("Train the first layer and the spike jumps to where the parameters, not the features, match the data", pad=24, fontsize=12)
    a1.text(0.0, 1.02, f"ReLU network, d = {d}, n = {n}, linear target + noise τ² = {float(z['tau2']):g}; dots: median of {res.shape[1]} seeds, bars: min–max; line: Mei–Montanari (frozen)",
            transform=a1.transAxes, fontsize=9, color=S.MUTED, va="bottom")
    med_tr = np.median(res[:, :, 2], 1)
    a2.plot(Ns, np.maximum(med_tr, 1e-14), "o-", color=FULL_C, ms=4, mec=S.PAPER, mew=0.7, lw=1.4)
    a2.set_yscale("log"); a2.set_ylim(1e-15, 3)
    a2.yaxis.set_major_locator(FixedLocator([1e-12, 1e-8, 1e-4, 1])); a2.yaxis.set_minor_locator(NullLocator())
    a2.set_ylabel("train MSE\n(trained net)")
    a2.set_xlabel("hidden units N")
    a2.xaxis.set_major_locator(FixedLocator([2, 5, 20, 50, 200, 400, 2000])); a2.set_xticklabels(["2", "5", "20", "50", "200", "400", "2000"])
    a2.xaxis.set_minor_locator(NullLocator())
    S.save(fig, HERE / "figures" / "twolayer.png")
    i_rf = int(np.argmax(np.median(res[:, :, 0], 1))); i_f = int(np.argmax(np.median(res[:, :, 1], 1)))
    print(f"RF peak N={Ns[i_rf]} ({np.median(res[i_rf, :, 0]):.3g}); full peak N={Ns[i_f]} (N*d={Ns[i_f]*d}, N(d+1)={Ns[i_f]*(d+1)}) ({np.median(res[i_f, :, 1]):.3g})")
    first_interp = Ns[np.argmax(med_tr < 1e-6)]
    print("first width with median train MSE < 1e-6:", first_interp, "N*d =", first_interp * d)
    for N, r in zip(Ns, res):
        print(N, np.round(np.median(r, 0), 4))


if __name__ == "__main__":
    main()
