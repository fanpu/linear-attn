"""Anisotropic, misspecified linear model: python render_aniso.py"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

import style as S
from compute_aniso import spectrum

HERE = pathlib.Path(__file__).resolve().parent


def main():
    z = np.load(HERE / "cache" / "aniso.npz")
    PS, alphas, th, sim, N = z["PS"], z["alphas"], z["th"], z["sim"], int(z["N"])
    S.use("light")
    fig = plt.figure(figsize=(13, 4.8))
    a0 = fig.add_axes([0.05, 0.14, 0.2, 0.7])
    a1 = fig.add_axes([0.33, 0.14, 0.64, 0.7])
    cols = S.ramp(S.INDIGO, len(alphas), lo=0.3, hi=1.0)
    j = np.arange(1, 2001)
    for c, a in zip(cols, alphas):
        a0.plot(j, spectrum(a), color=c, lw=2)
    a0.set_xscale("log"); a0.set_yscale("log")
    a0.axvline(N, color=S.GOLD, lw=1)
    a0.set_xlabel("covariate index j"); a0.set_title("variance of covariate j", pad=8)
    a0.text(260, 30, "n", color=S.INK, fontsize=10)
    for c, a in zip(cols, alphas):
        a0.text(1.2, spectrum(a)[0] * (1.25 if a > 0 else 0.55), f"α = {a:g}", fontsize=9.5, color=S.INK)
    li = 0
    for c, a, ia in zip(cols, alphas, range(len(alphas))):
        t = th[ia, li]
        lo = PS < N
        a1.plot(PS[lo], t[lo], color=c, lw=2); a1.plot(PS[~lo], t[~lo], color=c, lw=2)
        m = sim[ia, :, :, li].mean(0)
        ok = np.abs(PS - N) > 12
        a1.plot(PS[ok], m[ok], "o", ms=4, color=c, mec=S.PAPER, mew=0.8)
        px_, off = {0.0: (25, (0, 7)), 0.5: (25, (0, -8)), 1.0: (1300, (0, 6)), 1.5: (1300, (0, 6))}[float(a)]
        k = np.argmin(np.abs(PS - px_))
        a1.annotate(f"α = {a:g}", xy=(PS[k], t[k]), xytext=off, textcoords="offset points", ha="center",
                    va="bottom" if off[1] > 0 else "top", fontsize=9.5, color=S.INK)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlim(5, 2000); a1.set_ylim(0.05, 200)
    a1.axvline(N, color=S.GOLD, lw=1.1, zorder=0)
    a1.text(N * 1.06, 120, "p = n = 200", fontsize=10, color=S.INK)
    a1.xaxis.set_major_locator(FixedLocator([5, 10, 50, 200, 1000, 2000])); a1.set_xticklabels(["5", "10", "50", "200", "1000", "2000"])
    a1.xaxis.set_minor_formatter(NullFormatter())
    a1.set_xlabel("p = number of covariates the model is given (largest variance first)")
    a1.set_ylabel("test risk (log)")
    a1.set_title("Min-norm least squares when the model only sees the first p of 2,000 covariates", pad=24)
    a1.text(0.0, 1.015, "lines: theory (misspecified deterministic equivalent)  ·  dots: mean of 30 draws, n = 200, σ² = 0.05",
             fontsize=9.5, color=S.MUTED, transform=a1.transAxes, va="bottom")
    S.save(fig, HERE / "figures" / "aniso.png")


if __name__ == "__main__":
    main()
