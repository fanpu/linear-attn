"""Static plates for the isotropic linear model.  python render_linear.py [family|sizes|biasvar|ridge|all]"""
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

import dd_core as C
import style as S

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
CACHE = HERE / "cache"


def load(tag):
    z = np.load(CACHE / f"linear_{tag}.npz")
    return {k: z[k] for k in z.files}


def realized(parts, lam_idx, sigma2):
    s = np.sqrt(sigma2)
    P = parts[:, :, lam_idx, :]
    return P[..., 0] + 2 * s * P[..., 1] + sigma2 * P[..., 2]


def gamma_axis(ax, lo=0.1, hi=10):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def finite_n_risk(n, p, r2, sigma2):
    """Exact finite-n expected risk of min-norm LS with Gaussian design (inverse-Wishart moments)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.inf)
    lo = p < n - 1
    out[lo] = sigma2 * p[lo] / (n - p[lo] - 1)
    hi = p > n + 1
    out[hi] = r2 * (1 - n / p[hi]) + sigma2 * n / (p[hi] - n - 1)
    return out


def fig_family():
    c = S.use("light")
    d = load("n400")
    n, ps, g = int(d["n"]), d["p"], d["gamma"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    gg = np.concatenate([np.geomspace(0.1, 0.999, 400), np.geomspace(1.001, 10, 400)])
    cols = S.ramp(S.INDIGO, len(d["sig2"]), lo=0.3, hi=1.0)[::-1]   # low noise = light? no: high SNR = dark
    cols = S.ramp(S.INDIGO, len(d["sig2"]), lo=0.32, hi=1.0)
    ok = np.abs(ps - n) > 2
    for k, s2 in enumerate(d["sig2"]):
        snr = 1.0 / s2
        th = C.ridgeless_risk(gg, 1.0, s2)[0]
        col = cols[k]
        ax.plot(gg[gg < 1], th[gg < 1], color=col, lw=2, zorder=3)
        ax.plot(gg[gg > 1], th[gg > 1], color=col, lw=2, zorder=3)
        R = realized(d["parts"], 0, s2)
        mean = R.mean(1); se = R.std(1) / np.sqrt(R.shape[1])
        ax.errorbar(g[ok], mean[ok], yerr=2 * se[ok], fmt="o", ms=4.2, color=col, mec=S.PAPER, mew=0.9,
                    elinewidth=1, capsize=0, zorder=4)
        # direct label where the curves are well separated: next to the overparameterized minimum
        if snr > 1:
            gmin = np.sqrt(snr) / (np.sqrt(snr) - 1)
            ymin = C.ridgeless_risk(gmin, 1.0, s2)[0]
            ax.plot([gmin], [ymin], marker="v", ms=6, color=col, mec=S.PAPER, mew=0.8, zorder=6)
            ax.annotate(f"SNR {snr:g}", xy=(gmin, ymin), xytext=(0, -9), textcoords="offset points",
                        ha="center", va="top", fontsize=9.5, color=S.INK,
                        bbox=dict(boxstyle="square,pad=0.1", fc=S.PAPER, ec="none", alpha=0.85), zorder=7)
        else:
            ax.annotate(f"SNR {snr:g}  (no dip: noise dominates)", xy=(2.6, C.ridgeless_risk(2.6, 1.0, s2)[0]),
                        xytext=(6, 6), textcoords="offset points", fontsize=9.5, color=S.INK, va="bottom")
    ax.axhline(1.0, color=S.MUTED, lw=1, ls=(0, (1, 3)))
    ax.text(0.105, 1.04, "null risk $r^2$ (predict 0)", color=S.MUTED, fontsize=9.5, va="bottom")
    ax.axvline(1.0, color=S.GOLD, lw=1.2, zorder=1)
    ax.text(0.97, 2.93, "interpolation\nthreshold  $p = n$", color=S.INK, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="square,pad=0.2", fc=S.PAPER, ec="none"), zorder=8)
    gamma_axis(ax)
    ax.set_ylim(0, 3.0)
    ax.set_xlabel(r"$\gamma = p\,/\,n$   (parameters per sample)")
    ax.set_ylabel("test risk  $\\mathbb{E}\\,\\|\\hat\\beta-\\beta\\|^2$")
    ax.set_title("Min-norm least squares, isotropic features: theory vs. measurement", pad=26)
    ax.text(0.0, 1.02, f"lines: Hastie et al. closed form   ·   dots: mean of 50 random draws at n = {n} (±2 s.e.)   ·   triangles: best γ > 1",
            transform=ax.transAxes, fontsize=9.2, color=S.MUTED, va="bottom")
    S.save(fig, FIG / "linear_family.png")
    # numbers for the post
    for k, s2 in enumerate(d["sig2"]):
        R = realized(d["parts"], 0, s2); mean = R.mean(1)
        th = C.ridgeless_risk(g, 1.0, s2)[0]
        far = np.abs(g - 1) > 0.25
        rel = np.abs(mean[far] - th[far]) / th[far]
        print(f"SNR {1/s2:g}: |mean - theory|/theory for |gamma-1|>0.25: median {np.median(rel):.3f} max {rel.max():.3f}")


def fig_sizes():
    c = S.use("light")
    s2 = 0.25
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    gg = np.concatenate([np.geomspace(0.1, 0.999, 300), np.geomspace(1.001, 10, 300)])
    th = C.ridgeless_risk(gg, 1.0, s2)[0]
    for ax, n in zip(axs, [50, 200, 800]):
        d = load(f"n{n}")
        ps, g = d["p"], d["gamma"]
        R = realized(d["parts"], 0, s2)
        rng = np.random.default_rng(0)
        for i in range(len(ps)):
            jit = np.exp(rng.normal(0, 0.012, R.shape[1]))
            ax.scatter(g[i] * jit, np.clip(R[i], 0, 3.2), s=5, color=S.INDIGO, alpha=0.16, lw=0, zorder=2)
        q25, q50, q75 = np.percentile(R, [25, 50, 75], axis=1)
        ax.vlines(g, q25, q75, color=S.INDIGO, lw=1.8, zorder=3)
        ax.plot(g, q50, "o", ms=3.2, color=S.INDIGO, mec=S.PAPER, mew=0.6, zorder=4)
        ax.plot(gg[gg < 1], th[gg < 1], color=S.INK, lw=1.3, zorder=5)
        ax.plot(gg[gg > 1], th[gg > 1], color=S.INK, lw=1.3, zorder=5)
        pf = np.concatenate([np.arange(1, n - 1), np.arange(n + 2, 10 * n + 1)])
        ax.plot(pf / n, finite_n_risk(n, pf, 1.0, s2), color=S.VAR, lw=1.1, ls=(0, (4, 2)), zorder=5)
        ax.axvline(1, color=S.GOLD, lw=1, zorder=1)
        gamma_axis(ax)
        ax.set_ylim(0, 3.2)
        ax.set_title(f"n = {n}", pad=6)
        ax.set_xlabel(r"$\gamma = p/n$")
        # spread number at gamma ~ 2
        i2 = np.argmin(np.abs(g - 2.0))
        ax.text(0.97, 0.95, f"IQR at γ≈2: {q75[i2] - q25[i2]:.3f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=S.MUTED)
        print(f"n={n}: IQR at gamma={g[i2]:.2f}: {q75[i2]-q25[i2]:.4f}; median {q50[i2]:.4f} theory {C.ridgeless_risk(g[i2],1,s2)[0]:.4f}")
    axs[0].set_ylabel("test risk")
    axs[0].text(1.6, 2.3, "each faint dot = one random\ndataset; bars = middle 50%", fontsize=9.3, color=S.MUTED)
    axs[2].text(1.6, 2.55, "asymptotic formula", color=S.INK, fontsize=9.5)
    axs[2].text(1.6, 2.25, "exact finite-n mean", color=S.VAR, fontsize=9.5)
    fig.suptitle("The same experiment at three sample sizes (SNR = 4): the cloud collapses onto the curve",
                 x=0.07, ha="left", fontweight="bold", fontsize=12.5, y=1.02)
    S.save(fig, FIG / "linear_sizes.png")


def fig_biasvar():
    c = S.use("light")
    d = load("n400")
    n, ps, g = int(d["n"]), d["p"], d["gamma"]
    s2 = 0.25
    gg = np.concatenate([np.geomspace(0.1, 0.995, 300), np.geomspace(1.005, 10, 300)])
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    ok = np.abs(ps - n) > 2
    kopt = 7 + int(np.where(np.isclose(d["sig2"], s2))[0][0])
    for ax, mode in zip(axs, ["ridgeless", "optimal"]):
        if mode == "ridgeless":
            R, B, V = C.ridgeless_risk(gg, 1.0, s2)
            li = 0
        else:
            R, B, V = C.ridge_risk(gg, C.optimal_lambda(gg, 1.0, s2), 1.0, s2)
            li = kopt
        P = d["parts"][:, :, li, :]
        mb = P[..., 0].mean(1); mv = (s2 * P[..., 3]).mean(1)
        for arr, col, name in [(B, S.BIAS, "bias²"), (V, S.VAR, "variance")]:
            m = gg < 1
            ax.plot(gg[m], arr[m], color=col, lw=2.2); ax.plot(gg[~m], arr[~m], color=col, lw=2.2)
        ax.plot(gg[gg < 1], R[gg < 1], color=S.INK, lw=1.2); ax.plot(gg[gg > 1], R[gg > 1], color=S.INK, lw=1.2)
        ax.plot(g[ok][::2], mb[ok][::2], "o", ms=4.2, color=S.BIAS, mec=S.PAPER, mew=0.8)
        ax.plot(g[ok][::2], mv[ok][::2], "o", ms=4.2, color=S.VAR, mec=S.PAPER, mew=0.8)
        ax.axvline(1, color=S.GOLD, lw=1, zorder=0)
        gamma_axis(ax); ax.set_ylim(0, 2.0)
        ax.set_xlabel(r"$\gamma = p/n$")
        if mode == "ridgeless":
            ax.set_title("Min-norm interpolation (λ → 0)", pad=8)
            ax.text(0.6, 0.62, "variance", color=S.INK, fontsize=10.5, ha="right")
            ax.text(6.0, 0.93, "bias²", color=S.INK, fontsize=10.5, ha="center")
            ax.text(3.0, 1.35, "total", color=S.INK, fontsize=10.5)
        else:
            ax.set_title("Optimally tuned ridge (λ* = σ²γ/r²)", pad=8)
            ax.text(6.0, 0.62, "bias²", color=S.INK, fontsize=10.5, ha="center")
            ax.text(6.0, 0.08, "variance", color=S.INK, fontsize=10.5, ha="center")
            ax.text(0.8, 0.56, "total", color=S.INK, fontsize=10.5)
    axs[0].set_ylabel("risk components")
    fig.text(0.07, 1.0, "Where the peak comes from: the variance term alone (SNR = 4; lines theory, dots n = 400, conditional on X, 50 draws)",
             fontsize=12, fontweight="bold", ha="left")
    S.save(fig, FIG / "linear_biasvar.png")


def fig_ridge():
    c = S.use("light")
    d = load("n400")
    n, ps, g = int(d["n"]), d["p"], d["gamma"]
    s2 = 0.25
    gg = np.concatenate([np.geomspace(0.1, 0.999, 400), np.geomspace(1.001, 10, 400)])
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    lams = d["fixed_lams"]
    show = [2, 4, 6]          # 1e-2, 0.1, 1
    cols = S.ramp(S.TEAL, len(show), lo=0.4, hi=1.0)
    th0 = C.ridgeless_risk(gg, 1.0, s2)[0]
    ax.plot(gg[gg < 1], th0[gg < 1], color=S.MUTED, lw=1, ls=(0, (3, 2)))
    ax.plot(gg[gg > 1], th0[gg > 1], color=S.MUTED, lw=1, ls=(0, (3, 2)))
    ok = np.abs(ps - n) > 2
    label_at = {2: (1.0, (10, 4)), 4: (1.45, (-70, 40)), 6: (0.15, (0, 9))}
    for col, li in zip(cols, show):
        lam = lams[li]
        th = C.ridge_risk(gg, lam, 1.0, s2)[0]
        ax.plot(gg, th, color=col, lw=1.8)
        R = realized(d["parts"], li, s2).mean(1)
        ax.plot(g[ok][::2], R[ok][::2], "o", ms=3.6, color=col, mec=S.PAPER, mew=0.7)
        gx, off = label_at[li]
        j = np.argmin(np.abs(gg - gx))
        ax.annotate(f"λ = {lam:g}", xy=(gg[j], th[j]), xytext=off, textcoords="offset points",
                    fontsize=9.5, color=S.INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.7, shrinkA=2, shrinkB=2) if li == 4 else None)
    kopt = 7 + int(np.where(np.isclose(d["sig2"], s2))[0][0])
    tho = C.ridge_risk(gg, C.optimal_lambda(gg, 1.0, s2), 1.0, s2)[0]
    ax.plot(gg, tho, color=S.INK, lw=2.6)
    Ro = realized(d["parts"], kopt, s2).mean(1)
    ax.plot(g, Ro, "o", ms=4.5, color=S.INK, mec=S.PAPER, mew=0.9)
    ax.text(2.3, 0.47, "optimal λ*(γ)", fontsize=10.5, color=S.INK, ha="left", va="top", fontweight="bold")
    ax.text(1.12, 1.9, "ridgeless", fontsize=9.5, color=S.MUTED)
    ax.axvline(1, color=S.GOLD, lw=1, zorder=0)
    gamma_axis(ax); ax.set_ylim(0, 2.0)
    ax.set_xlabel(r"$\gamma = p/n$"); ax.set_ylabel("test risk")
    ax.set_title("A little ridge flattens the peak; the optimal ridge removes it (SNR = 4)", pad=26)
    ax.text(0.0, 1.02, "lines: MP / Stieltjes closed form ·  dots: n = 400, mean of 50 draws",
            transform=ax.transAxes, fontsize=9.2, color=S.MUTED, va="bottom")
    S.save(fig, FIG / "linear_ridge.png")
    dif = np.diff(tho)
    print("optimal ridge theory monotone increasing in gamma:", bool(np.all(dif > -1e-12)))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in ["family", "sizes", "biasvar", "ridge"]:
        if what in (name, "all"):
            globals()[f"fig_{name}"]()
