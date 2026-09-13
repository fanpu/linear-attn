"""Deep double descent plates from the CNN sweep.  python render_cnn.py [heat|slices|emc|all]"""
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
import cmcrameri.cm as cmc

import cnn_data as cd
import style as S

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
EPS = 0.10


def edges_log(v):
    lv = np.log(v)
    mid = (lv[1:] + lv[:-1]) / 2
    return np.exp(np.concatenate([[lv[0] - (mid[0] - lv[0])], mid, [lv[-1] + (lv[-1] - mid[-1])]]))


def k_axis(ax, ks):
    ax.set_xscale("log")
    t = [1, 2, 4, 8, 16, 32, 64]
    ax.xaxis.set_major_locator(FixedLocator(t)); ax.set_xticklabels([str(v) for v in t])
    ax.xaxis.set_minor_locator(NullLocator())


def e_axis(ax, E):
    ax.set_yscale("log")
    t = [v for v in [1, 3, 10, 30, 100, 300, 1000] if v <= E.max() * 1.01]
    ax.yaxis.set_major_locator(FixedLocator(t)); ax.set_yticklabels([str(v) for v in t])
    ax.yaxis.set_minor_locator(NullLocator())


def analysis(d, eps=EPS, kmin_peak=3):
    """Per epoch: measured model-wise peak (smoothed test error) and the EMC-predicted width (train error = eps)."""
    ks, E = d["k"], d["epoch"]
    te = cd.smooth_epochs(d["test_err"], 2)
    tr = cd.smooth_epochs(d["train_err"], 1)
    sel = ks >= kmin_peak
    kpk, kem = [], []
    for j in range(len(E)):
        # a double-descent peak: a width whose test error exceeds the best smaller AND the best larger width.
        # take the most prominent one; require prominence >= 0.01 (about 2x the epoch-to-epoch eval noise)
        col = te[:, j]
        prom = np.full(len(ks), -np.inf)
        for i in range(1, len(ks) - 1):
            prom[i] = col[i] - max(col[:i].min(), col[i + 1:].min())
        i = int(np.argmax(prom))
        if prom[i] >= 0.01 and ks[i] >= kmin_peak:
            lo, hi = max(0, i - 1), min(len(ks), i + 2)
            kp, _ = cd.peak_location(ks[lo:hi], col[lo:hi])
            kpk.append(kp)
        else:
            kpk.append(np.nan)
        kem.append(cd.crossing(ks, tr[:, j], eps))
    return te, tr, np.array(kpk), np.array(kem)


def fig_heat(d):
    S.use("light")
    ks, E = d["k"], d["epoch"]
    te, tr, kpk, kem = analysis(d)
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw=dict(wspace=0.18))
    for ax, M, title, cmap, vr in [(axs[0], te, "test error (clean labels)", cmc.lajolla_r, (0.3, 0.8)),
                                   (axs[1], tr, "train error (on the noisy labels it was given)", cmc.devon_r, (0, 0.9))]:
        pc = ax.pcolormesh(edges_log(ks), edges_log(E), M.T, cmap=cmap, vmin=vr[0], vmax=vr[1], shading="flat", rasterized=True)
        k_axis(ax, ks); e_axis(ax, E)
        ax.set_xlabel("CNN width k  (channels: k, 2k, 4k, 8k)")
        ax.set_title(title, pad=8)
        cs = ax.contour(ks, E, tr.T, levels=[EPS], colors=[S.INK], linewidths=1.6, linestyles="--")
        cb = fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9)
    axs[0].set_ylabel("epochs trained")
    ok = np.isfinite(kpk)
    axs[0].plot(kpk[ok], E[ok], "o", ms=3.4, color="white", mec=S.INK, mew=0.7, zorder=5)
    axs[0].text(0.02, 0.02, "dots: worst width at each epoch\ndashed: train error = 10% (EMC = n)", transform=axs[0].transAxes,
                fontsize=9.5, color=S.INK, va="bottom", bbox=dict(fc=S.PAPER, ec="none", alpha=0.85, boxstyle="round,pad=0.3"))
    fig.text(0.07, 1.0, "Deep double descent on 10,000 CIFAR-10 images with 20% label noise: 14 widths × 500 epochs",
             fontsize=12.5, fontweight="bold", ha="left")
    S.save(fig, FIG / "cnn_heat.png")


def fig_slices(d, epochs=(20, 60, 150, 500)):
    S.use("light")
    ks, E = d["k"], d["epoch"]
    te, tr, kpk, kem = analysis(d)
    epochs = [e for e in epochs if e <= E.max()]
    cols = S.ramp(S.VAR, len(epochs), lo=0.35, hi=1.0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1], hspace=0.08))
    for c, e in zip(cols, epochs):
        j = int(np.argmin(np.abs(E - e)))
        a1.plot(ks, te[:, j], "-o", color=c, ms=4, mec=S.PAPER, mew=0.8)
        a2.plot(ks, tr[:, j], "-o", color=c, ms=4, mec=S.PAPER, mew=0.8)
        a1.annotate(f"epoch {E[j]}", xy=(ks[-1], te[-1, j]), xytext=(6, 0), textcoords="offset points", fontsize=9.5, va="center", color=S.INK)
        if np.isfinite(kem[j]):
            a2.plot([kem[j]], [EPS], "v", ms=7, color=c, mec=S.PAPER, zorder=6)
            a1.axvline(kem[j], color=c, lw=1, ls=(0, (2, 2)), zorder=0)
        if np.isfinite(kpk[j]):
            yk = np.interp(np.log(kpk[j]), np.log(ks), te[:, j])
            a1.plot([kpk[j]], [yk], "*", ms=11, color=c, mec=S.INK, mew=0.6, zorder=7)
    a2.axhline(EPS, color=S.MUTED, lw=1, ls=(0, (1, 2)))
    a2.text(1.05, EPS + 0.02, "ε = 10%", fontsize=9.5, color=S.MUTED)
    a2.axhline(0.2, color=S.GOLD, lw=1)
    a2.text(1.05, 0.22, "fraction of labels that are wrong (20%)", fontsize=9.5, color=S.INK)
    k_axis(a2, ks)
    a1.set_ylabel("test error"); a2.set_ylabel("train error"); a2.set_xlabel("CNN width k")
    a1.set_title("Model-wise slices: the peak follows the width that can just barely fit the noisy labels", pad=24)
    a1.text(0.0, 1.02, "★ measured peak (smoothed) · dashed verticals / ▼: width where train error hits 10% (the EMC prediction)",
            transform=a1.transAxes, fontsize=9.2, color=S.MUTED, va="bottom")
    S.save(fig, FIG / "cnn_slices.png")


def fig_emc(d):
    S.use("light")
    ks, E = d["k"], d["epoch"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(wspace=0.28))
    te, tr, kpk, kem = analysis(d)
    ok = np.isfinite(kpk) & np.isfinite(kem)
    sc = a1.scatter(kem[ok], kpk[ok], c=np.log10(E[ok]), cmap=cmc.batlow, s=34, edgecolor=S.PAPER, lw=0.6, zorder=4)
    lim = [2, 64]
    a1.plot(lim, lim, color=S.INK, lw=1)
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(*lim); a1.set_ylim(*lim)
    for ax in (a1,):
        t = [2, 4, 8, 16, 32, 64]
        ax.xaxis.set_major_locator(FixedLocator(t)); ax.set_xticklabels(map(str, t)); ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(FixedLocator(t)); ax.set_yticklabels(map(str, t)); ax.yaxis.set_minor_locator(NullLocator())
    a1.set_xlabel("predicted: width where train error = 10%"); a1.set_ylabel("measured: width with the highest test error")
    a1.set_title("Model-wise peak vs. EMC prediction (one dot per epoch)", pad=8)
    cb = fig.colorbar(sc, ax=a1, fraction=0.045, pad=0.02); cb.set_label("log10 epoch", fontsize=9.5); cb.outline.set_visible(False)
    # which epsilon predicts best?
    epss = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    errs = []
    for eps in epss:
        _, _, kp, ke = analysis(d, eps)
        m = np.isfinite(kp) & np.isfinite(ke) & (E >= 30)
        errs.append(np.median(np.abs(np.log2(ke[m] / kp[m]))) if m.sum() > 3 else np.nan)
    a2.plot(epss, errs, "-o", color=S.INDIGO, ms=5, mec=S.PAPER)
    a2.set_xscale("log")
    a2.set_xlabel("threshold ε in the EMC definition"); a2.set_ylabel("median |log₂(predicted / measured)|")
    a2.set_title("How well does each ε locate the peak? (epochs ≥ 30)", pad=8)
    a2.axvline(0.1, color=S.GOLD, lw=1); a2.text(0.105, a2.get_ylim()[1] * 0.95, "ε = 0.1 (Nakkiran et al.)", fontsize=9.5, va="top")
    S.save(fig, FIG / "cnn_emc.png")
    print("eps sweep:", dict(zip(epss.tolist(), np.round(errs, 3).tolist())))
    m = ok & (E >= 30)
    print("eps=0.1 epochs>=30: pairs", list(zip(E[m].tolist(), np.round(kem[m], 1).tolist(), np.round(kpk[m], 1).tolist())))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    d = cd.load()
    print("epochs available:", d["epoch"].max())
    for name in ["heat", "slices", "emc"]:
        if what in (name, "all"):
            globals()[f"fig_{name}"](d)
