"""Render the static plates from cache (no recompute).

    python render_plates.py --analysis cache/main/ts_main_analysis --styles paper,dark,riso,spectral --pieces diptych,fan
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from style import STYLES, setup, log_paper, dcolor, save

p = argparse.ArgumentParser()
p.add_argument("--analysis", default="cache/main/ts_main_analysis")
p.add_argument("--styles", default="paper,dark,riso,spectral")
p.add_argument("--pieces", default="diptych,fan")
p.add_argument("--fam", default="relu0")
p.add_argument("--idw", type=int, default=45, help="student width whose N(r) curves are drawn")
p.add_argument("--out", default="gallery")
p.add_argument("--dpi", type=int, default=200)
a = p.parse_args()

A = json.load(open(a.analysis + ".json"))
C = np.load(a.analysis + ".npz")
radii = C["radii"]
Ns = np.array(A["Ns"]); dims = A["dims"]
order = np.argsort(Ns); Ns = Ns[order]


def id_of(f, d, what="twonn"):
    v = [x[what] for k, x in A["ids"].items() if k.startswith(f"{f}_{d}_")]
    return float(np.median(v)), (float(np.min(v)), float(np.max(v)))


def curve_for(f, d):
    k = f"curve_{f}_{d}_0_w{a.idw}"
    if k not in C.files:
        k = sorted([q for q in C.files if q.startswith(f"curve_{f}_{d}_0_")])[-1]
    return C[k]


def scaling_range(cnt):
    return (cnt >= 3) & (cnt <= 300)


def local_d(cnt):
    m = scaling_range(cnt)
    return np.polyfit(np.log(radii[m]), np.log(cnt[m]), 1)[0], m


def label(ax, x, y, s, S, color, **kw):
    ax.text(x, y, s, color=color, fontsize=kw.pop("fs", 9), va="center", ha=kw.pop("ha", "left"), **kw)


# --------------------------------------------------------------------------- diptych
def diptych(style):
    fig, S = setup(style, (16, 8.6), a.dpi)
    f = a.fam
    axL = fig.add_axes([0.06, 0.25, 0.41, 0.62]); axR = fig.add_axes([0.555, 0.25, 0.41, 0.62])
    Lall = np.array([A["fits"][f"{f}_{d}"]["L"] for d in dims])[:, order]
    log_paper(axL, style, (Ns[0] / 1.6, Ns[-1] * 3.2), (Lall.min() / 2.5, Lall.max() * 2.2))
    log_paper(axR, style, (0.25, 60), (0.4, 3000))
    for d in dims:
        F = A["fits"][f"{f}_{d}"]
        col = dcolor(style, d, dims)
        L = np.array(F["L"])[order]; La = np.array(F["L_all"])[order]
        xx = np.geomspace(Ns[0] / 1.3, Ns[-1] * 1.3, 50)
        axL.plot(xx, np.exp(F["logc"]) * xx ** -F["alpha"], color=col, lw=0.9, alpha=0.8, zorder=2)
        for s in range(La.shape[1]):
            axL.scatter(Ns, La[:, s], s=5, color=col, alpha=0.35, lw=0, zorder=3)
        axL.scatter(Ns, L, s=22 if style != "paper" else 18, facecolor=S["bg"], edgecolor=col, lw=1.1, zorder=4)
        label(axL, xx[-1] * 1.08, np.exp(F["logc"]) * xx[-1] ** -F["alpha"], f"d={d}", S, col, fs=8.5)
        cnt = curve_for(f, d)
        dd, m = local_d(cnt)
        ok = cnt > 0
        axR.plot(radii[ok], cnt[ok], color=col, lw=1.1, zorder=3)
        lo, hi = radii[m][0], radii[m][-1]
        rr = np.geomspace(lo, hi, 10)
        c0 = np.exp(np.polyfit(np.log(radii[m]), np.log(cnt[m]), 1)[1])
        axR.plot(rr, c0 * rr ** dd, color=S["accent"] if style in ("dark", "riso") else col, lw=3.2, alpha=0.35, zorder=2, solid_capstyle="butt")
        j = np.searchsorted(cnt, 2000)
        if j < len(radii):
            label(axR, radii[j] * 1.12, cnt[j], f"d={d}", S, col, fs=8.5)
    axL.set_xlabel("N  (student parameters)", fontsize=10); axL.set_ylabel("test loss  L  (relative MSE)", fontsize=10)
    axR.set_xlabel("r / (median nearest-neighbour distance)", fontsize=10)
    axR.set_ylabel("N(r)  (neighbours within r)", fontsize=10)
    fig.text(0.06, 0.915, "the α-line", fontsize=17, style="italic", color=S["text"])
    fig.text(0.06, 0.885, r"$L \propto N^{-\alpha}$ : a straight line on log paper, slope $-\alpha$", fontsize=10.5, color=S["text"])
    fig.text(0.555, 0.915, "the d-line", fontsize=17, style="italic", color=S["text"])
    fig.text(0.555, 0.885, r"$N(r) \propto r^{\,d}$ in the student's last hidden layer, slope $d$", fontsize=10.5, color=S["text"])
    # ledger
    xs = np.linspace(0.14, 0.93, len(dims))
    fig.text(0.06, 0.145, "teacher d", fontsize=9.5, color=S["text"]); fig.text(0.06, 0.105, r"4 / $\alpha$", fontsize=9.5, color=S["text"])
    fig.text(0.06, 0.065, "TwoNN d", fontsize=9.5, color=S["text"])
    for x, d in zip(xs, dims):
        F = A["fits"][f"{f}_{d}"]; idd, _ = id_of(f, d)
        col = dcolor(style, d, dims)
        fig.text(x, 0.145, f"{d}", fontsize=11, ha="center", color=col, family="DejaVu Sans Mono")
        fig.text(x, 0.105, f"{F['four_over_alpha']:.1f}", fontsize=11, ha="center", color=S["text"], family="DejaVu Sans Mono")
        fig.text(x, 0.065, f"{idd:.1f}", fontsize=11, ha="center", color=S["text"], family="DejaVu Sans Mono")
    fig.text(0.965, 0.02, f"teacher/student, ReLU teacher [24,600,600,1] on a d-cube; student [24,n,n,1], median of 3 seeds; ID from {a.idw}-wide students, 12k points",
             fontsize=7.5, ha="right", color=S["text"], alpha=0.75)
    return save(fig, f"{a.out}/diptych_{style}.png", riso=(style == "riso"))


# --------------------------------------------------------------------------- fan of slopes
def fan(style):
    fig, S = setup(style, (12, 12), a.dpi)
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84]); ax.set_facecolor(S["bg"])
    f = a.fam
    ax.set_xlim(-0.1, 2.35); ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    # decade grid (declared: log paper ruling, one unit = one decade)
    for v in np.arange(0, 2.4, 1):
        ax.axvline(v, color=S["grid_major"], lw=0.8, zorder=0)
    for v in np.arange(-2, 3, 1):
        ax.axhline(v, color=S["grid_major"], lw=0.8 if v else 1.2, zorder=0)
    for e in range(0, 3):
        for m in range(2, 10):
            x = e + np.log10(m)
            if x < 2.35: ax.axvline(x, color=S["grid_minor"], lw=0.35, zorder=0)
    for e in range(-3, 3):
        for m in range(2, 10):
            y = e + np.log10(m)
            if -2.6 < y < 2.6: ax.axhline(y, color=S["grid_minor"], lw=0.35, zorder=0)
    x1 = 2.2
    for d in dims:
        F = A["fits"][f"{f}_{d}"]; col = dcolor(style, d, dims)
        idd, _ = id_of(f, d)
        # lower fan: loss curves, each seed, shifted to start at the origin (measured)
        La = np.array(F["L_all"])[order]
        x = np.log10(Ns / Ns[0])
        for s in range(La.shape[1]):
            y = np.log10(La[:, s] / np.exp(F["logc"] + np.log(Ns[0]) * -F["alpha"]))
            ax.plot(x, y, color=col, lw=0.5, alpha=0.45, zorder=2)
        ax.plot([0, x1], [0, -F["alpha"] * x1], color=col, lw=1.3, zorder=3)
        ax.plot([0, x1], [0, -4 / idd * x1], color=col, lw=0.8, ls=(0, (1, 2.5)), zorder=3)
        label(ax, x1 + 0.03, -F["alpha"] * x1, f"d={d}  α={F['alpha']:.2f}", S, col, fs=8)
        # upper fan: neighbour counts vs radius (measured), slope d / 4 so both fans share a scale
        cnt = curve_for(f, d); dd, m = local_d(cnt)
        ok = cnt > 0.3
        r = np.log10(radii[ok] / radii[m][0]); c = np.log10(cnt[ok] / cnt[m][0])
        # compress the count axis by 4 so slope d/4 mirrors alpha = 4/d
        sel = (r > -0.4) & (r < x1 * 4 / max(dd, 1) + 1)
        ax.plot(r[sel] * 1.0, c[sel] / (4 * (np.log10(radii[m][-1] / radii[m][0]) / 1.0)) if False else c[sel] / 4 * 1.0, color=col, lw=0.6, alpha=0.6, zorder=2)
        ax.plot([0, x1], [0, dd / 4 * x1 / 4], color=col, lw=0.0)
    ax.text(0.02, 2.45, "upper fan: log N(r) / 4 against log r — slopes d/4", fontsize=9, color=S["text"])
    ax.text(0.02, -2.5, "lower fan: log L against log N — slopes −α; dotted rays −4/d(TwoNN)", fontsize=9, color=S["text"])
    return save(fig, f"{a.out}/fan_{style}.png", riso=(style == "riso"))


os.makedirs(a.out, exist_ok=True)
for piece in a.pieces.split(","):
    for st in a.styles.split(","):
        print(piece, st, globals()[piece](st))
