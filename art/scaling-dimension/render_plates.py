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
    fig.text(0.965, 0.02, f"teacher/student, ReLU teacher [24,600,600,1] on a d-cube; student [24,n,n,1], median of 3 seeds; TwoNN ID: median over widths 16/45/90 x 3 seeds, 12k points; N(r) curves from {a.idw}-wide seed-0 students",
             fontsize=7.5, ha="right", color=S["text"], alpha=0.75)
    return save(fig, f"{a.out}/diptych_{style}.png", riso=(style == "riso"))


# --------------------------------------------------------------------------- fan of slopes
def fan(style):
    """Two fans hinged at one origin. Lower: log(L/L0) vs log(N/N0), slope -alpha.
    Upper: 4*log(r/r0) vs log(N(r)/N0), slope 4/d.  If alpha = 4/d the fans are mirror images.
    Rays and curves are cut at a common radius RR (declared composition choice)."""
    fig, S = setup(style, (12, 12), a.dpi)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94]); ax.set_facecolor(S["bg"])
    f = a.fam
    RR = 2.25
    ax.set_ylim(-2.5, 2.55); ax.set_xlim(-0.75, 3.25)
    ax.set_aspect("equal")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    th = np.linspace(-np.pi / 2, np.pi / 2, 400)
    for R in (0.5, 1.0, 1.5, 2.0):             # declared: range rings every half decade
        ax.plot(R * np.cos(th), R * np.sin(th), color=S["grid_minor"], lw=0.5, zorder=0)
    ax.plot(RR * np.cos(th), RR * np.sin(th), color=S["grid_major"], lw=0.8, zorder=0)
    for e in range(0, 3):
        for m in range(1, 10):
            x = e + np.log10(m)
            if x <= RR:
                h = 0.07 if m == 1 else 0.03
                ax.plot([x, x], [-h, h], color=S["grid_major"], lw=0.8, zorder=1)
    ax.plot([0, RR], [0, 0], color=S["grid_major"], lw=1.2, zorder=1)
    riso = style == "riso"
    lower_col = lambda d: STYLES["riso"]["ink"] if riso else dcolor(style, d, dims)
    upper_col = lambda d: "#ff48b0" if riso else dcolor(style, d, dims)

    def clip(x, y):
        k = np.hypot(x, y) <= RR
        return np.where(k, x, np.nan), np.where(k, y, np.nan)

    def ray(slope, col, sign, text):
        ang = np.arctan(slope) * sign
        ax.plot([0, RR * np.cos(ang)], [0, RR * np.sin(ang)], color=col, lw=1.5, zorder=3, solid_capstyle="round")
        R2 = RR + 0.06
        ax.text(R2 * np.cos(ang), R2 * np.sin(ang), text, color=col, fontsize=8.5, rotation=np.degrees(ang),
                rotation_mode="anchor", va="center", ha="left")

    for d in dims:
        F = A["fits"][f"{f}_{d}"]
        La = np.array(F["L_all"])[order]; L = np.array(F["L"])[order]
        L0 = np.exp(F["logc"]) * Ns[0] ** -F["alpha"]
        x = np.log10(Ns / Ns[0])
        cl = lower_col(d)
        for sd in range(La.shape[1]):
            ax.plot(*clip(x, np.log10(La[:, sd] / L0)), color=cl, lw=0.45, alpha=0.3, zorder=2)
        xx, yy = clip(x, np.log10(L / L0))
        ax.scatter(xx, yy, s=10, color=cl, lw=0, zorder=4)
        ray(F["alpha"], cl, -1, f"α = {F['alpha']:.2f}")
        cnt = curve_for(f, d); dd, m = local_d(cnt)
        cu = upper_col(d)
        c0 = cnt[m][0]; r0 = radii[m][0]
        ok = cnt >= c0
        ax.plot(*clip(np.log10(cnt[ok] / c0), 4 * np.log10(radii[ok] / r0)), color=cu, lw=1.0, alpha=0.75, zorder=2)
        ray(4 / dd, cu, +1, f"d={d}   4/d̂ = {4 / dd:.2f}")
    t = S["text"]
    kw = dict(ha="right", transform=ax.transData)
    ax.text(3.2, -1.95, "the fan of slopes", fontsize=19, style="italic", color=t, **kw)
    ax.text(3.2, -2.07, "upper: 4·log r  vs  log N(r) in the student's last hidden layer — slope 4/d",
            fontsize=8.5, color=upper_col(12) if riso else t, **kw)
    ax.text(3.2, -2.17, "lower: log test loss  vs  log parameters N — slope −α", fontsize=8.5,
            color=lower_col(2) if riso else t, **kw)
    ax.text(3.2, -2.27, "if α = 4/d the two fans mirror each other across the hinge", fontsize=8.5, color=t, alpha=0.8, **kw)
    ax.text(3.2, -2.45, f"teacher '{f}', d = {', '.join(map(str, dims))}. hinge ruled in decades; curves & rays cut at 2.25 decades. "
            "thin: per-seed losses (lower), student N(r) (upper); dots: seed medians; rays: fits",
            fontsize=6.5, color=t, alpha=0.7, **kw)
    return save(fig, f"{a.out}/fan_{style}.png", riso=riso)


# --------------------------------------------------------------------------- agreement plate
def real_points():
    path = "cache/real/real.json"
    if not os.path.exists(path):
        return []
    R = json.load(open(path)); out = []
    for ds, v in R.items():
        ks = sorted([k for k in v if k.startswith("c")], key=lambda k: v[k]["N"])
        if len(ks) < 4: continue
        N = np.array([v[k]["N"] for k in ks]); L = np.array([v[k]["test_loss"] for k in ks])
        ids = np.array([v[k]["id_twonn"] for k in ks])
        al = -np.polyfit(np.log(N), np.log(L), 1)[0]
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            ii = np.sort(rng.integers(0, len(N), len(N)))
            if len(np.unique(ii)) < 3: continue
            bs.append(-np.polyfit(np.log(N[ii]), np.log(L[ii]), 1)[0])
        lo, hi = np.percentile(bs, [5, 95])
        top = ids[-3:]
        out.append(dict(name=ds, alpha=al, ci=(lo, hi), id=float(np.median(top)), id_rng=(top.min(), top.max()),
                        pix=v["pixel_id"]["twonn"]))
    return out


def agree(style):
    fig, S = setup(style, (12, 12), a.dpi)
    ax = fig.add_axes([0.1, 0.09, 0.84, 0.8])
    lim = (1.4, 140)
    log_paper(ax, style, lim, lim)
    ax.set_aspect("equal")
    riso = style == "riso"
    ax.plot(lim, lim, color=S["accent"] if style != "paper" else S["ink"], lw=1.0, ls=(0, (6, 3)), zorder=2)
    ax.text(100, 118, "4/α = d", rotation=45, fontsize=10, color=S["accent"] if style != "paper" else S["ink"],
            ha="center", va="center", rotation_mode="anchor")
    marks = {"relu0": dict(marker="o", filled=True), "relub": dict(marker="s", filled=False)}
    for f in A["fams"]:
        for d in dims:
            F = A["fits"][f"{f}_{d}"]
            idd, (imin, imax) = id_of(f, d)
            col = STYLES["riso"]["ink"] if riso else dcolor(style, d, dims)
            y = F["four_over_alpha"]; ylo, yhi = 4 / F["alpha_ci"][1], 4 / F["alpha_ci"][0]
            ax.plot([idd, idd], [ylo, yhi], color=col, lw=1.0, zorder=3)
            ax.plot([imin, imax], [y, y], color=col, lw=1.0, zorder=3)
            mk = marks[f]
            ax.scatter([idd], [y], s=60, marker=mk["marker"], facecolor=col if mk["filled"] else S["bg"],
                       edgecolor=col, lw=1.3, zorder=5)
            if f == "relu0":
                ax.text(idd * 0.93, y * 1.07, f"d={d}", fontsize=8, color=col, ha="right")
    pink = "#ff48b0" if riso else S["accent"] if style != "paper" else S["ink"]
    for P in real_points():
        ax.plot([P["id"], P["id"]], [4 / P["ci"][1], 4 / P["ci"][0]], color=pink, lw=1.0, zorder=3)
        ax.plot(P["id_rng"], [4 / P["alpha"]] * 2, color=pink, lw=1.0, zorder=3)
        ax.scatter([P["id"]], [4 / P["alpha"]], s=130, marker="*", color=pink, zorder=6, lw=0)
        ax.scatter([P["pix"]], [4 / P["alpha"]], s=40, marker="D", facecolor=S["bg"], edgecolor=pink, lw=1.0, zorder=5)
        ax.plot([P["id"], P["pix"]], [4 / P["alpha"]] * 2, color=pink, lw=0.6, ls=":", zorder=3)
        ax.text(P["pix"] * 1.1, 4 / P["alpha"], f"{P['name']}  (4/α={4 / P['alpha']:.1f})", fontsize=8.5, color=pink, va="center")
    # GPT-2 from Sharma & Kaplan 2022 (not measured here): 4/alpha ~ 53, first-layer ID 50-80, other layers > 90
    g = 4 / 0.076
    ax.plot([50, 80], [g, g], color=S["text"], lw=3.5, alpha=0.35, solid_capstyle="butt", zorder=3)
    ax.annotate("", xy=(138, g), xytext=(90, g), arrowprops=dict(arrowstyle="-|>", color=S["text"], lw=0.9), zorder=3)
    ax.text(12, g * 1.2, "GPT-2 small (Sharma & Kaplan 2022, not measured here):\n4/α≈53; first-layer ID 50–80 (bar), other layers > 90 (arrow)",
            fontsize=8, color=S["text"], alpha=0.85, ha="left")
    ax.set_xlabel("intrinsic dimension of the learned representation  (TwoNN, student's last hidden layer)", fontsize=10)
    ax.set_ylabel("4 / α   (from the measured loss scaling exponent)", fontsize=10)
    fig.text(0.1, 0.945, "the agreement plate", fontsize=19, style="italic", color=S["text"])
    fig.text(0.1, 0.915, r"Sharma & Kaplan: $\alpha \approx 4/d$ for piecewise-linear regression on a $d$-manifold",
             fontsize=10.5, color=S["text"])
    ax.text(0.97, 0.03, "●  teacher/student, zero-bias ReLU teacher        □  teacher with biases\n"
             "★  CNN on real images (final hidden layer ID)   ◇  pixel-space ID of the same data\n"
             "bars: 90% bootstrap CI on α (vertical); ID spread over widths & seeds (horizontal)",
             fontsize=8.5, ha="right", va="bottom", color=S["text"], family="DejaVu Sans", transform=ax.transAxes,
             bbox=dict(facecolor=S["bg"], edgecolor=S["grid_major"], pad=6))
    return save(fig, f"{a.out}/agree_{style}.png", riso=riso)


os.makedirs(a.out, exist_ok=True)
for piece in a.pieces.split(","):
    for st in a.styles.split(","):
        print(piece, st, globals()[piece](st))
