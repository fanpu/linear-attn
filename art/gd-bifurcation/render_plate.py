"""Scientific-plate idiom: annotated cascade + Lyapunov strip + delta table.

Everything drawn is measured, except typography/colours.  Annotations:
  * hairline at eta* = 2/s_min, s_min = k (sharpness of the balanced = flattest global minimum)
  * dashed hairline at 2/s_GF(x0): the rule "2/lambda_max at the minimum gradient flow would reach from x0"
  * eta_n from Newton-Floquet bisection (cache/feigenbaum_A.json), periodic windows (cache/windows.json)
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from render_lib import CACHE, GAL, density
from compute_threshold import s_gf

INK = "#1b1b22"
RED = "#b3261e"
PAPER = "#f3eee2"


def tone(c, sat_pct=98.5):
    nz = c[c > 0]
    c0 = np.percentile(nz, 30)
    sat = np.percentile(nz, sat_pct)
    return np.clip(np.log1p(c / c0) / np.log1p(sat / c0), 0, 1) ** 0.9


def plate(name="prod4", lo=0.45, hi=1.21, ylo=-0.03, yhi=2.2, W=6000, H=1900, out=None, k=4, x0=(1.1, 0.9, 1.05, 0.95),
          title=None, windows=True):
    d = np.load(f"{CACHE}/bif_{name}_hi.npz")
    e = d["etas"]
    m = (e >= lo) & (e <= hi)
    c = density(e[m], d["P"][m], lo, hi, ylo, yhi, W, H)
    t = tone(c)
    ink = np.array(matplotlib.colors.to_rgb(INK))
    pap = np.array(matplotlib.colors.to_rgb(PAPER))
    img = pap[None, None] * (1 - t[..., None]) + ink[None, None] * t[..., None]

    A = json.load(open(f"{CACHE}/feigenbaum_A.json"))
    key = f"A_bal{k}" if f"A_bal{k}" in A else f"A_{name}"
    etan = A[key]["eta"]
    dl = A[key]["delta"]
    ds = A[key]["delta_sigma"]

    fig = plt.figure(figsize=(W / 300 + 4.6, (H + 900) / 300 + 1.6), dpi=300, facecolor=PAPER)
    gs = gridspec.GridSpec(2, 2, height_ratios=[H, 620], width_ratios=[W, 1250], hspace=0.06, wspace=0.03,
                           left=0.05, right=0.985, top=0.9, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, extent=[lo, hi, ylo, yhi], aspect="auto", interpolation="nearest")
    ax.set_facecolor(PAPER)
    for sp in ax.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7, width=0.5, length=3, labelbottom=False)
    ax.set_ylabel(r"network output $P_t = x_1x_2x_3x_4$", color=INK, fontsize=8, family="serif")
    # rules
    es = 2.0 / k
    ax.axvline(es, color=RED, lw=0.5)
    sg = s_gf(x0)[0]
    ax.axvline(2 / sg, color=RED, lw=0.5, ls=(0, (3, 3)))
    ax.text(es + 0.002, yhi - 0.05, r"$\eta^\ast = 2/s_{\min}$" + f" = 2/{k}\n" +
            r"$s_{\min}$ = sharpness of the balanced (flattest) global minimum" + "\nmeasured first doubling (Newton-Floquet): "
            + f"{etan[0]:.6f}", color=RED, fontsize=6.5, family="serif", va="top")
    ax.annotate(r"$2/s_{GF}(x_0)$" + f" = {2 / sg:.4f}: 2/sharpness of the minimum gradient flow\nwould reach from the declared init (dashed). Not the onset.",
                xy=(2 / sg, 1.55), xytext=(0.53, 1.62), color=RED, fontsize=6.5, family="serif", va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.4))
    # eta_n ticks
    for n, en in enumerate(etan[1:6], start=2):
        ax.plot([en, en], [ylo, ylo + 0.06 + 0.03 * (n % 2)], color=RED, lw=0.4)
    for n, (en, dx, yy) in enumerate([(etan[1], -0.03, 0.30), (etan[2], -0.012, 0.42)], start=2):
        ax.annotate(rf"$\eta_{n}$ = {en:.10f}", xy=(en, ylo + 0.07), xytext=(en + dx, ylo + yy), color=RED, fontsize=7,
                    family="serif", ha="right", arrowprops=dict(arrowstyle="-", color=RED, lw=0.4))
    einf = etan[-1]
    ax.annotate(r"$\eta_\infty$" + f" = {einf:.10f}", xy=(einf, ylo + 0.02), xytext=(einf + 0.03, ylo + 0.30),
                color=RED, fontsize=7, family="serif", arrowprops=dict(arrowstyle="-", color=RED, lw=0.4))
    if windows:
        try:
            Wj = json.load(open(f"{CACHE}/windows.json"))["atlas"]
            for r in Wj:
                if r["base"] in (3, 4, 5, 6, 7) and r["hi"] - r["lo"] > 2e-4:
                    xm = 0.5 * (r["lo"] + r["hi"])
                    ax.text(xm, 1.93 if r["base"] % 2 else 1.85, f"{r['base']}", color=RED, fontsize=7.5, ha="center",
                            family="serif")
        except FileNotFoundError:
            pass
    # crisis marker
    ax.annotate("band touches P = 0 (the degenerate stationary point x = 0):\nlong laminar phases near P = 0, sparse bursts (intermittency)",
                xy=(0.99, 0.02), xytext=(1.02, 0.55), color=RED, fontsize=5.5, family="serif",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.4))
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo, yhi)

    # Lyapunov strip
    d2 = np.load(f"{CACHE}/bif_{name}.npz")
    axl = fig.add_subplot(gs[1, 0], sharex=ax)
    ee = d2["etas"]
    lb = d["lyap_bal"] if "lyap_bal" in d else d2["lyap_bal"]
    lf = d2["lyap"]
    axl.fill_between(d["etas"], 0, np.clip(lb, -1.2, 1), where=lb > 0, color=INK, lw=0, step="mid")
    axl.fill_between(d["etas"], 0, np.clip(lb, -1.2, 1), where=lb <= 0, color=INK, alpha=0.35, lw=0, step="mid")
    axl.plot(ee, np.clip(lf, -1.2, 1), color=RED, lw=0.25, alpha=0.9)
    axl.axhline(0, color=INK, lw=0.4)
    axl.set_ylim(-1.2, 1.0)
    axl.set_facecolor(PAPER)
    for sp in axl.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.6)
    axl.tick_params(colors=INK, labelsize=7, width=0.5, length=3)
    axl.set_xlabel(r"step size $\eta$", color=INK, fontsize=8, family="serif")
    axl.set_ylabel(r"Lyapunov $\lambda$", color=INK, fontsize=8, family="serif")
    axl.text(hi - 0.003, -1.05, "grey/black fill: exponent of the oscillating mode (balanced line, 8192 steps)\n"
             "red line: max exponent of the full 4-coordinate GD map (tangent propagation, 2048 steps);\n"
             "it sits at 0 below the first doubling (the manifold of minima is neutral). Past 0.99 both collapse toward 0:\norbits linger near the degenerate stationary point x = 0; near 1.158 the full GD is chaotic where the balanced orbit is periodic",
             color=INK, fontsize=5, family="serif", ha="right", va="bottom")

    # delta table
    axt = fig.add_subplot(gs[:, 1])
    axt.axis("off")
    lines = ["FEIGENBAUM RATIOS", r"$\delta_n = (\eta_n-\eta_{n-1})/(\eta_{n+1}-\eta_n)$", ""]
    for i, (v, s) in enumerate(zip(dl, ds), start=2):
        lines.append(f"δ{i:<2d} {v:.5f} ± {max(s, 1e-12):.0e}")
    lines += ["", "logistic map, same code:", f"δ12  {A['A_logistic']['delta'][-1]:.5f}",
              "Feigenbaum constant", "δ    4.66920", "",
              "η_n: Newton on G^p(x) = x,", "bisection on Floquet", "multiplier μ = −1;",
              "± = |float64 − quad|", "propagated",
              "", "declared init", f"x0 = {x0}", "16 000 η, burn-in 20 000,", "8 192 iterates per η, float64"]
    axt.text(0.04, 0.98, "\n".join(lines), transform=axt.transAxes, va="top", ha="left", fontsize=9,
             family="monospace", color=INK, linespacing=1.5)
    fig.suptitle(title or "CASCADE  ·  gradient descent on  f(x) = ½(x₁x₂x₃x₄ − 1)²  ·  every visited iterate, one dot each",
                 color=INK, fontsize=11, family="serif", x=0.05, ha="left", y=0.965)
    out = out or f"{GAL}/plate_scientific_prod4.png"
    fig.savefig(out, dpi=300, facecolor=PAPER)
    print("wrote", out)


if __name__ == "__main__":
    plate()
