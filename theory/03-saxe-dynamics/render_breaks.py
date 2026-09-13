"""Figures for 'where it breaks': relative error of the closed-form transition time under violated assumptions,
and the depth plate."""
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import saxe_core as sc
import style

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
S = np.array([5.0, 2.0, 1.0, 0.5])


def panel_style(ax, P, ylim=(-0.5, 0.5)):
    ax.axhspan(-0.05, 0.05, color=P["rule"], alpha=0.55, lw=0, zorder=0)
    ax.axhline(0, color=P["muted"], lw=0.8, zorder=1)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}" if v else "0"))


def breaks_figure():
    P = style.use("light")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw=dict(wspace=0.28))

    # (a) init scale, random Gaussian init
    d = np.load(HERE / "cache/break_init.npz")
    sig, B, tk, u0 = d["sigmas"], int(d["seeds"]), d["tk"], d["u0eff"]
    nkeep = int(np.sum(sig <= 0.11))
    ax = axs[0]
    panel_style(ax, P, (-0.1, 0.4))
    for k in range(4):
        rel = []
        for i in range(nkeep):
            sl = slice(i * B, (i + 1) * B)
            ok = u0[sl, k] < S[k] / 4
            pred = sc.t_half(S[k], u0[sl, k][ok])
            r = (tk[sl, k][ok] - pred) / pred
            rel.append((np.mean(r) if ok.sum() >= B // 2 else np.nan, np.std(r) if ok.sum() >= B // 2 else np.nan))
        rel = np.array(rel)
        sg = sig[:nkeep]
        ax.fill_between(sg, rel[:, 0] - rel[:, 1], rel[:, 0] + rel[:, 1], color=P["modes"][k], alpha=0.10, lw=0)
        ax.plot(sg, rel[:, 0], color=P["modes"][k], lw=2, marker="o", ms=4.5, mec=P["bg"], mew=1)
        ax.text(sg[-1] * 1.3, rel[-1, 0] + [0, 0.012, -0.012, 0][k], f"s={S[k]:g}", fontsize=9.5, color=P["ink"], va="center")
    ax.set_xscale("log")
    ax.set_xlabel("init scale σ  (entries ~ N(0, σ²))")
    ax.set_ylabel("error in predicted  $t_{1/2}$")
    ax.set_title("Random (not decoupled) init", pad=10)
    ax.axvspan(0.3, 1.2, color=P["rule"], alpha=0.8, lw=0)
    ax.text(0.6, 0.38, "init no\nlonger\nsmall:\nno\nplateau", ha="center", va="top", fontsize=9.5, color=P["muted"])
    ax.set_xlim(8e-5, 1.2)

    # (b) imbalance
    d = np.load(HERE / "cache/break_imbalance.npz")
    rho, tk, u0 = d["rhos"], d["tk"], float(d["u0"])
    ax = axs[1]
    panel_style(ax, P, (-1.0, 0.25))
    for k in range(4):
        a0, b0 = rho * np.sqrt(u0), np.sqrt(u0) / rho
        pn = sc.t_half(S[k], u0)
        pl = sc.t_half(S[k], (a0 + b0) ** 2 / 4)
        ax.plot(rho, (tk[:, k] - pn) / pn, color=P["modes"][k], lw=2, marker="o", ms=4.5, mec=P["bg"], mew=1)
        ax.plot(rho[:-1], ((tk[:, k] - pl) / pl)[:-1], color=P["modes"][k], lw=1.2, ls=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_xlabel("imbalance  $a_0 / b_0$  (product $a_0 b_0 = 10^{-5}$ fixed)")
    ax.set_title("Unbalanced layers", pad=10)
    ax.text(1.1, -0.93, "solid: plug in $u_0 = a_0 b_0$\ndashed: plug in $u_0 = ((a_0+b_0)/2)^2$", fontsize=9.5,
            color=P["ink"], va="bottom")

    # (c) non-whitened inputs
    d = np.load(HERE / "cache/break_whiten.npz")
    kap, var, kk, tm, lam = d["kappas"], d["variant"], d["kappa"], d["tm"], d["lam_modes"]
    u0 = float(d["u0"])
    ax = axs[2]
    panel_style(ax, P, (-0.02, 0.5))
    for k in range(4):
        for vname, ls, lw in (("rotated", "-", 2.0), ("aligned", (0, (2, 2)), 1.3)):
            m, sd = [], []
            for kp in kap:
                sel = (var == vname) & (kk == kp)
                pred = sc.t_half(S[k], u0)
                r = (tm[sel, k] - pred) / pred
                m.append(np.nanmean(np.abs(r))); sd.append(np.nanstd(np.abs(r)))
            ax.plot(kap, m, color=P["modes"][k], lw=lw, ls=ls, marker="o" if vname == "rotated" else None, ms=4.5,
                    mec=P["bg"], mew=1)
    ax.set_xscale("log")
    ax.set_xlabel("input covariance condition number κ")
    ax.set_ylabel("mean |error| in  $t_{1/2}$")
    ax.set_title("Non-whitened inputs", pad=10)
    ax.text(1.05, 0.47, "solid: covariance eigenvectors random\ndashed: aligned with the target's V\n"
            "(aligned + corrected formula: < 0.03% error)", fontsize=9.5, color=P["ink"], va="top")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    style.save(fig, FIG / "breaks.png", dpi=150)


def depth_figure():
    P = style.use("light")
    d = np.load(HERE / "cache/break_depth.npz")
    fig = plt.figure(figsize=(15, 5.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.3)
    u0 = float(d["u0"])
    for j, L in enumerate((2, 3, 4)):
        ax = fig.add_subplot(gs[0, j])
        t = d[f"dec_t_L{L}"]
        tt = np.linspace(0, t[-1], 500)
        for k in range(4):
            ax.plot(tt, sc.deep_mode(tt, S[k], u0, L) / S[k], color=P["theory"], lw=1.0, zorder=3)
            idx = np.arange(0, len(t), 12)
            ax.scatter(t[idx], d[f"dec_modes_L{L}"][idx, k] / S[k], s=12, color=P["modes"][k], edgecolor=P["bg"],
                       lw=0.5, zorder=4)
        ax.set_xlim(0, 12 if L == 2 else (12 if L == 3 else 12)); ax.set_ylim(-0.04, 1.08)
        ax.set_title(f"{L} weight matrices", pad=8)
        ax.set_xlabel("training time t")
        if j == 0:
            ax.set_ylabel("mode strength  u/s")
    fig.text(0.125, 0.965, "Decoupled init, $u_0 = 10^{-3}$: dots = gradient descent, lines = Saxe eq. 15   "
             "(depth makes each transition sharper and the wait longer)", fontsize=11, color=P["muted"])
    # scaling with s, random init: measured / balanced prediction
    ax = fig.add_subplot(gs[0, 3])
    s_many = d["s_many"]
    ax.axhspan(0.95, 1.05, color=P["rule"], alpha=0.55, lw=0)
    ax.axhline(1, color=P["muted"], lw=0.8)
    slopes = {}
    for j, L in enumerate((2, 3, 4)):
        tk = d[f"rnd_tk_L{L}"]
        c = P["modes"][[0, 2, 3][j]]
        sig = float(d[f"rnd_sigma_L{L}"]); W = int(d["width"])
        u0L = (sig ** 2 * W / 2) ** (L / 2)
        tg = np.linspace(0, 150, 30001)
        pred = np.array([sc.first_crossing(tg, sc.deep_mode(tg, si, u0L, L), si / 2) for si in s_many])
        ratio = tk / pred[None, :]
        ax.scatter(np.tile(s_many, (tk.shape[0], 1)).ravel(), ratio.ravel(), s=9, color=c, alpha=0.35, lw=0)
        ax.plot(s_many, np.nanmean(ratio, 0), color=c, lw=2, marker="o", ms=5, mec=P["bg"], mew=1)
        mean = np.nanmean(tk, 0)
        slopes[L] = (np.polyfit(np.log(s_many), np.log(mean), 1)[0], np.polyfit(np.log(s_many), np.log(pred), 1)[0])
        ax.text(s_many[0] * 1.1, np.nanmean(ratio, 0)[0], f"L={L}", ha="left", va="center", fontsize=10, color=P["ink"])
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(0.3, 6.0)
    ax.set_xlabel("mode strength s")
    ax.set_ylabel("measured / predicted  $t_{1/2}$")
    ax.set_title("Random init at depth: timing drifts with s", pad=8)
    txt = "\n".join(f"L={L}: slope of t½ vs s  {m:.2f}  (theory {p_:.2f})" for L, (m, p_) in slopes.items())
    ax.text(5.8, 2.3, txt, ha="right", va="top", fontsize=9, color=P["ink"], linespacing=1.4)
    print(slopes)
    style.save(fig, FIG / "depth.png", dpi=150)


if __name__ == "__main__":
    breaks_figure()
    depth_figure()
