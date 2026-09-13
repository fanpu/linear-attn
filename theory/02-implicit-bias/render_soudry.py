"""Static plates for the logistic-regression sections.  Reads cache/soudry_*.npz.

  figures/soudry_crawl.png     (a) 2D: w(t) - w_hat log t -> w_tilde (Soudry Thm 4, closed form)
                               (b) d=50: angle * ln t flattens  (the 1/log t rate)
                               (c) margin gap: GD vs normalized GD variants, with rate guides
  figures/geometry_margins.png  GD / normalized GD / sign GD, normalized margins in L2 and Linf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style as S
from core import angle, margin, svm

S.light()

H = np.load("cache/soudry_hero.npz")
G = np.load("cache/soudry_gd.npz")
NG = np.load("cache/soudry_ngd.npz")
NS = np.load("cache/soudry_ngd_sqrt.npz")
SG = np.load("cache/soudry_sign.npz")
Z = G["Z"]
Wh = G["W_hat"]
Wl = SG["W_linf"]
Sn = Z.shape[0]

fig, axs = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw=dict(wspace=0.32))
# ---------------------------------------------------------------- (a) 2D residual
ax = axs[0]
t, W = H["t"], H["W"]
m = t >= 1.5
LT = np.log10(t[m])
rho = W[m] - np.log(t[m])[:, None] * H["w_hat"][None]
wt = H["w_tilde"]
for j, (col, lab) in enumerate([(S.BLUE, r"$\rho_1(t)$"), (S.ORANGE, r"$\rho_2(t)$")]):
    ax.axhline(wt[j], color=S.INK, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.semilogx(LT, rho[:, j], color=col, lw=2.4, zorder=2)
    ax.annotate(lab, (LT[-1], rho[-1, j]), xytext=(-2, 8), textcoords="offset points", ha="right", color=S.INK2, fontsize=11)
eta = float(H["gd_eta"])
tau = H["gd_steps"] * eta
sel = tau >= 1.5
rg = H["gd_W"][sel] - np.log(tau[sel])[:, None] * H["w_hat"][None]
for j, col in enumerate([S.BLUE, S.ORANGE]):
    ax.scatter(np.log10(tau[sel])[::4], rg[::4, j], s=18, color=col, edgecolor=S.PAPER, lw=1, zorder=3)
ax.set_xlim(0.18, 100)
ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(["$10^{1}$", "$10^{3}$", "$10^{10}$", "$10^{30}$", "$10^{100}$"])
ax.set_xlabel("time t  (axis: log of log t)")
ax.set_title(r"(a) 2D: the residual $\rho(t) = w(t) - \hat w \ln t$ converges to $\tilde w$")
ax.text(0.36, 0.97, r"dashed: $\tilde w$ predicted from the SVM duals," "\n" r"$e^{-y_n x_n^\top \tilde w} = \alpha_n$ for each support vector",
        transform=ax.transAxes, fontsize=9.5, color=S.INK2, va="top")
ax.text(0.36, 0.84, "dots: discrete GD (η = 4/σ²max, 10⁷ steps)\nsolid: gradient flow in log-time", transform=ax.transAxes, fontsize=9.5, color=S.INK2, va="top")
final_err = np.abs(rho[-1] - wt).max()
ax.text(0.98, 0.03, f"|ρ(10¹⁰⁰) − w̃| = {final_err:.0e}", transform=ax.transAxes, ha="right", fontsize=9, color=S.MUTED)
ax.set_ylim(-0.1, 1.45)

# ---------------------------------------------------------------- (b) d = 50: angle * ln t
ax = axs[1]
steps = G["steps"]
etas = G["eta"]
fs = G["flow_s"]
theta_gd = np.degrees(angle(G["W"], Wh[None]))  # T, S
for i in range(Sn):
    # discrete GD in flow-time units tau = eta * k
    tau = steps * etas[i]
    ok = tau >= 10
    ax.semilogx(np.log10(tau[ok]), theta_gd[ok, i] * np.log(tau[ok]), color=S.BLUE, lw=1.8, alpha=0.95)
    th_f = np.degrees(angle(G["flow_W"][i], Wh[i][None]))
    tf = fs / np.log(10)
    okf = tf >= np.log10(tau[-1])
    ax.semilogx(tf[okf], th_f[okf] * fs[okf], color=S.BLUE, lw=1.3, alpha=0.55, ls=(0, (1, 1.5)))
ax.set_xlim(1, 100)
ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(["$10^{1}$", "$10^{3}$", "$10^{10}$", "$10^{30}$", "$10^{100}$"])
ax.set_xlabel("time t  (axis: log of log t)")
ax.set_ylabel(r"angle to SVM (degrees) $\times \ln t$")
ax.set_title(r"(b) d = 50: angle $\times \ln t$ levels off, so angle $\propto 1/\ln t$")
ax.text(0.03, 0.97, "8 random datasets.  solid: GD, 10⁸ steps (float64)\ndotted: the same gradient flow continued in log-time",
        transform=ax.transAxes, fontsize=9.5, color=S.INK2, va="top")
ax.set_ylim(40, 100)
end_theta = np.degrees(angle(G["W"][-1], Wh)).mean()
ax.text(0.98, 0.03, f"mean angle after 10⁸ GD steps: {end_theta:.2f}°", transform=ax.transAxes, ha="right", fontsize=9, color=S.MUTED)

# ---------------------------------------------------------------- (c) margin gap rates
ax = axs[2]
gam = 1 / np.linalg.norm(Wh, axis=1)


def gap(D):
    W = D["W"]
    return np.array([gam[i] - margin(Z[i], W[:, i], "l2") for i in range(Sn)]).T / gam  # relative


series = [(G, S.BLUE, "GD", 1.0), (NS, S.AQUA, r"normalized GD, $\eta_t \propto 1/\sqrt{t}$", 1.0), (NG, S.VIOLET, "normalized GD, constant η", 1.0)]
for D, col, lab, _ in series:
    g = gap(D)
    st = D["steps"]
    med = np.median(np.maximum(g, 1e-16), axis=1)
    for i in range(Sn):
        ax.loglog(st, np.maximum(g[:, i], 1e-16), color=col, lw=0.8, alpha=0.25)
    ax.loglog(st, med, color=col, lw=2.2)
    yl = {"GD": 1e-5, r"normalized GD, $\eta_t \propto 1/\sqrt{t}$": 1e-5 / 4, "normalized GD, constant η": 1e-5 / 16}[lab]
    ax.plot([1.5, 4], [yl, yl], color=col, lw=2.4)
    ax.text(5.5, yl, lab, color=S.INK2, fontsize=10, va="center")
# rate guides (theory shapes), anchored at the late-time medians
g_gd = np.median(gap(G), 1); st = G["steps"]
k0 = np.searchsorted(st, 1e4)
ax.loglog(st[k0:], g_gd[k0] * np.log(st[k0]) / np.log(st[k0:]), color=S.INK, lw=1, ls=(0, (4, 3)))
ax.text(st[-1], g_gd[-1] * 1.6, r"$\propto 1/\ln t$", ha="right", fontsize=10, color=S.INK)
g_ns = np.median(gap(NS), 1); st2 = NS["steps"]; k1 = np.searchsorted(st2, 1e3)
ax.loglog(st2[k1:], g_ns[k1] * (np.log(st2[k1:]) / np.sqrt(st2[k1:])) / (np.log(st2[k1]) / np.sqrt(st2[k1])), color=S.INK, lw=1, ls=(0, (4, 3)))
ax.text(st2[-1] * 1.3, g_ns[-1] * 0.5, r"$\propto \ln t/\sqrt{t}$", ha="left", fontsize=10, color=S.INK)
g_ng = np.median(gap(NG), 1); st3 = NG["steps"]; k2 = np.searchsorted(st3, 1e3)
ax.loglog(st3[k2:], g_ng[k2] * st3[k2] / st3[k2:], color=S.INK, lw=1, ls=(0, (4, 3)))
ax.text(st3[-1] * 1.3, g_ng[-1] * 0.8, r"$\propto 1/t$", ha="left", fontsize=10, color=S.INK)
ax.set_ylim(1e-7, 2); ax.set_xlim(1, 1e9)
ax.set_xlabel("steps t")
ax.set_ylabel(r"relative margin gap $(\gamma^\star - \gamma(w_t))/\gamma^\star$")
ax.set_title("(c) Normalizing the step removes the crawl")
fig.savefig("figures/soudry_crawl.png", bbox_inches="tight")
print("wrote soudry_crawl.png")

# =============================================================== geometry: which margin?
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.25))
gam2 = gam
gaminf = np.array([margin(Z[i], Wl[i], "linf") for i in range(Sn)])
for ax, nrm, gstar, title in [(axs[0], "l2", gam2, r"$L_2$-normalized margin  $\min_n y_n x_n^\top w / \|w\|_2$"),
                              (axs[1], "linf", gaminf, r"$L_\infty$-normalized margin  $\min_n y_n x_n^\top w / \|w\|_\infty$")]:
    for D, col, lab in [(G, S.BLUE, "GD"), (NG, S.VIOLET, "normalized GD"), (SG, S.ORANGE, "sign GD")]:
        W = D["W"]
        r = np.array([margin(Z[i], W[:, i], nrm) / gstar[i] for i in range(Sn)]).T
        st = D["steps"]
        for i in range(Sn):
            ax.semilogx(st, r[:, i], color=col, lw=0.8, alpha=0.25)
        ax.semilogx(st, np.median(r, 1), color=col, lw=2.2)
        ax.annotate(lab, (st[-1], np.median(r, 1)[-1]), xytext=(6, 6 if lab == "normalized GD" else 0), textcoords="offset points", va="bottom" if lab == "normalized GD" else "center", fontsize=10, color=S.INK2)
    # theory: the level each method should saturate at
    other = np.array([margin(Z[i], (Wl if nrm == "l2" else Wh)[i], nrm) / gstar[i] for i in range(Sn)])
    ax.axhline(1, color=S.INK, lw=1, ls=(0, (4, 3)))
    ax.axhline(np.median(other), color=S.INK, lw=1, ls=(0, (1, 2)))
    ax.text(12, 1.01, "1 = the max-margin solution of this norm", fontsize=9.5, color=S.INK2, va="bottom")
    ax.text(1e4 if nrm == "l2" else 12, np.median(other) + (-0.012 if nrm == "l2" else 0.008), ("margin of the $L_\\infty$ SVM, measured in $L_2$" if nrm == "l2" else "margin of the $L_2$ SVM, measured in $L_\\infty$"),
            fontsize=9.5, color=S.INK2, va="top" if nrm == "l2" else "bottom")
    ax.set_ylim(0.55, 1.06)
    ax.set_xlim(1, 3e8)
    ax.set_xlabel("steps t")
    ax.set_title(title, fontsize=12)
axs[0].set_ylabel("fraction of the best possible margin")
fig.savefig("figures/geometry_margins.png", bbox_inches="tight")
print("wrote geometry_margins.png")
