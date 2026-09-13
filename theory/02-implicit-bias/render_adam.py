"""Build-on figure: Adam's L_inf phase and its lifetime.  Reads cache/adam_*.npz.

  figures/adam.png   (a) 2D data: direction of w vs t for several eps, with the predicted crossover t_x and the
                         'GD with lr/eps' continuation (log-time gradient flow) after the last Adam step
                     (b) the beta2 throttle: log sqrt(vhat) decays at rate (1 - beta2)/2 per step
                     (c) d = 50: crossover time vs ln(1/eps), measured vs predicted slope
                     (d) d = 50: where Adam's direction sits between the L_inf and L_2 max-margin solutions
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import style as S
from core import flow_logtime, margin, angle

S.light()
EPS_CMAP = LinearSegmentedColormap.from_list("eps", ["#b9a6e8", "#7a67d6", "#4a3aa7", "#2a2170"])


def eps_color(e):
    if e == 0:
        return S.ORANGE
    return EPS_CMAP((np.log10(e) + 16) / 12)


def eps_label(e):
    return "ε = 0" if e == 0 else f"ε = 1e{int(np.log10(e))}"


def adam_rate(lr, gamma, b1=0.9, b2=0.999):
    """Self-consistent exponential decay rate r of Adam's gradient scale in the sign-descent phase (eps = 0, constant lr).
    If g(t) ~ e^{-rt}, the EMAs give |m/sqrt(v)| = (1-b1)/(1-b1 e^r) * sqrt((1-b2 e^{2r})/(1-b2)); the margin (and hence
    -log g) then grows by gamma * lr * that per step.  Solve r = gamma lr f(r) on (0, -ln(b2)/2)."""
    from scipy.optimize import brentq
    f = lambda r: r - gamma * lr * (1 - b1) / (1 - b1 * np.exp(r)) * np.sqrt(max(1 - b2 * np.exp(2 * r), 0) / (1 - b2))
    return brentq(f, 1e-14, -np.log(b2) / 2 * (1 - 1e-13))


def predicted_crossover(D, s, e_idx, gamma, anchor=-2.0):
    """t_x = t_a + ln(sqrt(vhat)(t_a)/eps)/r, anchored where the eps=0 run first has log10 sqrt(vhat) < anchor."""
    lv = D["log10_sqrtv"][:, s, 0]
    k = np.where(lv < anchor)[0][0]
    r = adam_rate(float(D["lr"]), gamma, b2=float(D["b2"]) if "b2" in D.files else 0.999)
    return D["steps"][k] + (lv[k] - np.log10(D["eps"][e_idx])) * np.log(10) / r


def crossover(D, s, e_idx):
    """first recorded step where sqrt(vhat) (mean over coords) drops below eps."""
    eps = D["eps"][e_idx]
    lv = D["log10_sqrtv"][:, s, e_idx]
    k = np.where(lv < np.log10(eps))[0]
    return D["steps"][k[0]] if len(k) else np.nan


fig = plt.figure(figsize=(15, 10.2))
gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.24)

# ------------------------------------------------------------------------------------------------ (a)
ax = fig.add_subplot(gs[0, 0])
D = np.load("cache/adam_geom_lr0.01_const.npz")
st, W, eps = D["steps"], D["W"][:, 0], D["eps"]
lr, b2 = float(D["lr"]), 0.999
Z = D["Z"][0]
for e_i, e in enumerate(eps):
    th = np.degrees(np.arctan2(W[:, e_i, 1], W[:, e_i, 0]))
    ax.semilogx(st, th, color=eps_color(e), lw=2.2)
    if e > 0:
        # continuation: once eps dominates, Adam ~ GD with step lr/eps -> gradient flow in tau = (lr/eps)(t - T)
        w_end = W[-1, e_i]
        T = st[-1]
        s_end = np.log(1e40 * lr / e)
        s_eval = np.linspace(np.log(1.0), s_end, 300)
        # flow_logtime integrates from tau = 1 (s = 0) starting at w_end: tau counts flow time since the last Adam step
        s_f, W_f = flow_logtime(Z, s_end, w0=w_end, s_eval=s_eval)
        t_f = T + np.exp(s_f) * e / lr
        thf = np.degrees(np.arctan2(W_f[:, 1], W_f[:, 0]))
        ok = t_f < 1e30
        ax.semilogx(t_f[ok], thf[ok], color=eps_color(e), lw=1.2, ls=(0, (1, 1.6)))
        tx = predicted_crossover(D, 0, e_i, 1.2)
        ax.scatter([tx], [np.interp(np.log(tx), np.log(st), th)], s=46, marker="|", color=S.INK, lw=1.6, zorder=5)
    if e == 0:
        ax.annotate(eps_label(e), (st[-1], th[-1]), xytext=(6, -2), textcoords="offset points", ha="left", fontsize=9.5, color=S.INK2)
    else:
        ax.annotate(eps_label(e), (t_f[ok][-1], thf[ok][-1]), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5, color=S.INK2)
ax.axhline(45, color=S.ORANGE, lw=1, ls=(0, (5, 4)))
ax.axhline(63.435, color=S.BLUE, lw=1, ls=(0, (5, 4)))
ax.text(1.5, 45.4, "$L_\\infty$ max-margin direction", fontsize=9.5, color=S.INK2)
ax.text(1.5, 63.8, "$L_2$ max-margin direction", fontsize=9.5, color=S.INK2)
ax.set_xlim(1, 1e37); ax.set_ylim(43, 66)
ax.set_xlabel("Adam steps t"); ax.set_ylabel("direction of w (degrees)")
ax.set_title("(a) Adam leaves the $L_\\infty$ solution once $\\sqrt{\\hat v}$ falls below ε")
ax.text(0.98, 0.04, "ticks: predicted crossover $t_\\times$ (see text)\n"
        "dotted: after the last Adam step, gradient flow with step lr/ε", transform=ax.transAxes, ha="right", fontsize=9, color=S.INK2)
ax.set_xticks([1, 1e5, 1e10, 1e15, 1e20, 1e25, 1e30])

# ------------------------------------------------------------------------------------------------ (b)
ax = fig.add_subplot(gs[0, 1])
rows_b = []
for fn, lrv, b2v, col in [("cache/adam_geom_lr0.01_const_b20.99.npz", 1e-2, 0.99, S.DEPTH_COLORS[1]),
                          ("cache/adam_geom_lr0.01_const.npz", 1e-2, 0.999, S.DEPTH_COLORS[2]),
                          ("cache/adam_geom_lr0.01_const_b20.9999.npz", 1e-2, 0.9999, S.DEPTH_COLORS[3]),
                          ("cache/adam_geom_lr0.001_const.npz", 1e-3, 0.999, S.AQUA)]:
    Db = np.load(fn)
    stb, lv = Db["steps"], Db["log10_sqrtv"][:, 0, 0] * np.log(10)
    r = adam_rate(lrv, 1.2, b2=b2v)
    x = r * stb
    k = np.where((lv < -20) & (lv > -300))[0]
    meas = -np.polyfit(stb[k], lv[k], 1)[0]
    b = np.median(lv[k] + x[k])
    m = lv > -700
    ax.plot(x[m], lv[m] - b, color=col, lw=[8, 5.5, 3.2, 1.4][len(rows_b)], solid_capstyle="round")
    rows_b.append((lrv, b2v, r, meas, col))
xx = np.array([0, 350])
ax.plot(xx, -xx, color=S.INK, lw=1.1, ls=(0, (4, 3)))
ax.set_xlim(0, 300); ax.set_ylim(-300, 30)
ax.set_xlabel(r"$r_{\mathrm{pred}}\cdot t$   (steps, rescaled by the predicted rate)")
ax.set_ylabel(r"$\ln\sqrt{\hat v}$  (shifted)")
ax.set_title(r"(b) Adam's gradient scale decays at a predictable rate $r$")
ytxt = -175
ax.text(8, ytxt, "     lr        β₂        predicted r     measured r", fontsize=9.5, color=S.INK, family=S.MONO)
for i_, (lrv, b2v, r, meas, col) in enumerate(rows_b):
    yy = ytxt - 22 * (i_ + 1)
    ax.plot([9, 22], [yy + 5, yy + 5], color=col, lw=2.6)
    ax.text(26, yy, f"{lrv:<8g}{b2v:<9g}{r:<15.4e}{meas:.4e}", fontsize=9.5, color=S.INK2, family=S.MONO)
ax.text(150, -20, "all runs collapse onto slope −1 (dashed)", fontsize=9.5, color=S.INK2)

# ------------------------------------------------------------------------------------------------ (c) & (d)
axc = fig.add_subplot(gs[1, 0])
axd = fig.add_subplot(gs[1, 1])
for fn, lrv, mk in [("cache/adam_gauss_lr0.001_const.npz", 1e-3, "o"), ("cache/adam_gauss_lr0.01_const.npz", 1e-2, "s")]:
    try:
        G = np.load(fn)
    except FileNotFoundError:
        continue
    Zg, st_g, eps_g = G["Z"], G["steps"], G["eps"]
    Sn = Zg.shape[0]
    ginf = np.array([margin(Zg[s], G["W_linf"][s], "linf") for s in range(Sn)])
    tx = np.array([[crossover(G, s_, e) for e in range(1, len(eps_g))] for s_ in range(Sn)])
    tp = np.array([[predicted_crossover(G, s_, e, ginf[s_]) for e in range(1, len(eps_g))] for s_ in range(Sn)])
    colc = S.AQUA if lrv == 1e-3 else S.VIOLET
    axc.scatter(tp.ravel(), tx.ravel(), s=30, marker=mk, color=colc, alpha=0.75, lw=0, label=f"lr = {lrv:g}")
    print("lr", lrv, "median |log10 measured/predicted crossover|", np.nanmedian(np.abs(np.log10(tx / tp))))
    # (d): position between Linf and L2 solutions for lr = 0.01 (the throttled, practical regime)
    if lrv == 1e-2:
        W2, Wi = G["W_l2"], G["W_linf"]
        for e_i, e in enumerate(eps_g):
            Wt = G["W"][:, :, e_i]  # T S d
            a2 = np.degrees(angle(Wt, W2[None]))
            ai = np.degrees(angle(Wt, Wi[None]))
            pos = ai / (ai + a2)  # 0 at Linf solution, 1 at L2 solution
            axd.semilogx(st_g, np.median(pos, 1), color=eps_color(e), lw=2.2)
            axd.annotate(eps_label(e), (st_g[-1], np.median(pos, 1)[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9.5, color=S.INK2)
lim = [3e3, 3e6]
axc.loglog(lim, lim, color=S.INK, lw=1, ls=(0, (4, 3)))
axc.set_xlim(lim); axc.set_ylim(lim)
axc.legend(loc="lower right", fontsize=9.5)
axc.set_xlabel(r"predicted crossover step $t_a + \ln(\sqrt{\hat v}(t_a)/\varepsilon)\,/\,r_{\mathrm{pred}}$")
axc.set_ylabel(r"crossover step $t_\times$  (first $t$ with $\sqrt{\hat v} < \varepsilon$)")
axc.set_title("(c) d = 50: when does ε take over? Predicted vs measured")
axc.text(0.03, 0.93, "8 datasets × 5 values of ε (1e-16 … 1e-4); dashed: y = x", transform=axc.transAxes, fontsize=9, color=S.INK2, va="top")
axd.axhline(0, color=S.ORANGE, lw=1, ls=(0, (5, 4))); axd.axhline(1, color=S.BLUE, lw=1, ls=(0, (5, 4)))
axd.text(1.5, 0.03, "$L_\\infty$ max-margin", fontsize=9.5, color=S.INK2)
axd.text(1.5, 0.95, "$L_2$ max-margin", fontsize=9.5, color=S.INK2, va="top")
axd.set_ylim(-0.05, 1.05); axd.set_xlim(1, 1e8)
axd.set_xlabel("Adam steps t  (lr = 0.01, β₂ = 0.999)")
axd.set_ylabel(r"$\angle(w, w_\infty) \,/\, (\angle(w, w_\infty) + \angle(w, w_2))$")
axd.set_title("(d) d = 50: which max-margin solution is Adam closer to?")
fig.savefig("figures/adam.png", bbox_inches="tight")
print("wrote adam.png")
