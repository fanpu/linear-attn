"""Build-on figures: a softmax attention head trained on copy-and-transform, vs the reduced Saxe-style model."""
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import saxe_core as sc
import style

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"


def reduced_loss(s, u, al, T):
    A2 = al ** 2 + (1 - al) ** 2 / (T - 1)
    return 0.5 * np.sum(s[None, :] ** 2 - 2 * s[None, :] * u * al[:, None] + u ** 2 * A2[:, None], axis=1)


def showcase():
    P = style.use("light")
    d = np.load(HERE / "cache/attn_showcase.npz")
    r = np.load(HERE / "cache/attn_reduced_showcase.npz")
    t, tr = d["t"], r["t"]
    T = int(d["T"])
    s = d["S"][0][:3]
    iL, iU, iO = 0, 3, 4
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 9.6), sharex=True, gridspec_kw=dict(hspace=0.22, height_ratios=[1, 1.15, 0.8]))
    xmax = 45
    # (a) loss
    ax = axs[0]
    ax.plot(t, d["loss"][:, iU], color=P["muted"], lw=1.6, ls=(0, (3, 2)))
    ax.plot(t, d["loss"][:, iO], color=P["modes"][4], lw=1.6)
    for i in (0, 1, 2):
        ax.plot(t, d["loss"][:, i], color=style.ACCENT, lw=2.4 if i == 0 else 1.0, alpha=1 if i == 0 else 0.45)
    ax.plot(tr, reduced_loss(s, r[f"u_{iL}"], r[f"a1_{iL}"], T), color=P["theory"], lw=0.9)
    ax.plot(tr, reduced_loss(s, r[f"u_{iU}"], r[f"a1_{iU}"], T), color=P["theory"], lw=0.9)
    ax.set_yscale("log"); ax.set_ylim(3e-4, 12)
    ax.set_ylabel("loss")
    ax.text(44, 5.6, "attention frozen uniform", ha="right", va="bottom", fontsize=10, color=P["ink"])
    ax.text(3.2, 1.2e-3, "attention frozen on token 1\n(= a plain 2-layer linear net)", fontsize=10, color=P["ink"])
    ax.text(30.5, 2.2e-2, "learned attention\n(3 seeds)", fontsize=10, color=P["ink"])
    ax.set_title("A softmax attention head learning  $y = Mx_1$  (singular values of M: 3, 1.5, 0.75)", pad=10)
    # (b) OV singular values
    ax = axs[1]
    for k in range(3):
        c = P["modes"][[0, 2, 3][k]]
        ax.plot(t, d["sv_dov"][:, iU, k], color=c, lw=1.4, ls=(0, (3, 2)), alpha=0.7)
        ax.plot(t, d["sv_dov"][:, iL, k], color=c, lw=2.6)
        ax.plot(tr, r[f"u_{iL}"][:, k], color=P["theory"], lw=0.9)
        ax.axhline(s[k], color=P["rule"], lw=1, zorder=0)
        ax.text(xmax - 0.3, s[k] + 0.06, f"$s_{k + 1}={s[k]:g}$", ha="right", va="bottom", fontsize=10, color=P["muted"])
    ax.set_ylabel("singular values of  $\\Delta W_{OV}$")
    ax.set_ylim(-0.1, 4.6)
    ax.text(0.5, 4.2, "solid: learned attention   ·   dashed: attention frozen uniform   ·   thin black: reduced model",
            fontsize=10, color=P["muted"])
    # (c) attention
    ax = axs[2]
    for i in (0, 1, 2):
        ax.plot(t, d["a1"][:, i], color=style.ACCENT, lw=2.4 if i == 0 else 1.0, alpha=1 if i == 0 else 0.45)
    ax.plot(tr, r[f"a1_{iL}"], color=P["theory"], lw=0.9)
    ax.axhline(1 / T, color=P["rule"], lw=1, zorder=0)
    ax.set_ylabel("attention on token 1")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("training time  t  (learning rate × steps)")
    ax.set_xlim(0, xmax)
    # phase annotations
    t_ov = sc.first_crossing(t, d["sv_dov"][:, iL, 0], s[0] / 2)
    t_at = sc.first_crossing(t, d["a1"][:, iL], 0.5)
    for ax in axs:
        ax.axvline(t_ov, color=P["muted"], lw=0.8, ls=(0, (1, 2)))
        ax.axvline(t_at, color=P["muted"], lw=0.8, ls=(0, (1, 2)))
    axs[2].text(t_ov / 2, 0.55, "(1) value/output circuit learns\nthe strong mode through\nblurry attention (T× slower)",
                ha="center", fontsize=10, color=P["ink"])
    axs[2].text((t_ov + t_at) / 2, 0.55, "(2) attention logit\ngrows from its\nown saddle", ha="center", fontsize=10,
                color=P["ink"])
    axs[2].text(t_at + 8, 0.35, "(3) attention locks on,\nweak modes finish fast", ha="left", fontsize=10, color=P["ink"])
    style.save(fig, FIG / "attn_showcase.png", dpi=150)


def fit_slope(x, y):
    ok = np.isfinite(y) & (y > 0)
    X = np.log(x[ok]); Y = np.log(y[ok])
    A = np.vstack([X, np.ones_like(X)]).T
    coef, res, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    se = np.sqrt(np.sum(resid ** 2) / max(1, len(X) - 2) / np.sum((X - X.mean()) ** 2))
    return coef[0], se


def scaling():
    P = style.use("light")
    d = np.load(HERE / "cache/attn_sweep_rank1.npz")
    r = np.load(HERE / "cache/attn_reduced_sweep_rank1.npz")
    t, tr = d["t"], r["t"]
    S, att = d["S"][:, 0], d["attn"]
    rows = dict(ov=[], qk=[], orc=[], ov_r=[], qk_r=[], orc_r=[], s_l=[], s_o=[])
    for i in range(len(S)):
        m, mr = d["modes"][:, i, 0], r[f"u_{i}"][:, 0]
        if att[i] == "learn":
            tov = sc.first_crossing(t, m, S[i] / 2); tat = sc.first_crossing(t, d["a1"][:, i], 0.5)
            tov_r = sc.first_crossing(tr, mr, S[i] / 2); tat_r = sc.first_crossing(tr, r[f"a1_{i}"], 0.5)
            rows["ov"].append(tov); rows["qk"].append(tat - tov); rows["ov_r"].append(tov_r); rows["qk_r"].append(tat_r - tov_r)
            rows["s_l"].append(S[i])
        elif att[i] == "oracle":
            rows["orc"].append(sc.first_crossing(t, m, S[i] / 2)); rows["orc_r"].append(sc.first_crossing(tr, mr, S[i] / 2))
            rows["s_o"].append(S[i])
    R = {k: np.array(v) for k, v in rows.items()}
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw=dict(wspace=0.28, width_ratios=[1.25, 1]))
    ax = axs[0]
    series = [("qk", "(2) attention (QK) escape", P["modes"][1], "s"), ("ov", "(1) OV mode through uniform attention", P["modes"][0], "o"),
              ("orc", "attention fixed on token 1 (plain Saxe)", P["modes"][3], "^")]
    out = {}
    for key, lab, c, mk in series:
        x = R["s_o"] if key == "orc" else R["s_l"]
        y = R[key]; yr = R[key + "_r"]
        order = np.argsort(x)
        xu = np.unique(x)
        yr_u = np.array([np.mean(yr[x == v]) for v in xu])
        ax.plot(xu, yr_u, color=P["theory"], lw=1.0, zorder=2)
        ax.scatter(x, y, s=46, marker=mk, color=c, edgecolor=P["bg"], lw=1.2, zorder=3)
        sl, se = fit_slope(x, y)
        out[key] = (sl, se)
        pos = dict(qk=(1.05, 330, "left"), ov=(0.47, 15.5, "left"), orc=(1.25, 5.2, "left"))[key]
        ax.text(pos[0], pos[1], f"{lab}\nfitted slope {sl:.2f} ± {se:.2f}", fontsize=10, color=P["ink"], va="center",
                ha=pos[2])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.42, 5)
    ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("target strength  s  (rank-1 M)")
    ax.set_ylabel("duration of the phase")
    ax.set_title("Two saddles, two scaling laws", pad=10)
    ax.text(0.45, 0.62, "markers: trained attention head (3 seeds)   ·   thin lines: reduced model", fontsize=9.5,
            color=P["muted"])

    # second panel: weak mode after the attention click
    d = np.load(HERE / "cache/attn_sweep_mode3.npz")
    r = np.load(HERE / "cache/attn_reduced_sweep_mode3.npz")
    t, tr = d["t"], r["t"]
    T = int(d["T"])
    ax = axs[1]
    xs, ys, yrs, piece = [], [], [], []
    for i in range(len(d["attn"])):
        if d["attn"][i] != "learn":
            continue
        w = d["S"][i, 1]
        t2 = sc.first_crossing(t, d["modes"][:, i, 1], w / 2)
        t2r = sc.first_crossing(tr, r[f"u_{i}"][:, 1], w / 2)
        xs.append(w); ys.append(t2); yrs.append(t2r)
    xs, ys, yrs = map(np.array, (xs, ys, yrs))
    ax.scatter(xs, ys, s=46, color=P["modes"][3], edgecolor=P["bg"], lw=1.2, zorder=3, label="trained head")
    xu = np.unique(xs)
    ax.plot(xu, [np.mean(yrs[xs == v]) for v in xu], color=P["theory"], lw=1.0, label="reduced model")
    # uniform-attention-only Saxe prediction and oracle prediction for reference
    u0 = 8e-4
    ww = np.linspace(0.28, 1.25, 100)
    ax.plot(ww, T * np.log(ww / u0 - 1) / (2 * ww), color=P["muted"], lw=1.2, ls=(0, (3, 2)))
    ax.text(0.33, T * np.log(0.33 / u0 - 1) / 0.66 * 0.96, "if attention stayed uniform:  $T\\ln(s/u_0)/2s$", fontsize=9.5,
            color=P["muted"], va="top")
    t_click = np.mean([sc.first_crossing(t, d["a1"][:, i], 0.5) for i in range(len(d["attn"])) if d["attn"][i] == "learn"])
    ax.axhline(t_click, color=P["rule"], lw=1.2)
    ax.text(0.29, t_click - 2.8, "attention locks on (driven by $s_1 = 3$)", ha="left", fontsize=9.5, color=P["muted"])
    ax.set_xlabel("strength of the weaker mode  $s_2$  (with $s_1 = 3$)")
    ax.set_ylabel("time the weak mode is half-learned")
    ax.set_title("Weak modes wait for the attention", pad=10)
    ax.set_ylim(15, 75)
    style.save(fig, FIG / "attn_scaling.png", dpi=150)
    return out


if __name__ == "__main__":
    showcase()
    print(scaling())
