"""Figure: mode-by-mode analytic sigmoids with measured dots (toy target s = 5, 2, 1, 0.5)."""
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import saxe_core as sc
import style

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def main():
    d = np.load(HERE / "cache/reproduce.npz")
    t, s, u0 = d["t"], d["s"], float(d["dec_u0"])
    P = style.use("light")
    fig = plt.figure(figsize=(12.5, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[1.55, 1], hspace=0.42, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    tt = np.linspace(0, t[-1], 1200)
    for a, sa in enumerate(s):
        c = P["modes"][a]
        # random-init seeds, faint
        for b in range(d["rnd_modes"].shape[1]):
            ax.plot(t, d["rnd_modes"][:, b, a] / sa, color=c, lw=0.8, alpha=0.18, zorder=1)
        ax.plot(tt, sc.sigmoid_mode(tt, sa, u0) / sa, color=P["theory"], lw=1.0, zorder=3)
        idx = np.arange(0, len(t), 9)
        ax.scatter(t[idx], d["dec_modes"][idx, a] / sa, s=16, color=c, edgecolor=P["bg"], linewidth=0.7, zorder=4)
        th = sc.t_half(sa, u0)
        ax.plot([th, th], [-0.04, 0.5], color=c, lw=1, ls=(0, (1, 2)), zorder=2)
        ax.text(th + 0.12, 0.08, f"$s={sa:g}$\n$t_{{1/2}}={th:.2f}$", color=P["ink"], fontsize=10,
                va="bottom", ha="left")
    ax.set_xlim(0, t[-1]); ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("training time  $t$  (learning rate × steps)")
    ax.set_ylabel("mode strength  $u_\\alpha(t)\\,/\\,s_\\alpha$")
    ax.set_title("Each singular mode is learned along its own sigmoid, strongest first", pad=12)
    ax.text(0.455, 0.56, "thin black: closed form (Saxe et al. 2014, eq. 12)\n"
            "dots: gradient descent, decoupled init, float64\n"
            "faint lines: 16 small random inits", transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color=P["muted"], linespacing=1.5)

    # loss staircase
    ax2 = fig.add_subplot(gs[1, 0])
    const = float(d["loss_const"])
    loss_th = 0.5 * sum((sa - sc.sigmoid_mode(tt, sa, u0)) ** 2 for sa in s)
    for b in range(d["rnd_loss"].shape[1]):
        ax2.plot(t, d["rnd_loss"][:, b] + const, color=P["muted"], lw=0.8, alpha=0.25)
    ax2.plot(tt, loss_th, color=P["theory"], lw=1.0)
    idx = np.arange(0, len(t), 9)
    ax2.scatter(t[idx], d["dec_loss"][idx] + const, s=10, color=style.ACCENT, edgecolor=P["bg"], lw=0.5, zorder=3)
    ths = [0.0] + [sc.t_half(sa, u0) for sa in s] + [t[-1] + 2]
    names = ["nothing learned", "1 mode learned", "2 modes", "3 modes", ""]
    for k in range(1, 4):
        lv = 0.5 * np.sum(s[k:] ** 2)
        ax2.text(ths[k] + 0.6, lv * 1.3, names[k], ha="left", va="bottom", fontsize=9.5, color=P["muted"])
    ax2.set_yscale("symlog", linthresh=0.1)
    ax2.set_yticks([0, 0.1, 1, 10]); ax2.set_yticklabels(["0", "0.1", "1", "10"])
    ax2.set_xlim(0, t[-1])
    ax2.set_xlabel("training time  $t$"); ax2.set_ylabel("loss")
    ax2.set_title("…so the loss is a staircase of plateaus and drops", pad=10)

    # t_half vs 1/s
    ax3 = fig.add_subplot(gs[1, 1])
    inv = np.linspace(0.15, 2.1, 200)
    ax3.plot(inv, sc.t_half(1 / inv, u0), color=P["theory"], lw=1.0)
    for a, sa in enumerate(s):
        th = sc.first_crossing(t, d["dec_modes"][:, a], sa / 2)
        ax3.scatter([1 / sa], [th], s=64, color=P["modes"][a], edgecolor=P["bg"], lw=1.5, zorder=3)
        thr = [sc.first_crossing(t, d["rnd_modes"][:, b, a], sa / 2) for b in range(d["rnd_modes"].shape[1])]
        ax3.scatter(np.full(len(thr), 1 / sa) + 0.06, thr, s=9, color=P["modes"][a], alpha=0.5, lw=0, zorder=2)
    ax3.text(1.05, sc.t_half(1 / 1.3, u0) + 2.2, "$t_{1/2} = \\frac{1}{2s}\\ln\\!\\left(\\frac{s}{u_0}-1\\right)$",
             fontsize=12, color=P["ink"])
    ax3.set_xlabel("1 / mode strength  $1/s$"); ax3.set_ylabel("time to half-learned  $t_{1/2}$")
    ax3.set_title("Transition time grows like 1/s (times a log)", pad=10)
    ax3.set_xlim(0, 2.2); ax3.set_ylim(0, 13)
    style.save(fig, FIG / "modes_overlay.png", dpi=150)


if __name__ == "__main__":
    main()
