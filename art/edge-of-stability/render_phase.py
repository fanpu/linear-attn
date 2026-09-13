"""Phase portraits and return maps of self-stabilisation (render step).

usage: python render_phase.py <cache.npz> <tag> [models comma]
All plates use steps t > t_edge + 100 (after the run first reaches the edge).
r_t  = (lambda_1(t) - 2/eta) / (2/eta)          relative distance above the edge
A_t  = sqrt((c_t^2 + c_{t+1}^2) / 2)              phase-free amplitude of the oscillation coordinate c
       (windowed-PCA coordinate by default)

orbit_*   (r_t, log10 A_t) with a declared 9-step centred mean on both (removes the period-2 flicker of
          lambda_1 itself). night: colour = step; paper: single ink; spectral: colour = smoothed growth
          d log A / dt, split at 0 (declared Spectral split). If the quadratic model held, the orbit would
          rise right of the line and fall left of it.
growth_*  measured one-step growth g_t = log(A_{t+1}/A_t) against r_t for all step sizes on ONE axis,
          with binned medians (dots) and the quadratic-model prediction g = log|1 + 2r| (one curve serves all
          step sizes in these relative units).
return_*  burst return maps: successive burst peak amplitudes (A_n, A_{n+1}) and successive inter-burst
          intervals (tau_n, tau_{n+1}), consecutive points joined (cobweb idiom).
"""
import sys

from matplotlib.collections import LineCollection
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from eos_common import *  # noqa: F401,F403

INKS = ["#d53e4f", "#f08a3e", "#3aa17e", "#3a6fb0"]  # declared categorical inks per step size


def series(d, m):
    inv = float(d["invs"][m])
    lam = ffill(d["evals"][:, m].astype(float))
    c = braid(d, m)
    T = len(c)
    A = np.sqrt(0.5 * (c ** 2 + np.r_[c[1:], np.nan] ** 2))
    r = (lam[:, 0] - inv) / inv
    ab = np.flatnonzero((lam[:, 0] >= inv) & (np.arange(T) > 5))
    below = np.flatnonzero(lam[:, 0] < inv)
    first_below = below[0] if len(below) else 0
    ab = ab[ab > first_below]
    te = int(ab[0]) if len(ab) else T
    s = slice(te + 100, T - 2)
    return dict(inv=inv, t=np.arange(T)[s], r=r[s], A=A[s], te=te)


def plate_orbit(d, ms, style, tag, k=9):
    bg = {"night": NIGHT, "paper": PAPER, "spectral": "#0d0d12"}[style]
    fg = {"night": "#9a968a", "paper": "#6b675e", "spectral": "#8a877e"}[style]
    fig, axs = plt.subplots(1, len(ms), figsize=(5.0 * len(ms), 6.4), facecolor=bg, squeeze=False)
    for ax, m in zip(axs[0], ms):
        o = series(d, m)
        r = uniform_filter1d(o["r"], k, mode="nearest")
        la = uniform_filter1d(np.log10(np.maximum(o["A"], 1e-12)), k, mode="nearest")
        p = np.stack([r, la], 1)
        seg = np.stack([p[:-1], p[1:]], 1)
        if style == "night":
            tt = o["t"][:-1]
            c = plt.get_cmap("cmc.lajolla_r")((tt - tt.min()) / max(1, np.ptp(tt)) * 0.8 + 0.15)
            lc = LineCollection(seg, colors=c, linewidths=0.35, alpha=0.6)
        elif style == "paper":
            lc = LineCollection(seg, colors=INK, linewidths=0.22, alpha=0.4)
        else:
            g = np.diff(la)
            rgb = spectral_split(g[None, :], near_boundary="small")[0]
            lc = LineCollection(seg, colors=rgb, linewidths=0.4, alpha=0.75)
        ax.set_facecolor(bg)
        ax.add_collection(lc)
        ax.axvline(0, color={"night": "#e8e2d0", "paper": "#c0392b", "spectral": "#f0ead8"}[style], lw=0.6)
        ax.set_xlim(min(-0.02, np.percentile(r, 0.3) - 0.01), np.percentile(r, 99.7) + 0.01)
        ax.set_ylim(np.percentile(la, 0.3) - 0.1, np.percentile(la, 99.9) + 0.1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=fg, labelsize=7, length=2)
        ax.set_xlabel("(λ₁ − 2/η) / (2/η)", color=fg, fontsize=9, family=MONO)
        ax.set_title(f"η = 2/{o['inv']:.0f}", color=fg, fontsize=12, family=SERIF)
    axs[0, 0].set_ylabel("log₁₀ oscillation amplitude", color=fg, fontsize=9, family=MONO)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.01, 0.012, f"steps after t_edge+100 · {k}-step centred mean on both axes (declared) · vertical line = the edge · "
             + {"night": "colour = step (declared)", "paper": "single ink",
                "spectral": "colour = amplitude growth d log A/dt, split at 0 (declared Spectral split)"}[style],
             color=fg, fontsize=8, family=MONO)
    return save(fig, f"orbit_{tag}_{style}.png", dpi=300, facecolor=bg)


def plate_growth(d, ms, tag):
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    rr = np.linspace(-0.2, 0.3, 300)
    for i, m in enumerate(ms):
        o = series(d, m)
        g = np.log(o["A"][1:] / o["A"][:-1])
        r = o["r"][:-1]
        ok = np.isfinite(g) & np.isfinite(r)
        ax.scatter(r[ok], g[ok], s=0.4, c=INKS[i], alpha=0.12, lw=0)
        bins = np.linspace(np.percentile(r[ok], 1), np.percentile(r[ok], 99), 16)
        idx = np.digitize(r[ok], bins)
        bc = [(r[ok][idx == b].mean(), np.median(g[ok][idx == b])) for b in range(1, len(bins)) if np.sum(idx == b) > 30]
        bc = np.array(bc)
        ax.plot(bc[:, 0], bc[:, 1], "o-", color=INKS[i], ms=4, lw=1.0, label=f"η = 2/{o['inv']:.0f}  (binned median)")
    ax.plot(rr, np.log(np.abs(1 + 2 * rr)), color=INK, lw=1.0, ls="--", label="quadratic model  log|1 + 2r|")
    ax.axvline(0, color="#c0392b", lw=0.6)
    ax.axhline(0, color=INK, lw=0.4)
    ax.set_xlim(-0.12, 0.25)
    ax.set_ylim(-0.6, 0.6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("r = (λ₁ − 2/η) / (2/η)", family=MONO)
    ax.set_ylabel("measured growth per step  log(Aₜ₊₁ / Aₜ)", family=MONO)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.01, f"steps after t_edge+100 · A from the {coord_label(short=True)}", fontsize=8, family=MONO, color="#6b675e")
    return save(fig, f"growth_{tag}_paper.png", dpi=250, facecolor=PAPER)


def peaks(o):
    env = envelope(np.nan_to_num(o["A"]), 3)
    pk, _ = find_peaks(env, distance=15, prominence=0.3 * np.median(env))
    return o["t"][pk], env[pk]


def plate_return(d, ms, tag, style="paper"):
    bg, fg, ink = (PAPER, "#6b675e", INK) if style == "paper" else (NIGHT, "#9a968a", "#e8e2d0")
    fig, axs = plt.subplots(2, len(ms), figsize=(4.4 * len(ms), 9), facecolor=bg, squeeze=False)
    for j, m in enumerate(ms):
        o = series(d, m)
        tp, ap = peaks(o)
        la = np.log10(ap)
        tau = np.diff(tp)
        for row, (u, lab) in enumerate([(la, "log₁₀ Aₙ (burst peak)"), (tau, "τₙ (steps between bursts)")]):
            ax = axs[row, j]
            ax.set_facecolor(bg)
            col = INKS[j] if style == "night" else ink
            ax.plot(u[:-1], u[1:], "-", color=col, lw=0.35, alpha=0.5)
            ax.plot(u[:-1], u[1:], "o", color=col, ms=2.2, alpha=0.85, mew=0)
            lo, hi = np.percentile(u, 0.5), np.percentile(u, 99.5)
            pad = 0.06 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.plot([lo, hi], [lo, hi], color=fg, lw=0.4, ls=":")
            ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(colors=fg, labelsize=7, length=2)
            ax.set_xlabel(lab, color=fg, fontsize=8, family=MONO)
            if j == 0:
                ax.set_ylabel(lab.replace("ₙ", "ₙ₊₁"), color=fg, fontsize=8, family=MONO)
            if row == 0:
                ax.set_title(f"η = 2/{o['inv']:.0f}   ({len(tp)} bursts)", color=fg, fontsize=11, family=SERIF)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.01, 0.008, "bursts = peaks of the 7-step running max of the oscillation amplitude (prominence ≥ 0.3× median, ≥ 15 steps apart; "
             "declared detector) · dotted = identity", color=fg, fontsize=8, family=MONO)
    return save(fig, f"return_{tag}_{style}.png", dpi=250, facecolor=bg)


if __name__ == "__main__":
    f, tag = sys.argv[1], sys.argv[2]
    d = load(f)
    ms = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(len(d["invs"])))
    for st in ["night", "paper", "spectral"]:
        plate_orbit(d, ms, st, tag)
    plate_growth(d, ms, tag)
    plate_return(d, ms, tag, "paper")
    plate_return(d, ms, tag, "night")
