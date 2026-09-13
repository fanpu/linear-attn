"""Phase portraits of self-stabilisation (render step).

usage: python render_phase.py <cache.npz> <tag> [models comma] [t_start]

Orbit plane (all measured per step):
  horizontal  lambda_1(t) - 2/eta                           (above / below the edge)
  vertical    log10 A_t,  A_t = sqrt((x_t^2 + x_{t+1}^2)/2)   (phase-free oscillation amplitude)
The loop: above the edge the amplitude grows; large oscillation lowers the sharpness
(self-stabilisation, Damian et al. 2022); below the edge it decays; progressive sharpening pushes
lambda_1 back up. Colour in the spectral style = growth rate g_t = log A_{t+1} - log A_t, split at
g = 0 (declared aesthetic mapping: each side rank-normalised onto one half of Spectral).

Return map plate: g_t against lambda_1(t) - 2/eta, with the quadratic-model prediction
g = log|1 - eta*lambda_1| drawn as the one straight-ish curve.
"""
import sys

from matplotlib.collections import LineCollection

from eos_common import *  # noqa: F401,F403


def orbit_data(d, m, ts=30):
    inv = float(d["invs"][m])
    lam = ffill(d["evals"][:, m, 0].astype(float)[:, None])[:, 0]
    x = braid(d, m)
    A = np.sqrt(0.5 * (x[:-1] ** 2 + x[1:] ** 2))
    n = len(A)
    t = np.arange(n)
    ok = np.isfinite(A) & (A > 0) & (t >= ts)
    la = np.full(n, np.nan)
    la[ok] = np.log10(A[ok])
    g = np.full(n, np.nan)
    g[:-1] = (la[1:] - la[:-1]) * np.log(10)
    return dict(t=t, dl=lam[:n] - inv, la=la, g=g, inv=inv, eta=2 / inv, lam=lam[:n])


def segs(a, b):
    p = np.stack([a, b], 1)
    s = np.stack([p[:-1], p[1:]], 1)
    ok = np.all(np.isfinite(s), axis=(1, 2))
    return s, ok


def plate_orbit(d, ms, style, tag, ts=30, amin=-6):
    n = len(ms)
    bg = {"night": NIGHT, "paper": PAPER, "spectral": "#0d0d12"}[style]
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 6.2), facecolor=bg, squeeze=False)
    for ax, m in zip(axs[0], ms):
        o = orbit_data(d, m, ts)
        # start the orbit once the oscillation exists (amplitude above the float32 floor)
        s, ok = segs(o["dl"], o["la"])
        ok &= o["la"][:-1] > amin
        s = s[ok]
        tt = o["t"][:-1][ok]
        ax.set_facecolor(bg)
        if style == "night":
            c = plt.get_cmap("cmc.lajolla_r")((tt - tt.min()) / max(1, np.ptp(tt)) * 0.85 + 0.1)
            lc = LineCollection(s, colors=c, linewidths=0.35, alpha=0.55)
            fg = "#9a968a"
        elif style == "paper":
            lc = LineCollection(s, colors=INK, linewidths=0.25, alpha=0.35)
            fg = "#6b675e"
        else:
            gg = o["g"][:-1][ok]
            rgb = spectral_split(gg[None, :], near_boundary="small")[0]
            lc = LineCollection(s, colors=rgb, linewidths=0.4, alpha=0.7)
            fg = "#8a877e"
        ax.add_collection(lc)
        ax.axvline(0, color={"night": "#e8e2d0", "paper": "#c0392b", "spectral": "#f0ead8"}[style], lw=0.6)
        if len(s):
            xs = np.percentile(np.abs(s[:, 0, 0]), 99.5) * 1.1
            ax.set_xlim(-xs, xs)
            ax.set_ylim(np.percentile(s[:, 0, 1], 0.2) - 0.1, np.percentile(s[:, 0, 1], 99.9) + 0.2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=fg, labelsize=7, length=2)
        ax.set_xlabel("λ₁ − 2/η", color=fg, fontsize=9, family=MONO)
        ax.set_title(f"η = 2/{o['inv']:.0f}", color=fg, fontsize=11)
    axs[0, 0].set_ylabel("log₁₀ oscillation amplitude", color=fg, fontsize=9, family=MONO)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.01, 0.01, "each step t is one segment; the edge 2/η is the vertical line; "
             + {"night": "colour = step (declared)", "paper": "single ink",
                "spectral": "colour = amplitude growth rate, split at 0 (declared Spectral split)"}[style],
             color=fg, fontsize=8, family=MONO)
    return save(fig, f"orbit_{tag}_{style}.png", dpi=300, facecolor=bg)


def plate_return(d, ms, tag, ts=30, amin=-5):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    inks = ["#0078bf", "#ff48b0", "#00a95c", "#f15060", "#765ba7"]
    for i, m in enumerate(ms):
        o = orbit_data(d, m, ts)
        ok = np.isfinite(o["g"]) & (o["la"] > amin)
        ax.scatter(o["dl"][ok], o["g"][ok], s=0.3, c=inks[i % 5], alpha=0.25, lw=0, label=f"η = 2/{o['inv']:.0f}")
        lg = np.linspace(-0.3 * o["inv"], 0.3 * o["inv"], 400)
        ax.plot(lg, np.log(np.abs(1 - o["eta"] * (o["inv"] + lg))), color=inks[i % 5], lw=0.8)
    ax.axvline(0, color=INK, lw=0.5)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xlabel("λ₁ − 2/η", family=MONO)
    ax.set_ylabel("measured growth per step  log(A₍ₜ₊₁₎ / Aₜ)", family=MONO)
    lgd = ax.legend(markerscale=20, frameon=False)
    for h in lgd.legend_handles:
        h.set_alpha(1)
    return save(fig, f"return_{tag}_paper.png", dpi=250, facecolor=PAPER)


if __name__ == "__main__":
    f, tag = sys.argv[1], sys.argv[2]
    d = load(f)
    ms = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(len(d["invs"])))
    for st in ["night", "paper", "spectral"]:
        plate_orbit(d, ms, st, tag)
    plate_return(d, ms, tag)
