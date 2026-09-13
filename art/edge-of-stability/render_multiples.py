"""Four step sizes as small multiples (render step).

usage: python render_multiples.py <cache.npz> [t0 t1]
ladder_*   one plate, absolute lambda axis: the four 2/eta hairlines and the four lambda_1 strokes over
           the whole run; each stroke climbs (progressive sharpening) and then clings to its own hairline.
multiples_* four stacked rows over the same step window [t0, t1): top strip lambda_1..3 relative to the
           edge, (lambda - 2/eta)/(2/eta), on ONE shared scale for all rows; below it the braid (windowed-PCA
           oscillation coordinate, even/odd strands) with a per-row gain (declared: the oscillation
           amplitude differs between step sizes; the gain is printed).
           spectral: the thin step-to-step zigzag coloured by lambda_1 - 2/eta split at 0 (declared).
"""
import sys

from matplotlib.collections import LineCollection

from eos_common import *  # noqa: F401,F403

STY = {
    "paper": dict(bg=PAPER, hair="#c0392b", l1=INK, l23="#a39e92", zig=INK, ev=INK, od=INK, fg="#6b675e"),
    "night": dict(bg=NIGHT, hair="#e8e2d0", l1="#f4ead2", l23="#5d5a55", zig="#8d8a80", ev="#f2a541",
                  od="#58b4c4", fg="#9a968a"),
    "spectral": dict(bg="#0d0d12", hair="#f0ead8", l1="#f0ead8", l23="#55524d", zig=None, ev="#fbf8ef",
                     od="#fbf8ef", fg="#8a877e"),
}
INKS = ["#f46d43", "#fee08b", "#66c2a5", "#3288bd"]  # declared categorical inks (from Spectral) per step size


def ladder(d, style):
    S = STY[style]
    fig = plt.figure(figsize=(16, 9), facecolor=S["bg"])
    ax = fig.add_axes([0.06, 0.1, 0.83, 0.84], facecolor=S["bg"])
    T = len(d["loss"])
    t = np.arange(T)
    for m, inv in enumerate(d["invs"]):
        lam = ffill(d["evals"][:, m, 0].astype(float)[:, None])[:, 0]
        c = S["l1"] if style == "paper" else INKS[m] if style == "spectral" else S["l1"]
        ax.axhline(inv, color=S["hair"], lw=0.5, alpha=0.9)
        ax.plot(t, lam, color=c, lw=0.45, alpha=0.95)
        ax.text(T * 1.003, inv, f"2/η = {inv:.0f}", color=S["fg"], fontsize=10, family=MONO, va="center")
    ax.set_xlim(0, T)
    ax.set_ylim(0, max(d["invs"]) * 1.12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=S["fg"], labelsize=9, length=2)
    ax.set_xlabel("GD step", color=S["fg"], family=MONO)
    ax.set_ylabel("λ₁  (top Hessian eigenvalue)", color=S["fg"], family=MONO)
    fig.text(0.06, 0.02, "fc-tanh 3072-200-200-10 · CIFAR-10 first 5000 · full-batch GD, MSE · four step sizes, one "
             "seed · hairlines 2/η, strokes measured λ₁ every step" +
             (" · stroke colours declared" if style == "spectral" else ""), color=S["fg"], fontsize=9, family=MONO)
    return save(fig, f"ladder_{style}.png", dpi=250, facecolor=S["bg"])


def multiples(d, style, t0, t1):
    S = STY[style]
    M = len(d["invs"])
    fig = plt.figure(figsize=(16, 4.2 * M), facecolor=S["bg"])
    t = np.arange(t0, t1)
    rel_all = []
    for m in range(M):
        inv = d["invs"][m]
        lam = ffill(d["evals"][:, m].astype(float))[t0:t1]
        rel_all.append((lam - inv) / inv)
    lo = min(np.nanpercentile(r[:, 2], 1) for r in rel_all)
    hi = max(np.nanmax(r[:, 0]) for r in rel_all)
    rowh = 1.0 / M
    for m in range(M):
        inv = d["invs"][m]
        y0 = 1 - (m + 1) * rowh
        at = fig.add_axes([0.05, y0 + 0.60 * rowh, 0.9, 0.30 * rowh], facecolor=S["bg"])
        ab = fig.add_axes([0.05, y0 + 0.10 * rowh, 0.9, 0.46 * rowh], facecolor=S["bg"])
        for a in (at, ab):
            a.set_xlim(t0, t1 - 1)
            a.axis("off")
        rel = rel_all[m]
        at.axhline(0, color=S["hair"], lw=0.5)
        for j, (c, lw) in enumerate([(S["l1"], 0.8), (S["l23"], 0.45), (S["l23"], 0.45)]):
            at.plot(t, rel[:, j], color=c, lw=lw, zorder=3 - j)
        at.set_ylim(lo, hi * 1.05)
        x = np.nan_to_num(braid(d, m)[t0:t1])
        A = np.percentile(np.abs(x), 99.7) * 1.1 + 1e-12
        ab.set_ylim(-A, A)
        seg = np.stack([np.stack([t[:-1], x[:-1]], 1), np.stack([t[1:], x[1:]], 1)], 1)
        if style == "spectral":
            # rank normalisation pooled per row (each row has its own edge)
            rgb = spectral_split(rel[:-1, 0][None, :], near_boundary="small")[0]
            ab.add_collection(LineCollection(seg, colors=rgb, linewidths=0.5, alpha=0.95))
        else:
            ab.add_collection(LineCollection(seg, colors=S["zig"], linewidths=0.25, alpha=0.45))
        ev = t % 2 == 0
        ab.plot(t[ev], x[ev], color=S["ev"], lw=0.8)
        ab.plot(t[~ev], x[~ev], color=S["od"], lw=0.8)
        fig.text(0.05, y0 + 0.92 * rowh, f"η = 2/{inv:.0f}", color=S["fg"], fontsize=13, family=SERIF)
        fig.text(0.95, y0 + 0.92 * rowh, f"braid gain ±{A:.1e}", color=S["fg"], fontsize=9, family=MONO, ha="right")
    fig.text(0.05, 0.004, f"steps {t0}–{t1 - 1} · strips: (λ₁,λ₂,λ₃ − 2/η)/(2/η), one shared scale, hairline = the edge · "
             f"{coord_label(short=True)}, even/odd strands, per-row gain" +
             (" · zigzag colour = λ₁ − 2/η split at 0 (declared Spectral)" if style == "spectral" else ""),
             color=S["fg"], fontsize=9, family=MONO)
    return save(fig, f"multiples_{t0}_{style}.png", dpi=200, facecolor=S["bg"])


if __name__ == "__main__":
    f = sys.argv[1]
    d = load(f)
    T = len(d["loss"])
    t0 = int(sys.argv[2]) if len(sys.argv) > 2 else T - 600
    t1 = int(sys.argv[3]) if len(sys.argv) > 3 else T
    for st in STY:
        ladder(d, st)
        multiples(d, st, t0, t1)
