"""The three-layer print: 2/eta hairline, lambda_max stroke, braid ribbon (render step).

usage: python render_print.py <cache.npz> <model index> <tag> [t0 t1]
Layers (all measured, per step t, no smoothing):
  hairline   2/eta
  strokes    lambda_1 (bright), lambda_2, lambda_3 (dim): top Hessian eigenvalues at theta_t
  braid      oscillation coordinate c_t: by default the windowed-PCA coordinate (eos_common.pca_braid);
             EOS_COORD=u1 gives <theta_t - thetabar_t, u1(t)> along the current top eigenvector. Even steps form one strand, odd steps the other; the thin zigzag joins
             consecutive steps. Strands swap (a crossing) when the period-2 oscillation slips phase.
Styles: night, paper (single ink + one red hairline), riso (two spot inks, misregistered),
spectral (zigzag coloured by lambda_1 - 2/eta, split at 0: declared aesthetic mapping).
"""
import sys

from matplotlib.collections import LineCollection

from eos_common import *  # noqa: F401,F403

STYLES = {
    "night": dict(bg=NIGHT, hair="#e8e2d0", lam1="#f4ead2", lam23="#6d6a62", zig="#8d8a80",
                  even="#f2a541", odd="#58b4c4", text="#9a968a", zig_a=0.55),
    "paper": dict(bg=PAPER, hair="#c0392b", lam1=INK, lam23="#9a958a", zig=INK,
                  even=INK, odd=INK, text="#6b675e", zig_a=0.45),
    "riso": dict(bg=PAL.RISO_PAPER, hair="#ff48b0", lam1="#0078bf", lam23="#8fb8de", zig="#0078bf",
                 even="#ff48b0", odd="#0078bf", text="#0078bf", zig_a=0.5),
    "spectral": dict(bg="#0d0d12", hair="#f0ead8", lam1="#f0ead8", lam23="#5d5a55", zig=None,
                     even="#fbf8ef", odd="#fbf8ef", text="#8a877e", zig_a=0.9),
}


def draw(d, m, style, t0=0, t1=None, W=18, H=8, label=True):
    S = STYLES[style]
    inv = float(d["invs"][m])
    lam = ffill(d["evals"][:, m])
    x = braid(d, m)
    T = len(x) if t1 is None else t1
    t = np.arange(t0, T)
    lam, x = lam[t0:T], x[t0:T]
    x = np.where(np.isfinite(x), x, 0.0)
    fig = plt.figure(figsize=(W, H), facecolor=S["bg"])
    at = fig.add_axes([0.04, 0.56, 0.92, 0.36], facecolor=S["bg"])
    ab = fig.add_axes([0.04, 0.10, 0.92, 0.40], facecolor=S["bg"])
    for a in (at, ab):
        a.set_xlim(t0, T - 1)
        a.axis("off")
    # --- layer 1: the law
    at.axhline(inv, color=S["hair"], lw=0.5, zorder=1)
    # --- layer 2: sharpness strokes
    for j, (c, lw) in enumerate([(S["lam1"], 0.9), (S["lam23"], 0.5), (S["lam23"], 0.5)]):
        at.plot(t, lam[:, j], color=c, lw=lw, zorder=3 - j * 0.5, solid_joinstyle="round")
    lo = np.nanmin(lam[:, 2])
    hi = max(inv + 0.35 * (inv - lo), np.nanmax(lam[:, 0]) + 0.04 * (inv - lo))
    at.set_ylim(lo - 0.05 * (inv - lo), hi)
    # --- layer 3: braid
    A = np.percentile(np.abs(x), 99.7) * 1.15 + 1e-12
    ab.set_ylim(-A, A)
    seg = np.stack([np.stack([t[:-1], x[:-1]], 1), np.stack([t[1:], x[1:]], 1)], 1)
    if style == "spectral":
        s = lam[:-1, 0] - inv
        rgb = spectral_split(s[None, :], near_boundary="small")[0]
        lc = LineCollection(seg, colors=rgb, linewidths=0.35, alpha=S["zig_a"])
    else:
        lc = LineCollection(seg, colors=S["zig"], linewidths=0.25, alpha=S["zig_a"])
    ab.add_collection(lc)
    ev, od = t % 2 == 0, t % 2 == 1
    off = 0.0
    if style == "riso":  # declared misregistration: the odd plate shifted right by 0.6 px-equivalent
        off = 0.0009 * (T - t0)
    ab.plot(t[ev], x[ev], color=S["even"], lw=0.7, alpha=0.95)
    ab.plot(t[od] + off, x[od], color=S["odd"], lw=0.7, alpha=0.95)
    if style == "riso":
        at.plot(t + off, lam[:, 0], color=S["even"], lw=0.5, alpha=0.6)
    if label:
        kw = dict(color=S["text"], fontsize=9, family=MONO, transform=fig.transFigure)
        fig.text(0.04, 0.05, f"fc-tanh 3072-200-200-10  ·  CIFAR-10 first 5000  ·  full-batch GD, MSE  ·  "
                 f"η = 2/{inv:.0f}  ·  steps {t0}–{T - 1}", **kw)
        fig.text(0.04, 0.022, "hairline 2/η   ·   strokes λ₁ λ₂ λ₃ of the loss Hessian   ·   " +
                 coord_label() + ", even / odd steps", **kw)
        at.text(t0, inv, " 2/η", color=S["hair"], fontsize=9, family=MONO, va="bottom")
    return fig


if __name__ == "__main__":
    f, m, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    t0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    t1 = int(sys.argv[5]) if len(sys.argv) > 5 else None
    styles = sys.argv[6].split(",") if len(sys.argv) > 6 else list(STYLES)
    d = load(f)
    for st in styles:
        fig = draw(d, m, st, t0, t1)
        save(fig, f"print_{tag}_{st}.png", dpi=300, facecolor=fig.get_facecolor())
