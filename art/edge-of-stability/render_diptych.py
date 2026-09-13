"""Honesty plate: the braid in the rotating frame (current top eigenvector) vs the data frame (windowed PCA).

usage: python render_diptych.py <cache.npz> <model> <tag> t0 t1
Top: x_t = <theta_t - thetabar_t, u1(t)>, even/odd strands; ticks below it mark steps where the refreshed top
eigenvector turned by more than acos(0.95) (|<u1(t),u1(t-1)>| < 0.95, identity swap of near-degenerate
eigenvalues). Bottom: windowed-PCA coordinate over the same steps. Both measured; each panel has its own gain.
"""
import sys

from eos_common import *  # noqa: F401,F403


def main(f, m, tag, t0, t1):
    d = load(f)
    inv = float(d["invs"][m])
    t = np.arange(t0, t1)
    ov = d["u1_overlap_prev"][t0:t1, m]
    fig, axs = plt.subplots(2, 1, figsize=(18, 9), facecolor=PAPER)
    for ax, kind, title in [(axs[0], "u1", "rotating frame  ·  ⟨θₜ − θ̄ₜ, u₁(t)⟩ along the current top eigenvector"),
                            (axs[1], "pca", "data frame  ·  windowed-PCA oscillation coordinate")]:
        x = np.nan_to_num(braid(d, m, kind=kind)[t0:t1])
        A = np.percentile(np.abs(x), 99.5) * 1.15
        ax.set_facecolor(PAPER)
        ax.plot(t, x, color=INK, lw=0.25, alpha=0.4)
        ev = t % 2 == 0
        ax.plot(t[ev], x[ev], color="#c0392b", lw=0.9)
        ax.plot(t[~ev], x[~ev], color="#1f4e79", lw=0.9)
        ax.set_xlim(t0, t1 - 1)
        ax.set_ylim(-1.25 * A, A)
        ax.axis("off")
        ax.text(t0, A * 1.02, title, family=MONO, fontsize=11, color="#4a463f", va="bottom")
        if kind == "u1":
            sw = t[np.nan_to_num(ov, nan=1) < 0.95]
            ax.vlines(sw, -1.22 * A, -1.12 * A, color=INK, lw=0.6)
            ax.text(t1 - 1, -1.25 * A, f"ticks: eigenvector swaps |⟨u₁(t),u₁(t−1)⟩| < 0.95 ({len(sw)} of {len(t)} steps)",
                    family=MONO, fontsize=9, color="#6b675e", ha="right", va="top")
    fig.text(0.125, 0.04, f"η = 2/{inv:.0f}, steps {t0}–{t1 - 1} · red = even steps, blue = odd steps, grey = step-to-step "
             "zigzag · a strand swap is a crossing", family=MONO, fontsize=9, color="#6b675e")
    save(fig, f"diptych_frames_{tag}.png", dpi=250, facecolor=PAPER)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
