"""Plotter sheet: the measured {10/k} star polygon of the C=10 train class means at ~N log-spaced
checkpoints, overdrawn in one ink (line weight grows with epoch), ideal star dashed.

  python plotter_stars.py c10 [--n 48] [--ink ink|spectral]
Each polygon is the Procrustes-aligned mean configuration at one checkpoint (rotation/reflection +
one global scale per checkpoint). ink=spectral colours each polygon by epoch along Spectral (declared,
sequential use: a labelled variant only).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nclib as N, artlib as A

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("--n", type=int, default=48); ap.add_argument("--ink", default="ink")
ap.add_argument("--res", type=int, default=2400)
a = ap.parse_args()
meta, mets, ck = N.load_run(a.tag)
E = np.array([e for e, _ in ck])
want = np.r_[0, np.geomspace(0.013, E.max(), a.n - 1)]
idx = sorted(set(int(np.argmin(np.abs(E - w))) for w in want))
confs = []
for i in idx:
    z = np.load(ck[i][1])
    al = N.Aligner(z["mu"], z["muG"])
    confs.append((ck[i][0], al.coords_means, al.residual))
F, planes = N.fourier_etf(len(meta["classes"]))
C = F.shape[0]
dark = a.ink == "spectral"
bg = "#0e0e12" if dark else A.PAPER
fig = A.canvas(a.res, int(a.res * 1.06), bg)
cm = plt.get_cmap("Spectral_r")
ext = 0.95
for q, k in enumerate([1, 2, 3, 4]):
    g = 0.03; s = (1 - 3 * g) / 2
    ax = fig.add_axes([g + (q % 2) * (s + g), 0.06 / 1.06 + (1 - q // 2) * (s + g) / 1.06 + g / 1.06 * 0, s, s / 1.06])
    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext); ax.set_axis_off(); ax.set_aspect("equal")
    c = planes[k]; o = np.arange(C + 1) % C
    for n, (e, m, r) in enumerate(confs):
        t = n / (len(confs) - 1)
        col = cm(t) if dark else (0.11, 0.11, 0.12, 0.25 + 0.6 * t)
        ax.plot(m[o, c[0]], m[o, c[1]], color=col, lw=(0.18 + 0.9 * t ** 3) * a.res / 2400, solid_joinstyle="miter")
    ax.plot(F[o, c[0]], F[o, c[1]], color="#c0392b" if not dark else "#ffffff", lw=0.5 * a.res / 2400, ls=(0, (6, 4)), alpha=0.8)
    ax.text(-ext, ext, f"{{10/{k}}}", color="#8a857a", fontsize=9 * a.res / 2400, family="serif", va="top")
txt = (f"{len(confs)} checkpoints, epoch 0 → {confs[-1][0]:.0f}, log-spaced; thin/faint = early, heavy = late; "
       f"dashed = ideal simplex ETF.  misfit {confs[0][2]:.2f} → {confs[-1][2]:.3f}")
fig.text(0.5, 0.02, txt, ha="center", color="#8a857a" if not dark else "#b9b4a6", fontsize=8 * a.res / 2400, family="serif")
print(A.save(fig, f"plotter_stars_{a.tag}_{a.ink}.png"))
