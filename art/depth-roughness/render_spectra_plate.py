"""Scientific plate: angular power per log-multipole, l(l+1)C_l/2pi, of the depth-L Heaviside kernels
(L = 1..12) and, inset, the regular activations. For Heaviside C_l ~ l^-(2+2^(1-L)) so the curves
flatten toward l^-2 (area-filling level sets) as depth grows."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

sp = np.load("cache/spectra.npz"); names = list(sp["names"]); C = sp["C"]
l = np.arange(C.shape[1])
SEP = "#6b3417"; PAP = "#f4efe4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["C059", "DejaVu Serif"], "font.size": 13})
fig = plt.figure(figsize=(12, 13), facecolor=PAP)
ax = fig.add_axes([0.1, 0.38, 0.84, 0.54], facecolor=PAP)
ends = {}
for L in range(1, 13):
    c = C[names.index(f"heaviside_L{L}")]
    m = (l >= 1) & (c > 1e-14)
    if L == 1:
        m &= (l % 2 == 1)
    y = l * (l + 1) * c / (2 * np.pi)
    ax.plot(l[m], y[m], color=SEP, lw=0.9 + 0.08 * L, alpha=0.35 + 0.05 * L)
    ends[L] = np.log10(y[m][-1])
order = sorted(ends, key=lambda k: ends[k])
pos, last = {}, -99
for k in order:
    pos[k] = max(ends[k], last + 0.12); last = pos[k]
for L in ends:
    ax.text(8192 * 1.12, 10 ** pos[L], f"L={L}   slope −{2**(1-L):.3g}", color=SEP, fontsize=10, va="center")
    ax.plot([8192 * 1.0, 8192 * 1.1], [10 ** ends[L], 10 ** pos[L]], color=SEP, lw=0.5, alpha=0.5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 8192 * 3.6)
for s in ax.spines.values(): s.set_color(SEP)
ax.tick_params(colors=SEP, which="both")
ax.set_xlabel("multipole l", color=SEP); ax.set_ylabel("l(l+1) C$_l$ / 2π", color=SEP)
ax.grid(True, which="major", color=SEP, alpha=0.12, lw=0.6)
fig.text(0.1, 0.955, "Plate I.  Angular power of the Heaviside network, by depth", fontsize=24, color=SEP)
fig.text(0.1, 0.932, "Infinite-width limit on S², Γ$_b$ = 0; kernel κ(u) = 1 − arccos(u)/π iterated L times; Legendre transform to l = 8192 (odd l only for L = 1).",
         fontsize=11.5, color=SEP, style="italic")
ax2 = fig.add_axes([0.1, 0.06, 0.84, 0.2], facecolor=PAP)
for a, ls in [("relu", "-"), ("gelu", "--"), ("tanh", "-."), ("sin", ":")]:
    for L in [1, 4, 8, 12]:
        c = C[names.index(f"{a}_L{L}")]
        m = (l >= 1) & (c > 1e-13)
        ax2.plot(l[m], (l * (l + 1) * c / (2 * np.pi))[m], color=SEP, ls=ls, lw=0.8 + 0.1 * L, alpha=0.4 + 0.04 * L)
    ax2.text(1.1, 1e-13 * 10 ** (["relu", "gelu", "tanh", "sin"].index(a) * 0), "", color=SEP)
ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlim(1, 8192 * 2.6); ax2.set_ylim(1e-12, 3)
for s in ax2.spines.values(): s.set_color(SEP)
ax2.tick_params(colors=SEP, which="both"); ax2.grid(True, color=SEP, alpha=0.12, lw=0.6)
ax2.set_xlabel("multipole l", color=SEP)
fig.text(0.1, 0.275, "Regular activations at L = 1, 4, 8, 12 (heavier = deeper): ReLU (solid, C$_l$ ~ l$^{-5}$ tail), GELU (dashed), tanh (dash-dot), sin (dotted).\nTheir spectra fall off a cliff and their level sets stay smooth curves.",
         fontsize=11.5, color=SEP, style="italic")
fig.savefig("gallery/plate_spectra.png", dpi=220, facecolor=PAP)
