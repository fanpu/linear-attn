"""Verification sheet (fractals doc section 11): box counting across zoom levels, null model,
precision and resolution checks, threshold dependence.

  python render_verification.py
"""
import json, os
import numpy as np
from render_common import *

RB = json.load(open(os.path.join(CACHE, "fractal_report_B_N100_sync.json")))
RA = json.load(open(os.path.join(CACHE, "fractal_report_A_N100_tau.json")))
W, H = 4200, 2300
fig = fig_px(W, H, bg=PAPER)
fig.text(0.04, 0.965, "Verification: is the finite-width frontier fractal?  (erf MLP, N = 100, depth 1000)",
         fontsize=22, color=INK, va="center")
c_sync, c_tau, c_null, c_f32 = "#b8473a", "#2f6aa3", INK, "#8e9c94"

# (a) local slope vs zoom
ax = fig.add_axes([0.045, 0.55, 0.27, 0.36]); ax.set_facecolor(PAPER)
for R, col, lab in ((RB, c_sync, "sync frontier (pair merged within 1000 layers?)"), (RA, c_tau, "paper threshold L = 1e-5")):
    z = [d["zoom"] for d in R["levels"]]
    ax.plot(z, [d["slope"] for d in R["levels"]], "o-", color=col, lw=1.4, label=lab + ", float64")
    ax.plot(z, [d.get("slope_f32", np.nan) for d in R["levels"]], "x--", color=col, lw=0.8, alpha=0.6, label=lab + ", float32")
    for rs, mk in (("512", "s"), ("1024", "D")):
        zz = [d["zoom"] for d in R["levels"] if rs in d["rescheck"]]
        ss = [d["rescheck"][rs]["slope_fine"] for d in R["levels"] if rs in d["rescheck"]]
        if zz:
            ax.plot(zz, ss, mk, mfc="none", mec=col, ms=9, label=f"{lab.split(' (')[0]}, same windows at {rs} px")
zn = [4.0 ** d["level"] for d in RB["null"]]
ax.plot(zn, [d["slope"] for d in RB["null"]], "^-", color=c_null, lw=1.2, label="null model: infinite width, same pipeline")
ax.axhline(1, color=INK, lw=0.4, ls=":"); ax.axhline(2, color=INK, lw=0.4, ls=":")
ax.set_xscale("log"); ax.set_ylim(0.05, 2.15); ax.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0])
ax.set_xlabel("zoom factor (window side 4 / zoom)"); ax.set_ylabel("local box-counting slope (box 2 to R/8 px)")
ax.legend(fontsize=7.5, frameon=False, loc="lower right", ncol=2, columnspacing=0.8)
ax.set_title("(a) slope inside each boundary-centred window", loc="left", fontsize=13)

# (b) stitched N(eps)
ax = fig.add_axes([0.71, 0.55, 0.27, 0.36]); ax.set_facecolor(PAPER)
for R, col, lab in ((RB, c_sync, "sync frontier"), (RA, c_tau, "L = 1e-5")):
    e = np.array(R["stitched"]["eps"]); n = np.array(R["stitched"]["logN"])
    ax.plot(1 / e, np.exp(n), ".", color=col, ms=5, label=f"{lab}: global slope {R['stitched_slope']:.2f} over {np.log10(e.max()/e.min()):.1f} decades")
x = np.logspace(0, 8, 10)
for sl, ls in ((1, ":"), (2, "--")):
    ax.plot(x, 3 * x ** sl, ls, color=INK, lw=0.6, label=f"slope {sl}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2, 1e8); ax.set_ylim(1, 1e17)
ax.set_xlabel("1 / eps  (eps in units of the full [0,4] window)"); ax.set_ylabel("N(eps), stitched across zoom levels")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("(c) stitched box count (upper-biased: windows at max mixing)", loc="left", fontsize=13)

# (c) precision
ax = fig.add_axes([0.045, 0.10, 0.27, 0.34]); ax.set_facecolor(PAPER)
for R, col, lab in ((RB, c_sync, "sync"), (RA, c_tau, "L = 1e-5")):
    z = [d["zoom"] for d in R["levels"]]
    ax.plot(z, [100 * d.get("mismatch_f32", np.nan) for d in R["levels"]], "o-", color=col, label=f"{lab}: float32 vs float64")
    zp = [d["zoom"] for d in R["levels"] if "mismatch_pert" in d]
    ax.plot(zp, [100 * d["mismatch_pert"] for d in R["levels"] if "mismatch_pert" in d], "s--", color=col, mfc="none",
            label=f"{lab}: float64 vs float64 with inputs perturbed 1e-13")
ax.set_xscale("log"); ax.set_ylabel("% pixels with a different outcome"); ax.set_xlabel("zoom factor")

ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("(d) precision floor", loc="left", fontsize=13)

# (d) threshold scan
ax = fig.add_axes([0.375, 0.10, 0.27, 0.34]); ax.set_facecolor(PAPER)
cm = plt.get_cmap("cmc.batlow")
for d in RA["levels"]:
    t = np.array([x[0] for x in d["tau_scan"]]); s = np.array([x[1] if x[1] is not None else np.nan for x in d["tau_scan"]], float)
    ax.plot(t, s, "-", color=cm(d["level"] / 9), lw=1.2, label=f"x{d['zoom']:,.0f}")
ax.axvspan(1e-5, 1, color="#e6d9c3", alpha=0.5, zorder=0); ax.text(2e-5, 0.85, "paper's threshold range (max taken)", fontsize=8)
ax.set_xscale("log"); ax.set_ylim(0.6, 2.1); ax.set_xlabel("threshold tau on L"); ax.set_ylabel("local slope")
ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower left")
ax.set_title("(e) slope vs threshold tau (L = 1e-5 chain windows)", loc="left", fontsize=13)

# (b) neighbour correlation of the label vs zoom
ax = fig.add_axes([0.375, 0.55, 0.27, 0.36]); ax.set_facecolor(PAPER)
for R, col, lab in ((RB, c_sync, "sync frontier"), (RA, c_tau, "paper threshold L = 1e-5")):
    ax.plot([d["zoom"] for d in R["levels"]], [d["adj_corr"] for d in R["levels"]], "o-", color=col, lw=1.4, label=lab)
ax.plot(zn, [d["adj_corr"] for d in RB["null"]], "^-", color=c_null, lw=1.2, label="null model (infinite width)")
ax.axhline(0, color=INK, lw=0.4, ls=":")
ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("zoom factor"); ax.set_ylabel("correlation of the label between adjacent pixels")
ax.text(1.3, 0.04, "0 = neighbouring pixels independent: the map is unresolved noise at this grid", fontsize=8.5, color=INK)
ax.legend(fontsize=8, frameon=False, loc="center left")
ax.set_title("(b) is the structure resolved? (256 px windows)", loc="left", fontsize=13)

# (f) resolution check: box counts at matched physical box sizes, 256 vs 512 vs 1024 grids
ax = fig.add_axes([0.71, 0.10, 0.27, 0.34]); ax.set_facecolor(PAPER)
mk = {"512": "s", "1024": "D"}
for R, col in ((RB, c_sync),):
    levs = [d for d in R["levels"] if d["rescheck"]]
    for j, d in enumerate(levs):
        cc = plt.get_cmap("cmc.batlow")(d["level"] / 9)
        for rs, v in d["rescheck"].items():
            phys = np.array(v["sizes_coarse_px"], float)
            ax.plot(1 / phys, v["counts_coarse"], "o-", color=cc, lw=1.2, ms=5)
            ax.plot(1 / phys, v["counts_fine"], mk[rs], color=cc, mfc="none", ms=10, mew=1.2)
        ax.plot([], [], "o-", color=cc, label=f"x{d['zoom']:,.0f}: 256 px (line)  vs  " + ", ".join(f"{rs} px" for rs in d["rescheck"]))
ax.plot([], [], "s", mfc="none", color=INK, label="same window at 512 px"); ax.plot([], [], "D", mfc="none", color=INK, label="same window at 1024 px")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("1 / box size (in 256-grid pixels; the finer grid uses 2x or 4x more pixels per box)"); ax.set_ylabel("occupied boxes N")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("(f) resolution check (sync): matched physical box sizes", loc="left", fontsize=13)
fig.text(0.04, 0.03, "Local slopes: least-squares fit of log N vs log(1/box) for box sizes 2 ... R/8 px on the 2x2-mixing edge cells of the binary map (same rule as the paper's code).\n"
         "Stitched curve: level k+1 counts multiplied by the number of occupied (R/4)-boxes at level k; windows chosen at maximal mixing, so biased toward dense regions.",
         fontsize=10.5, color=INK)
for a in fig.axes:
    a.tick_params(labelsize=9)
savefig(fig, "verification_sheet.png")
print("ok")
