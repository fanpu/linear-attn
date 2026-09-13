"""Verification sheet (fractals doc section 11): box counting across zoom levels, null model,
precision and resolution checks, threshold dependence.

  python render_verification.py
"""
import json, os
import numpy as np
from render_common import *

RB = json.load(open(os.path.join(CACHE, "fractal_report_B_N100_sync.json")))
RA = json.load(open(os.path.join(CACHE, "fractal_report_A_N100_tau.json")))
W, H = 3000, 2200
fig = fig_px(W, H, bg=PAPER)
fig.text(0.04, 0.965, "Verification: is the finite-width frontier fractal?  (erf MLP, N = 100, depth 1000)",
         fontsize=22, color=INK, va="center")
c_sync, c_tau, c_null, c_f32 = "#b8473a", "#2f6aa3", INK, "#8e9c94"

# (a) local slope vs zoom
ax = fig.add_axes([0.06, 0.55, 0.40, 0.36]); ax.set_facecolor(PAPER)
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
ax.set_xscale("log"); ax.set_ylim(0.8, 2.15)
ax.set_xlabel("zoom factor (window side 4 / zoom)"); ax.set_ylabel("local box-counting slope (box 2 to R/8 px)")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_title("(a) slope inside each boundary-centred window", loc="left", fontsize=13)

# (b) stitched N(eps)
ax = fig.add_axes([0.55, 0.55, 0.40, 0.36]); ax.set_facecolor(PAPER)
for R, col, lab in ((RB, c_sync, "sync frontier"), (RA, c_tau, "L = 1e-5")):
    e = np.array(R["stitched"]["eps"]); n = np.array(R["stitched"]["logN"])
    ax.plot(1 / e, np.exp(n), ".", color=col, ms=5, label=f"{lab}: global slope {R['stitched_slope']:.2f} over {np.log10(e.max()/e.min()):.1f} decades")
x = np.logspace(0, 8, 10)
for sl, ls in ((1, ":"), (2, "--")):
    ax.plot(x, 3 * x ** sl, ls, color=INK, lw=0.6, label=f"slope {sl}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2, 1e8); ax.set_ylim(1, 1e17)
ax.set_xlabel("1 / eps  (eps in units of the full [0,4] window)"); ax.set_ylabel("N(eps), stitched across zoom levels")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("(b) stitched box count (upper-biased: windows at max mixing)", loc="left", fontsize=13)

# (c) precision
ax = fig.add_axes([0.06, 0.10, 0.40, 0.34]); ax.set_facecolor(PAPER)
for R, col, lab in ((RB, c_sync, "sync"), (RA, c_tau, "L = 1e-5")):
    z = [d["zoom"] for d in R["levels"]]
    ax.plot(z, [100 * d.get("mismatch_f32", np.nan) for d in R["levels"]], "o-", color=col, label=f"{lab}: float32 vs float64")
    zp = [d["zoom"] for d in R["levels"] if "mismatch_pert" in d]
    ax.plot(zp, [100 * d["mismatch_pert"] for d in R["levels"] if "mismatch_pert" in d], "s--", color=col, mfc="none",
            label=f"{lab}: float64 vs float64 with inputs perturbed 1e-13")
ax.set_xscale("log"); ax.set_ylabel("% pixels with a different outcome"); ax.set_xlabel("zoom factor")

ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("(c) precision floor", loc="left", fontsize=13)

# (d) threshold scan
ax = fig.add_axes([0.55, 0.10, 0.40, 0.34]); ax.set_facecolor(PAPER)
cm = plt.get_cmap("cmc.batlow")
for d in RA["levels"]:
    t = np.array([x[0] for x in d["tau_scan"]]); s = np.array([x[1] if x[1] is not None else np.nan for x in d["tau_scan"]], float)
    ax.plot(t, s, "-", color=cm(d["level"] / 9), lw=1.2, label=f"x{d['zoom']:,.0f}")
ax.axvspan(1e-5, 1, color="#e6d9c3", alpha=0.5, zorder=0); ax.text(2e-5, 0.85, "paper's threshold range (max taken)", fontsize=8)
ax.set_xscale("log"); ax.set_ylim(0.6, 2.1); ax.set_xlabel("threshold tau on L"); ax.set_ylabel("local slope")
ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower left")
ax.set_title("(d) slope vs threshold tau (L = 1e-5 chain windows)", loc="left", fontsize=13)
fig.text(0.04, 0.03, "Local slopes: least-squares fit of log N vs log(1/box) for box sizes 2 ... R/8 px on the 2x2-mixing edge cells of the binary map (same rule as the paper's code).\n"
         "Stitched curve: level k+1 counts multiplied by the number of occupied (R/4)-boxes at level k; windows chosen at maximal mixing, so biased toward dense regions.",
         fontsize=10.5, color=INK)
for a in fig.axes:
    a.tick_params(labelsize=9)
savefig(fig, "verification_sheet.png")
print("ok")
