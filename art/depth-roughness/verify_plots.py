"""Verification figures + numbers (cache -> gallery/verify_*.png, cache/verify_summary.json)."""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import fit_dim, theory, ACTS

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
TXT, TXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": TXT2, "axes.labelcolor": TXT,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2, "savefig.dpi": 200})
summary = {}
bc = json.load(open("cache/boxcount_tiles.json"))
cal = json.load(open("cache/calibration.json"))

# ---- V1: dimension vs depth (tile protocol) ------------------------------------------------------
Ls = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(8.5, 5.2))
th = 2 - 2.0 ** -Ls
exp_ = [cal[f"tile_H{2.0**-L:.5f}"]["D_meas"] if f"tile_H{2.0**-L:.5f}" in cal else np.nan for L in Ls]
meas = [fit_dim(np.array(bc["n2048"][f"heaviside_L{L}"]["sizes"]), np.array(bc["n2048"][f"heaviside_L{L}"]["counts"]), 4, 64)[0] for L in Ls]
meas4 = [fit_dim(np.array(bc["n4096"][f"heaviside_L{L}"]["sizes"]), np.array(bc["n4096"][f"heaviside_L{L}"]["counts"]), 8, 128)[0]
         if f"heaviside_L{L}" in bc["n4096"] else np.nan for L in Ls]
ax.plot(Ls, th, color=TXT, ls="--", lw=1.5)
ax.text(10.15, th[-1], "paper: 2 - 2$^{-L}$", color=TXT, va="center")
ax.plot(Ls, exp_, color=S2, lw=2)
ax.text(8.15, exp_[7] - 0.035, "estimator on fields of\nexactly that dimension", color=TXT2, va="top", fontsize=9.5)
ax.plot(Ls, meas, "o", color=S1, ms=8, mec=SURF, mew=2)
ax.plot(Ls, meas4, "s", color=S1, ms=7, mfc="none", mew=1.6)
ax.text(1.25, meas[0] - 0.01, "Heaviside GP sample, 2048² (4-64 px)", color=TXT2, va="top", fontsize=9.5)
ax.text(3.2, meas4[2] - 0.05, "4096², l ≤ 8192 (8-128 px)", color=TXT2, fontsize=9.5)
reg = {}
for a in ["relu", "gelu", "tanh", "sin"]:
    reg[a] = [fit_dim(np.array(bc["n2048"][f"{a}_L{L}"]["sizes"]), np.array(bc["n2048"][f"{a}_L{L}"]["counts"]), 4, 64)[0] for L in Ls]
    ax.plot(Ls, reg[a], "o", color="#9a9890", ms=5, alpha=0.8)
rbf = fit_dim(np.array(bc["n2048"]["rbf"]["sizes"]), np.array(bc["n2048"]["rbf"]["counts"]), 4, 64)[0]
ax.text(1.0, 1.1, "ReLU, GELU, tanh, sin (Kac-Rice class): dimension 1", color=TXT2, fontsize=9.5)
ax.set_xlabel("depth L"); ax.set_ylabel("box-counting dimension of level set")
ax.set_xticks(Ls); ax.set_ylim(0.9, 2.08); ax.set_xlim(0.6, 11.9)
ax.set_title("Measured dimension vs depth, same random draw", loc="left", color=TXT, fontsize=13)
fig.tight_layout(); fig.savefig("gallery/verify_dimension_vs_depth.png"); plt.close(fig)
summary["tile"] = dict(L=Ls.tolist(), theory=th.tolist(), estimator_expectation=exp_, heaviside_2048=meas,
                       heaviside_4096=meas4, regular=reg, rbf_null=rbf)

# ---- V2: multi-scale windows ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.4))
ms = {}
cols = {"heaviside_L1": S1, "heaviside_L2": S2, "heaviside_L3": S3, "heaviside_L6": S4}
for k in ["heaviside_L1", "heaviside_L2", "heaviside_L3", "heaviside_L4", "heaviside_L6", "relu_L2", "rbf"]:
    files = sorted(glob.glob(f"cache/multiscale_{k}_s*.npz"))
    if not files:
        continue
    allD = []
    for f in files:
        d = np.load(f)
        allD.append([fit_dim(d["sizes"], c, 8, 64)[0] for c in d["counts"]])
        F = d["F"]
    allD = np.array(allD)
    m = allD.mean(0)
    col = cols.get(k, "#9a9890")
    ax.plot(F, m, "-o", color=col, ms=4, lw=1.6 if k in cols else 1.2)
    lab = k.replace("heaviside_L", "Heaviside L=").replace("relu_L2", "ReLU L=2").replace("rbf", "RBF (smooth null)")
    ax.text(F[-1] / 1.6, m[-1], lab, color=TXT2 if k not in cols else col, va="center", fontsize=9.5)
    if k.startswith("heaviside"):
        L = int(k.split("_L")[1])
        e = cal.get(f"window_H{2.0**-L:.5f}")
        ms[k] = dict(F=F.tolist(), D_mean_per_window=m.tolist(), D_all_windows_mean=float(allD.mean()),
                     D_all_windows_sd=float(allD.std()), n_windows=int(allD.size), theory=2 - 2.0**-L,
                     estimator_expectation=e["D_meas"] if e else None, estimator_sd=e["D_sd"] if e else None)
    else:
        ms[k] = dict(F=F.tolist(), D_mean_per_window=m.tolist(), D_all_windows_mean=float(allD.mean()), n_windows=int(allD.size))
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xlabel("window side (radians on S²)  →  zooming in"); ax.set_ylabel("box D in window (8-64 px)")
ax.set_ylim(0.9, 1.9); ax.set_xlim(2, 3e-9)
ax.set_title("Twelve nested windows, 6.6 decades: the roughness does not run out", loc="left", fontsize=13)
fig.tight_layout(); fig.savefig("gallery/verify_multiscale.png"); plt.close(fig)
summary["multiscale"] = ms

# ---- V3: width cutoff: local slope vs physical box size, and collapse in eps * n -----------------
def local_slopes(counts, F, npx, smin, smax):
    out = []
    for c, Fw, nw in zip(counts, F, npx):
        for k in range(len(c) - 1):
            s = 2 ** k
            if s < smin or 2 * s > smax or c[k + 1] <= 0:
                continue
            out.append((s * np.sqrt(2) * Fw / nw, np.log2(c[k] / c[k + 1])))
    return np.array(out)

def binned(pts, xs, width=0.25):
    lx = np.log10(xs)
    edges = np.arange(np.floor(lx.min()), np.ceil(lx.max()) + width, width)
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lx >= a) & (lx < b)
        if m.sum() >= 2:
            cx.append(10 ** ((a + b) / 2)); cy.append(pts[m, 1].mean())
    return np.array(cx), np.array(cy)

wc = {}
ramp = {256: "#a9c9f0", 1024: "#6fa3e6", 4096: "#2a78d6", 16384: "#1b4f94", 64: "#d6e5f7", 65536: "#0f2f5c"}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for f in sorted(glob.glob("cache/nets_zoom_L2_n*.npz"), key=lambda q: int(q.split("_n")[1][:-4])):
    n = int(f.split("_n")[1][:-4]); d = np.load(f)
    pts = local_slopes(d["counts"], d["F"], d["npx"], 2, 64)
    x, y = binned(pts, pts[:, 0])
    axs[0].plot(x, y, "-", color=ramp.get(n, S1), lw=1.8)
    axs[0].text(x[-1] * 1.25, y[-1], f"n={n:,}", color=TXT2, fontsize=8.5, ha="right", va="center")
    x2, y2 = binned(pts, pts[:, 0] * n)
    axs[1].plot(x2, y2, "-", color=ramp.get(n, S1), lw=1.8)
    # crossover: largest eps*n where binned slope <= 1.15
    order = np.argsort(-x2); xs_, ys_ = x2[order], y2[order]      # from large eps*n downwards
    cross = None
    for q in range(len(xs_) - 1):
        if ys_[q] > 1.3 >= ys_[q + 1]:
            t = (ys_[q] - 1.3) / (ys_[q] - ys_[q + 1])
            cross = float(10 ** (np.log10(xs_[q]) + t * (np.log10(xs_[q + 1]) - np.log10(xs_[q]))))
            break
    wc[f"n{n}"] = dict(eps=x.tolist(), slope=y.tolist(), eps_n=x2.tolist(), slope_collapse=y2.tolist(),
                       eps_n_where_slope_falls_to_1p3=cross)
gp = []
for f in sorted(glob.glob("cache/multiscale_heaviside_L2_s*.npz")):
    d = np.load(f)
    gp.append(local_slopes(d["counts"], d["F"], np.full(len(d["F"]), 2048), 8, 128))
gp = np.concatenate(gp); x, y = binned(gp, gp[:, 0], 0.5)
axs[0].plot(x, y, "-", color=S2, lw=2.2)
axs[0].text(x[-1] / 1.3, y[-1] + 0.04, "n = ∞ (GP)", color=S2, fontsize=9.5, ha="left")
axs[0].set_xscale("log"); axs[0].invert_xaxis(); axs[0].set_ylim(0.9, 2.1); axs[0].axhline(1, color=TXT2, lw=0.8, ls=":")
axs[0].set_xlabel("box side ε (rad)"); axs[0].set_ylabel("local box-counting slope")
axs[0].set_title("Heaviside depth 2: finite widths flatten, the limit does not", loc="left", fontsize=12)
axs[1].set_xscale("log"); axs[1].invert_xaxis(); axs[1].set_ylim(0.9, 2.1)
axs[1].set_xlabel("ε · n  (box side in units of 1/width)"); axs[1].set_ylabel("local box-counting slope")
axs[1].axhline(1, color=TXT2, lw=0.8, ls=":")
axs[1].set_title("…and they collapse on ε·n: the cutoff scale is ∝ 1/n", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig("gallery/verify_width_cutoff.png"); plt.close(fig)
summary["width_cutoff"] = wc

# ---- V4: resolution check -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
px2048 = 1.4 / 2048; px4096 = 1.4 / 4096
rc = {}
for k, col in [("heaviside_L1", S1), ("heaviside_L2", S2), ("heaviside_L3", S3), ("rbf", "#9a9890")]:
    for res, px, mk in [("n2048", px2048, "o"), ("n4096", px4096, "s")]:
        s = np.array(bc[res][k]["sizes"]); c = np.array(bc[res][k]["counts"])
        # normalise counts by patch area in boxes so both grids share units: N(eps) per unit area
        ax.plot(s * px, c, mk + ("-" if res == "n2048" else ""), color=col, ms=5, mfc=col if res == "n2048" else "none", lw=1.2)
    rc[k] = {res: dict(eps=(np.array(bc[res][k]["sizes"]) * px).tolist(), N=bc[res][k]["counts"]) for res, px in [("n2048", px2048), ("n4096", px4096)]}
    ax.text(1.5 * px4096, bc["n4096"][k]["counts"][0], k.replace("heaviside_L", "Heaviside L=").replace("rbf", "RBF null"), color=col, fontsize=9.5, ha="right", va="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("box side ε (rad)"); ax.set_ylabel("occupied boxes N(ε)")
ax.set_title("Resolution check: 2048² (l≤4096, filled) vs 4096² (l≤8192, open)", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig("gallery/verify_resolution.png"); plt.close(fig)
summary["resolution"] = rc
json.dump(summary, open("cache/verify_summary.json", "w"), indent=1)
print(json.dumps({k: {kk: (vv if not isinstance(vv, list) else [round(x, 3) if isinstance(x, float) else x for x in vv][:12]) for kk, vv in v.items() if kk not in ("F",)} if isinstance(v, dict) and k != "resolution" else "" for k, v in summary.items()}, indent=0)[:4000])
