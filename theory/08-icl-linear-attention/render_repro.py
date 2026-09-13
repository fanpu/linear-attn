"""Reproduction plates for one-layer LSA (ZFB 2024) and deep LSA vs GD-k.

  figures/lsa1_plate.png   (a) exact gradient flow -> W*, (b) Adam training, (c) risk vs test length M, (d) covariate scaling
  figures/randcov.png      random-covariance training: error plateau vs the closed-form limit
  figures/depth.png        deep linear attention vs k-step GD (iso and correlated inputs)

  .venv/bin/python 08-icl-linear-attention/render_repro.py
"""
import json, pathlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import style
from icl_core import effective_preconditioner, make_cov, zfb_gamma

HERE = pathlib.Path(__file__).parent
FIG = HERE / "figures"
style.use_light()
COR, ISO = style.C["linear"], "#8f887c"
ev = json.load(open(HERE / "cache" / "lsa1_eval.json"))
summary = {}

# ------------------------------------------------------------------------------------------ lsa1 plate
fig, axs = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw=dict(wspace=0.34))
ax = axs[0]
for cov, col, name in [("ar1", COR, "correlated"), ("iso", ISO, "isotropic")]:
    z = np.load(HERE / "cache" / f"gradflow_{cov}_d20_N40_s0.npz")
    t, err = z["t"][1:], z["err_kq"][1:]
    ax.plot(t, err, color=col, lw=2.2)
    style.label_end(ax, t[-1], err[-1], name, col, dx=-4, dy=10, ha="right")
    summary[f"gradflow_{cov}_final_relerr"] = float(err[-1])
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-12, 3)
ax.set_xlabel("gradient-flow time"); ax.set_ylabel("$\\|W^{KQ}(t) - W^{KQ}_*\\| \\,/\\, \\|W^{KQ}_*\\|$")
ax.set_title("a  exact gradient flow hits the theorem")
ax = axs[1]
for cov, col, name in [("ar1", COR, "correlated"), ("iso", ISO, "isotropic")]:
    z = np.load(HERE / "cache" / f"lsa1_{cov}_d20_N40_s0.npz")
    Lam = torch.tensor(z["Lam"]); Gi = torch.linalg.inv(zfb_gamma(Lam, 40)).numpy()
    B = np.array([effective_preconditioner(torch.tensor(pv), torch.tensor(kq), 20).numpy() for pv, kq in zip(z["Wpv"], z["Wkq"])])
    e = np.linalg.norm(B - Gi, axis=(1, 2)) / np.linalg.norm(Gi)
    ax.plot(z["steps"][1:], e[1:], color=col, lw=2.2)
    style.label_end(ax, z["steps"][-1], e[-1], f"{name}: {e[-1]:.0%}", col, dx=-4, dy=-11, ha="right")
    summary[f"adam_{cov}_relerr_B"] = float(e[-1])
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-3, 3)
ax.set_xlabel("Adam steps (batch 4096, all 882 entries free)"); ax.set_ylabel("$\\|A - \\Gamma^{-1}\\| / \\|\\Gamma^{-1}\\|$")
ax.set_title("b  minibatch training gets most of the way")
ax = axs[2]
Ms = np.array(ev["Ms"])
for cov, col in [("ar1", COR), ("iso", ISO)]:
    r = ev[cov]["risk_M"]
    ax.plot(Ms, r["theory_star"], color=style.INK, lw=1.2)
    ax.scatter(Ms, r["trained_mc"], color=col, s=34, zorder=4, edgecolor=style.PAPER, lw=1.2)
    if cov == "ar1":
        ax.plot(Ms, r["gd1_scalar"], color=col, lw=1.2, ls=(0, (4, 3)), alpha=0.8)
        style.label_end(ax, Ms[-1], r["gd1_scalar"][-1], "plain GD step\n(correlated)", col, dx=-2, dy=16, ha="right", fontweight="normal", fontsize=9)
ax.axvline(40, color="#c9c4ba", lw=1); ax.text(42, 17, "train N", color=style.MUTED, fontsize=9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([2, 3, 5, 10, 20, 40])
style.label_end(ax, Ms[0], ev["iso"]["risk_M"]["trained_mc"][0], "isotropic", ISO, dx=8, dy=0)
style.label_end(ax, Ms[0], ev["ar1"]["risk_M"]["trained_mc"][0], "correlated", COR, dx=8, dy=-2)
ax.set_xlabel("test context length  $M$"); ax.set_ylabel("in-context excess risk")
ax.set_title("c  risk = closed form (ZFB Thm 4.2)")
ax.text(0.97, 0.95, "line: theory at $W_*$\ndots: trained layer (MC)", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=style.MUTED)
ax = axs[3]
Cs = np.array(ev["Cs"])
for cov, col in [("ar1", COR), ("iso", ISO)]:
    r = ev[cov]["risk_scale"]
    ax.plot(Cs, r["theory_star"], color=style.INK, lw=1.2)
    ax.scatter(Cs, r["trained_mc"], color=col, s=34, zorder=4, edgecolor=style.PAPER, lw=1.2)
ax.axhline(0, color=style.INK, lw=0)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([0.5, 0.71, 1, 1.41, 2]); ax.set_xticklabels(["½", "", "1", "", "2"])
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([5, 10, 20, 50, 100])
style.label_end(ax, 1.0, ev["iso"]["risk_scale"]["trained_mc"][3], "isotropic", ISO, dx=0, dy=12, ha="center")
style.label_end(ax, 1.0, ev["ar1"]["risk_scale"]["trained_mc"][3], "correlated", COR, dx=0, dy=-12, ha="center")
ax.set_xlabel("test inputs rescaled  $x \\to c\\,x$"); ax.set_ylabel("excess risk / $c^2$")
ax.set_title("d  rescaling breaks it (least squares = 0)")
fig.savefig(FIG / "lsa1_plate.png")
plt.close(fig)
for cov in ["iso", "ar1"]:
    r = ev[cov]["risk_M"]
    i40 = list(Ms).index(40)
    summary[f"{cov}_risk40_trained"] = r["trained_mc"][i40]
    summary[f"{cov}_risk40_theory"] = r["theory_star"][i40]
    summary[f"{cov}_risk40_gd1scalar"] = r["gd1_scalar"][i40]
    summary[f"{cov}_scale2_trained"] = ev[cov]["risk_scale"]["trained_mc"][-1]
    summary[f"{cov}_scale2_theory"] = ev[cov]["risk_scale"]["theory_star"][-1]

# ------------------------------------------------------------------------------------------ random covariances
rc = ev["randexp"]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(wspace=0.3))
ax = axs[0]
ax.plot(rc["M"], rc["risk"], color=COR, lw=2.2, marker="o", ms=6, mec=style.PAPER)
ax.axhline(rc["plateau_theory"], color=style.INK, lw=1.2)
ax.axhline(rc["null_risk"], color="#c9c4ba", lw=1.2)
ax.text(rc["M"][0], rc["plateau_theory"] * 0.9, f"closed-form plateau  {rc['plateau_theory']:.2f}", color=style.INK, fontsize=10, va="top")
ax.text(rc["M"][0], rc["null_risk"] * 1.03, "always predict 0", color=style.MUTED, fontsize=10, va="bottom")
ax.set_xscale("log"); ax.set_ylim(0, rc["null_risk"] * 1.15)
ax.set_xlabel("test context length  $M$"); ax.set_ylabel("excess risk  ($d=20$)")
ax.set_title("more examples don't help")
ax = axs[1]
ys, yh = np.array(rc["scatter"][0]), np.array(rc["scatter"][1])
lim = np.percentile(np.abs(ys), 98)
ax.scatter(ys, yh, s=6, color=COR, alpha=0.35, lw=0)
xx = np.array([-lim, lim])
ax.plot(xx, xx, color="#c9c4ba", lw=1.2); ax.plot(xx, rc["slope_theory"] * xx, color=style.INK, lw=1.2)
ax.text(lim * 0.95, lim * 0.95, "perfect", color=style.MUTED, ha="right", va="top", fontsize=10)
ax.text(lim * 0.95, rc["slope_theory"] * lim * 0.95 - lim * 0.08, f"theory slope {rc['slope_theory']:.2f}", color=style.INK, ha="right", va="top", fontsize=10)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel("true  $w^\\top x_q$"); ax.set_ylabel("layer's prediction")
ax.set_title(f"M = {rc['M'][-1]}: measured slope {rc['slope'][-1]:.2f}")
fig.savefig(FIG / "randcov.png")
plt.close(fig)
summary["randexp"] = {k: rc[k] for k in ["M", "slope", "risk", "b_finiteN", "slope_theory", "plateau_theory"]}

# ------------------------------------------------------------------------------------------ depth
de = json.load(open(HERE / "cache" / "deep_eval.json"))
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(wspace=0.62))
for ax, tag, ttl in [(axs[0], "iso", "isotropic inputs"), (axs[1], "ar1", "correlated inputs ($\\Lambda_{ij} = 0.8^{|i-j|}$)")]:
    rows = [v for k, v in de.items() if k.startswith(f"lsa_deep_{tag}|")]
    d, n = rows[0]["d"], rows[0]["n"]
    series = [("gd", "GD, learned step sizes", "#8f887c", "o"), ("pgd", "GD, learned preconditioners $A_\\ell$", style.INK, "s"),
              ("dense", "linear attention (all weights free)", COR, "o")]
    for kind, name, col, mk in series:
        pts = sorted([(v["L"], v["mean"] / d, v["sem"] / d, v["trim999"] / d) for v in rows if v["kind"] == kind])
        if not pts:
            continue
        L_, mu, se, tr = map(np.array, zip(*pts))
        if kind == "dense":
            ax.plot(L_, tr, color=col, lw=1.4, ls=(0, (3, 2)), marker="o", ms=5, mfc=style.PAPER, mec=col, zorder=3)
            ax.errorbar(L_, mu, yerr=np.minimum(2 * se, mu * 0.95), color=col, lw=2.4, marker=mk, ms=8, mec=style.PAPER, mew=1.5, zorder=4, capsize=0)
            style.label_end(ax, L_[-1], mu[-1], "linear attention", col, dx=10, dy=6 if tag == "iso" else -8, fontsize=10)
            style.label_end(ax, L_[-1], tr[-1], "same, dropping worst 0.1%\nof prompts", col, dx=10, fontsize=8.5, fontweight="normal")
            summary[f"depth_{tag}_dense_trim"] = dict(zip(map(int, L_), tr)); summary[f"depth_{tag}_dense_mean"] = dict(zip(map(int, L_), mu))
            summary[f"depth_{tag}_dense_sem"] = dict(zip(map(int, L_), se))
        else:
            ax.errorbar(L_, mu, yerr=2 * se, color=col, lw=2.2, marker=mk, ms=7, mec=style.PAPER, mew=1.5, capsize=0)
            style.label_end(ax, L_[-1], mu[-1], name, col, dx=10, fontsize=9.5)
            summary[f"depth_{tag}_{kind}"] = dict(zip(map(int, L_), mu))
    if tag == "iso":
        g = d / n
        k = np.linspace(1, 4, 50)
        ax.plot(k, g**k * (1 - g) / (1 - g ** (k + 1)), color=style.INK, lw=1, ls=(0, (2, 2)))
        ax.annotate("dotted: GD, proportional limit\n$\\gamma^k(1-\\gamma)/(1-\\gamma^{k+1})$", (1.05, 0.004), xytext=(0, 0),
                    textcoords="offset points", color=style.MUTED, fontsize=9)
    ax.set_yscale("log"); ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.8, 4.2); ax.set_ylim(2e-3, 1.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("layers = gradient steps  $k$"); ax.set_title(ttl)
axs[0].set_ylabel(f"excess risk / d   (d = {d}, n = {n}, noiseless)")
fig.savefig(FIG / "depth.png")
plt.close(fig)
json.dump(summary, open(HERE / "cache" / "repro_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1)[:3000])
