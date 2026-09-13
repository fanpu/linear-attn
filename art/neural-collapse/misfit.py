"""Procrustes misfit to the ideal simplex ETF vs epoch, with a random-means null.

compute:  python misfit.py compute c10 c4     -> cache/misfit_<tag>.npz
render:   python misfit.py render c10 c4 --style paper|night

misfit = ||(M U R)/s - F||_F / ||F||_F  (rotation/reflection + one global scale; 0 = exact ETF).
Null: C i.i.d. N(0, I_d) means, d = feature width (256) and d = 64, 4000 draws -> 5/50/95 %.
Also reported: nearest-neighbour-free NC2 stats of the same null (cosine std, norm spread).
"""
import sys, argparse, os
import numpy as np
import matplotlib.pyplot as plt
import nclib as N
import artlib as A

ST = {
    "paper": dict(bg=A.PAPER, ink=A.INK, muted="#8a857a", tr="#1b1b1f", te="#c0392b", w="#2f6f8f", null="#b9b09c"),
    "night": dict(bg="#0b0c10", ink="#e8e4d8", muted="#7d7a70", tr="#e8e4d8", te="#f0a35e", w="#6fb7d6", null="#3a3c44"),
}


def null_stats(C, d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    r, cs, en = [], [], []
    iu = np.triu_indices(C, 1)
    for _ in range(n):
        mu = rng.standard_normal((C, d))
        r.append(N.Aligner(mu, mu.mean(0)).residual)
        M = mu - mu.mean(0); nr = np.linalg.norm(M, axis=1)
        cs.append(((M @ M.T) / np.outer(nr, nr))[iu].std()); en.append(nr.std() / nr.mean())
    q = lambda a: np.percentile(a, [5, 50, 95])
    return dict(resid=q(r), cos_std=q(cs), equinorm=q(en))


def compute(tag):
    meta, mets, ck = N.load_run(tag)
    C = len(meta["classes"])
    ep, rtr, rte, rw = [], [], [], []
    for e, p in ck:
        z = np.load(p)
        ep.append(e)
        rtr.append(N.Aligner(z["mu"], z["muG"]).residual)
        rte.append(N.Aligner(z["mu_test"], z["mu_test"].mean(0)).residual)
        rw.append(N.Aligner(z["W"], z["W"].mean(0)).residual)  # classifier rows, centred (declared)
    d = z["mu"].shape[1]
    out = dict(epoch=np.array(ep), train=np.array(rtr), test=np.array(rte), W=np.array(rw), d=d, C=C)
    for dd in (d, 64):
        for k, v in null_stats(C, dd).items():
            out[f"null{dd}_{k}"] = v
    np.savez(os.path.join(N.HERE, "cache", f"misfit_{tag}.npz"), **out)
    print(tag, f"final train {out['train'][-1]:.4f} test {out['test'][-1]:.4f} W {out['W'][-1]:.4f} | "
          f"null d={d} 5/50/95% {np.round(out[f'null{d}_resid'], 4)}  d=64 {np.round(out['null64_resid'], 4)}")
    i = np.argmin(out["train"]); print(tag, f"min train misfit {out['train'][i]:.4f} at ep {out['epoch'][i]:.2f}")
    below = out["epoch"][out["train"] < out[f"null{d}_resid"][1]]
    print(tag, "first epoch with train misfit below null median:", below[:1])


def render(tags, style):
    s = ST[style]
    plt.rcParams.update({"font.family": "serif"})
    fig, axs = plt.subplots(1, len(tags), figsize=(7.2 * len(tags), 4.6), dpi=260, facecolor=s["bg"], squeeze=False)
    for ax, tag in zip(axs[0], tags):
        z = np.load(os.path.join(N.HERE, "cache", f"misfit_{tag}.npz"))
        d, C = int(z["d"]), int(z["C"])
        x = np.where(z["epoch"] <= 0, 0.004, z["epoch"])
        ax.set_facecolor(s["bg"])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=s["muted"], labelsize=7.5, length=2)
        for dd, a in ((d, 0.55), (64, 0.25)):
            lo, md, hi = z[f"null{dd}_resid"]
            ax.axhspan(lo, hi, color=s["null"], alpha=a, lw=0)
            ax.axhline(md, color=s["muted"], lw=0.6, ls=(0, (1, 2)))
            ax.text(0.0045, hi * 1.04, f"random i.i.d. Gaussian means, d={dd}: 5–95 %", color=s["muted"], fontsize=6.5, va="bottom")
        for key, col, ls, lab in (("train", s["tr"], "-", "train means"), ("test", s["te"], (0, (4, 2)), "test means"),
                                  ("W", s["w"], (0, (1, 1.5)), "classifier rows W")):
            ax.plot(x, z[key], color=col, lw=1.3, ls=ls)
            ax.text(x[-1] * 1.1, z[key][-1], f"{lab} {z[key][-1]:.3f}", color=col, fontsize=7, va="center")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.003, x[-1] * 6); ax.set_ylim(0.01, 1.3)
        ax.set_title(f"C = {C}: Procrustes misfit to the simplex ETF (feature width d = {d})", loc="left", color=s["ink"], fontsize=9.5)
        ax.set_xlabel("epoch (log; initialisation drawn at left edge)", color=s["muted"], fontsize=8)
    fig.tight_layout()
    return A.save(fig, f"misfit_{'_'.join(tags)}_{style}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("tags", nargs="+"); ap.add_argument("--style", default="paper")
    a = ap.parse_args()
    if a.mode == "compute":
        for t in a.tags:
            compute(t)
    else:
        print(render(a.tags, a.style))
