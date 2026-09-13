"""Reduce cache/width_*.npz (and alpha_*.npz) to small summaries in cache/ntk_summary.npz (render scripts read these)."""
import glob, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntk_core import cifar_binary, kernel_alignment, linearized_gd, relu_ntk_analytic

HERE = os.path.dirname(os.path.abspath(__file__))
C = f"{HERE}/cache"
WIDTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def slope(x, y):
    A = np.stack([np.log(x), np.ones(len(x))], 1)
    coef, res, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    r = np.log(y) - A @ coef
    se = np.sqrt((r @ r) / max(1, len(x) - 2) / ((np.log(x) - np.log(x).mean()) ** 2).sum())
    return coef[0], se, coef[1]


def widths():
    xtr, ytr, xte, yte = cifar_binary()
    n = len(ytr)
    Xall = np.concatenate([xtr, xte])
    Kinf = relu_ntk_analytic(Xall, xtr)
    rows = []
    for m in WIDTHS:
        for f in sorted(glob.glob(f"{C}/width_m{m}_s*.npz")):
            d = np.load(f)
            K0, KT = d["K0"].astype(np.float64), d["KT"].astype(np.float64)
            lr, T = float(d["lr"]), int(d["step"][-1])
            f_all = d["f_all"].astype(np.float64)
            steps = d["step"]
            flin_traj = linearized_gd(K0, K0[:n], f_all[0], f_all[0][:n], ytr, lr, None, record=list(steps))
            flin = flin_traj[-1]
            dfun = np.linalg.norm(f_all[-1][n:] - flin[n:]) / np.linalg.norm(flin[n:] - f_all[0][n:])
            dfun_sup = max(np.linalg.norm(f_all[k][n:] - flin_traj[k][n:]) for k in range(len(steps))) / \
                np.linalg.norm(flin[n:] - f_all[0][n:])
            Ksub = d["Ksub"].astype(np.float64)
            rows.append(dict(
                m=m, seed=int(d["seed"]),
                dW=d["dW"][-1], da=d["da"][-1],
                dtheta=np.sqrt((d["dW"][-1] ** 2 * 192 + d["da"][-1] ** 2) / 193),  # |dtheta|/|theta0|, E|W|^2=192|a|^2
                dK=np.linalg.norm(KT - K0[:n]) / np.linalg.norm(K0[:n]),
                dKinit=np.linalg.norm(K0[:n] - Kinf[:n]) / np.linalg.norm(Kinf[:n]),
                dfun=dfun, dfun_sup=dfun_sup,
                err_net=np.mean(np.sign(f_all[-1][n:]) != yte), err_lin=np.mean(np.sign(flin[n:]) != yte),
                loss=d["loss"][-1],
                ksub_rel=np.array([np.linalg.norm(K - Ksub[0]) / np.linalg.norm(Ksub[0]) for K in Ksub]),
                steps=steps, fnet_test=f_all[-1][n:], flin_test=flin[n:],
                align0=kernel_alignment(K0[:n], ytr), alignT=kernel_alignment(KT, ytr),
            ))
    finf = linearized_gd(Kinf, Kinf[:n], np.zeros(2 * n), np.zeros(n), ytr, 2.0, 3000)
    err_inf = np.mean(np.sign(finf[n:]) != yte)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("ksub_rel", "steps", "fnet_test", "flin_test")}
    for k in ("ksub_rel", "fnet_test", "flin_test"):
        out[k] = np.stack([r[k] for r in rows])
    out["steps"] = rows[0]["steps"]
    out["err_inf"] = err_inf
    out["yte"] = yte
    np.savez(f"{C}/ntk_widths_summary.npz", **out)
    big = out["m"] >= 512
    for k in ["dW", "da", "dtheta", "dK", "dKinit", "dfun", "dfun_sup"]:
        s, se, _ = slope(out["m"][big], out[k][big])
        print(f"{k:8s} slope (m>=512) {s:+.3f} +- {se:.3f}")
    for m in WIDTHS:
        sel = out["m"] == m
        print(m, f"err_net {out['err_net'][sel].mean():.3f} err_lin {out['err_lin'][sel].mean():.3f} "
                 f"dK {out['dK'][sel].mean():.4f} dfun {out['dfun'][sel].mean():.4f} align {out['align0'][sel].mean():.3f}->{out['alignT'][sel].mean():.3f}")
    print("err_inf", err_inf)


def alpha():
    xtr, ytr, xte, yte = cifar_binary()
    n = len(ytr)
    rows = []
    for f in sorted(glob.glob(f"{C}/alpha_a*_s*.npz")):
        d = np.load(f)
        K0, KT = d["K0"].astype(np.float64), d["KT"].astype(np.float64)
        f_all = d["f_all"]
        rows.append(dict(alpha=float(d["alpha"]), seed=int(d["seed"]),
                         dW=d["dW"][-1], da=d["da"][-1],
                         dK=np.linalg.norm(KT - K0[:n]) / np.linalg.norm(K0[:n]),
                         err=np.mean(np.sign(f_all[-1][n:]) != yte),
                         err_traj=np.mean(np.sign(f_all[:, n:]) != yte[None], 1),
                         loss=d["loss"], steps=d["step"],
                         align_traj=np.array([kernel_alignment(K, ytr[d["sub"]]) for K in d["Ksub"].astype(np.float64)]),
                         ksub=d["Ksub"].astype(np.float32)))
    rows.sort(key=lambda r: (r["alpha"], r["seed"]))
    out = {k: np.array([r[k] for r in rows]) for k in ("alpha", "seed", "dW", "da", "dK", "err")}
    for k in ("err_traj", "loss", "align_traj", "ksub"):
        out[k] = np.stack([r[k] for r in rows])
    out["steps"] = rows[0]["steps"]
    np.savez(f"{C}/ntk_alpha_summary.npz", **out)
    for r in rows:
        print(f"alpha {r['alpha']:7.2f} s{r['seed']} dW {r['dW']:.3e} dK {r['dK']:.3e} err {r['err']:.3f} "
              f"loss {r['loss'][-1]:.4f} align {r['align_traj'][0]:.3f}->{r['align_traj'][-1]:.3f}")


if __name__ == "__main__":
    {"widths": widths, "alpha": alpha}[sys.argv[1]]()
