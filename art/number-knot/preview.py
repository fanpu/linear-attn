"""M1 previews (matplotlib only; r3d is for M2). Reads cache/ only, writes cache/preview/.

  python preview.py [--tiny] [--template K] [--layer L] [--period T]

Declared choices: last digit and calendar position are residues, so they use the cyclic map
'twilight'; margin heatmaps use sequential 'viridis' on ΔR² / (larger per-cell null 99th pct),
clipped to [0, 10]. Axes of the helix plots are the least-squares decoded fitted directions
(fits.helix_coords), not PCA.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C
import fits as F

Ts = [2, 5, 10, 100]


def heatmaps(out, fits):
    Rs = sorted(fits)
    nt = fits[Rs[0]]["r2"].shape[1]
    fig, axs = plt.subplots(len(Rs), nt, figsize=(3.2 * nt, 4.2 * len(Rs)), squeeze=False)
    for i, R in enumerate(Rs):
        r2, null = fits[R]["r2"], fits[R]["null"]
        dr2 = r2[..., 1:] - r2[..., :1]
        dn = null[..., 1:, :] - null[..., :1, :]
        thr = np.maximum(np.quantile(dn[0], .99, -1), np.quantile(dn[1], .99, -1))
        ratio = dr2[0] / np.maximum(thr, 1e-9)
        for k in range(nt):
            ax = axs[i, k]
            im = ax.imshow(np.clip(ratio[k], 0, 10), aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=10)
            for l in range(ratio.shape[1]):
                for j in range(4):
                    ax.text(j, l, f"{dr2[0, k, l, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="w" if ratio[k, l, j] < 6 else "k")
            ax.set_xticks(range(4), [f"T={T}" for T in Ts])
            ax.set_ylabel("layer (0 = embedding)")
            ax.set_title(f"0–{R - 1}  `{C.NUMBER_TEMPLATES[k]}`", fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.6, label="ΔR² / max(shuffle99, random-token99)")
    fig.suptitle("OLMo-2-0425-1B: helix circle ΔR² per layer and period (text) and margin over nulls (colour)")
    fig.savefig(out / "numbers_margin_heatmap.png", dpi=110)
    plt.close(fig)


def curves(out, fits, k):
    fig, axs = plt.subplots(1, len(fits), figsize=(6.5 * len(fits), 4.2), squeeze=False)
    cols = dict(zip(Ts, ["#4a7fb0", "#d08a2e", "#3f9b62", "#b0476a"]))
    for ax, R in zip(axs[0], sorted(fits)):
        r2, null = fits[R]["r2"], fits[R]["null"]
        L = np.arange(r2.shape[2])
        for j, T in enumerate(Ts):
            d = r2[0, k, :, j + 1] - r2[0, k, :, 0]
            dn = null[:, k, :, j + 1] - null[:, k, :, 0]
            ax.plot(L, d, "-o", ms=3, lw=2, color=cols[T], label=f"T={T}")
            ax.plot(L, np.quantile(dn[0], .99, -1), ":", lw=1, color=cols[T])
            ax.plot(L, np.quantile(dn[1], .99, -1), "--", lw=1, color=cols[T])
        ax.set_xlabel("layer (0 = embedding)"); ax.set_ylabel("ΔR² over linear")
        ax.set_title(f"numbers 0–{R - 1}, `{C.NUMBER_TEMPLATES[k]}`  (dotted: shuffle 99%, dashed: random-token 99%)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / f"numbers_curves_t{k}.png", dpi=110); plt.close(fig)


def fourier(out, fits, k, layers):
    fig, axs = plt.subplots(len(fits), 1, figsize=(11, 3.4 * len(fits)), squeeze=False)
    for ax, R in zip(axs[:, 0], sorted(fits)):
        l = layers[R]
        sp, sn = fits[R]["spec"][0, k, l, 1], fits[R]["spec_null99"][k, l, 1]
        f = np.arange(1, len(sp))
        ax.plot(f, sp[1:], lw=1, color="#333", label="numbers (detrended)")
        ax.plot(f, fits[R]["spec"][1, k, l, 1][1:], lw=.8, color="#c77", alpha=.7, label="random tokens")
        ax.plot(f, sn[1:], lw=.8, color="#6a9", label="shuffled-a 99%")
        for T in Ts:
            if R // T < len(sp):
                ax.axvline(R // T, color="#bbb", lw=.6, zorder=0); ax.text(R // T, ax.get_ylim()[1] * .9, f"T={T}", fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("frequency bin (period = N / bin)"); ax.set_ylabel("power fraction")
        ax.set_title(f"Fourier power over a, 0–{R - 1}, layer {l}, `{C.NUMBER_TEMPLATES[k]}`", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"fourier_t{k}.png", dpi=110); plt.close(fig)


def helix(out, hs, k, l, R=1000):
    X = np.load(hs / f"olmo_numbers_t{k}.npy")[:R, l].astype(np.float64)
    R = len(X); a = np.arange(R)
    Y, *_ = F.pca(X, 100)
    p = np.random.default_rng(7).permutation(R)
    a_sh = np.empty(R); a_sh[p] = a
    fig = plt.figure(figsize=(16, 8.5))
    for row, (lab, aa) in enumerate([("measured", a), ("shuffled-label null", a_sh)]):
        for c, T in enumerate([10, 100, 5]):
            xyz, *_ = F.helix_coords(Y, aa, T)
            if c == 0:
                ax = fig.add_subplot(2, 4, row * 4 + 1, projection="3d")
                ax.scatter(xyz[:, 1], xyz[:, 2], xyz[:, 0], c=a % 10, cmap="twilight", s=4, vmin=0, vmax=10)
                ax.plot(xyz[:, 1], xyz[:, 2], xyz[:, 0], lw=.2, color="k", alpha=.3)
                ax.set_xlabel("cos10"); ax.set_ylabel("sin10"); ax.set_zlabel("linear")
                ax.set_title(f"{lab}: T=10 helix (3D)", fontsize=9)
            ax = fig.add_subplot(2, 4, row * 4 + 2 + c)
            colour = a % T if T != 100 else a % 100
            ax.scatter(xyz[:, 1], xyz[:, 2], c=colour, cmap="twilight", s=5, vmin=0, vmax=T)
            ax.set_aspect("equal"); ax.set_title(f"{lab}: fitted T={T} plane, colour a mod {T}", fontsize=9)
            ax.set_xlabel(f"cos {T}"); ax.set_ylabel(f"sin {T}")
    fig.suptitle(f"OLMo-2 numbers 0–{R - 1}, layer {l}, `{C.NUMBER_TEMPLATES[k]}` (axes: least-squares decoded fitted directions)")
    fig.tight_layout(); fig.savefig(out / f"helix_t{k}_L{l}.png", dpi=110); plt.close(fig)


def calendar(out, d):
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    for i, name in enumerate(["qwen_days", "qwen_months"]):
        z = np.load(d / f"fits_{name}.npz")
        obs, no, npnt, W = z["obs"], z["n_order"], z["n_point"], list(z["words"])
        K = len(W); L = np.arange(len(obs))
        ax = axs[i, 0]
        ax.plot(L, obs[:, 1], "-o", ms=3, color="#222", lw=2, label="R² held-out templates")
        ax.plot(L, obs[:, 0], "-", color="#888", lw=1, label="R² in-sample")
        ax.plot(L, np.quantile(no[:, 1], .99, -1), "--", color="#b0476a", label="order-shuffle 99% (held-out)")
        ax.plot(L, np.quantile(npnt[:, 1], .99, -1), ":", color="#4a7fb0", label="point-shuffle 99% (held-out)")
        cyc = obs[:, 2] > 0
        ax.scatter(L[cyc], obs[cyc, 1], s=40, facecolors="none", edgecolors="#3f9b62", label="cyclic order exact")
        ax.set_xlabel("layer (0 = embedding)"); ax.set_title(f"Qwen3-0.6B {name[5:]}: circle R²", fontsize=9)
        ax.legend(frameon=False, fontsize=7); ax.grid(alpha=.2)
        margin = obs[:, 1] - np.maximum(np.quantile(no[:, 1], .99, -1), np.quantile(npnt[:, 1], .99, -1))
        for ax, l in zip(axs[i, 1:], [int(np.argmax(margin)), len(obs) - 1]):
            P, Pm, lab = z["proj"][l], z["proj_means"][l], z["labels"]
            ax.scatter(P[:, 0], P[:, 1], c=lab, cmap="twilight", vmin=0, vmax=K, s=12, alpha=.7)
            ax.plot(np.r_[Pm[:, 0], Pm[0, 0]], np.r_[Pm[:, 1], Pm[0, 1]], "-", color="k", lw=.8)
            for w, (x, y) in zip(W, Pm):
                ax.text(x, y, w[:3], fontsize=8, weight="bold")
            ax.set_aspect("equal")
            ax.set_title(f"{name[5:]} layer {l}: mean-difference plane (held-out R² {obs[l, 1]:.2f})", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "calendar.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--template", type=int, default=None)
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()
    d = C.CACHE / "tiny" if args.tiny else C.CACHE
    hs = C.CACHE / ("tiny" if args.tiny else "hs")
    out = C.CACHE / "preview" / ("tiny" if args.tiny else "")
    out.mkdir(parents=True, exist_ok=True)
    fits = {R: dict(np.load(f)) for R in (20, 100, 1000) if (f := d / f"fits_numbers_{R}.npz").exists()}
    summ = json.loads((d / "summary_M1.json").read_text())
    Rmax = max(fits)
    k = args.template if args.template is not None else int(
        np.argmax([summ[f"numbers_{Rmax}"][str(t)]["10"]["ratio"] for t in range(len(C.NUMBER_TEMPLATES))]))
    heatmaps(out, fits)
    for t in range(len(C.NUMBER_TEMPLATES)):
        curves(out, fits, t)
    layers = {R: (args.layer if args.layer is not None else summ[f"numbers_{R}"][str(k)]["10"]["layer"]) for R in fits}
    fourier(out, fits, k, layers)
    helix(out, hs, k, layers[Rmax])
    calendar(out, d)
    print("template", k, "layers", layers)


if __name__ == "__main__":
    main()
