"""Combed M1 trajectory previews (matplotlib, reads cache/ only). Not gallery renders.

  art/.venv/bin/python art/combed/preview.py

Chart: identity R^3 -> R^3 (data coordinates), matplotlib 3D axes, elev 35 azim -60, equal aspect.
Declared: colour = t via cividis (sequential), line alpha, point sizes.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import combed_common as C

OUT = C.CACHE / "preview"
N_HAIR = 1500
LIM = 1.8


def hair(ax, z, n_hair=N_HAIR, title="", lim=LIM, tmin=0.0):
    sel = z["t_keep"] >= tmin
    traj = z["traj"][sel, :n_hair]  # (T, B, 3)
    t = z["t_keep"][sel]
    seg = np.stack([traj[:-1], traj[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 3)
    tc = np.tile(0.5 * (t[:-1] + t[1:]), n_hair)
    lc = Line3DCollection(seg, cmap="cividis", linewidths=0.25, alpha=0.35)
    lc.set_array(tc)
    lc.set_clim(0, 1)
    ax.add_collection(lc)
    end = z["xhat256"][:n_hair]
    ax.scatter(*end.T, s=3, c="#ff3b1f", depthshade=False, linewidths=0)
    ax.scatter(*z["data"].T, s=4 if len(z["data"]) <= 256 else 0.5, c="k", depthshade=False, linewidths=0)
    style(ax, title, lim)


def style(ax, title, lim=LIM):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=35, azim=-60)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {(r["N"], r["field"]): r for r in json.loads((C.CACHE / "summary.json").read_text())}
    OUT.mkdir(parents=True, exist_ok=True)
    for n in C.NS:
        fig = plt.figure(figsize=(12, 12.5), dpi=130)
        for j, kind in enumerate(("closed", "mlp")):
            f = C.CACHE / f"flow_{kind}_N{n}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            r = rows[(n, kind)]
            name = "closed-form v*" if kind == "closed" else "trained MLP 4x256"
            ax = fig.add_subplot(2, 2, j + 1, projection="3d")
            hair(ax, z, title=f"N={n}  {name}\n{N_HAIR} of 20k trajectories, colour = t (cividis); "
                              f"memorised {100 * r['mem_xhat256']:.1f}%")
            ax = fig.add_subplot(2, 2, j + 3, projection="3d")
            hair(ax, z, title="same trajectories, only t >= 0.6, axes |x| < 1.05", lim=1.05, tmin=0.6)
        fig.suptitle("Combed M1 preview: RK4 256 steps, t in [0, 1-1e-3]; orange = x_hat endpoints, black = training points",
                     fontsize=9)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93, wspace=0.05, hspace=0.08)
        fig.savefig(OUT / f"hair_N{n}.png")
        plt.close(fig)

    # endpoints only, zoomed on the knot, all N
    fig = plt.figure(figsize=(10, 22), dpi=130)
    for i, n in enumerate(C.NS):
        for j, kind in enumerate(("closed", "mlp")):
            f = C.CACHE / f"flow_{kind}_N{n}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            ratio = z["xhat256_d1"] / z["xhat256_d2"]
            mem = ratio < C.MEM_RATIO
            ax = fig.add_subplot(5, 2, 2 * i + j + 1, projection="3d")
            e = z["xhat256"]
            ax.scatter(*e[~mem].T, s=0.3, c="#1f77b4", depthshade=False, linewidths=0)
            ax.scatter(*e[mem].T, s=0.3, c="#d62728", depthshade=False, linewidths=0)
            style(ax, f"N={n} {kind}: red memorised (d1/d2<1/3) {100 * mem.mean():.1f}%", lim=1.1)
    fig.tight_layout()
    fig.savefig(OUT / "endpoints_all.png")
    plt.close(fig)

    # memorised fraction vs N
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for kind, col in (("closed", "#222222"), ("mlp", "#d62728")):
        ns = [n for n in C.NS if (n, kind) in rows]
        ax.plot(ns, [rows[(n, kind)]["mem_xhat256"] for n in ns], "o-", c=col, label=f"{kind}, x_hat endpoint")
        ax.plot(ns, [rows[(n, kind)]["mem_end256"] for n in ns], "s--", c=col, alpha=0.5, label=f"{kind}, raw endpoint t=1-1e-3")
        ax.plot(ns, [rows[(n, kind)]["mem_xhat_tail96"] for n in ns], "^:", c=col, alpha=0.8, label=f"{kind}, x_hat at t=1-1e-6")
    ax.plot(C.NS, [rows[(n, "null_fresh_knot")]["mem_xhat256"] for n in C.NS], "x-", c="#1f77b4",
            label="null: fresh points on the knot")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N training points on the trefoil")
    ax.set_ylabel("memorised fraction (d1 < d2/3)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "memorised_vs_N.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
