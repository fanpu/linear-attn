"""Star-polygon plates: last-layer features of a C=10 run projected onto the discrete-Fourier
planes of the ideal simplex ETF, where the ideal class means form regular {10/k} star polygons.

  python render_stars.py c10 --epoch 200 --style night|riso|spectral|plotter [--planes 1 2 3 4]
"""
import argparse, sys
import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
import nclib as N
import artlib as A
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P

EXT = 0.92


def project(tag, epoch):
    meta, mets, ck = N.load_run(tag)
    e, path = N.pick_ckpts(ck, [epoch])[0]
    z = np.load(path)
    ytr, yte = N.labels(meta)
    al = N.Aligner(z["mu"], z["muG"])
    out = dict(epoch=e, al=al, ytr=ytr, yte=yte, Xtr=al(z["h_train"]), Xte=al(z["h_test"]))
    for nm, H, y in [("tr", z["h_train"], ytr), ("te", z["h_test"], yte)]:
        L = H.astype(np.float64) @ z["W"].T + z["b"]
        Lc = L[np.arange(len(y)), y]
        L2 = L.copy(); L2[np.arange(len(y)), y] = -np.inf
        out["margin_" + nm] = Lc - L2.max(1)
    return out


def class_colors(C):
    cm = cc.cm["cyclic_rygcbmr_50_90_c64"]
    return np.array([cm(j / C)[:3] for j in range(C)])


def night_layer(d, al, k, pan_px, ext, cols, gain=0.9):
    """Additive class-coloured light: counts per class -> RGB -> 3-scale glow -> 1-exp tone map."""
    raw = np.zeros((pan_px, pan_px, 3))
    for X, y, w in [(d["Xte"], d["yte"], 1.0), (d["Xtr"], d["ytr"], 0.6)]:
        for j in range(al.C):
            raw += w * A.splat(plane_xy(X[y == j], al, k), ext, pan_px)[..., None] * cols[j]
    s = pan_px / 1200
    img = np.stack([A.glow(raw[..., c], (0.7 * s, 3 * s, 14 * s)) for c in range(3)], -1)
    q = d.get("norm") or np.percentile(img.max(-1), 99.7)  # auto exposure per plate; films pass a fixed norm
    d.setdefault("norm_used", {})[k] = q
    return A.tonemap(img / q, gain * 2.0)


def plane_xy(X, al, k):
    c = al.planes[k]
    return X[:, c]


def star_lines(ax, al, k, color, lw, ideal_color=None, ideal_lw=None, ls="-"):
    o = N.star_order(al.C, k)
    c = al.planes[k]
    if ideal_color is not None:
        ax.plot(al.F[o, c[0]], al.F[o, c[1]], color=ideal_color, lw=ideal_lw, ls=ls, zorder=2, solid_joinstyle="miter")
    m = al.coords_means
    ax.plot(m[o, c[0]], m[o, c[1]], color=color, lw=lw, zorder=3, solid_joinstyle="round")


def layout(n):
    if n == 1:
        return [[0, 0, 1, 1]]
    g = 0.02
    s = (1 - 3 * g) / 2
    return [[g + (i % 2) * (s + g), g + (1 - i // 2) * (s + g), s, s] for i in range(n)]


def render(d, style, planes, res_px):
    al = d["al"]; C = al.C
    n = len(planes)
    pan_px = int(res_px * (1.0 if n == 1 else (1 - 3 * 0.02) / 2))
    ext = (-EXT, EXT, -EXT, EXT)
    if style == "night":
        bg = "#06070b"
        fig = A.canvas(res_px, res_px, bg)
        cols = class_colors(C)
        for rect, k in zip(layout(n), planes):
            img = night_layer(d, al, k, pan_px, ext, cols, gain=d.get("gain", 0.9))
            img = A.rgb(bg) + (1 - A.rgb(bg)) * img
            ax = A.panel(fig, rect, ext)
            ax.imshow(img, extent=ext, interpolation="lanczos", zorder=1)
            star_lines(ax, al, k, color=(1, 1, 1, 0.55), lw=0.5 * res_px / 2400, ideal_color=(1, 1, 1, 0.12), ideal_lw=2.2 * res_px / 2400)
    elif style == "riso":
        paper = P.RISO_PAPER
        fig = A.canvas(res_px, res_px, paper)
        pink, blue = A.rgb(P.RISO["fluo_pink"]), A.rgb(P.RISO["blue"])
        for rect, k in zip(layout(n), planes):
            s = pan_px / 1200
            gtr = A.glow(A.splat(plane_xy(d["Xtr"], al, k), ext, pan_px), (0.9 * s, 3.5 * s), (1, .25))
            gte = A.glow(A.splat(plane_xy(d["Xte"], al, k), ext, pan_px), (0.9 * s, 3.5 * s), (1, .25))
            cov_tr = A.tonemap(gtr / np.percentile(gtr, 99.5), 1.6)  # auto exposure (declared)
            cov_te = A.tonemap(gte / np.percentile(gte, 99.5), 1.6)
            sh = max(1, int(round(4 * s)))  # declared misregistration of the blue drum
            cov_te = np.roll(cov_te, (sh, -sh), (0, 1))
            img = A.rgb(paper) * (1 - cov_tr[..., None] * (1 - pink)) * (1 - cov_te[..., None] * (1 - blue))
            ax = A.panel(fig, rect, ext)
            ax.imshow(np.clip(img, 0, 1), extent=ext, interpolation="lanczos", zorder=1)
            star_lines(ax, al, k, color=P.RISO["blue"], lw=0.6 * res_px / 2400)
    elif style == "spectral":
        bg = "#101014"
        fig = A.canvas(res_px, res_px, bg)
        cmap = P.split_cmap(P.SIDES["spectral_red"], P.SIDES["spectral_purple"])
        mall = np.r_[d["margin_te"], d["margin_tr"]]
        v = P.signed_rank_normalize(mall, near_boundary="small")
        X = np.r_[d["Xte"], d["Xtr"]]
        order = np.argsort(-np.abs(v))  # confident (pale) first, near-boundary (dark) on top
        for rect, k in zip(layout(n), planes):
            ax = A.panel(fig, rect, ext)
            xy = plane_xy(X, al, k)[order]
            ax.scatter(xy[:, 0], xy[:, 1], s=(0.9 * res_px / 2400) ** 2 * (4 if n > 1 else 9), c=cmap((v[order] + 1) / 2),
                       linewidths=0, alpha=0.8, zorder=1, rasterized=True)
            star_lines(ax, al, k, color=(0.95, 0.93, 0.85, 0.5), lw=0.5 * res_px / 2400)
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--epoch", type=float, default=1e9)
    ap.add_argument("--style", default="night"); ap.add_argument("--planes", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--res", type=int, default=2400); ap.add_argument("--name", default=None)
    a = ap.parse_args()
    d = project(a.tag, a.epoch)
    print(f"epoch {d['epoch']:.3f} procrustes residual {d['al'].residual:.4f}")
    fig = render(d, a.style, a.planes, a.res)
    name = a.name or f"stars_{a.tag}_{a.style}_k{''.join(map(str, a.planes))}_ep{d['epoch']:06.2f}.png"
    print(A.save(fig, name))
