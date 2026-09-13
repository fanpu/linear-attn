"""Fractals-doc Sec. 11 verification for the initialisation basin maps.

Boundary = pixels whose converge/diverge label differs from a 4-neighbour.
 1. Box counting at the finest resolution: N(eps) for eps = 2^j pixels; least-squares slope of
    log N vs log(1/eps) over a declared eps range, with local slopes reported.
 2. Resolution check: the same region at 1024, 2048, 4096, 8192 px.  N_boundary(R) ~ R^D
    (a smooth curve gives D = 1; a fractal refines and gives D > 1).  Also: fraction of labels that
    change when a 2x-finer grid is subsampled back onto the coarse grid points.
 3. Null models through the identical pipeline:
      prod2  (GD on 1/2(xy-1)^2, proven smooth convergence region, Liang & Montufar Thm 1),
      analytic ellipse D'_eta from the same theorem rasterised (pure rendering null),
    and a positive control: lmreg (Liang & Montufar Fig. 3, their estimate 1.249).
Output: cache/boxcount.json
"""
import json
import numpy as np

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"


def boundary(div):
    b = np.zeros_like(div, bool)
    b[:-1] |= div[:-1] != div[1:]
    b[1:] |= div[:-1] != div[1:]
    b[:, :-1] |= div[:, :-1] != div[:, 1:]
    b[:, 1:] |= div[:, :-1] != div[:, 1:]
    return b


def boxcount(b, jmax=None):
    n = b.shape[0]
    jmax = jmax or int(np.log2(n))
    out = []
    cur = b
    for j in range(0, jmax + 1):
        out.append((2 ** j, int(cur.sum())))
        if cur.shape[0] < 2:
            break
        s = cur.shape[0] // 2
        cur = cur[: 2 * s, : 2 * s].reshape(s, 2, s, 2).any(axis=(1, 3))
    return out


def fit(eps_counts, n, jlo, jhi):
    e = np.array([c[0] for c in eps_counts], float) / n     # box side as fraction of the region
    N = np.array([c[1] for c in eps_counts], float)
    m = (np.arange(len(e)) >= jlo) & (np.arange(len(e)) <= jhi) & (N > 0)
    x = np.log(1 / e[m])
    y = np.log(N[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    s2 = np.sum((y - yhat) ** 2) / max(1, len(x) - 2)
    se = np.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    local = np.diff(np.log(N)) / np.diff(np.log(1 / e))
    return dict(D=float(coef[0]), se=float(se), eps_min=float(e[m].min()), eps_max=float(e[m].max()),
                local_slopes=[float(v) for v in local], counts=[int(v) for v in N], eps=[float(v) for v in e])


def ellipse_null(res, eta=0.2, lim=4.5):
    xs = -lim + 2 * lim * (np.arange(res) + 0.5) / res
    u, v = np.meshgrid(xs, xs)
    s = u ** 2 + v ** 2
    Q = s + np.sqrt(s ** 2 - 16 * 1.0 * (u * v - 1.0) + 0j).real
    return ~(Q < 8 / eta)


def main():
    out = {}
    for name, tag in [("zhu4", "z1"), ("prod2", "null"), ("lmreg", "pos")]:
        per_res = {}
        labels = {}
        for R in [1024, 2048, 4096, 8192]:
            try:
                d = np.load(f"{CACHE}/basin_{name}_{tag}_{R}.npz")
            except FileNotFoundError:
                continue
            div = d["status"] == 2
            labels[R] = div
            b = boundary(div)
            per_res[R] = int(b.sum())
        if not per_res:
            continue
        Rs = sorted(per_res)
        Rmax = Rs[-1]
        bc = boxcount(boundary(labels[Rmax]))
        j_lo, j_hi = 1, int(np.log2(Rmax)) - 3          # 2 px ... R/8 px boxes
        f = fit(bc, Rmax, j_lo, j_hi)
        # resolution scaling of boundary pixel count
        x = np.log(np.array(Rs, float))
        y = np.log(np.array([per_res[r] for r in Rs], float))
        Dres = float(np.polyfit(x, y, 1)[0]) if len(Rs) > 1 else float("nan")
        # label consistency: coarse grid centres coincide with every other fine pixel? (pixel centres differ by
        # half a fine pixel; compare coarse labels with the 2x2 fine block majority)
        flips = {}
        for r in Rs:
            if 2 * r in labels:
                fine = labels[2 * r].reshape(r, 2, r, 2).mean(axis=(1, 3))
                flips[f"{r}->{2 * r}"] = float(np.mean((fine > 0.5) != labels[r]))
        out[name] = dict(tag=tag, boundary_pixels=per_res, D_resolution_scaling=Dres, boxcount=f, label_flip=flips)
        print(f"{name}: box-count D = {f['D']:.3f} ± {f['se']:.3f} over eps in [{f['eps_min']:.1e}, {f['eps_max']:.1e}] "
              f"(res {Rmax});  D from boundary-pixel scaling over R={Rs}: {Dres:.3f}; flips {flips}")
        print("   local slopes:", np.round(f["local_slopes"], 3))
    # analytic ellipse null (rendering pipeline only)
    for R in [1024, 8192]:
        div = ellipse_null(R)
        f = fit(boxcount(boundary(div)), R, 1, int(np.log2(R)) - 3)
        out[f"ellipse_analytic_{R}"] = dict(boxcount=f)
        print(f"analytic ellipse {R}: D = {f['D']:.3f} ± {f['se']:.3f}")
    json.dump(out, open(f"{CACHE}/boxcount.json", "w"), indent=1)


if __name__ == "__main__":
    main()
