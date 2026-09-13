"""Zoomed bifurcation plates computed with the FULL 4-coordinate GD (float64).

For a rectangle [eta_lo, eta_hi] x [P_lo, P_hi]: N etas (2 per pixel column), fixed declared init
x0 = (1.1, 0.9, 1.05, 0.95), burn-in, then `rec` iterates whose P = x1x2x3x4 are binned into a W x H
count raster (exact counts of visited iterates; only binning is done here).  Also stores the max
Lyapunov exponent of the full map (tangent propagation over the last min(rec, 6000) steps) and the
exponent along the balanced line, per eta.

Used for (a) the window atlas (cache/atlas/*.npz) and (b) the zoom film (cache/zoom/frame_*.npz).

Usage: python compute_plates.py atlas            # all atlas windows from cache/windows.json
       python compute_plates.py zoom --nframes 600 --workers 4
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
import numpy as np
import npmaps as nm

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"
X0 = (1.1, 0.9, 1.05, 0.95)
K = 4


def counts(lo, hi, ylo, yhi, W, H, burn, rec, spp=2, lyap_steps=6000, seed=0):
    N = W * spp
    e = lo + (hi - lo) * (np.arange(N) + 0.5) / N
    x = np.broadcast_to(np.asarray(X0, float), (N, K)).copy()
    u = np.full(N, 1.0001)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, K))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    col = np.arange(N) // spp
    C = np.zeros(W * H, np.int64)
    lyap = np.zeros(N)
    lyap_b = np.zeros(N)
    alive = np.ones(N, bool)
    buf = np.empty((256, N))
    nb = 0
    L0 = max(0, rec - lyap_steps)
    with np.errstate(all="ignore"):
        for t in range(burn + rec):
            if t >= burn + L0 - 200:
                v = nm.prod_tangent(x, v, e)
                nv = np.linalg.norm(v, axis=1)
                if t >= burn + L0:
                    lyap += np.log(nv)
                    lyap_b += np.log(np.abs(nm.bal_deriv(u, e, K)))
                v /= nv[:, None]
            u = nm.bal_step(u, e, K)
            x = nm.prod_step(x, e)
            if t % 64 == 0:
                bad = ~np.isfinite(x).all(1) | (np.abs(x) > 1e6).any(1)
                alive &= ~bad
                x[bad] = 0.5
            if t >= burn:
                buf[nb] = x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3]
                nb += 1
                if nb == 256 or t == burn + rec - 1:
                    B = buf[:nb]
                    cy = np.floor((yhi - B) / (yhi - ylo) * H)
                    ok = np.isfinite(cy) & (cy >= 0) & (cy < H) & alive[None]
                    cx = np.broadcast_to(col, B.shape)
                    C += np.bincount((cx[ok] + cy[ok].astype(np.int64) * W), minlength=W * H)
                    nb = 0
    n_l = rec - L0
    ly = np.where(alive, lyap / n_l, np.nan)
    lb = lyap_b / n_l
    return C.reshape(H, W), e, ly, lb, alive


def atlas_job(r):
    p = r["base"]
    w = r["hi"] - r["lo"]
    lo, hi = r["lo"] - 0.12 * w, r["hi"] + 0.80 * w
    h = r["yhi"] - r["ylo"]
    ylo, yhi = r["ylo"] - 0.55 * h, r["yhi"] + 0.55 * h
    fn = f"{CACHE}/atlas/p{p:02d}.npz"
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    C, e, ly, lb, al = counts(lo, hi, ylo, yhi, 2400, 1500, 30000, int(min(2500 * p, 60000)))
    np.savez_compressed(fn, C=C.astype(np.int32), etas=e, lyap=ly, lyap_bal=lb, alive=al, rect=(lo, hi, ylo, yhi),
                        period=p, win=(r["lo"], r["hi"]))
    print(f"atlas p={p} done {time.time() - t0:.0f}s", flush=True)
    return fn


def tower_job(r):
    p = r["base"]
    w = r["hi"] - r["lo"]
    lo, hi = r["lo"] - 0.12 * w, r["hi"] + 0.80 * w
    h = r["yhi"] - r["ylo"]
    ylo, yhi = r["ylo"] - 0.55 * h, r["yhi"] + 0.55 * h
    fn = f"{CACHE}/atlas/tower_p{p:03d}.npz"
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    C, e, ly, lb, al = counts(lo, hi, ylo, yhi, 2400, 1500, 30000, int(min(2500 * p, 400000)), spp=1)
    np.savez_compressed(fn, C=C.astype(np.int32), etas=e, lyap=ly, lyap_bal=lb, alive=al, rect=(lo, hi, ylo, yhi),
                        period=p, win=(r["lo"], r["hi"]))
    print(f"tower p={p} done {time.time() - t0:.0f}s", flush=True)
    return fn


def zoom_path(tower, nframes, full=(0.45, 0.99, -0.03, 1.80)):
    """Keyframes: full diagram, then each tower window's central branch.  Log-linear interpolation of
    centre offsets and widths; returns list of (lo, hi, ylo, yhi, p_eff)."""
    keys = [dict(c=0.5 * (full[0] + full[1]), w=full[1] - full[0], yc=0.5 * (full[2] + full[3]), h=full[3] - full[2], p=1)]
    for r in tower:
        w = r["hi"] - r["lo"]
        lo, hi = r["lo"] - 0.12 * w, r["hi"] + 0.80 * w
        hh = r["yhi"] - r["ylo"]
        ylo, yhi = r["ylo"] - 0.55 * hh, r["yhi"] + 0.55 * hh
        # keep the film's aspect: height follows the branch, width follows the window
        keys.append(dict(c=0.5 * (lo + hi), w=hi - lo, yc=0.5 * (ylo + yhi), h=yhi - ylo, p=r["base"]))
    segs = len(keys) - 1
    frames = []
    for i in range(nframes):
        s = i / (nframes - 1) * segs
        j = min(int(s), segs - 1)
        f = s - j
        f = 0.5 - 0.5 * np.cos(np.pi * f)  # ease in/out between keyframes
        A, B = keys[j], keys[j + 1]
        lw = np.log(A["w"]) + f * (np.log(B["w"]) - np.log(A["w"]))
        lh = np.log(A["h"]) + f * (np.log(B["h"]) - np.log(A["h"]))
        w, h = np.exp(lw), np.exp(lh)
        # move the centre so that the target stays in view while zooming (fraction of zoom completed)
        g = (A["w"] - w) / (A["w"] - B["w"]) if A["w"] != B["w"] else f
        gy = (A["h"] - h) / (A["h"] - B["h"]) if A["h"] != B["h"] else f
        c = A["c"] + g * (B["c"] - A["c"])
        yc = A["yc"] + gy * (B["yc"] - A["yc"])
        p = np.exp(np.log(A["p"]) + f * (np.log(B["p"]) - np.log(A["p"])))
        frames.append((c - w / 2, c + w / 2, yc - h / 2, yc + h / 2, p))
    return frames


def zoom_job(args):
    i, (lo, hi, ylo, yhi, p) = args
    fn = f"{CACHE}/zoom/frame_{i:04d}.npz"
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    rec = int(min(2000 * p, 300000))
    C, e, ly, lb, al = counts(lo, hi, ylo, yhi, 1920, 860, 30000, rec, spp=1, lyap_steps=3000, seed=i)
    np.savez_compressed(fn, C=np.minimum(C, 65535).astype(np.uint16), etas=e, lyap=ly.astype(np.float32),
                        lyap_bal=lb.astype(np.float32), rect=(lo, hi, ylo, yhi), p_eff=p, rec=rec)
    print(f"frame {i} p_eff={p:.1f} rec={rec} {time.time() - t0:.0f}s", flush=True)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--nframes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--levels", type=int, default=4, help="tower levels to zoom through (p = 3^levels)")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.what == "atlas":
        src = f"{CACHE}/windows.json" if os.path.exists(f"{CACHE}/windows.json") else f"{CACHE}/windows_atlas_from_log.json"
        Wj = json.load(open(src))
        os.makedirs(f"{CACHE}/atlas", exist_ok=True)
        with Pool(a.workers) as pool:
            list(pool.imap_unordered(atlas_job, Wj["atlas"]))
    elif a.what == "tower":
        os.makedirs(f"{CACHE}/atlas", exist_ok=True)
        Wj = json.load(open(f"{CACHE}/windows.json"))
        with Pool(a.workers) as pool:
            list(pool.imap_unordered(tower_job, Wj["tower"][::-1]))
    else:
        Wj = json.load(open(f"{CACHE}/windows.json"))
        os.makedirs(f"{CACHE}/zoom", exist_ok=True)
        fr = zoom_path(Wj["tower"][: a.levels], a.nframes)
        json.dump(fr, open(f"{CACHE}/zoom/path.json", "w"))
        # expensive (deep) frames first so the pool is balanced
        jobs = sorted(enumerate(fr), key=lambda t: -t[1][4])
        with Pool(a.workers) as pool:
            list(pool.imap_unordered(zoom_job, jobs, chunksize=1))


if __name__ == "__main__":
    main()
