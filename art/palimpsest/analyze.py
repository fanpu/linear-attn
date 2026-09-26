"""Summarise every run in a directory: per-band ghost of A, the decoy (chance) level, half-lives,
what is left at the end, what is left *between* the pixels, and relearning savings.

    python analyze.py cache/runs  ->  cache/runs/summary.json
"""
import glob
import json
import os
import sys

import numpy as np

from metrics import Ghost, psnr
from pages import load, BASE

HERE = os.path.dirname(os.path.abspath(__file__))
DECOYS = ["D1", "D2", "D3", "D4"]


def offset_pages(n):
    """Pages sampled half a pixel down-right of the training grid (box filter from the 1024 master)."""
    path = os.path.join(HERE, "cache", f"pages_{n}_off.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    P = load(BASE)
    s = BASE // n
    out = {}
    for k, v in P.items():
        v = np.roll(v, (-(s // 2), -(s // 2)), axis=(0, 1))
        out[k] = v.reshape(n, s, n, s).mean(axis=(1, 3)).astype(np.float32)
    np.savez_compressed(path, **out)
    return out


_G = {}


def ghost_for(n, off=False):
    key = (n, off)
    if key not in _G:
        P = offset_pages(n) if off else load(n)
        _G[key] = (Ghost({k: P[k] for k in ["A", "B", "C"] + DECOYS}, n), P)
    return _G[key]


def half_life(steps, y):
    """First (log-interpolated) step at which y falls to half its step-0 value."""
    y0 = y[0]
    if not np.isfinite(y0) or y0 <= 0.05:
        return np.nan
    below = np.where(y <= 0.5 * y0)[0]
    if len(below) == 0:
        return np.inf
    i = below[0]
    if i == 0:
        return 0.0
    s0, s1 = max(steps[i - 1], 0.5), steps[i]
    f = (y[i - 1] - 0.5 * y0) / (y[i - 1] - y[i] + 1e-12)
    return float(np.exp(np.log(s0) + f * (np.log(s1) - np.log(s0))))


def summarise(path):
    d = np.load(path, allow_pickle=False)
    args = json.loads(str(d["args"]))
    names = [str(x) for x in d["template_names"]]
    labels = [str(x) for x in d["band_labels"]]
    ix = {k: i for i, k in enumerate(names)}
    g = d["ghost"]  # (T, P, K)
    steps = d["snap_steps"]
    n = args["res"]
    G, P = ghost_for(n)
    gA = g[:, ix["A"]]
    gD = g[:, [ix[k] for k in DECOYS]]
    letters = gA - gD.mean(1)  # A-specific: beyond what A's hand and ruling explain
    S = dict(
        name=os.path.basename(path)[:-4], args=args, bands=labels, steps=steps.tolist(),
        psnr1=float(d["psnr1"]) if "psnr1" in d else None,
        psnrB_final=float(d["psnrB"][-1]), psnrA_final=float(d["psnrA"][-1]),
        wall=float(d["wall"]),
        gA=gA.tolist(), gD_mean=gD.mean(1).tolist(), gD_std=gD.std(1).tolist(), gC=g[:, ix["C"]].tolist(),
        gB=g[:, ix["B"]].tolist(), letters=letters.tolist(),
        gall_A=d["ghost_all"][:, ix["A"]].tolist(), gall_D=d["ghost_all"][:, [ix[k] for k in DECOYS]].mean(1).tolist(),
    )
    # which bands of A were learned in phase 1 (coefficient of A in the phase-1 output)
    if "f_end1" in d and args["first"] == "A":
        fit, fitall = G(d["f_end1"])
        S["fitA"] = fit[ix["A"]].tolist()
        S["fitA_letters"] = (fit[ix["A"]] - fit[[ix[k] for k in DECOYS]].mean(0)).tolist()
    # half-lives of the A-specific ghost and of B's unwritten fraction, per band
    S["half_life_letters"] = [half_life(steps, letters[:, k]) for k in range(letters.shape[1])]
    S["half_life_gA"] = [half_life(steps, gA[:, k]) for k in range(gA.shape[1])]
    # tenth-life: the last step at which the A-specific ghost is still >= 10% of its step-0 value
    # (captures tails and resurfacing, which a half-life misses)
    def last_above(y, frac=0.1):
        if y[0] <= 0.05:
            return np.nan
        above = np.where(y >= frac * y[0])[0]
        return float(steps[above[-1]]) if len(above) else 0.0
    S["tenth_life_letters"] = [last_above(letters[:, k]) for k in range(letters.shape[1])]
    # palimpsest index: does B's coarse writing coexist with A's fine detail (the hypothesis), or
    # A's coarse structure with B's fine detail (the reverse)? max over t of the smaller of the two.
    lo = np.array([float(l.split("-")[0]) for l in labels])
    coarse = (lo >= 2) & (lo < 16)
    fine = lo >= n / 8
    Bw = 1 + g[:, ix["B"]]
    An = gA / np.maximum(gA[0:1], 1e-6)
    fwd = np.minimum(Bw[:, coarse].mean(1), An[:, fine].mean(1))
    rev = np.minimum(Bw[:, fine].mean(1), An[:, coarse].mean(1))
    S["pal_fwd"], S["pal_fwd_step"] = float(fwd.max()), int(steps[fwd.argmax()])
    S["pal_rev"], S["pal_rev_step"] = float(rev.max()), int(steps[rev.argmax()])
    S["half_life_Bunwritten"] = [half_life(steps, -g[:, ix["B"], k]) for k in range(gA.shape[1])]
    # between the pixels: the final network sampled half a pixel off the grid
    Go, Po = ghost_for(n, off=True)
    go, goall = Go(d["f_final_off"] - Po["B"])
    oi = {k: i for i, k in enumerate(Go.names)}
    S["off_gA"] = go[oi["A"]].tolist()
    S["off_gD_mean"] = go[[oi[k] for k in DECOYS]].mean(0).tolist()
    S["off_gD_std"] = go[[oi[k] for k in DECOYS]].std(0).tolist()
    S["off_psnrB"] = psnr(d["f_final_off"], Po["B"])
    S["off_all_A"] = float(goall[oi["A"]]); S["off_all_D"] = float(np.mean([goall[oi[k]] for k in DECOYS]))
    # relearning A from the final network: PSNR after k steps
    if "loss3" in d:
        l3 = d["loss3"]
        S["relearn_psnr"] = {str(k): float(-10 * np.log10(l3[k - 1])) for k in (1, 3, 10, 30, 100, len(l3)) if k <= len(l3)}
    return S


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cache", "runs")
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*.npz"))):
        if p.endswith(("summary.npz", "_kick.npz", "_kernel.npz", "_kernel_init.npz", "_kernel_final.npz")):
            continue
        try:
            out.append(summarise(p))
        except Exception as e:  # keep going; report
            print("skip", p, e)
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(out, f, default=float)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    for S in out:
        L = np.array(S["letters"])
        print(f"{S['name']:48s} p1={S['psnr1'] or 0:5.1f} pB={S['psnrB_final']:5.1f} "
              f"letters@end={np.array2string(L[-1], precision=3)} offA-D={np.array2string(np.array(S['off_gA'])-np.array(S['off_gD_mean']), precision=3)}")


if __name__ == "__main__":
    main()
