"""Exact functional graphs of x -> 4x(1-x) (correctly rounded) over all representable values in [0,1].

    python compute_graphs.py          # formats + minifloat family + null models -> cache/graphs.npz, cache/graph_stats.json

For each format: successor array over the sorted value list, cycles, basin of every node,
depth (distance to its cycle), the cyclic node each tree hangs from, in-degrees.
Null model: random rewiring that preserves the exact in-degree sequence (succ[perm]).
"""
import json
import time
from pathlib import Path

import numpy as np

from graph import analyze, null_model
from minifloat import BF16, E4M3, E5M2, FP4, FP16, Minifloat

HERE = Path(__file__).parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)


def invariant_weights(vals):
    """Invariant measure of the r=4 logistic map, mu(dx) = dx / (pi sqrt(x(1-x))), integrated over each
    value's round-to-nearest cell (midpoints to neighbours). CDF = (2/pi) asin(sqrt(x))."""
    mids = np.concatenate([[0.0], (vals[1:] + vals[:-1]) / 2, [1.0]])
    cdf = 2 / np.pi * np.arcsin(np.sqrt(np.clip(mids, 0, 1)))
    return np.diff(cdf)


def build(mf: Minifloat, order="4x(1-x)"):
    X = mf.values_X()
    Y = mf.logistic_X(X, order)
    if X.dtype == object:
        pos = {int(x): i for i, x in enumerate(X.tolist())}
        succ = np.array([pos[int(y)] for y in Y.tolist()], dtype=np.int64)
    else:
        succ = np.searchsorted(X, Y)
        assert np.array_equal(X[succ], Y)
    vals = mf.to_float(X)
    return vals, succ


def summarize(name, mf, vals, succ, a, null=None):
    w = invariant_weights(vals)
    n_eff = 1.0 / np.sum(w ** 2)
    bfrac = a["basin_sizes"] / a["n"]
    s = dict(
        name=name, E=mf.E, M=mf.M, N=int(a["n"]), N_eff_invariant=float(n_eff),
        n_cycles=int(a["n_cycles"]), cycle_lengths=a["cycle_lengths"].tolist(),
        basin_sizes=a["basin_sizes"].tolist(), n_cyclic=int(a["n_cyclic"]),
        max_depth=int(a["max_depth"]), mean_depth=float(a["mean_depth"]),
        basin_weighted_period=float(np.sum(bfrac * a["cycle_lengths"])),
        longest_cycle=int(a["cycle_lengths"].max()),
        longest_cycle_basin_frac=float(bfrac[np.argmax(a["cycle_lengths"])]),
        zero_basin_frac=float(bfrac[a["cycle_id"][0]]) if "cycle_id" in a else None,
        indeg_hist=a["indeg_hist"].tolist(),
    )
    if null is not None:
        s["null_mean"] = dict(zip(["n_cycles", "n_cyclic", "max_depth", "mean_depth", "longest_cycle", "max_basin_frac"],
                                  null.mean(0).round(3).tolist()))
        s["null_std"] = dict(zip(["n_cycles", "n_cyclic", "max_depth", "mean_depth", "longest_cycle", "max_basin_frac"],
                                 null.std(0).round(3).tolist()))
    return s


def main():
    rng = np.random.default_rng(0)
    stats = {}
    arrays = {}
    for mf in (FP4, E5M2, E4M3, FP16, BF16):
        for order in ("4x(1-x)", "4x-4xx"):
            t = time.time()
            vals, succ = build(mf, order)
            a = analyze(succ)
            null = null_model(succ, rng, reps=30) if len(succ) > 10 else None
            key = f"{mf.name}|{order}"
            stats[key] = summarize(key, mf, vals, succ, a, null)
            print(key, {k: stats[key][k] for k in ("N", "n_cycles", "cycle_lengths", "basin_sizes", "max_depth")},
                  f"{time.time() - t:.1f}s")
            if order == "4x(1-x)":
                tag = mf.name.replace(" ", "_")
                arrays.update({f"{tag}_vals": vals, f"{tag}_succ": succ, f"{tag}_depth": a["depth"],
                               f"{tag}_root": a["root"], f"{tag}_basin": a["basin"], f"{tag}_cycle_id": a["cycle_id"],
                               f"{tag}_indeg": a["indeg"], f"{tag}_cyclens": a["cycle_lengths"]})
    # minifloat family (exact, vectorised int64): scaling of cycle structure with number of states
    fam = []
    for E in (3, 4, 5):
        bias = (1 << (E - 1)) - 1
        for M in range(1, 40):
            S = bias + M - 1
            N = bias * (1 << M)
            if 2 * S + 3 >= 62 or N > 5e6:
                break
            t = time.time()
            mf = Minifloat(E, M)
            vals, succ = build(mf)
            a = analyze(succ, want_node_arrays=True)
            s = summarize(mf.name, mf, vals, succ, a,
                          null_model(succ, rng, reps=10) if 10 < N < 3e5 else None)
            s["seconds"] = time.time() - t
            fam.append(s)
            print(mf.name, "N", N, "cycles", s["n_cycles"], "longest", s["longest_cycle"], "cyclic", s["n_cyclic"],
                  "maxdepth", s["max_depth"], "bw_period %.1f" % s["basin_weighted_period"], f"{s['seconds']:.1f}s")
    stats["family"] = fam
    np.savez_compressed(CACHE / "graphs.npz", **arrays)
    (CACHE / "graph_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
