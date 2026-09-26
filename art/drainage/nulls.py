"""Baselines for the basin census.

1. Markov-1 map: f(t) = the model's greedy next token after the single token t (= the first
   generated token of every run, read from the production data; no extra compute). Iterating f is
   what greedy decoding would do if the model could only see the last token. f is a function on
   the vocabulary, so its drainage structure (in-trees onto cycles) is guaranteed; the question is
   how many seas it has and how big they are, compared with the full-context model.
2. Random mapping on n points (uniform random function), the classical null for functional graphs:
   expected number of cycles ~ 0.5*ln(n), largest basin ~ 0.76 n on average.

Writes cache/<run>/nulls.json and markov1.npz (f, sea id, depth for every token).
"""
import json
import sys

import numpy as np

from engine import N_REGULAR, EOS_IDS


def functional_graph(f, sinks):
    """f: int array on 0..n-1 (values may be >= n => absorbing sink). Returns (sea, depth, cycles)
    with sea id per node, depth = steps to reach the cycle (or sink), cycles = list of node lists."""
    n = len(f)
    sea = np.full(n, -1)
    depth = np.full(n, -1)
    state = np.zeros(n, np.int8)       # 0 new, 1 on stack, 2 done
    cycles = []
    sink_id = {}
    for s0 in range(n):
        if state[s0]:
            continue
        path = []
        x = s0
        while x < n and state[x] == 0:
            state[x] = 1
            path.append(x)
            x = f[x]
        if x >= n:                      # ran into a sink token
            if x not in sink_id:
                sink_id[x] = len(cycles); cycles.append([int(x)])
            sid, d0 = sink_id[x], 0
            for k, y in enumerate(reversed(path)):
                sea[y] = sid; depth[y] = k + 1; state[y] = 2
            continue
        if state[x] == 1:               # new cycle found, starting at x
            i = path.index(x)
            cyc = path[i:]
            sid = len(cycles); cycles.append([int(c) for c in cyc])
            for y in cyc:
                sea[y] = sid; depth[y] = 0; state[y] = 2
            path = path[:i]
            base_d = 0
        else:                           # reached an already-finished node
            sid, base_d = sea[x], depth[x]
        for k, y in enumerate(reversed(path)):
            sea[y] = sid; depth[y] = base_d + k + 1; state[y] = 2
    return sea, depth, cycles


def summarize(sea, cycles, n):
    sizes = np.bincount(sea, minlength=len(cycles))
    order = np.argsort(-sizes)
    return dict(n=int(n), n_seas=len(cycles), largest_frac=float(sizes.max() / n),
                top10=[int(sizes[i]) for i in order[:10]],
                cycle_lens=[len(cycles[i]) for i in order[:10]],
                sizes_sorted=[int(v) for v in sizes[order]])


if __name__ == "__main__":
    d = sys.argv[1]
    z = np.load(d + "/analysis.npz")
    from analyze import load_run
    starts, L, R, seqs = load_run(d)
    f = np.full(N_REGULAR, -1, np.int64)
    for s in seqs:
        f[s[0]] = s[1]
    assert (f >= 0).all(), "missing starts"
    sea, depth, cycles = functional_graph(f, None)
    res = dict(markov1=summarize(sea, cycles, N_REGULAR))
    res["markov1"]["cycles_top"] = [cycles[i] for i in np.argsort(-np.bincount(sea))[:20]]
    res["markov1"]["depth_pct"] = np.percentile(depth, [10, 50, 90, 99]).tolist()
    np.savez(d + "/markov1.npz", f=f, sea=sea, depth=depth)
    # the same files the census renderer reads, so the null gets the identical treatment
    import os
    md = d + "_markov1"; os.makedirs(md, exist_ok=True)
    sizes = np.bincount(sea, minlength=len(cycles))
    order = np.argsort(-sizes)
    remap = np.empty(len(cycles), int); remap[order] = np.arange(len(cycles))
    basins = []
    for new, old in enumerate(order):
        cyc = cycles[old]
        is_sink = len(cyc) == 1 and cyc[0] in EOS_IDS      # other special ids: absorbing, shown as the token
        basins.append(dict(id=new, key="EOS" if is_sink else cyc, rep=cyc, count=int(sizes[old]), period=len(cyc),
                           entry_median=float(np.median(depth[sea == old])), entry_mean=float(depth[sea == old].mean())))
    cls_m = np.ones(N_REGULAR, int)
    np.savez(md + "/analysis.npz", cls=cls_m, sea=remap[sea], entry=depth, period=np.array([len(cycles[s]) for s in sea]))
    json.dump(basins, open(md + "/basins.json", "w"))
    rng = np.random.default_rng(0)
    rm = []
    for k in range(20):
        g = rng.integers(0, N_REGULAR, N_REGULAR)
        s2, d2, c2 = functional_graph(g, None)
        sm = summarize(s2, c2, N_REGULAR)
        rm.append(dict(n_seas=sm["n_seas"], largest_frac=sm["largest_frac"],
                       depth_median=float(np.median(d2)), top10=sm["top10"]))
    res["random_mapping"] = rm
    json.dump(res, open(d + "/nulls.json", "w"))
    m = res["markov1"]
    print("Markov-1: seas", m["n_seas"], "largest frac", round(m["largest_frac"], 3), "top10", m["top10"],
          "cycle lens", m["cycle_lens"], "depth pct", m["depth_pct"])
    print("random mapping: seas", [r["n_seas"] for r in rm], "largest", [round(r["largest_frac"], 2) for r in rm])
