"""Functional-graph analysis for a map on a finite set, vectorised (works to ~1e8 nodes)."""
from __future__ import annotations

import numpy as np


def cyclic_mask(succ: np.ndarray) -> np.ndarray:
    """Nodes on cycles = the limit of repeatedly taking the image of the whole set."""
    n = len(succ)
    alive = np.ones(n, bool)
    idx = np.arange(n, dtype=succ.dtype)
    while True:
        new = np.zeros(n, bool)
        new[succ[idx]] = True
        new &= alive
        if new.sum() == len(idx):
            return new
        alive = new
        idx = np.flatnonzero(alive).astype(succ.dtype)


def label_cycles(succ, cyc):
    """Return cycle_id per node (-1 if not cyclic), list of cycles (arrays of node ids in orbit order)."""
    cid = np.full(len(succ), -1, np.int64)
    cycles = []
    for start in np.flatnonzero(cyc):
        if cid[start] >= 0:
            continue
        orbit = [start]
        cid[start] = len(cycles)
        x = succ[start]
        while x != start:
            cid[x] = len(cycles)
            orbit.append(x)
            x = succ[x]
        cycles.append(np.array(orbit, dtype=np.int64))
    return cid, cycles


def reverse_csr(succ):
    order = np.argsort(succ, kind="stable")
    counts = np.bincount(succ, minlength=len(succ))
    ptr = np.concatenate([[0], np.cumsum(counts)])
    return ptr, order, counts


def children_of(frontier, ptr, order):
    starts = ptr[frontier]
    lens = ptr[frontier + 1] - starts
    tot = int(lens.sum())
    if tot == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    rep_parent = np.repeat(frontier, lens)
    offs = np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens)
    kids = order[np.repeat(starts, lens) + offs]
    return kids, rep_parent


def analyze(succ: np.ndarray, want_node_arrays=True):
    succ = np.asarray(succ, dtype=np.int64)
    n = len(succ)
    cyc = cyclic_mask(succ)
    cid, cycles = label_cycles(succ, cyc)
    ptr, order, indeg = reverse_csr(succ)
    depth = np.full(n, -1, np.int64)
    root = np.full(n, -1, np.int64)       # cyclic node the tree hangs from
    depth[cyc] = 0
    root[cyc] = np.flatnonzero(cyc)
    frontier = np.flatnonzero(cyc)
    d = 0
    level_sizes = [len(frontier)]
    while len(frontier):
        kids, par = children_of(frontier, ptr, order)
        keep = depth[kids] < 0
        kids, par = kids[keep], par[keep]
        d += 1
        depth[kids] = d
        root[kids] = root[par]
        frontier = kids
        if len(kids):
            level_sizes.append(len(kids))
    basin = cid[root]
    cyc_len = np.array([len(c) for c in cycles])
    basin_size = np.bincount(basin, minlength=len(cycles))
    tree_size = np.bincount(root, minlength=n)[cyc]    # per cyclic node, includes the node itself
    out = dict(
        n=n, n_cycles=len(cycles), cycle_lengths=cyc_len, basin_sizes=basin_size,
        n_cyclic=int(cyc.sum()), max_depth=int(depth.max()), mean_depth=float(depth.mean()),
        indeg_hist=np.bincount(indeg), level_sizes=np.array(level_sizes), tree_sizes=tree_size,
    )
    if want_node_arrays:
        out.update(depth=depth, root=root, basin=basin, cycle_id=cid, indeg=indeg)
        out["cycles"] = cycles
    return out


def null_model(succ, rng, reps=20):
    """Random rewiring preserving the in-degree sequence: succ_null = succ[perm]."""
    stats = []
    for _ in range(reps):
        s = succ[rng.permutation(len(succ))]
        a = analyze(s, want_node_arrays=False)
        stats.append((a["n_cycles"], a["n_cyclic"], a["max_depth"], a["mean_depth"], int(a["cycle_lengths"].max()),
                      float(a["basin_sizes"].max() / a["n"])))
    return np.array(stats, dtype=np.float64)
