"""Layouts + 'traffic' for the functional-graph drawings (reads cache/graphs.npz).

traffic(node) = Lebesgue measure of seeds in [0,1] whose (rounded) orbit passes through the node
             = width of the node's round-to-nearest cell + sum of traffic over its tree children.
This is exact for uniformly random real seeds rounded on input.

    python compute_layout.py   # -> cache/layout_<format>.npz
"""
import numpy as np

from compute_graphs import CACHE
from layout import component_layout

FORMATS = ["FP8_E5M2", "FP8_E4M3", "float16", "bfloat16"]
ORDER, ALPHA = "center", 0.75   # declared layout choices (see layout.py)


def cell_widths(vals):
    mids = np.concatenate([[0.0], (vals[1:] + vals[:-1]) / 2, [1.0]])
    return np.diff(mids)


def traffic(succ, depth, cell):
    t = cell.copy()
    order = np.argsort(-depth, kind="stable")
    for c in order:
        if depth[c] > 0:
            t[succ[c]] += t[c]
    return t


def main():
    G = np.load(CACHE / "graphs.npz", allow_pickle=True)
    for f in FORMATS:
        vals, succ, depth, root, basin, cid = (G[f"{f}_{k}"] for k in ("vals", "succ", "depth", "root", "basin", "cycle_id"))
        cell = cell_widths(vals)
        tr = traffic(succ, depth, cell)
        out = dict(vals=vals, succ=succ, depth=depth, basin=basin, cell=cell, traffic=tr, cycle_id=cid)
        ncyc = int(cid.max()) + 1
        basin_measure = np.bincount(basin, weights=cell, minlength=ncyc)
        out["basin_measure"] = basin_measure
        for k in range(ncyc):
            # orbit order of cycle k
            start = int(np.flatnonzero(cid == k)[0])
            cyc = [start]
            x = int(succ[start])
            while x != start:
                cyc.append(x)
                x = int(succ[x])
            mask = basin == k
            lay = component_layout(mask, succ, depth, root, np.array(cyc), vals, dr=1.0, order=ORDER, alpha=ALPHA)
            out[f"c{k}_nodes"] = lay["nodes"]
            out[f"c{k}_theta"] = lay["theta"]
            out[f"c{k}_r"] = lay["r"]
            out[f"c{k}_edges"] = lay["edges"]
            out[f"c{k}_cyc_edges"] = lay["cyc_edges"]
            out[f"c{k}_cycle"] = np.array(cyc)
            print(f, "cycle", k, "len", len(cyc), "basin nodes", int(mask.sum()),
                  "seed measure %.4f" % basin_measure[k], "min x on cycle %.6g" % vals[cyc].min())
        np.savez_compressed(CACHE / f"layout_{f}.npz", **out)


if __name__ == "__main__":
    main()
