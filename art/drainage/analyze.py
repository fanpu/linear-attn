"""Offline analysis of a decode run: exact cycles, basins, and the confluence tree.

python analyze.py cache/q06b            -> cache/q06b/analysis.npz, basins.json, tree.pkl
Terminal classes:  1 = entered a loop (cycle), 2 = emitted EOS, 0 = unresolved at cap.
"""
import collections
import glob
import json
import os
import pickle
import sys

import numpy as np

from cycles import detect_at_end, canonical

MASK = (1 << 61) - 1
MUL = 1_000_003


def load_run(d):
    starts, lengths, reasons, seqs = [], [], [], []
    for f in sorted(glob.glob(os.path.join(d, "shard*/b*.npz"))):
        z = dict(np.load(f))
        off = np.concatenate([[0], np.cumsum(z["length"])])
        starts.append(z["starts"]); lengths.append(z["length"]); reasons.append(z["reason"])
        tk = z["tokens"]
        seqs += [tk[off[i]:off[i + 1]] for i in range(len(z["starts"]))]
    return (np.concatenate(starts), np.concatenate(lengths), np.concatenate(reasons), seqs)


def classify(seqs, reasons, pmax=80):
    """Returns per-sequence (cls, p, e, cycle_key) with cycle_key the canonical token tuple."""
    out = []
    for s, r in zip(seqs, reasons):
        if r == 2:
            out.append((2, 0, len(s) - 1, ("EOS",)))
        elif r == 1:
            p, e = detect_at_end(s, pmax)
            assert p > 0
            out.append((1, p, e, canonical(s[e:e + p].tolist())))
        else:
            out.append((0, 0, len(s), None))
    return out


def build_tree(seqs, cls, cyc_index):
    """Reversed-transient trie. Node key = hash of (sea, entry token, s[e-1], s[e-2], ... ).
    Returns dict node -> [count, depth, parent, token, sea] and per-seq list of node ids."""
    nodes = {}
    paths = []
    for i, (s, (c, p, e, key)) in enumerate(zip(seqs, cls)):
        if c == 0:
            paths.append(None)
            continue
        sea = cyc_index[key]
        # the sea is entered at token s[e] (for EOS: the EOS token itself)
        h = (sea * 7919 + 1) & MASK
        h = (h * MUL + int(s[e]) + 7) & MASK
        path = []
        parent = ("sea", sea)
        root = h
        if root not in nodes:
            nodes[root] = [0, 0, parent, int(s[e]), sea]
        nodes[root][0] += 1
        path.append(root)
        prev = root
        for k in range(1, e + 1):
            tok = int(s[e - k])
            h = (h * MUL + tok + 7) & MASK
            if h not in nodes:
                nodes[h] = [0, k, prev, tok, sea]
            nodes[h][0] += 1
            path.append(h)
            prev = h
        paths.append(path)
    return nodes, paths


def main(d, pmax=80):
    starts, lengths, reasons, seqs = load_run(d)
    print(d, len(starts), "sequences", flush=True)
    cls = classify(seqs, reasons, pmax)
    cnt = collections.Counter(k for c, p, e, k in cls if c != 0)
    order = [k for k, _ in cnt.most_common()]
    cyc_index = {k: i for i, k in enumerate(order)}
    nodes, paths = build_tree(seqs, cls, cyc_index)
    # per-sequence stats
    C = np.array([c for c, p, e, k in cls])
    P = np.array([p for c, p, e, k in cls])
    E = np.array([e for c, p, e, k in cls])
    SEA = np.array([cyc_index[k] if c != 0 else -1 for c, p, e, k in cls])
    shared = np.zeros(len(seqs), int)       # deepest node on the path used by >= 2 starts
    for i, path in enumerate(paths):
        if path is None:
            continue
        sd = 0
        for h in path:
            if nodes[h][0] >= 2:
                sd = nodes[h][1] + 1        # number of path tokens (incl. entry token) shared
            else:
                break
        shared[i] = sd
    # entry-phase representative text: most common entry rotation
    phase = collections.defaultdict(collections.Counter)
    for s, (c, p, e, k) in zip(seqs, cls):
        if c == 1:
            phase[k][tuple(s[e:e + p].tolist())] += 1
    basins = []
    for k in order:
        rep = list(phase[k].most_common(1)[0][0]) if k != ("EOS",) else []
        idx = SEA == cyc_index[k]
        basins.append(dict(id=cyc_index[k], key=list(k) if k != ("EOS",) else "EOS",
                           rep=rep, count=int(idx.sum()), period=int(P[idx][0]),
                           entry_median=float(np.median(E[idx])), entry_mean=float(E[idx].mean())))
    np.savez(os.path.join(d, "analysis.npz"), starts=starts, cls=C, period=P, entry=E, sea=SEA,
             shared=shared, length=lengths)
    json.dump(basins, open(os.path.join(d, "basins.json"), "w"))
    with open(os.path.join(d, "tree.pkl"), "wb") as f:
        pickle.dump(dict(nodes=nodes, paths=paths), f, protocol=5)
    print("classes: cycle %d  eos %d  unresolved %d" % ((C == 1).sum(), (C == 2).sum(), (C == 0).sum()))
    print("distinct seas", len(order), " nodes", len(nodes))


if __name__ == "__main__":
    main(sys.argv[1])
