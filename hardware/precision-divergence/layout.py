"""Radial layout of a functional graph: each cycle on a ring, trees hanging outward from cyclic nodes.

Angular span of every subtree is proportional to its number of leaves; children are ordered by
their representable value x (so neighbouring angles are neighbouring numbers where possible).
"""
import numpy as np


def component_layout(nodes_mask, succ, depth, root, cycle_nodes, vals, R0=None, dr=1.0, gamma=1.0, order="value"):
    """Returns dict node -> (x, y) arrays for nodes in the component, plus edges (child, parent)."""
    idx = np.flatnonzero(nodes_mask)
    n_local = len(idx)
    # children lists (non-cyclic nodes only as children)
    is_cyc = depth == 0
    kids = {}
    for c in idx:
        if is_cyc[c]:
            continue
        kids.setdefault(int(succ[c]), []).append(int(c))
    # leaves count, post-order by depth descending
    leaves = {}
    for c in idx[np.argsort(-depth[idx], kind="stable")]:
        c = int(c)
        ch = kids.get(c, [])
        leaves[c] = max(1, sum(leaves[k] for k in ch)) if ch else 1
    L = len(cycle_nodes)
    if R0 is None:
        R0 = 0.0 if L == 1 else max(1.0, L * dr * 0.25)
    tot = sum(leaves[int(c)] for c in cycle_nodes)
    theta = {}
    span = {}
    a = 0.0
    for c in cycle_nodes:
        c = int(c)
        w = 2 * np.pi * leaves[c] / tot
        span[c] = (a, a + w)
        theta[c] = a + w / 2
        a += w
    # BFS
    frontier = [int(c) for c in cycle_nodes]
    while frontier:
        nxt = []
        for p in frontier:
            ch = kids.get(p, [])
            if not ch:
                continue
            if order == "value":
                ch = sorted(ch, key=lambda k: vals[k])
            else:
                ch = sorted(ch, key=lambda k: -leaves[k])
            s0, s1 = span[p]
            totp = sum(leaves[k] for k in ch)
            b = s0
            for k in ch:
                w = (s1 - s0) * leaves[k] / totp
                span[k] = (b, b + w)
                theta[k] = b + w / 2
                b += w
                nxt.append(k)
        frontier = nxt
    nodes = np.array(sorted(theta.keys()))
    th = np.array([theta[k] for k in nodes])
    r = R0 + dr * depth[nodes].astype(float) ** gamma
    if L > 1:
        # cyclic nodes sit on the ring at evenly spaced angles in orbit order, overriding leaf-weighted angle
        pass
    pos = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    P = {int(k): i for i, k in enumerate(nodes)}
    edges = np.array([(P[int(c)], P[int(succ[c])]) for c in nodes if not is_cyc[c]], dtype=np.int64).reshape(-1, 2)
    cyc_edges = np.array([(P[int(cycle_nodes[i])], P[int(cycle_nodes[(i + 1) % L])]) for i in range(L)]) if L > 1 else np.zeros((0, 2), int)
    return dict(nodes=nodes, pos=pos, theta=th, r=r, edges=edges, cyc_edges=cyc_edges, R0=R0, leaves=leaves)
