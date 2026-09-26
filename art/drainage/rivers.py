"""Layout of the reversed-transient trees as river networks.

Each sea's tree: nodes are token-prefixes of the path *up from the sea* (depth = tokens upstream
of the cycle entry). A start token is a leaf; its stream runs down its transient and joins other
streams where their texts coincide, token for token, until the loop.

Layout (declared): the sea is at y = 0 and depth runs upward (y = depth, measured, in tokens).
Every leaf gets an equal horizontal slot (so a river's catchment width is proportional to the
number of starting tokens it drains, measured); children are ordered with the largest in the
middle and smaller ones alternating outward; an internal node sits at the count-weighted mean x of
its children. Chains of single-child nodes are drawn as one smooth curve (cubic, vertical tangents)
from the upper node to the confluence below it.
"""
import collections

import numpy as np


class SeaTree:
    def __init__(self, nodes, sea):
        """nodes: dict h -> [count, depth, parent, token, sea] (from analyze.build_tree)."""
        self.sea = sea
        items = [(h, v) for h, v in nodes.items() if v[4] == sea]
        self.ids = {h: i for i, (h, v) in enumerate(items)}
        n = len(items)
        self.count = np.array([v[0] for h, v in items])
        self.depth = np.array([v[1] for h, v in items])
        self.token = np.array([v[3] for h, v in items])
        self.parent = np.array([self.ids[v[2]] if not isinstance(v[2], tuple) else -1 for h, v in items])
        kids = collections.defaultdict(list)
        for i, p in enumerate(self.parent):
            kids[p].append(i)
        self.kids = kids
        self.n = n

    @staticmethod
    def centre_order(ch, cnt):
        """Largest child in the middle, the rest alternating right/left by decreasing size."""
        ch = sorted(ch, key=lambda c: (-cnt[c], c))
        left, right = [], []
        for k, c in enumerate(ch):
            (right if k % 2 == 0 else left).append(c)
        return left[::-1] + right

    def layout(self, x0=0.0, slot=1.0):
        """Assign x to every node; every starting token gets an equal slot. A node can itself be a
        starting token *and* be passed through by others (its text is a suffix of theirs): its own
        slot is placed among its children as if it were a child of size `own`.
        Total width = count(all) * slot."""
        x = np.zeros(self.n)
        cursor = [x0]
        cnt = self.count
        csum = np.zeros(self.n)
        np.add.at(csum, self.parent[self.parent >= 0], cnt[self.parent >= 0])
        self.own = (cnt - csum).astype(int)          # starts that end exactly at this node
        self.own_x = np.full(self.n, np.nan)
        order = self.centre_order(self.kids[-1], cnt)
        stack = [(c, 0) for c in reversed(order)]
        while stack:
            i, st = stack.pop()
            if st == 2:                              # own slot of node i
                self.own_x[i] = cursor[0] + 0.5 * slot * self.own[i]
                cursor[0] += slot * self.own[i]
                continue
            ch = self.kids.get(i, [])
            if not ch:
                x[i] = cursor[0] + 0.5 * slot * cnt[i]
                self.own_x[i] = x[i]
                cursor[0] += slot * cnt[i]
                continue
            if st == 1:
                w = [cnt[c] for c in ch]; xs = [x[c] for c in ch]
                if self.own[i] > 0:
                    w.append(self.own[i]); xs.append(self.own_x[i])
                w = np.array(w, float)
                x[i] = float((w * np.array(xs)).sum() / w.sum())
                continue
            stack.append((i, 1))
            items = list(ch) + ([("own", i)] if self.own[i] > 0 else [])
            wt = {c if not isinstance(c, tuple) else c: (cnt[c] if not isinstance(c, tuple) else self.own[i]) for c in items}
            srt = sorted(items, key=lambda c: (-wt[c], str(c)))
            left, right = [], []
            for k, c in enumerate(srt):
                (right if k % 2 == 0 else left).append(c)
            for c in reversed(left[::-1] + right):
                stack.append((i, 2) if isinstance(c, tuple) else (c, 0))
        self.x = x
        self.width = cursor[0] - x0
        return x

    def edges(self):
        """Compressed edges from every drawn node (leaf, branch point or root) down to the nearest
        drawn ancestor; roots get a stub down to the sea line (depth -1).
        Returns array rows (x_up, depth_up, x_low, depth_low, count)."""
        nk = np.array([len(self.kids.get(i, [])) for i in range(self.n)])
        drawn = (nk != 1) | (self.parent == -1)
        out = []
        for i in np.nonzero(drawn)[0]:
            p = self.parent[i]
            if p == -1:
                out.append((self.x[i], self.depth[i], self.x[i], -1, self.count[i]))
                continue
            while not drawn[p]:
                p = self.parent[p]
            out.append((self.x[i], self.depth[i], self.x[p], self.depth[p], self.count[i]))
        return np.array(out, float)


def bezier(xu, yu, xl, yl, n=24):
    t = np.linspace(0, 1, n)[:, None]
    h = yu - yl
    P0 = np.array([xu, yu]); P1 = np.array([xu, yu - 0.5 * h])
    P2 = np.array([xl, yl + 0.5 * h]); P3 = np.array([xl, yl])
    return ((1 - t) ** 3) * P0 + 3 * ((1 - t) ** 2) * t * P1 + 3 * (1 - t) * t * t * P2 + t ** 3 * P3
