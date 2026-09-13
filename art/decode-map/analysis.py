"""Measurements derived from a decode-map cache (no GPU): cells, stable colouring, metrics,
boundaries and box counting."""
import numpy as np

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def load(path):
    d = dict(np.load(path, allow_pickle=True))
    for k in ("grid", "rule", "prompt", "prompt_key", "stats", "engine"):
        if k in d:
            d[k] = str(d[k])
    return d


def prefix_hashes(tokens):
    """tokens [H,W,L] (-1 after EOS) -> uint64 hashes [L, H, W]; h[l] identifies the prefix of
    length l+1. Collision probability for 1e6 distinct prefixes is ~3e-8."""
    H, W, L = tokens.shape
    h = np.full((H, W), np.uint64(1469598103934665603))
    out = np.empty((L, H, W), np.uint64)
    with np.errstate(over="ignore"):
        for t in range(L):
            x = tokens[..., t].astype(np.int64).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
            h = (h ^ x) * np.uint64(0x100000001B3)
            h ^= h >> np.uint64(29)
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(32)
            out[t] = h
    return out


def labels_from_hash(h):
    _, lab = np.unique(h.ravel(), return_inverse=True)
    return lab.reshape(h.shape).astype(np.int32)


def cell_labels(tokens, l=None):
    """Integer cell labels for prefix length l (default: full output)."""
    hs = prefix_hashes(tokens)
    return labels_from_hash(hs[(l or tokens.shape[2]) - 1])


def n_cells_per_length(tokens):
    hs = prefix_hashes(tokens)
    return np.array([len(np.unique(hs[t])) for t in range(hs.shape[0])])


def boundary(lab):
    """Boolean [H,W]: pixel differs from its right or upper neighbour (edge set, one side)."""
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    return b


def boundary_full(lab):
    """Boolean [H,W]: pixel differs from any 4-neighbour (two-sided, for drawing)."""
    b = boundary(lab)
    b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b[1:, :] |= lab[1:, :] != lab[:-1, :]
    return b


def first_divergence(tokens, ref=None):
    """Position of first token that differs from the reference output (default: the greedy
    output, i.e. the pixel at the lowest (T, p) corner, where the nucleus is a single token).
    Returns L if identical over the whole horizon."""
    H, W, L = tokens.shape
    if ref is None:
        ref = tokens[0, 0]
    diff = tokens != ref[None, None]
    return np.where(diff.any(-1), diff.argmax(-1), L)


def text_metrics(tokens, eos_pos):
    """distinct-1, distinct-2 ratios, repetition rate (fraction of 3-grams seen earlier in the same
    output), generated length."""
    H, W, L = tokens.shape
    flat = tokens.reshape(-1, L)
    n = eos_pos.reshape(-1).astype(int)
    lens = np.minimum(n + 1, L)
    d1 = np.zeros(len(flat)); d2 = np.zeros(len(flat)); rep = np.zeros(len(flat))
    cache = {}
    for i, row in enumerate(flat):
        k = row.tobytes()
        if k in cache:
            d1[i], d2[i], rep[i] = cache[k]
            continue
        s = row[:lens[i]]
        a = len(set(s.tolist())) / max(1, len(s))
        bg = list(zip(s[:-1].tolist(), s[1:].tolist()))
        b = len(set(bg)) / max(1, len(bg))
        tg = list(zip(s[:-2].tolist(), s[1:-1].tolist(), s[2:].tolist()))
        seen, r = set(), 0
        for g in tg:
            r += g in seen
            seen.add(g)
        rr = r / max(1, len(tg))
        cache[k] = (a, b, rr)
        d1[i], d2[i], rep[i] = a, b, rr
    sh = (H, W)
    return dict(distinct1=d1.reshape(sh), distinct2=d2.reshape(sh), rep=rep.reshape(sh), length=lens.reshape(sh))


def mean_entropy(d, key="H_model"):
    x = d[key].astype(np.float64)
    return np.nanmean(x, -1)


def stable_colours(hs, n_colours, lengths=None, seed=0):
    """Proper colouring of the cell partition at every prefix length, with inheritance:
    each cell keeps its colour when it survives, the largest child of a split inherits the
    parent's colour, and other children get the colour least used among their already coloured
    neighbours (ties broken by a hash of the cell). Adjacent cells never share a colour as long
    as n_colours is large enough for the greedy step (fallback: least-conflicting colour).
    Returns colour index arrays [len(lengths), H, W]."""
    Lh, H, W = hs.shape
    lengths = list(range(1, Lh + 1)) if lengths is None else lengths
    rng = np.random.default_rng(seed)
    prev_lab, prev_col = None, None
    out = np.zeros((len(lengths), H, W), np.int16)
    col_of = {}
    for fi, l in enumerate(range(1, Lh + 1)):
        lab = labels_from_hash(hs[l - 1])
        nl = lab.max() + 1
        area = np.bincount(lab.ravel(), minlength=nl)
        # adjacency (4-neighbour)
        pairs = np.concatenate([
            np.stack([lab[:, :-1].ravel(), lab[:, 1:].ravel()], 1),
            np.stack([lab[:-1, :].ravel(), lab[1:, :].ravel()], 1)])
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        pairs = np.unique(np.sort(pairs, 1), axis=0)
        nbrs = [[] for _ in range(nl)]
        for a, b in pairs:
            nbrs[a].append(b); nbrs[b].append(a)
        colour = np.full(nl, -1, np.int32)
        if prev_lab is None:
            parent = np.zeros(nl, np.int64) - 1
        else:
            parent = np.zeros(nl, np.int64)
            parent[lab.ravel()] = prev_lab.ravel()
        # inheritance: largest child of each parent
        if prev_lab is not None:
            best = {}
            for c in range(nl):
                p = parent[c]
                if p not in best or area[c] > area[best[p]]:
                    best[p] = c
            for p, c in best.items():
                colour[c] = prev_col[p]
        order = np.argsort(-area, kind="stable")
        for c in order:
            if colour[c] >= 0:
                continue
            used = np.zeros(n_colours)
            for b in nbrs[c]:
                if colour[b] >= 0:
                    used[colour[b]] += 1
            free = np.flatnonzero(used == 0)
            if prev_lab is not None:
                free = free[free != prev_col[parent[c]]] if len(free) > 1 else free
            if len(free):
                colour[c] = free[rng.integers(len(free))]
            else:
                colour[c] = int(np.argmin(used))
        if l in lengths:
            out[lengths.index(l)] = colour[lab]
        prev_lab, prev_col = lab, colour
    return out


def box_count(b, eps_list):
    """Occupied boxes of side eps (pixels) for a boolean set b. Grid offset averaged over 4 shifts."""
    H, W = b.shape
    res = []
    for e in eps_list:
        cnts = []
        for oy, ox in [(0, 0), (e // 2, 0), (0, e // 2), (e // 2, e // 2)] if e > 1 else [(0, 0)]:
            bb = np.pad(b, ((oy, (-(H + oy)) % e), (ox, (-(W + ox)) % e)))
            h2, w2 = bb.shape
            cnts.append(bb.reshape(h2 // e, e, w2 // e, e).any((1, 3)).sum())
        res.append(np.mean(cnts))
    return np.array(res, float)


def fit_slope(eps, N, lo, hi):
    sel = (eps >= lo) & (eps <= hi) & (N > 0)
    x, y = np.log(1 / eps[sel]), np.log(N[sel])
    A = np.stack([x, np.ones_like(x)], 1)
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - pred) ** 2).sum() / ss if ss > 0 else 1.0
    # local slopes
    return coef[0], r2
