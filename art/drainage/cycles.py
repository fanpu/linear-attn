"""Exact offline cycle analysis of stored greedy sequences (numpy, CPU)."""
import numpy as np

MIN_SPAN = 16


def trailing_runs(s, pmax):
    """run[p][i] = number of consecutive equalities s[j]==s[j+p] ending at j=i (i indexes eq_p)."""
    out = {}
    for p in range(1, pmax + 1):
        if p >= len(s):
            break
        eq = (s[p:] == s[:-p]).astype(np.int32)
        # run length of trues ending at each index
        idx = np.arange(len(eq))
        last_false = np.maximum.accumulate(np.where(eq == 0, idx, -1))
        out[p] = idx - last_false
    return out


def detect_at_end(s, pmax=170, min_span=MIN_SPAN, confirm=0):
    """Smallest p such that the tail of s is p-periodic over >= max(3p, min_span)+confirm tokens.
    Returns (p, e) with e the minimal entry index (s[e:] is p-periodic), or (0, len(s))."""
    n = len(s)
    for p in range(1, min(pmax, n - 1) + 1):
        need = max(2 * p, min_span - p) + confirm
        if need + p > n:
            break
        if np.array_equal(s[n - need:], s[n - need - p:n - p]):
            # extend back: minimal e with s[e:] p-periodic
            eq = s[p:] == s[:-p]                 # eq[j]: s[j]==s[j+p]
            bad = np.nonzero(~eq)[0]
            e = 0 if len(bad) == 0 else bad[-1] + 1
            return p, int(e)
    return 0, n


def first_detection(s, pmax=128, min_span=MIN_SPAN, confirm=0):
    """Earliest prefix length L at which the online criterion fires; returns (L, p) or (0, 0)."""
    runs = trailing_runs(s, pmax)
    n = len(s)
    best = (0, 0)
    for p, r in runs.items():
        need = max(2 * p, min_span - p) + confirm
        # at prefix length L, the tail equalities are eq[L-p-1] ending run; need run >= need
        ok = np.nonzero(r >= need)[0]
        if len(ok):
            L = ok[0] + p + 1
            if best[0] == 0 or L < best[0] or (L == best[0] and p < best[1]):
                best = (int(L), p)
    return best


def canonical(cyc):
    """Lexicographically smallest rotation (as a tuple)."""
    c = list(cyc)
    k = len(c)
    rots = [tuple(c[i:] + c[:i]) for i in range(k)]
    return min(rots)
