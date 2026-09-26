"""Compare two decode runs on their common starting tokens (precision change, or another model).

python compare_runs.py cache/q06b cache/q17b  -> prints and writes cache/compare_<a>_<b>.json
"""
import collections
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

from analyze import load_run, classify

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)


def per_start(d):
    starts, L, R, seqs = load_run(d)
    cls = classify(seqs, R)
    return {int(s): (q, c) for s, q, c in zip(starts, seqs, cls)}


A, Bd = sys.argv[1], sys.argv[2]
a, b = per_start(A), per_start(Bd)
common = sorted(set(a) & set(b))
n = len(common)
res = dict(a=A, b=Bd, n_common=n)
same_first = same_tokens = 0
fd = []
same_term = 0
ca = collections.Counter(); cb = collections.Counter()
both = both_same = 0
for s in common:
    (qa, xa), (qb, xb) = a[s], b[s]
    same_first += qa[1] == qb[1]
    m = min(len(qa), len(qb))
    dv = np.nonzero(qa[:m] != qb[:m])[0]
    fd.append(int(dv[0]) if len(dv) else m)
    same_tokens += len(dv) == 0
    same_term += (xa[0] == xb[0]) and (xa[3] == xb[3])
    if xa[0] == 1:
        ca[tuple(xa[3])] += 1
    if xb[0] == 1:
        cb[tuple(xb[3])] += 1
    if xa[0] == 1 and xb[0] == 1:
        both += 1; both_same += xa[3] == xb[3]
res["same_first_token"] = same_first / n
res["identical_until_one_stops"] = same_tokens / n
res["first_divergence_median"] = float(np.median(fd))
res["same_terminal_state"] = same_term / n
res["both_looped"] = both
res["both_looped_same_loop"] = both_same / max(1, both)
res["looped_frac"] = [sum(ca.values()) / n, sum(cb.values()) / n]
res["n_loops"] = [len(ca), len(cb)]
shared = set(ca) & set(cb)
res["loops_in_both"] = len(shared)
res["starts_in_shared_loops"] = [sum(ca[k] for k in shared) / max(1, sum(ca.values())),
                                 sum(cb[k] for k in shared) / max(1, sum(cb.values()))]
topa = [k for k, _ in ca.most_common(20)]; topb = [k for k, _ in cb.most_common(20)]
res["top20_overlap"] = len(set(topa) & set(topb))
res["top_a"] = [(ca[k], tok.decode(list(k))) for k in topa[:15]]
res["top_b"] = [(cb[k], tok.decode(list(k))) for k in topb[:15]]
fa = collections.Counter(int(a[s][0][1]) for s in common); fb = collections.Counter(int(b[s][0][1]) for s in common)
res["first_words_a"] = [(tok.decode([k]), v) for k, v in fa.most_common(8)]
res["first_words_b"] = [(tok.decode([k]), v) for k, v in fb.most_common(8)]
out = f"cache/compare_{os.path.basename(A)}_{os.path.basename(Bd)}.json"
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
for k, v in res.items():
    print(k, v)
