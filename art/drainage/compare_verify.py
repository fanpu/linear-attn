"""Compare the batched bf16 production run with unbatched fp32 / bf16 runs on the same 200 starts."""
import json
import sys

import numpy as np

from analyze import load_run, classify


def by_start(d, ids=None):
    starts, L, R, seqs = load_run(d)
    cls = classify(seqs, R)
    return {int(s): (q, c) for s, q, c in zip(starts, seqs, cls)}


ids = np.load("cache/verify_ids.npy")
prod = by_start("cache/q06b")
res = {}
for name in sys.argv[1:]:
    other = by_start(f"cache/{name}")
    same_tokens = same_terminal = same_class = 0
    firstdiv = []
    rows = []
    for t in ids:
        (a, ca), (b, cb) = prod[int(t)], other[int(t)]
        n = min(len(a), len(b))
        dv = np.nonzero(a[:n] != b[:n])[0]
        fd = int(dv[0]) if len(dv) else n
        firstdiv.append(fd)
        same_tokens += len(dv) == 0
        same_class += ca[0] == cb[0]
        same_terminal += (ca[0] == cb[0]) and (ca[3] == cb[3])
        rows.append(dict(start=int(t), firstdiv=fd, cls=(int(ca[0]), int(cb[0])), same_sea=bool(ca[3] == cb[3])))
    fdv = np.array(firstdiv)
    res[name] = dict(n=len(ids), identical_prefix=int(same_tokens), same_class=int(same_class),
                     same_terminal=int(same_terminal),
                     firstdiv_pct=np.percentile(fdv[fdv < 256], [10, 50, 90]).tolist() if (fdv < 256).any() else [],
                     n_diverged=int((np.array([not r["firstdiv"] >= min(len(prod[r["start"]][0]), len(other[r["start"]][0])) for r in rows])).sum()))
    # among starts where BOTH runs reached a loop: same loop?
    both = [r for r in rows if r["cls"] == (1, 1)]
    res[name]["both_cycled"] = len(both)
    res[name]["both_cycled_same_sea"] = sum(r["same_sea"] for r in both)
    print(name, json.dumps(res[name]))
json.dump(res, open("cache/verify_compare.json", "w"), indent=1)
