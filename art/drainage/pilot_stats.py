import glob, numpy as np, collections
from cycles import first_detection, detect_at_end, canonical
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
S = []
for f in sorted(glob.glob("cache/pilot/shard00/b*.npz")):
    d = np.load(f); off = np.concatenate([[0], np.cumsum(d["length"])])
    for i in range(len(d["starts"])):
        S.append(d["tokens"][off[i]:off[i+1]])
print(len(S))
res = collections.Counter(); L3 = []; Lc = []; per = []; ent = []; cyc = collections.Counter(); ex={}
for s in S:
    L0, p0 = first_detection(s, 128, 16, 0)
    L1, p1 = first_detection(s, 128, 16, 32)
    pe, e = detect_at_end(s, 170, 16, 0)
    if L0:
        # does detected cycle at L0 persist to end?
        eq = s[p0:] == s[:-p0]
        persist = eq[L0 - 3*p0 if L0-3*p0>0 else 0:].all() if True else None
        # simpler: s[L0-p0:] periodic with p0?
        persist = np.all(s[L0 - p0:][p0:] == s[L0 - p0:][:-p0])
        res['det0_persist' if persist else 'det0_break'] += 1
        L3.append(L0)
    else: res['nodet0'] += 1
    if L1:
        persist = np.all(s[L1 - p1:][p1:] == s[L1 - p1:][:-p1])
        res['det32_persist' if persist else 'det32_break'] += 1
        Lc.append(L1)
    else: res['nodet32'] += 1
    if pe:
        per.append(pe); ent.append(e); c = canonical(s[e:e+pe]); cyc[c] += 1; ex.setdefault(c, s)
print(res)
for nm, a in [("first det L (3rep)", L3), ("first det L (+32)", Lc), ("period at end", per), ("entry at end", ent)]:
    a = np.array(a); print(nm, "n", len(a), "pct 10/50/90/99", np.percentile(a, [10,50,90,99]).round(1), "max", a.max())
print("distinct cycles at end:", len(cyc))
for c, n in cyc.most_common(25):
    print(n, len(c), repr(tok.decode(list(c)))[:150])
