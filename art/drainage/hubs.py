import sys, json, pickle, collections, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
d = sys.argv[1]
B = json.load(open(d + "/basins.json")); T = pickle.load(open(d + "/tree.pkl", "rb")); N = T["nodes"]
kids = collections.defaultdict(list)
for h, v in N.items(): kids[v[2]].append(h)
def stem_text(h):  # text from node h down to sea
    toks = []
    while not isinstance(h, tuple):
        toks.append(N[h][3]); h = N[h][2]
    return tok.decode(toks)
for sea in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6):
    b = B[sea]
    print(f"=== sea {sea} n={b['count']} p={b['period']} {tok.decode(b['rep'])!r}")
    # walk main stem: follow the child with max count while the count >= 10% of sea
    roots = sorted(kids[("sea", sea)], key=lambda h: -N[h][0])
    for r in roots[:3]:
        h = r
        while True:
            ch = kids.get(h, [])
            big = [c for c in ch if N[c][0] >= 2]
            if len(ch) >= 2 and len(big) >= 1:
                print(f"   confluence at depth {N[h][1]} count {N[h][0]} children {len(ch)} (>=2: {len(big)})  ..{stem_text(h)[-70:]!r}")
            if not big: break
            h = max(big, key=lambda c: N[c][0])
            if N[h][0] < 0.02 * b['count']: break
