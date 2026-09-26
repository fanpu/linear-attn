"""Numbers for the README: census, entry times, periods, confluence, first words."""
import collections
import json
import pickle
import sys

import numpy as np
from transformers import AutoTokenizer

d = sys.argv[1]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
z = np.load(d + "/analysis.npz")
B = json.load(open(d + "/basins.json"))
C, E, P, S, SEA = z["cls"], z["entry"], z["period"], z["shared"], z["sea"]
N = len(C)
out = {}
out["N"] = int(N)
out["classes"] = dict(cycle=int((C == 1).sum()), eos=int((C == 2).sum()), unresolved=int((C == 0).sum()))
cnt = np.array([b["count"] for b in B])
out["n_seas"] = len(B)
out["singleton_seas"] = int((cnt == 1).sum())
drained = cnt.sum()
out["top_share"] = {k: float(cnt[:k].sum() / drained) for k in (1, 3, 10, 100, 1000)}
out["top_share_of_all"] = {k: float(cnt[:k].sum() / N) for k in (1, 3, 10, 100)}
out["seas_holding_half_drained"] = int(np.searchsorted(np.cumsum(cnt), drained / 2) + 1)
# discrete power-law MLE on sizes >= xmin (Clauset et al. approximation)
fits = {}
for xmin in (1, 2, 5, 10):
    x = cnt[cnt >= xmin].astype(float)
    alpha = 1 + len(x) / np.sum(np.log(x / (xmin - 0.5)))
    fits[xmin] = dict(n=int(len(x)), alpha=float(alpha))
out["powerlaw_alpha"] = fits
m1 = C == 1
out["entry_pct"] = np.percentile(E[m1], [10, 25, 50, 75, 90, 99]).tolist()
out["period_pct"] = np.percentile(P[m1], [10, 25, 50, 75, 90, 99]).tolist()
pc = collections.Counter(P[m1].tolist())
out["period_top"] = pc.most_common(12)
out["period1_frac"] = float((P[m1] == 1).mean())
m = C != 0
out["shared_frac_of_path_tokens"] = float(S[m].sum() / (E[m] + 1).sum())
out["starts_sharing_ge2"] = float((S[m] >= 2).mean())
out["starts_sharing_ge5"] = float((S[m] >= 5).mean())
out["starts_sharing_ge20"] = float((S[m] >= 20).mean())
out["private_len_median"] = float(np.median((E - S + 1)[m]))
# first generated token (s[1]) across all starts
from analyze import load_run
starts, L, R, seqs = load_run(d)
first = collections.Counter(int(s[1]) for s in seqs)
out["distinct_first_tokens"] = len(first)
out["first_top"] = [(tok.decode([k]), v) for k, v in first.most_common(25)]
# tree
T = pickle.load(open(d + "/tree.pkl", "rb"))["nodes"]
kids = collections.Counter(v[2] for v in T.values())
out["tree_nodes"] = len(T)
out["confluences"] = int(sum(1 for k, n in kids.items() if n >= 2 and not isinstance(k, tuple)))
big = sorted(((n, k) for k, n in kids.items() if not isinstance(k, tuple)), reverse=True)[:15]
def text_down(h):
    toks = []
    while not isinstance(h, tuple):
        toks.append(T[h][3]); h = T[h][2]
    return tok.decode(toks)
out["biggest_confluences"] = [(n, T[h][0], text_down(h)[:80]) for n, h in big]
out["top_seas"] = [(b["count"], b["period"], b["entry_median"], tok.decode(b["rep"]) if b["key"] != "EOS" else "<EOS>") for b in B[:40]]
json.dump(out, open(d + "/stats.json", "w"), indent=1, ensure_ascii=False)
for k, v in out.items():
    if k not in ("top_seas", "biggest_confluences", "first_top"):
        print(k, v)
print("first tokens:", out["first_top"][:12])
print("biggest confluences:")
for r in out["biggest_confluences"][:10]:
    print("  ", r)
print("top seas:")
for r in out["top_seas"][:30]:
    print("  ", r[0], r[1], r[2], repr(r[3])[:90])
