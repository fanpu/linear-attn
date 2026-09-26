"""Follow-up (2026-09-26 evening): the drainage census of Qwen3-0.6B (post-trained) beside
Qwen3-0.6B-Base, whole vocabulary, identical engine and settings (bf16 body, fp32 logits, B=1024,
cap 256, pmax 80, confirm 32, min_span 16).

Needs: analyze.py, stats.py and nulls.py already run on both run directories.
python followup_compare_base.py cache/q06b cache/followup_q06b_base -> cache/followup_compare_base.json
"""
import json
import sys

import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
A, Bd = sys.argv[1], sys.argv[2]


def summary(d):
    st = json.load(open(d + "/stats.json"))
    B = json.load(open(d + "/basins.json"))
    nl = json.load(open(d + "/nulls.json"))["markov1"]
    loops = [b for b in B if b["key"] != "EOS"]
    cnt = np.array([b["count"] for b in loops])
    N = st["N"]
    r = np.arange(1, len(cnt) + 1)
    sel = (r >= 10) & (r <= min(3000, len(cnt)))
    slope = float(np.polyfit(np.log(r[sel]), np.log(cnt[sel] / N), 1)[0]) if sel.sum() > 5 else float("nan")
    first = st["first_top"]
    return dict(
        N=N, looped=st["classes"]["cycle"], eos=st["classes"]["eos"], unresolved=st["classes"]["unresolved"],
        looped_frac=st["classes"]["cycle"] / N, n_loops=st["n_seas"], singleton_loops=st["singleton_seas"],
        seas_holding_half=st["seas_holding_half_drained"], alpha=st["powerlaw_alpha"], ranksize_slope_10_3000=slope,
        entry_pct=st["entry_pct"], period_pct=st["period_pct"], period1_frac=st["period1_frac"],
        neck=st["distinct_first_tokens"], first_top=[(w, c, c / N) for w, c in first[:12]],
        top_seas=[(b["count"], b["count"] / max(1, st["classes"]["cycle"]), b["period"], tok.decode(b["rep"])) for b in loops[:15]],
        starts_sharing_ge5=st["starts_sharing_ge5"], confluences=st["confluences"],
        biggest_confluences=st["biggest_confluences"][:6],
        markov1=dict(n_seas=nl["n_seas"], largest_frac=nl["largest_frac"],
                     largest=tok.decode(nl["cycles_top"][0]) if nl["cycles_top"] else ""),
        keys={tuple(b["key"]): b["count"] for b in loops})


a, b = summary(A), summary(Bd)
ka, kb = a.pop("keys"), b.pop("keys")
shared = set(ka) & set(kb)
ov = dict(loops_in_both=len(shared),
          looped_runs_in_shared_loops=[sum(ka[k] for k in shared) / sum(ka.values()),
                                       sum(kb[k] for k in shared) / sum(kb.values())],
          top10_overlap=len(set(sorted(ka, key=ka.get)[-10:]) & set(sorted(kb, key=kb.get)[-10:])),
          top100_overlap=len(set(sorted(ka, key=ka.get)[-100:]) & set(sorted(kb, key=kb.get)[-100:])),
          jaccard_loop_sets=len(shared) / len(set(ka) | set(kb)),
          biggest_shared=[(ka[k], kb[k], tok.decode(list(k))) for k in sorted(shared, key=lambda k: -(ka[k] + kb[k]))[:15]])
res = dict(chat=a, base=b, overlap=ov)
json.dump(res, open("cache/followup_compare_base.json", "w"), indent=1, ensure_ascii=False)
for k in a:
    if k in ("first_top", "top_seas", "biggest_confluences"):
        continue
    print(f"{k:28s} {str(a[k])[:60]:62s} {str(b[k])[:60]}")
print("\nfirst words (chat | base):")
for x, y in zip(a["first_top"], b["first_top"]):
    print(f"  {x[0]!r:20s} {x[2]*100:5.1f}%   |  {y[0]!r:20s} {y[2]*100:5.1f}%")
print("\ntop seas (chat | base):")
for x, y in zip(a["top_seas"], b["top_seas"]):
    print(f"  {x[0]:6d} {x[1]*100:4.1f}% {x[3][:40]!r:44s} | {y[0]:6d} {y[1]*100:4.1f}% {y[3][:40]!r}")
print("\nbiggest confluences, base:", b["biggest_confluences"])
print("\noverlap:", {k: v for k, v in ov.items() if k != "biggest_shared"})
for r in ov["biggest_shared"]:
    print("   shared", r[0], r[1], repr(r[2])[:60])
