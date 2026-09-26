import sys, json, numpy as np
from analyze import load_run
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
d = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
starts, L, R, seqs = load_run(d); z = np.load(d + "/analysis.npz"); B = json.load(open(d + "/basins.json"))
for b in B[:8]:
    print("=== sea", b["id"], "count", b["count"], "period", b["period"], repr(tok.decode(b["rep"]))[:120])
    idx = np.nonzero(z["sea"] == b["id"])[0][:k]
    for i in idx:
        s = seqs[i]; e = z["entry"][i]
        print("   ", repr(tok.decode(s[:e].tolist()))[:160], "||", repr(tok.decode(s[e:e+b['period']].tolist()))[:40])
