"""Follow-up diagnostic: after the k95 committee (or the single super weight) is zeroed, does the
massive activation disappear, or does a later layer write it again?

Reads cache/followup_sw_<tag>.json (layer, row, top contributors, k95). Forwards fineweb-edu val docs
[0, n_docs) intact and with the top-K contributors zeroed (K = max(k95, 8) so that >= 99.5% of the
spike is removed), and records per decoder layer the largest |h| at position 0 and its dim.
Writes cache/followup_regrow_<tag>.json.
"""
import argparse
import json
import os

import numpy as np
import torch

from common import CACHE, decoder_layers, fineweb_docs, load, pasar_job

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--n_docs", type=int, default=4)
ap.add_argument("--dtype", default="fp32")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
sw = json.load(open(os.path.join(CACHE, f"followup_sw_{args.tag}.json")))
tok, model = load(args.model, {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype], args.device)
layers = decoder_layers(model)
L = len(layers)
docs = [tok(t, return_tensors="pt").input_ids[:, :512].to(args.device) for t in fineweb_docs(0, args.n_docs)]
cur = {}
for i in range(L):
    layers[i].register_forward_hook(lambda m, inp, out, i=i: cur.__setitem__(i, (out[0] if isinstance(out, (tuple, list)) else out).detach()[0]))


@torch.no_grad()
def profile():
    top = np.zeros(L); dim = np.zeros(L, int); med = np.zeros(L)
    for ids in docs:
        model(input_ids=ids)
        for i in range(L):
            a = cur[i][0].float().abs()
            if a.max().item() > top[i]:
                top[i], dim[i] = a.max().item(), int(a.argmax())
            med[i] += cur[i].float().abs().median().item() / len(docs)
    return top.tolist(), dim.tolist(), med.tolist()


W = layers[sw["layer"]].mlp.down_proj.weight
K = max(sw["k95"], 8)
cols = sw["contrib_order"][:K]
res = {"tag": args.tag, "layer": sw["layer"], "row": sw["row"], "K": K}
res["intact"] = profile()
pasar_job.progress(1, 3)
with torch.no_grad():
    old = [W[sw["row"], c].item() for c in cols]
    for c in cols:
        W[sw["row"], c] = 0.0
res["zeroed"] = profile()
pasar_job.progress(2, 3)
with torch.no_grad():
    W[sw["row"], cols[0]] = old[0]
    for c in cols[1:]:
        W[sw["row"], c] = 0.0
res["zeroed_all_but_top1"] = profile()
json.dump(res, open(os.path.join(CACHE, f"followup_regrow_{args.tag}.json"), "w"), indent=1)
for k in ["intact", "zeroed", "zeroed_all_but_top1"]:
    t, d, m = res[k]
    print(k, " ".join(f"L{i}:{t[i]:.0f}/{d[i]}" for i in range(L)))
pasar_job.progress(3, 3)
