"""Massive activations: per-layer residual-stream magnitude statistics for one model.

For each decoder layer output h (the residual stream after the block), over N_DOCS documents
of real text (each forwarded alone, no padding), we record
  - top-1, top-2, top-10 |h| and the median |h| over all (token, dim) entries (Sun et al. 2024, Fig. 1),
  - the same with position 0 excluded (is it only the first token?),
  - where the top entries sit: (doc, position, dim, token string),
  - per-dim max |h| (which dims are massive),
and for document 0 the full residual stream at every layer (first SAVE_T tokens) for rendering.
Also records max |input| / max |output| (with channel index) of every MLP down projection, the
Yu et al. (2024) super-weight locator.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import CACHE, SHORT, decoder_layers, fineweb_docs, load, pasar_job

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n_docs", type=int, default=16)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--save_t", type=int, default=256)
ap.add_argument("--random_init", action="store_true")
ap.add_argument("--dtype", default="bf16")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

name = SHORT.get(args.model, args.model.split("/")[-1]) + ("-randinit" if args.random_init else "")
dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
t0 = time.time()
tok, model = load(args.model, dtype, args.device, random_init=args.random_init)
layers = decoder_layers(model)
L = len(layers)
print(name, "layers", L, "load", round(time.time() - t0, 1), "s", flush=True)

docs = fineweb_docs(0, args.n_docs)

cur = {}


def mk_hook(i):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        cur[i] = h.detach().float()[0].cpu()  # (T, D)
    return hook


hooks = [layers[i].register_forward_hook(mk_hook(i)) for i in range(L)]

# down projections (name contains down_proj; rwkv7 ffn uses 'value')
down = {}
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and (n.endswith("down_proj") or n.endswith("ffn.value")):
        down[n] = m
dstat = {n: {"in_max": 0.0, "in_arg": None, "out_max": 0.0, "out_arg": None} for n in down}


def mk_dhook(n):
    def hook(mod, inp, out):
        x = inp[0].detach().float()[0].abs()
        y = out.detach().float()[0].abs()
        xm = x.max().item()
        ym = y.max().item()
        s = dstat[n]
        if xm > s["in_max"]:
            j = int(x.argmax().item())
            s["in_max"], s["in_arg"] = xm, [cur_doc[0], j // x.shape[1], j % x.shape[1]]
        if ym > s["out_max"]:
            j = int(y.argmax().item())
            s["out_max"], s["out_arg"] = ym, [cur_doc[0], j // y.shape[1], j % y.shape[1]]
    return hook


cur_doc = [0]
for n, m in down.items():
    hooks.append(m.register_forward_hook(mk_dhook(n)))

tokmax, tokarg, tokpos, tokdoc, tokstr = [], [], [], [], []
all_abs = [[] for _ in range(L)]  # per layer list of (T,D) abs arrays (float16 to save RAM)
dim_max = None
tokens_per_doc = []
doc0 = None
nll = []
with torch.no_grad():
    for d, text in enumerate(docs):
        cur_doc[0] = d
        ids = tok(text, return_tensors="pt").input_ids[:, : args.seq_len].to(args.device)
        tokens_per_doc.append(tok.convert_ids_to_tokens(ids[0].tolist()))
        out = model(input_ids=ids)
        lg = out.logits.float()[0, :-1]
        nll.append(torch.nn.functional.cross_entropy(lg, ids[0, 1:], reduction="sum").item() / (ids.shape[1] - 1))
        H = torch.stack([cur[i] for i in range(L)])  # (L, T, D)
        A = H.abs()
        for i in range(L):
            all_abs[i].append(A[i].numpy().astype(np.float32))
        mx, am = A.max(dim=2)  # (L, T): per-token max |h| and its dim
        tokmax.append(mx.numpy())
        tokarg.append(am.numpy().astype(np.int16))
        tokpos.append(np.arange(ids.shape[1]))
        tokdoc.append(np.full(ids.shape[1], d))
        tokstr.extend(tokens_per_doc[-1])
        dm = A.amax(dim=1).numpy()  # (L, D)
        dim_max = dm if dim_max is None else np.maximum(dim_max, dm)
        if d == 0:
            doc0 = H[:, : args.save_t].numpy().astype(np.float32)
        pasar_job.progress(d + 1, len(docs))
        print("doc", d, ids.shape[1], "tokens", flush=True)

stats = []
for i in range(L):
    A = np.concatenate([a.reshape(-1) for a in all_abs[i]])
    A_no0 = np.concatenate([a[1:].reshape(-1) for a in all_abs[i]])
    srt = np.sort(A)[::-1]
    srt0 = np.sort(A_no0)[::-1]
    # locate top-10 entries
    tops = []
    offs = np.cumsum([0] + [a.size for a in all_abs[i]])
    for k in np.argsort(A)[::-1][:10]:
        d = int(np.searchsorted(offs, k, side="right") - 1)
        r = k - offs[d]
        D = all_abs[i][d].shape[1]
        t, c = int(r // D), int(r % D)
        tops.append({"doc": d, "pos": t, "dim": c, "val": float(A[k]), "tok": tokens_per_doc[d][t]})
    # per-token max, to see which tokens carry massive values
    stats.append({
        "layer": i,
        "top1": float(srt[0]), "top2": float(srt[1]), "top10": float(srt[9]),
        "median": float(np.median(A)),
        "top1_no0": float(srt0[0]), "median_no0": float(np.median(A_no0)),
        "tops": tops,
    })
    print(i, f"top1 {srt[0]:.1f} top2 {srt[1]:.1f} median {np.median(A):.4f} ratio {srt[0]/np.median(A):.0f}"
          f" | excl pos0: top1 {srt0[0]:.1f} ratio {srt0[0]/np.median(A_no0):.0f} | {tops[0]}", flush=True)

res = {
    "model": args.model, "name": name, "random_init": args.random_init, "dtype": args.dtype,
    "n_docs": len(docs), "n_tokens": int(sum(len(t) for t in tokens_per_doc)),
    "first_tokens": [t[0] for t in tokens_per_doc],
    "mean_nll": float(np.mean(nll)),
    "layers": stats,
    "down": dstat,
}
os.makedirs(CACHE, exist_ok=True)
json.dump(res, open(os.path.join(CACHE, f"acts_{name}.json"), "w"), indent=1)
np.savez_compressed(os.path.join(CACHE, f"acts_{name}.npz"), doc0=doc0, dim_max=dim_max,
                    tokens0=np.array(tokens_per_doc[0][: args.save_t]),
                    tokmax=np.concatenate(tokmax, 1), tokarg=np.concatenate(tokarg, 1),
                    tokpos=np.concatenate(tokpos), tokdoc=np.concatenate(tokdoc), tokstr=np.array(tokstr))
print("mean nll", np.mean(nll), "done", round(time.time() - t0, 1), "s")
