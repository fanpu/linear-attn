"""Follow-up (2026-09-26 evening): committee vs single super weight across Qwen3 sizes and Base vs chat.

One lean job per model, fp32, the same recipe as superweight.py:
  1. measurement pass over fineweb-edu val docs [0, 16) (the same pages measure.py uses): per decoder
     layer, top-1 |h| and median |h| of the residual stream -> massive-activation ratio (peak layer);
     plus max |in| / |out| of every mlp.down_proj (the Yu et al. locator).
  2. locate: first down_proj whose max |out| > 50x the median over layers; row = spiking output dim,
     col = largest |input| channel. Decompose the spike at doc 0 into per-weight contributions
     W[row, c] * x[pos, c]; k95 = the fewest top-|contribution| weights whose sum reaches 95% of it.
  3. held-out perplexity (docs from index 1000, n_held pages): base; each of the top-8 contributors
     zeroed alone; the top-k zeroed together for k = 1..max(8, k95); the null = n_null weights drawn
     uniformly from the same-|w| band as the top-1 weight (superweight.py's band rule), each zeroed
     alone for the first n_null_each, and all n_null zeroed together.
Writes cache/followup_sw_<tag>.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, decoder_layers, fineweb_docs, load, pasar_job

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--n_docs", type=int, default=16)
ap.add_argument("--n_held", type=int, default=8)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--n_null", type=int, default=100)
ap.add_argument("--n_null_each", type=int, default=30)
ap.add_argument("--dtype", default="fp32")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
torch.manual_seed(0)
rng = np.random.default_rng(0)
t0 = time.time()
tok, model = load(args.model, {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype], args.device)
layers = decoder_layers(model)
L = len(layers)
print(args.tag, "layers", L, "load", round(time.time() - t0, 1), "s", flush=True)


def enc(docs):
    return [tok(t, return_tensors="pt").input_ids[:, : args.seq_len].to(args.device) for t in docs]


cal = enc(fineweb_docs(0, args.n_docs))
held = enc(fineweb_docs(1000, args.n_held))
n_eval_total = 60
done_evals = [0]


@torch.no_grad()
def mean_nll(seqs):
    tot, n = 0.0, 0
    for ids in seqs:
        lg = model(input_ids=ids).logits.float()[0, :-1]
        tot += F.cross_entropy(lg, ids[0, 1:], reduction="sum").item()
        n += ids.shape[1] - 1
    done_evals[0] += 1
    pasar_job.progress(min(done_evals[0], n_eval_total - 1), n_eval_total)
    return tot / n


# ---- 1. measurement pass ----
cur = {}
hooks = []
for i in range(L):
    def hk(mod, inp, out, i=i):
        h = out[0] if isinstance(out, (tuple, list)) else out
        cur[i] = h.detach()[0].abs()
    hooks.append(layers[i].register_forward_hook(hk))
dstat = {i: {"in": 0.0, "in_arg": None, "out": 0.0, "out_arg": None} for i in range(L)}
for i in range(L):
    def dh(mod, inp, out, i=i):
        x = inp[0].detach()[0].abs()
        y = out.detach()[0].abs()
        s = dstat[i]
        if x.max().item() > s["in"]:
            j = int(x.argmax())
            s["in"], s["in_arg"] = x.max().item(), (cur_doc[0], j // x.shape[1], j % x.shape[1])
        if y.max().item() > s["out"]:
            j = int(y.argmax())
            s["out"], s["out_arg"] = y.max().item(), (cur_doc[0], j // y.shape[1], j % y.shape[1])
    hooks.append(layers[i].mlp.down_proj.register_forward_hook(dh))
cur_doc = [0]
top1 = np.zeros(L)
top1_arg = [None] * L
top1_no0 = np.zeros(L)
absvals = [[] for _ in range(L)]   # fp16 copies for the median (clipping at 65504 cannot move a median)
absvals_no0 = [[] for _ in range(L)]
nll = []
with torch.no_grad():
    for d, ids in enumerate(cal):
        cur_doc[0] = d
        lg = model(input_ids=ids).logits.float()[0, :-1]
        nll.append(F.cross_entropy(lg, ids[0, 1:]).item())
        for i in range(L):
            A = cur[i].float()
            m = A.max().item()
            if m > top1[i]:
                j = int(A.argmax())
                top1[i], top1_arg[i] = m, (d, j // A.shape[1], j % A.shape[1])
            top1_no0[i] = max(top1_no0[i], A[1:].max().item())
            a16 = A.half().cpu().numpy()
            absvals[i].append(a16.reshape(-1))
            absvals_no0[i].append(a16[1:].reshape(-1))
for h in hooks:
    h.remove()
med = np.array([float(np.median(np.concatenate(absvals[i]).astype(np.float32))) for i in range(L)])
med_no0 = np.array([float(np.median(np.concatenate(absvals_no0[i]).astype(np.float32))) for i in range(L)])
del absvals, absvals_no0
ratios = top1 / med
peak = int(np.argmax(ratios))
res = {"model": args.model, "tag": args.tag, "dtype": args.dtype, "n_docs": len(cal),
       "n_tokens": int(sum(x.shape[1] for x in cal)), "mean_nll_cal": float(np.mean(nll)),
       "ratios": ratios.tolist(), "top1": top1.tolist(), "median": med.tolist(),
       "top1_arg": top1_arg, "peak_layer": peak, "ratio": float(ratios[peak]),
       "ratio_excl_pos0": float(top1_no0[peak] / med_no0[peak]),
       "down_stats": {str(i): dstat[i] for i in range(L)}}
print(f"massive ratio {ratios[peak]:.0f} at L{peak} (top1 {top1[peak]:.1f} @ doc,pos,dim {top1_arg[peak]}; "
      f"median {med[peak]:.4f}); excl pos0 {res['ratio_excl_pos0']:.0f}; mean nll {np.mean(nll):.3f}", flush=True)
for i in range(L):
    s = dstat[i]
    print(f"L{i:2d} ratio {ratios[i]:9.0f} | down max|in| {s['in']:9.1f} @{s['in_arg']}  max|out| {s['out']:9.1f} @{s['out_arg']}")


def save():
    os.makedirs(CACHE, exist_ok=True)
    json.dump(res, open(os.path.join(CACHE, f"followup_sw_{args.tag}.json"), "w"), indent=1)


# ---- 2. locate ----
outs = np.array([dstat[i]["out"] for i in range(L)])
spike = [i for i in range(L) if outs[i] > 50 * np.median(outs)]
if not spike:
    res["no_spike"] = True
    save()
    print("no down_proj output spike > 50x the median over layers")
    raise SystemExit(0)
li = spike[0]
doc, pos, row = dstat[li]["out_arg"]
col = dstat[li]["in_arg"][2]
W = layers[li].mlp.down_proj.weight
w_sw = W[row, col].item()
Wabs = W.detach().abs()
rank_mag = int((Wabs > abs(w_sw)).sum().item()) + 1
cap = {}
hk = layers[li].mlp.down_proj.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()[0], y=o.detach()[0]))
with torch.no_grad():
    model(input_ids=cal[doc])
hk.remove()
out_val = cap["y"][pos, row].item()
contribs = (W[row].detach().float() * cap["x"][pos].float()).cpu().numpy()
corder = np.argsort(-np.abs(contribs))
share = np.cumsum(contribs[corder]) / contribs.sum()
k95 = int(np.argmax(share >= 0.95)) + 1 if (share >= 0.95).any() else len(share)
top_w = [float(W[row, int(c)].item()) for c in corder[:16]]
# where the top contributors' |w| sit in the whole matrix (magnitude rank)
mag_ranks = [int((Wabs > abs(w)).sum().item()) + 1 for w in top_w[:8]]
res.update({"layer": li, "row": int(row), "col": int(col), "spike_doc": int(doc), "spike_pos": int(pos),
            "w_sw": w_sw, "mag_rank": rank_mag, "shape": list(W.shape), "numel": W.numel(),
            "spike_layers": spike, "out_at_pos": out_val, "contrib_sum": float(contribs.sum()),
            "contrib_order": [int(c) for c in corder[:16]], "contrib_top": [float(contribs[c]) for c in corder[:16]],
            "contrib_w": top_w, "contrib_x": [float(cap["x"][pos, int(c)]) for c in corder[:16]],
            "contrib_share_cum": share[:16].tolist(), "k95": k95, "top_mag_ranks": mag_ranks})
print(f"spike: layers.{li}.mlp.down_proj out[{row}] = {out_val:.1f} at doc {doc} pos {pos}; recipe weight [{row},{col}] "
      f"= {w_sw:+.4f} (|w| rank {rank_mag}); k95 = {k95}", flush=True)
for j, c in enumerate(corder[:12]):
    print(f"   c={c:6d} W={W[row, c].item():+.4f} x={cap['x'][pos, c].item():+10.1f} Wx={contribs[c]:+9.1f} "
          f"cum {share[j]:.3f}")
save()


# ---- 3. ablations ----
def ablate_eval(idx_list, seqs):
    with torch.no_grad():
        old = [W[r, c].item() for r, c in idx_list]
        for r, c in idx_list:
            W[r, c] = 0.0
        v = mean_nll(seqs)
        for (r, c), o in zip(idx_list, old):
            W[r, c] = o
    return v


base = mean_nll(held)
res["base_held_nll"] = base
K = max(8, k95)
top = [int(c) for c in corder[:K]]
res["each_held_nll"] = [ablate_eval([(row, c)], held) for c in top[:8]]
res["cum_held_nll"] = [ablate_eval([(row, c) for c in top[:k]], held) for k in range(1, K + 1)]
res["k95_held_nll"] = res["cum_held_nll"][k95 - 1]
res["recipe_held_nll"] = ablate_eval([(row, col)], held) if col not in top[:1] else res["cum_held_nll"][0]
print(f"held ppl: base {np.exp(base):.2f} | top-1 zeroed {np.exp(res['cum_held_nll'][0]):.2f} | all k95={k95} zeroed "
      f"{np.exp(res['k95_held_nll']):.2f}", flush=True)
print("  each of top 8:", np.round(np.exp(res["each_held_nll"]), 2))
print("  top-k together:", np.round(np.exp(res["cum_held_nll"]), 2), flush=True)
save()

# null: same |w| band as the top-1 contributor (superweight.py's rule), excluding the row's top contributors
w1 = abs(top_w[0])
flat = Wabs.float().flatten().cpu().numpy()
excl = {row * W.shape[1] + c for c in top}
band = 0.10
while True:
    pool = np.where(np.abs(flat - w1) <= band * w1)[0]
    pool = np.array([k for k in pool if k not in excl])
    if len(pool) >= 3 * args.n_null or band > 0.9:
        break
    band += 0.05
near = rng.choice(pool, min(args.n_null, len(pool)), replace=False)
null_idx = [(int(k // W.shape[1]), int(k % W.shape[1])) for k in near]
res["null_band"], res["null_pool_size"] = band, int(len(pool))
res["null_mag_range"] = [float(flat[near].min()), float(flat[near].max())]
res["null_each_held_nll"] = [ablate_eval([rc], held) for rc in null_idx[: args.n_null_each]]
res["null_all_held_nll"] = ablate_eval(null_idx, held)
res["null_idx"] = null_idx
ne = np.exp(res["null_each_held_nll"])
print(f"null: pool {len(pool)} (band {band:.2f}, |w| {res['null_mag_range']}); each ppl {ne.min():.3f}-{ne.max():.3f}; "
      f"all {len(null_idx)} together {np.exp(res['null_all_held_nll']):.3f}", flush=True)

prompt = "My favourite condiment is"
ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
with torch.no_grad():
    g0 = model.generate(ids, max_new_tokens=30, do_sample=False)
    old = [W[row, c].item() for c in top[:1]]
    W[row, top[0]] = 0.0
    g1 = model.generate(ids, max_new_tokens=30, do_sample=False)
    W[row, top[0]] = old[0]
res["gen_base"], res["gen_top1_zeroed"] = tok.decode(g0[0]), tok.decode(g1[0])
print("base:", res["gen_base"])
print("top-1 zeroed:", res["gen_top1_zeroed"])
res["seconds"] = time.time() - t0
save()
pasar_job.progress(n_eval_total, n_eval_total)
print("done", round(time.time() - t0, 1), "s")
