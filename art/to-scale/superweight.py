"""The super weight in Qwen3-0.6B: locate (Yu et al. 2024 recipe), verify by ablation, null controls,
and a first-order fragility map of the whole down_proj matrix calibrated against exact ablation.

Data: calibration = fineweb-edu val docs [0, N_CAL) (the same docs measure.py uses);
held-out = docs starting at index 1000. All in fp32.
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
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--tag", default="qwen3-0.6b")
ap.add_argument("--n_cal", type=int, default=8)
ap.add_argument("--n_held", type=int, default=16)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--n_top", type=int, default=50)
ap.add_argument("--n_rand", type=int, default=50)
ap.add_argument("--n_null", type=int, default=100)
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", default="fp32")
ap.add_argument("--quick", action="store_true", help="locate + top-contributor ablations only")
ap.add_argument("--subsets", type=int, default=6, help="ablate every subset of the top-k contributors")
args = ap.parse_args()
torch.manual_seed(0)
rng = np.random.default_rng(0)

t0 = time.time()
tok, model = load(args.model, {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype], args.device)
layers = decoder_layers(model)
L = len(layers)


def enc(docs):
    return [tok(t, return_tensors="pt").input_ids[:, : args.seq_len].to(args.device) for t in docs]


cal = enc(fineweb_docs(0, args.n_cal))
held = enc(fineweb_docs(1000, args.n_held))
print("cal tokens", sum(x.shape[1] for x in cal), "held tokens", sum(x.shape[1] for x in held), flush=True)


@torch.no_grad()
def mean_nll(seqs):
    tot, n = 0.0, 0
    for ids in seqs:
        lg = model(input_ids=ids).logits.float()[0, :-1]
        tot += F.cross_entropy(lg, ids[0, 1:], reduction="sum").item()
        n += ids.shape[1] - 1
    return tot / n


# ---- 1. locate: max |in| and |out| of every down_proj over the calibration set ----
stat = {}
hooks = []
for i in range(L):
    m = layers[i].mlp.down_proj

    def h(mod, inp, out, i=i):
        x = inp[0].detach()[0].abs()
        y = out.detach()[0].abs()
        s = stat.setdefault(i, {"in": 0.0, "in_arg": None, "out": 0.0, "out_arg": None})
        if x.max().item() > s["in"]:
            j = int(x.argmax())
            s["in"], s["in_arg"] = x.max().item(), (j // x.shape[1], j % x.shape[1])
        if y.max().item() > s["out"]:
            j = int(y.argmax())
            s["out"], s["out_arg"] = y.max().item(), (j // y.shape[1], j % y.shape[1])
    hooks.append(m.register_forward_hook(h))
base_cal = mean_nll(cal)
for hk in hooks:
    hk.remove()
for i in range(L):
    s = stat[i]
    print(f"L{i:2d} down_proj max|in| {s['in']:9.1f} @(pos,ch){s['in_arg']}  max|out| {s['out']:9.1f} @(pos,dim){s['out_arg']}")

# the super-weight layer: first layer where the down_proj output spikes (> 50x the median over layers)
outs = np.array([stat[i]["out"] for i in range(L)])
spike = [i for i in range(L) if outs[i] > 50 * np.median(outs)]
if not spike:
    os.makedirs(CACHE, exist_ok=True)
    json.dump({"no_spike": True, "down_stats": {str(i): stat[i] for i in range(L)}, "base_cal_nll": base_cal},
              open(os.path.join(CACHE, f"superweight_{args.tag}.json"), "w"), indent=1)
    print("no down_proj output spike > 50x the median over layers: no super-weight candidate")
    raise SystemExit(0)
li = spike[0]
row = stat[li]["out_arg"][1]
col = stat[li]["in_arg"][1]
W = layers[li].mlp.down_proj.weight  # (hidden, intermediate): out = W @ x
w_sw = W[row, col].item()
print(f"super weight candidate: layers.{li}.mlp.down_proj.weight[{row}, {col}] = {w_sw:.4f}; spike layers {spike}")
Wabs = W.detach().abs()
rank_mag = int((Wabs > abs(w_sw)).sum().item()) + 1
imax = int(Wabs.argmax())
r_max, c_max = imax // W.shape[1], imax % W.shape[1]
print(f"|w_sw| magnitude rank in matrix: {rank_mag} of {W.numel()}; largest |w| = {W[r_max, c_max].item():.4f} at [{r_max},{c_max}]")

# contribution check: at the spike position, how much of out[row] is w_sw * x[col]?
cap = {}
hk = layers[li].mlp.down_proj.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()[0], y=o.detach()[0]))
with torch.no_grad():
    model(input_ids=cal[0])
hk.remove()
pos = int(cap["y"][:, row].abs().argmax())
contrib = (w_sw * cap["x"][pos, col]).item()
print(f"at pos {pos}: out[{row}] = {cap['y'][pos, row].item():.1f}; w_sw*x[{col}] = {contrib:.1f}")
# decompose the super activation into per-weight contributions W[row, c] * x[pos, c]
contribs = (W[row].detach() * cap["x"][pos]).cpu().numpy()
corder = np.argsort(-np.abs(contribs))
print("top contributors to out[row] at the spike position (c, W, x, W*x, cumulative share):")
cum = 0.0
for c in corder[:12]:
    cum += contribs[c]
    print(f"   c={c:5d} W={W[row, c].item():+.4f} x={cap['x'][pos, c].item():+10.1f} Wx={contribs[c]:+9.1f} cum {cum / contribs.sum():.3f}")


# ---- helpers for ablation ----
def ablate_eval(idx_list, seqs):
    """zero the listed (r,c) entries together, eval, restore"""
    with torch.no_grad():
        old = [W[r, c].item() for r, c in idx_list]
        for r, c in idx_list:
            W[r, c] = 0.0
        v = mean_nll(seqs)
        for (r, c), o in zip(idx_list, old):
            W[r, c] = o
    return v


base_held = mean_nll(held)
res = {"layer": li, "row": row, "col": col, "w_sw": w_sw, "mag_rank": rank_mag, "numel": W.numel(),
       "shape": list(W.shape), "largest": [r_max, c_max, W[r_max, c_max].item()],
       "contrib": contrib, "out_at_pos": cap["y"][pos, row].item(), "spike_pos": pos,
       "down_stats": {str(i): stat[i] for i in range(L)},
       "base_cal_nll": base_cal, "base_held_nll": base_held}
res["sw_held_nll"] = ablate_eval([(row, col)], held)
res["sw_cal_nll"] = ablate_eval([(row, col)], cal)
print(f"held-out ppl: base {np.exp(base_held):.2f}  super weight zeroed {np.exp(res['sw_held_nll']):.2f}", flush=True)
if (r_max, c_max) != (row, col):
    res["largest_held_nll"] = ablate_eval([(r_max, c_max)], held)
    print(f"largest-|w| zeroed: {np.exp(res['largest_held_nll']):.2f}")

# is it one weight? ablate the top contributors individually and cumulatively
res["contribs"] = contribs.tolist()
res["contrib_order"] = [int(c) for c in corder[:16]]
res["topc_each_held_nll"] = [ablate_eval([(row, int(c))], held) for c in corder[:8]]
res["topc_cum_held_nll"] = [ablate_eval([(row, int(c)) for c in corder[:k]], held) for k in range(1, 9)]
print("top contributors zeroed one at a time, ppl:", np.round(np.exp(res["topc_each_held_nll"]), 2))
print("top-k contributors zeroed together, ppl:", np.round(np.exp(res["topc_cum_held_nll"]), 2), flush=True)

# every subset of the top-k contributors (2^k held-out evals): is it one weight or a committee?
import itertools
K = args.subsets
top = [int(c) for c in corder[:K]]
res["subset_cols"] = top
res["subsets"] = {}
for r_ in range(K + 1):
    for sub in itertools.combinations(range(K), r_):
        res["subsets"]["".join(str(j) for j in sub)] = ablate_eval([(row, top[j]) for j in sub], held)
print("subsets done; all-K:", np.exp(res["subsets"]["".join(str(j) for j in range(K))]), flush=True)
if args.quick:
    os.makedirs(CACHE, exist_ok=True)
    json.dump(res, open(os.path.join(CACHE, f"superweight_{args.tag}.json"), "w"), indent=1)
    print("quick done", round(time.time() - t0, 1), "s")
    raise SystemExit(0)

# restore the super activation (Yu et al.): zero the weight but put back the original down_proj output at [pos, row]
with torch.no_grad():
    orig = {}
    def grab(m, i, o):
        orig[len(orig)] = o[0, :, row].clone()
    hk = layers[li].mlp.down_proj.register_forward_hook(grab)
    for ids in held:
        model(input_ids=ids)
    hk.remove()
    cnt = [0]

    def restore(m, i, o):
        o = o.clone()
        v = orig[cnt[0]]
        p = int(v.abs().argmax())  # the super activation: the single largest entry of that channel
        o[0, p, row] = v[p]
        cnt[0] += 1
        return o
    hk = layers[li].mlp.down_proj.register_forward_hook(restore)
    old = W[row, col].item()
    W[row, col] = 0.0
    res["sw_zeroed_act_restored_held_nll"] = mean_nll(held)
    W[row, col] = old
    hk.remove()
print(f"super weight zeroed but its activation row restored: {np.exp(res['sw_zeroed_act_restored_held_nll']):.2f}")

# null: 100 random weights of the same magnitude (the |w| nearest |w_sw|), same matrix
flat = Wabs.flatten().cpu().numpy()
# band of |w| within 10% of |w_sw| (widened until it holds >= 3x n_null), excluding the row's top-8 contributors;
# then a uniform random draw of n_null from the band
excl = {row * W.shape[1] + int(c) for c in corder[:8]}
band = 0.10
while True:
    pool = np.where(np.abs(flat - abs(w_sw)) <= band * abs(w_sw))[0]
    pool = np.array([k for k in pool if k not in excl])
    if len(pool) >= 3 * args.n_null or band > 0.9:
        break
    band += 0.05
near = rng.choice(pool, min(args.n_null, len(pool)), replace=False)
res["null_band"] = band
res["null_pool_size"] = int(len(pool))
print(f"null pool: {len(pool)} weights with |w| within {band:.2f} of |w_sw|", flush=True)
null_idx = [(int(k // W.shape[1]), int(k % W.shape[1])) for k in near]
res["null_mag_range"] = [float(flat[near].min()), float(flat[near].max())]
res["null_each_held_nll"] = []
for j, rc in enumerate(null_idx):
    res["null_each_held_nll"].append(ablate_eval([rc], held))
    pasar_job.progress(j + 1, 400)
res["null_all_held_nll"] = ablate_eval(null_idx, held)
ne = np.exp(res["null_each_held_nll"])
print(f"null (100 same-|w| weights): each zeroed ppl min {ne.min():.3f} median {np.median(ne):.3f} max {ne.max():.3f}; "
      f"all 100 together {np.exp(res['null_all_held_nll']):.3f}", flush=True)
res["null_idx"] = null_idx

# ---- 2. first-order fragility map: |w * dL/dw| over the whole matrix (calibration set) ----
for p in model.parameters():
    p.requires_grad_(False)
W.requires_grad_(True)
G = torch.zeros_like(W)
ntok = sum(x.shape[1] - 1 for x in cal)
for ids in cal:
    lg = model(input_ids=ids).logits.float()[0, :-1]
    loss = F.cross_entropy(lg, ids[0, 1:], reduction="sum") / ntok
    g, = torch.autograd.grad(loss, W)
    G += g
W.requires_grad_(False)
pred = (-(W * G)).detach()  # first-order change in mean NLL from setting w -> 0
score = pred.abs()
sflat = score.flatten().cpu().numpy()
order = np.argsort(sflat)[::-1]
print("first-order: super weight score", sflat[row * W.shape[1] + col], "rank",
      int((sflat > sflat[row * W.shape[1] + col]).sum()) + 1, "; top5", sflat[order[:5]], flush=True)

# exact ablation of the top-N first-order candidates and a random sample, on the calibration set
cand = [(int(k // W.shape[1]), int(k % W.shape[1])) for k in order[: args.n_top]]
rnd = rng.choice(W.numel(), args.n_rand, replace=False)
randc = [(int(k // W.shape[1]), int(k % W.shape[1])) for k in rnd]
exact_top, exact_rand = [], []
for j, rc in enumerate(cand):
    exact_top.append(ablate_eval([rc], cal) - base_cal)
    pasar_job.progress(100 + j + 1, 400)
for j, rc in enumerate(randc):
    exact_rand.append(ablate_eval([rc], cal) - base_cal)
    pasar_job.progress(200 + j + 1, 400)
res["cand"] = cand
res["cand_pred"] = [float(pred[r, c]) for r, c in cand]
res["cand_exact"] = exact_top
res["rand"] = randc
res["rand_pred"] = [float(pred[r, c]) for r, c in randc]
res["rand_exact"] = exact_rand
for (r, c), p_, e in list(zip(cand, res["cand_pred"], exact_top))[:12]:
    print(f"  [{r:4d},{c:4d}] w={W[r, c].item():+.4f} pred dNLL {p_:+.5f} exact {e:+.5f}")

# greedy samples with and without the super weight
prompt = "My favourite condiment is"
ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
with torch.no_grad():
    g0 = model.generate(ids, max_new_tokens=30, do_sample=False)
    old = W[row, col].item()
    W[row, col] = 0.0
    g1 = model.generate(ids, max_new_tokens=30, do_sample=False)
    W[row, col] = old
res["gen_base"] = tok.decode(g0[0])
res["gen_sw0"] = tok.decode(g1[0])
print("base:", res["gen_base"])
print("no SW:", res["gen_sw0"])

os.makedirs(CACHE, exist_ok=True)
json.dump(res, open(os.path.join(CACHE, f"superweight_{args.tag}.json"), "w"), indent=1)
np.savez_compressed(os.path.join(CACHE, f"superweight_{args.tag}.npz"), pred=pred.cpu().numpy(),
                    W=W.detach().cpu().numpy())
pasar_job.progress(400, 400)
print("done", round(time.time() - t0, 1), "s")
