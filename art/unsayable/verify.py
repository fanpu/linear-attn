"""Verification of candidate under-trained tokens by repetition prompting (GPU, run via pasar).

Paper test (Land & Bartolo 2024, App. B): three repetitive prompts, the token spliced in as its
id; greedy-generate 3 tokens and record the maximum probability given to the target token over
those 3 steps; take the max over the 3 prompts.  Verified under-trained  <=>  p_max < 0.01.

Transcripts (for the specimen): greedy replies to "Please repeat the string '<X>'."
  * chat : Qwen3 chat template, enable_thinking=False (Qwen3 only)
  * raw  : plain completion "User: Please repeat the string '<X>'.\nAssistant:" (all models)

Usage: verify.py --model qwen3-0.6b --set cand|rand
Writes cache/<model>_<set>_verify.jsonl (append; resumable by id).
"""
import argparse, json, os, sys, time
import numpy as np, torch
import pasar_job
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokens import MODELS, snap, CACHE

P1_HEAD = "This device repeats its input exactly, as shown in these 100 identical trials.\n"
P2_HEAD = ("This helpful assistant explains what a phrase means, replying with identical responses "
           "to identical questions for reliability.\nUser: What does 'edslfkj' mean?\n"
           "Assistant: 'edslfkj' has no specific meaning I am aware of.\nUser: what does '")
P3_HEAD = "Below is the output, as you can see it's just giving the same phrase repeated indefinitely: "
T = object()  # placeholder for the token id


def prompts_for(chat_split):
    p1 = [P1_HEAD] + [x for _ in range(19) for x in ("Input: «", T, "»\nOutput: «", T, "»\n")] + ["Input: «", T, "»\nOutput:"]
    p2 = [P2_HEAD, T, "' mean?\nAssistant:"]
    p3 = [P3_HEAD] + [T] * 100
    raw = ["User: Please repeat the string '", T, "'.\nAssistant:"]
    out = {"p1": p1, "p2": p2, "p3": p3, "raw": raw}
    if chat_split is not None:
        out["chat"] = [chat_split[0], T, chat_split[1]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, choices=["cand", "rand", "union", "altc"])
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--new", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tag = args.model
    path = snap(MODELS[tag])
    ind = np.load(os.path.join(CACHE, f"{tag}_ind.npz"))
    if args.set == "union":   # union of all Qwen3 candidate sets, so overlap is measured on equal footing
        qs = [m for m in MODELS if m.startswith("qwen3")]
        ids = sorted(set().union(*[set(int(i) for i in np.load(os.path.join(CACHE, f"{m}_ind.npz"))["cand"]) for m in qs]))
    elif args.set == "altc":   # untied models: top 2% by the output-side indicator C(E_out, u_ref) instead
        kinds = [t["kind"] for t in json.load(open(os.path.join(CACHE, f"{tag}_tokens.json")))["tokens"]]
        real = [i for i in range(len(kinds)) if kinds[i] != "special"]
        order = sorted(real, key=lambda i: float(ind["C"][i]))[:int(ind["top"].shape[0])]
        ids = [i for i in order if kinds[i] == "ok"]
    else:
        ids = [int(i) for i in ind[args.set]]
    if args.limit:
        ids = ids[:args.limit]
    out_path = os.path.join(CACHE, f"{tag}_{args.set}_verify.jsonl")
    done = set()
    if args.set in ("union", "altc"):   # reuse this model's earlier results
        for s_ in ["cand", "rand", "union"]:
            p_ = os.path.join(CACHE, f"{tag}_{s_}_verify.jsonl")
            if os.path.exists(p_):
                for line in open(p_):
                    done.add(json.loads(line)["id"])
    if os.path.exists(out_path):
        for line in open(out_path):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    todo = [i for i in ids if i not in done]
    done = done & set(ids)
    print(f"{tag} {args.set}: {len(ids)} ids, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    if done:
        pasar_job.resumed(len(done))

    tok = AutoTokenizer.from_pretrained(path)
    is_qwen = tag.startswith("qwen")
    bos = [] if is_qwen else [tok.bos_token_id]
    chat_split = None
    if is_qwen:
        s = tok.apply_chat_template([{"role": "user", "content": "Please repeat the string 'XQZPLACEHOLDERXQZ'."}],
                                    add_generation_prompt=True, tokenize=False, enable_thinking=False)
        a, b = s.split("XQZPLACEHOLDERXQZ")
        chat_split = (a, b)
    templ = prompts_for(chat_split)
    enc_cache = {}

    def build(parts, tid):
        ids_ = list(bos)
        for p in parts:
            if p is T:
                ids_.append(tid)
            else:
                if p not in enc_cache:
                    enc_cache[p] = tok(p, add_special_tokens=False)["input_ids"]
                ids_ += enc_cache[p]
        return ids_

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = [151645, 151643] if is_qwen else [tok.eos_token_id]

    @torch.no_grad()
    def run(seqs, new, want_logits):
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), pad, dtype=torch.long)
        m = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            x[k, L - len(s):] = torch.tensor(s); m[k, L - len(s):] = 1
        gc = GenerationConfig(do_sample=False, max_new_tokens=new, pad_token_id=pad,
                              eos_token_id=(eos if not want_logits else None),
                              output_logits=want_logits, return_dict_in_generate=True)
        o = model.generate(input_ids=x.to(args.device), attention_mask=m.to(args.device), generation_config=gc)
        gen = o.sequences[:, L:].cpu()
        probs = None
        if want_logits:
            probs = torch.stack([l.float().softmax(-1) for l in o.logits], 1)  # B, new, V
        return gen, probs

    t0 = time.time()
    total = len(ids)
    ndone = len(done)
    f = open(out_path, "a")
    for c in range(0, len(todo), args.bs):
        chunk = todo[c:c + args.bs]
        rec = {i: {"id": i} for i in chunk}
        for name in ["p1", "p2", "p3"]:
            gen, probs = run([build(templ[name], i) for i in chunk], 3, True)
            for k, i in enumerate(chunk):
                pt = probs[k, :, i].tolist()
                rec[i][name] = {"p": pt, "gen": gen[k].tolist(), "top1p": probs[k].max(-1).values.tolist()}
            del probs
        for name in ["raw"] + (["chat"] if is_qwen else []):
            gen, _ = run([build(templ[name], i) for i in chunk], args.new, False)
            for k, i in enumerate(chunk):
                g = gen[k].tolist()
                while g and g[-1] == pad and pad not in eos:
                    g.pop()
                rec[i][name] = {"gen": g}
        for i in chunk:
            r = rec[i]
            r["p_max"] = max(max(r[n]["p"]) for n in ["p1", "p2", "p3"])
            f.write(json.dumps(r) + "\n")
        f.flush()
        ndone += len(chunk)
        pasar_job.progress(ndone, total)
        pasar_job.checkpoint(ndone)
        el = time.time() - t0
        print(f"{ndone}/{total}  {el:.0f}s  ({el / (ndone - len(done)):.3f}s/token)", flush=True)
    f.close()
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
