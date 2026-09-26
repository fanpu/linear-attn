"""Greedy-decode Qwen3 from every ordinary vocabulary token until the text enters a cycle.

One pasar job = one shard of a fixed random permutation of token ids 0..151642.
Each batch of B starting tokens is decoded in lock-step (all rows share the same position, so
no padding). A row stops when
  * it emits EOS (<|endoftext|> or <|im_end|>)                      -> reason 2
  * its trailing text is p-periodic over >= max(3p, min_span) + confirm tokens  -> reason 1
  * it reaches `cap` generated tokens                                -> reason 0
Finished rows are compacted out of the KV cache. Output: one npz per batch (resumable: existing
batch files are skipped). The exact cycle/entry analysis is redone offline by analyze.py from the
stored tokens.
"""
import argparse
import os
import time

import numpy as np
import torch

import pasar_job
from engine import Qwen3, EOS_IDS, N_REGULAR

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen3-0.6B")
ap.add_argument("--dtype", default="bf16")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=8)
ap.add_argument("--subset", type=int, default=0, help="only the first N ids of the permutation")
ap.add_argument("--B", type=int, default=1024)
ap.add_argument("--cap", type=int, default=512)
ap.add_argument("--min_span", type=int, default=16)
ap.add_argument("--confirm", type=int, default=32)
ap.add_argument("--pmax", type=int, default=128)
ap.add_argument("--check_every", type=int, default=4)
ap.add_argument("--stop", type=int, default=1, help="0 = never stop on cycles (pilot)")
ap.add_argument("--max_batches", type=int, default=0)
ap.add_argument("--ids", default="", help="npy file of explicit start ids (overrides sharding)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

perm = np.random.default_rng(0).permutation(N_REGULAR)
if args.subset:
    perm = perm[:args.subset]
shard = np.array_split(perm, args.nshards)[args.shard]
if args.ids:
    shard = np.load(args.ids)
batches = [shard[i:i + args.B] for i in range(0, len(shard), args.B)]
if args.max_batches:
    batches = batches[:args.max_batches]
out = os.path.join(args.out, f"shard{args.shard:02d}")
os.makedirs(out, exist_ok=True)
print(f"shard {args.shard}/{args.nshards}: {len(shard)} ids, {len(batches)} batches -> {out}", flush=True)

m = Qwen3(args.model, dtype=dtype)
m.alloc(args.B, args.cap + 1)
eos = torch.tensor(EOS_IDS, device="cuda")

t_job = time.time()
for bi, starts in enumerate(batches):
    fn = os.path.join(out, f"b{bi:03d}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time()
    n = len(starts)
    seq = torch.zeros(n, args.cap + 1, dtype=torch.int32, device="cuda")
    seq[:, 0] = torch.as_tensor(starts, device="cuda")
    length = torch.full((n,), args.cap + 1, dtype=torch.long, device="cuda")
    reason = torch.zeros(n, dtype=torch.int8, device="cuda")
    alive = torch.arange(n, device="cuda")              # cache row r holds batch row alive[r]
    cur = seq[:, 0].long()
    fin = torch.zeros(n, dtype=torch.bool, device="cuda")   # per cache row: finished, not yet removed
    work = 0
    for t in range(args.cap):                           # consume token at position t, emit t+1
        logits = m.step(cur, t)
        work += alive.numel()
        nxt = logits.argmax(-1)
        L = t + 2
        seq[alive[~fin], t + 1] = nxt[~fin].int()
        new = torch.isin(nxt, eos) & ~fin
        reason[alive[new]] = 2
        length[alive[new]] = L
        fin |= new
        if args.stop and (t + 1) % args.check_every == 0:
            # trailing text p-periodic over >= max(3p, min_span) + confirm tokens
            sub = seq[alive]
            found = torch.zeros(alive.numel(), dtype=torch.bool, device="cuda")
            for pp in range(1, args.pmax + 1):
                need = max(2 * pp, args.min_span - pp) + args.confirm
                if need + pp > L:
                    break
                found |= (sub[:, L - need:L] == sub[:, L - need - pp:L - pp]).all(1)
            cyc = found & ~fin
            reason[alive[cyc]] = 1
            length[alive[cyc]] = L
            fin |= cyc
        nfin = int(fin.sum())
        if nfin == alive.numel():
            break
        if nfin > 0 and (nfin >= 0.125 * alive.numel() or (t + 1) % 32 == 0):
            keep = (~fin).nonzero().squeeze(1)
            m.compact(keep, t + 1)
            alive = alive[keep]
            nxt = nxt[keep]
            fin = fin[keep]
        cur = nxt
    seq_np = seq.cpu().numpy()
    length_np = length.cpu().numpy()
    toks = np.concatenate([seq_np[i, :length_np[i]] for i in range(n)])
    np.savez(fn + ".tmp.npz", starts=starts.astype(np.int32), length=length_np.astype(np.int32),
             reason=reason.cpu().numpy(), tokens=toks.astype(np.int32))
    os.replace(fn + ".tmp.npz", fn)
    dt = time.time() - t0
    r = reason.cpu().numpy()
    print(f"batch {bi}: n={n} steps={t + 1} work={work} rows*steps  {dt:.1f}s  "
          f"cycle={np.mean(r == 1):.3f} eos={np.mean(r == 2):.3f} cap={np.mean(r == 0):.3f} "
          f"mean_len={length_np.mean():.1f}", flush=True)
    pasar_job.progress(bi + 1, len(batches))
    pasar_job.checkpoint(bi + 1)
print(f"done in {time.time() - t_job:.0f}s", flush=True)
