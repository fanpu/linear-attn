"""Trie engine vs brute-force sampler (sample.py) vs a naive per-pixel reference loop.

The naive reference decodes each pixel alone (batch 1, fp32 body) with a full float64 sort
at every step. This is the ground truth for the sampling rule. The engines are bf16 by default,
so this test uses --fp32 for all of them, and reports token-level agreement."""
import sys
import numpy as np
import torch
from transformers import AutoTokenizer
from qwen import Qwen3
from sample import chat_ids, PROMPTS, shared_noise, EOS
from decode import Engine
torch.cuda.set_per_process_memory_fraction(0.10)

grid = sys.argv[1] if len(sys.argv) > 1 else "tp"
rule = sys.argv[2] if len(sys.argv) > 2 else "icdf"
R, L = 12, 12
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
pids = chat_ids(tok, PROMPTS["list"])
m = Qwen3(dtype=torch.float32)
xs = (np.arange(R) + .5) / R * 2.0
ys = (np.arange(R) + .5) / R
YY, XX = np.meshgrid(ys, xs, indexing="ij")
T = XX.ravel(); P = YY.ravel() if grid == "tp" else np.full(R * R, 1.0); RHO = np.ones(R * R) if grid == "tp" else 1 + YY.ravel()
if grid == "tr":
    P = np.where(np.arange(R * R) % 3 == 0, 0.9, 1.0)   # mix truncated and untruncated
f = lambda a: torch.tensor(a, dtype=torch.float64, device="cuda")
E = Engine(m, pids, L, rule, 0, K=16, Ncap=8, Fb=4, use_pen=grid == "tr", pix_batch=37)
E.run(f(T), f(P), f(RHO), [torch.arange(R * R)])
eng = E.out_tok.cpu().numpy()
print("engine stats", E.stats)

u, gum = shared_noise(0, L, E.V, rule)
u = u.cuda(); gum = gum.cuda() if gum is not None else None
m.alloc(1, len(pids) + L)
mism, first_bad, tail_ranks = 0, None, []
for i in range(R * R):
    ids = list(pids)
    seen = torch.zeros(E.V, dtype=torch.bool, device="cuda"); seen[torch.tensor(pids)] = True
    logits = m.forward(torch.tensor(ids, device="cuda")[None], 0)[0]
    out, ranks = [], []
    for t in range(L):
        if t > 0:
            logits = m.forward(torch.tensor([[out[-1]]], device="cuda"), len(pids) + t - 1)[0]
        z = logits.double()
        z = torch.where(seen, torch.where(z > 0, z / RHO[i], z * RHO[i]), z) / T[i]
        zs, ix = torch.sort(z, descending=True)
        q = torch.softmax(zs, 0); C = torch.cumsum(q, 0)
        if P[i] < 1:
            n = int(((C - q) < P[i]).sum().clamp_min(1)); Z = C[n - 1]
        else:
            n = E.V; Z = torch.tensor(1.0, dtype=torch.float64, device="cuda")
        if rule == "icdf":
            r = min(int(torch.searchsorted(C, (u[t] * Z)[None])[0]), n - 1)
            tk = int(ix[r])
            ranks.append(r)
        else:
            tk = int(ix[:n][(zs[:n] + gum[t][ix[:n]]).argmax()])
        out.append(tk); seen[tk] = True
        if tk in EOS:
            break
    ref = np.full(L, -1); ref[:len(out)] = out
    if not np.array_equal(ref, eng[i]):
        mism += 1
        d = int(np.argmax(ref != eng[i]))
        tail_ranks.append(ranks[d] if ranks else -1)
        if first_bad is None:
            first_bad = (i, ref, eng[i])
print(f"grid={grid} rule={rule}: {mism}/{R*R} pixels differ from naive reference")
print("sorted-rank of the reference token at the first differing step:", tail_ranks)
print("(ranks in the thousands = near-tied tail tokens whose order is set by ~1e-6 logit noise)")
