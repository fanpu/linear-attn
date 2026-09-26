"""Check the hand-written engine against transformers' Qwen3ForCausalLM (fp32, greedy, no template)."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from engine import Qwen3, N_REGULAR

torch.backends.cuda.matmul.allow_tf32 = False
ids = np.random.default_rng(1).choice(N_REGULAR, 16, replace=False)
T = 96
hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True,
                                          dtype=torch.float32).cuda().eval()
m = Qwen3("Qwen3-0.6B", dtype=torch.float32)
m.alloc(1, T + 1)
agree = []
with torch.no_grad():
    for tid in ids:
        x = torch.tensor([[int(tid)]], device="cuda")
        ref = hf.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=T, do_sample=False,
                          eos_token_id=None, pad_token_id=0)[0, 1:].cpu().numpy()
        cur = x[0]
        mine = []
        for t in range(T):
            nt = m.step(cur, t).argmax(-1)
            mine.append(int(nt))
            cur = nt
        mine = np.array(mine)
        n = min(len(ref), len(mine))
        d = np.nonzero(ref[:n] != mine[:n])[0]
        first = int(d[0]) if len(d) else n
        agree.append(first)
        print(int(tid), "first disagreement at", first, "of", n, flush=True)
print("engine vs HF fp32: identical", sum(a == T for a in agree), "/", len(ids))
