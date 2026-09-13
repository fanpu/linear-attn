"""Part B compute (small probes for the woven full-grid tapestries).

Short repeated random sequences so that the full 28 x 16 grid can be rendered
with several pixels per attention cell. Also stores per-head scores averaged
over 16 random draws of each probe shape.

  python compute_qwen_tapestry.py   -> cache/qwen_tapestry.npz
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM

torch.cuda.set_per_process_memory_fraction(0.10)
dev = 'cuda'
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True, dtype=torch.float32,
                                             attn_implementation='eager').to(dev).eval()
NORMAL_VOCAB = 151643
rs = np.random.default_rng(1729)
save = {}
for P, R in [(12, 4), (8, 6), (16, 3), (6, 8)]:
    T = P * R
    toks = np.stack([np.tile(rs.choice(NORMAL_VOCAB, P, replace=False), R) for _ in range(16)])
    with torch.no_grad():
        out = model(torch.tensor(toks, device=dev), output_attentions=True)
    att = torch.stack(out.attentions, 1).float()       # B,L,H,T,T
    idx = torch.arange(T, device=dev)
    ii = idx[P:]
    ind = att[:, :, :, ii, ii - P + 1].mean((0, 3))
    prev = att[:, :, :, idx[1:], idx[:-1]].mean((0, 3))
    key = f'P{P}R{R}'
    save[f'{key}_attn'] = att[0].half().cpu().numpy()
    save[f'{key}_mean_attn'] = att.mean(0).half().cpu().numpy()
    save[f'{key}_ind'] = ind.cpu().numpy()
    save[f'{key}_prev'] = prev.cpu().numpy()
    save[f'{key}_tokens'] = toks
    print(key, 'max ind', ind.max().item(), flush=True)
np.savez_compressed('cache/qwen_tapestry.npz', **save)
