"""Check the fast top-K sampler against a naive full-sort float64 reference on real logits."""
import torch
from transformers import AutoTokenizer
from qwen import Qwen3
from sample import Sampler, chat_ids, PROMPTS
torch.cuda.set_per_process_memory_fraction(0.10)


def reference(logits, T, p, rho, seen, u):
    z = logits.double()
    z = torch.where(seen, torch.where(z > 0, z / rho[:, None], z * rho[:, None]), z) / T[:, None]
    zs, idx = torch.sort(z, -1, descending=True)
    q = torch.softmax(zs, -1)
    C = torch.cumsum(q, -1)
    n = ((C - q) < p[:, None]).sum(-1).clamp(1, z.shape[1])
    Z = torch.where(p >= 1, torch.ones_like(p), C.gather(1, (n - 1)[:, None])[:, 0])
    r = torch.searchsorted(C, (u * Z)[:, None])[:, 0]
    r = torch.where(p >= 1, r.clamp_max(z.shape[1] - 1), torch.minimum(r, n - 1))
    return idx.gather(1, r[:, None])[:, 0]


tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
m = Qwen3()
B = 256
S = Sampler(m, chat_ids(tok, PROMPTS["story"]), 8, B, K=64, use_pen=True)
g = torch.Generator(device="cuda").manual_seed(1)
mism = 0
for trial in range(20):
    T = (torch.rand(B, generator=g, device="cuda", dtype=torch.float64) * 2 + 1e-3)
    p = torch.rand(B, generator=g, device="cuda", dtype=torch.float64)
    p[:64] = 1.0
    rho = 1 + torch.rand(B, generator=g, device="cuda", dtype=torch.float64)
    seen = torch.rand(B, S.V, generator=g, device="cuda") < 0.001
    # varied logits: prefill logits plus noise, some rows duplicated to exercise sharing
    logits = S.logits0[None] + 2 * torch.randn(B, S.V, generator=g, device="cuda")
    logits[B // 2:] = logits[0]
    seen[B // 2:] = seen[0]
    key = torch.cat([rho.view(torch.int64)[:, None], torch.arange(B, device="cuda")[:, None].clamp_max(B // 2)], 1)
    rho[B // 2:] = rho[0]; key[B // 2:, 0] = key[0, 0]; key[B // 2:, 1] = 0; key[0, 1] = 0
    t = trial % 8
    tk = S.step_sample(logits, T, p, rho, seen, t, key)[0]
    ref = reference(logits, T, p, rho, seen, S.u[t])
    mism += (tk != ref).sum().item()
print("fallback rows", S.n_fallback, "of", 20 * B, " token mismatches vs reference:", mism)
