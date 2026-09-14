"""Piece 2 (+3): same prompt, greedy decoding (temperature 0), different batch sizes.

B identical copies of one chat prompt are prefilled and decoded together with decode-map's hand-written
Qwen3-0.6B (bf16 body, fp32 final norm + head, static KV cache, eager, no CUDA graphs). Row 0's tokens
are recorded for every B. Per step we also keep the top-2 logit margin and, against the B = 1 logits
at the same step, max and median |delta logit| (meaningful while the prefixes still agree).
Kernel signatures (torch.profiler) of one prefill and one decode step are stored per B.

  python divergence.py --prompt feynman --L 320 --out cache/div_feynman.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

import common as C

sys.path.insert(0, C.DECODE_MAP)
from qwen import Qwen3  # noqa: E402

PROMPTS = {
    "feynman": "Tell me about Richard Feynman",          # the Thinking Machines prompt
    "story": "Write a short story about a lighthouse keeper.",   # decode-map's prompt
    "sky": "Explain why the sky is blue.",
}
DEFAULT_BS = list(range(1, 65)) + list(range(72, 129, 8)) + list(range(160, 513, 32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="feynman")
    ap.add_argument("--L", type=int, default=320)
    ap.add_argument("--bs", type=int, nargs="*", default=DEFAULT_BS)
    ap.add_argument("--noprof", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(C.ROOT)
    out = a.out or f"cache/div_{a.prompt}.npz"
    tag = os.path.splitext(os.path.basename(out))[0]
    before = C.smi(f"before_{tag}")
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    s = tok.apply_chat_template([{"role": "user", "content": PROMPTS[a.prompt]}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    pid = torch.tensor(tok(s).input_ids, device="cuda")
    P, L = pid.numel(), a.L
    m = Qwen3()
    Bs = list(a.bs)
    if Bs[0] != 1:
        Bs = [1] + Bs
    NB = len(Bs)
    toks = np.zeros((NB, L), np.int64)
    margin = np.zeros((NB, L), np.float32)
    dmax = np.full((NB, L), np.nan, np.float32)
    dmed = np.full((NB, L), np.nan, np.float32)
    rows_disagree = np.zeros((NB, L), np.int32)
    wall = np.zeros(NB)
    kern_pre, kern_dec = [], []
    ref_logits = None

    def decode(B, record=True):
        m.alloc(B, P + L)
        ids = pid[None].expand(B, P).contiguous()
        lg = m.forward(ids, 0)
        tk = torch.empty(B, L, dtype=torch.long, device="cuda")
        lgs = []
        for t in range(L):
            nxt = lg.argmax(-1)
            tk[:, t] = nxt
            lgs.append(lg[0])
            if t < L - 1:
                lg = m.forward(nxt[:, None], P + t)
        return tk, torch.stack(lgs)

    t00 = time.time()
    with torch.inference_mode():
        for j, B in enumerate(Bs):
            torch.cuda.synchronize()
            t0 = time.time()
            tk, lgs = decode(B)
            torch.cuda.synchronize()
            wall[j] = time.time() - t0
            toks[j] = tk[0].cpu().numpy()
            rows_disagree[j] = (tk != tk[:1]).sum(0).cpu().numpy()
            top2 = lgs.topk(2, -1).values
            margin[j] = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            if ref_logits is None:
                ref_logits = lgs.clone()
            d = (lgs.double() - ref_logits.double()).abs()
            dmax[j] = d.max(-1).values.cpu().numpy()
            dmed[j] = d.median(-1).values.cpu().numpy()
            div = np.nonzero(toks[j] != toks[0])[0]
            first = int(div[0]) if len(div) else -1
            if not a.noprof:
                m.alloc(B, P + 2)
                ids = pid[None].expand(B, P).contiguous()
                kern_pre.append(C.cuda_kernels(lambda: m.forward(ids, 0)))
                nx = torch.zeros(B, 1, dtype=torch.long, device="cuda")
                kern_dec.append(C.cuda_kernels(lambda: m.forward(nx, P)))
            print(f"B={B:4d} first_div={first:4d} max|dlogit|@t0={dmax[j,0]:.4f} rows_disagree={rows_disagree[j].max()} "
                  f"{wall[j]:.1f}s", flush=True)
            del tk, lgs, d
            torch.cuda.empty_cache()   # every B allocates a different KV-cache size; don't let the allocator keep them all
        # run-to-run repeats
        rep = {}
        for B in (1, 37):
            tk, _ = decode(B)
            j = Bs.index(B) if B in Bs else None
            rep[B] = bool(j is not None and np.array_equal(tk[0].cpu().numpy(), toks[j]))
        print("run-to-run identical:", rep)
    after = C.smi(f"after_{tag}")
    np.savez_compressed(out, Bs=np.array(Bs), toks=toks, margin=margin, dmax=dmax, dmed=dmed,
                        rows_disagree=rows_disagree, wall=wall, prompt_ids=pid.cpu().numpy())
    C.save_json(out.replace(".npz", ".json"), dict(
        prompt=PROMPTS[a.prompt], P=P, L=L, stack=C.stack(), smi_before=before, smi_after=after,
        run_to_run=rep, wall_s=time.time() - t00, kern_prefill=kern_pre, kern_decode=kern_dec,
        texts={str(B): tok.decode(toks[j]) for j, B in enumerate(Bs)}))
    print("saved", out, f"{time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
