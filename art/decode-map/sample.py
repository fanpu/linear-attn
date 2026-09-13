"""Decode map sweep: one full generation per pixel of a 2D grid of decoding knobs.

Sampling rule (declared):  INVERSE-CDF on the probability-sorted vocabulary with
per-position uniforms u_t shared by every pixel of every run of a prompt.

  z      = fp32 logits -> float64
  z_i   <- z_i / rho  if z_i > 0  else z_i * rho     for tokens already seen   (HF repetition penalty,
                                                     prompt + generated tokens, rho = 1 means off)
  q      = softmax(z / T)   in float64, sorted descending (ties: kernel order)
  n      = #{k : sum_{j<k} q_j < p}                 (HF top-p: keep smallest prefix with mass >= p)
  Z      = sum_{k<n} q_k
  token  = sorted index r with  C_{r-1} < u_t Z <= C_r ,  C = cumsum(q)

so for fixed u the output is a piecewise-constant function of (T, p, rho), and a
cell boundary is exactly where some cumulative sum crosses u_t Z or p.

--rule gumbel instead uses Gumbel-max with shared noise g_{t,v} (indexed by vocab id):
  token = argmax_{v in top-p set} z_v / T + g_{t,v}
which defines a different partition (see README).

Numerics: fp32 model, float64 softmax/CDF, no padding (every row has the same
length; the shared prompt is prefilled once and its KV copied to every row), fixed
batch size (last chunk padded with copies of its last pixel).
"""
import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from qwen import Qwen3

PROMPTS = {
    "story": "Write a short story about a lighthouse keeper.",
    "list": "List some animals that live in the ocean.",
    "fact": "What is the capital of Australia?",
}
EOS = (151645, 151643)


def chat_ids(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(s).input_ids


def shared_noise(seed, L, V, rule):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(L, generator=g, dtype=torch.float64)
    gum = None
    if rule == "gumbel":
        e = torch.rand(L, V, generator=g, dtype=torch.float64).clamp_min(1e-300)
        gum = -torch.log(-torch.log(e))
    return u, gum


class Sampler:
    def __init__(self, model, prompt_ids, L, batch, rule="icdf", seed=0):
        self.m, self.L, self.B, self.rule = model, L, batch, rule
        self.P = len(prompt_ids)
        self.V = model.w["lm_head.weight"].shape[0]
        u, gum = shared_noise(seed, L, self.V, rule)
        self.u = u.cuda()
        self.gum = gum.cuda() if gum is not None else None
        # prefill once (batch 1), keep KV and the first logits
        model.alloc(1, self.P)
        ids = torch.tensor(prompt_ids, device="cuda")[None]
        self.logits0 = model.forward(ids, 0)[0]
        self.k0 = model.cache_k.clone()
        self.v0 = model.cache_v.clone()
        self.prompt_mask = torch.zeros(self.V, dtype=torch.bool, device="cuda")
        self.prompt_mask[ids[0]] = True
        model.alloc(batch, self.P + L)

    @torch.no_grad()
    def step_sample(self, logits, T, p, rho, seen, t):
        B = logits.shape[0]
        zr = logits.double()
        lsm = torch.log_softmax(zr, -1)
        H_model = -(lsm.exp() * lsm).sum(-1)
        z = zr
        if (rho != 1).any():
            pen = rho[:, None]
            z = torch.where(seen, torch.where(z > 0, z / pen, z * pen), z)
        z = z / T[:, None]
        zs, idx = torch.sort(z, dim=-1, descending=True)
        q = torch.softmax(zs, -1)
        C = torch.cumsum(q, -1)
        before = C - q
        n = (before < p[:, None]).sum(-1).clamp(1, self.V)          # kept prefix length
        Z = C.gather(1, (n - 1)[:, None])[:, 0]
        if self.rule == "icdf":
            target = (self.u[t] * Z)[:, None]
            r = torch.searchsorted(C, target, right=False)[:, 0]
            r = torch.minimum(r, n - 1)
        else:
            y = zs + self.gum[t][idx]
            ar = torch.arange(self.V, device="cuda")[None]
            y = torch.where(ar < n[:, None], y, torch.tensor(-torch.inf, dtype=y.dtype, device="cuda"))
            r = y.argmax(-1)
        token = idx.gather(1, r[:, None])[:, 0]
        keep = torch.arange(self.V, device="cuda")[None] < n[:, None]
        qn = torch.where(keep, q / Z[:, None], torch.zeros((), dtype=q.dtype, device="cuda"))
        H_samp = -(qn * torch.log(qn.clamp_min(1e-300))).sum(-1)
        logp = lsm.gather(1, token[:, None])[:, 0]
        return token, H_model, H_samp, logp

    @torch.no_grad()
    def run_chunk(self, T, p, rho):
        """T, p, rho: float64 cuda tensors of length B. Returns dict of [B, L] arrays."""
        B, L, m = self.B, self.L, self.m
        assert T.shape[0] == B
        m.cache_k[:, :, :, :self.P] = self.k0
        m.cache_v[:, :, :, :self.P] = self.v0
        seen = self.prompt_mask[None].repeat(B, 1)
        toks = torch.empty(B, L, dtype=torch.long, device="cuda")
        Hm = torch.empty(B, L, device="cuda")
        Hs = torch.empty(B, L, device="cuda")
        lp = torch.empty(B, L, device="cuda")
        logits = self.logits0[None].expand(B, -1)
        ar = torch.arange(B, device="cuda")
        for t in range(L):
            if t > 0:
                logits = m.forward(toks[:, t - 1:t], self.P + t - 1)
            tk, a, b, c = self.step_sample(logits, T, p, rho, seen, t)
            toks[:, t] = tk
            Hm[:, t], Hs[:, t], lp[:, t] = a, b, c
            seen[ar, tk] = True
        return dict(tokens=toks, H_model=Hm, H_samp=Hs, logp=lp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="story")
    ap.add_argument("--grid", default="tp", choices=["tp", "tr"])
    ap.add_argument("--rule", default="icdf", choices=["icdf", "gumbel"])
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--x", type=float, nargs=2, default=[0.0, 2.0], help="temperature range")
    ap.add_argument("--y", type=float, nargs=2, default=None, help="top-p or penalty range")
    ap.add_argument("--p_fixed", type=float, default=1.0, help="top-p used in the tr grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.y is None:
        args.y = [0.0, 1.0] if args.grid == "tp" else [1.0, 2.0]

    torch.cuda.set_per_process_memory_fraction(0.10)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    pids = chat_ids(tok, PROMPTS[args.prompt])
    model = Qwen3()
    S = Sampler(model, pids, args.L, args.batch, args.rule, args.seed)

    R = args.res
    xs = args.x[0] + (np.arange(R) + 0.5) / R * (args.x[1] - args.x[0])
    ys = args.y[0] + (np.arange(R) + 0.5) / R * (args.y[1] - args.y[0])
    YY, XX = np.meshgrid(ys, xs, indexing="ij")          # row i <-> y, col j <-> x
    Tall = XX.ravel()
    if args.grid == "tp":
        Pall, Rall = YY.ravel(), np.ones(R * R)
    else:
        Pall, Rall = np.full(R * R, args.p_fixed), YY.ravel()

    N = R * R
    out = {k: np.zeros((N, args.L), dt) for k, dt in
           [("tokens", np.int32), ("H_model", np.float16), ("H_samp", np.float16), ("logp", np.float16)]}
    t0 = time.time()
    nchunks = (N + args.batch - 1) // args.batch
    for c in range(nchunks):
        sl = np.arange(c * args.batch, min(N, (c + 1) * args.batch))
        pad = np.concatenate([sl, np.full(args.batch - len(sl), sl[-1])])
        f = lambda a: torch.tensor(a[pad], dtype=torch.float64, device="cuda")
        res = S.run_chunk(f(Tall), f(Pall), f(Rall))
        for k in out:
            out[k][sl] = res[k][:len(sl)].cpu().numpy()
        if c % 8 == 0 or c == nchunks - 1:
            el = time.time() - t0
            print(f"chunk {c + 1}/{nchunks}  {el:.0f}s  eta {el / (c + 1) * (nchunks - c - 1):.0f}s", flush=True)

    toks = out["tokens"]
    iseos = np.isin(toks, EOS)
    has = iseos.any(1)
    first = np.where(has, iseos.argmax(1), args.L)
    after = np.arange(args.L)[None] > first[:, None]
    toks[after] = -1                                     # nothing exists after EOS
    for k in ("H_model", "H_samp", "logp"):
        out[k][after] = np.nan
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out, **{k: v.reshape(R, R, args.L) for k, v in out.items()},
        eos_pos=first.reshape(R, R).astype(np.int16), xs=xs, ys=ys, grid=args.grid, rule=args.rule,
        prompt=PROMPTS[args.prompt], prompt_key=args.prompt, prompt_ids=np.array(pids),
        u=S.u.cpu().numpy(), p_fixed=args.p_fixed, seed=args.seed, L=args.L,
        wall=time.time() - t0, batch=args.batch)
    print("saved", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
