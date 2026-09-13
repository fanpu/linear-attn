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

Numerics: bf16 transformer body (--fp32 for full fp32), fp32 final norm + unembedding, float64 softmax/CDF, no padding (every row has the same
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
    def __init__(self, model, prompt_ids, L, batch, rule="icdf", seed=0, K=1024, use_pen=False):
        self.m, self.L, self.B, self.rule = model, L, batch, rule
        self.K, self.use_pen, self.n_fallback = K, use_pen, 0
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
    def step_sample(self, logits, T, p, rho, seen, t, key):
        """Exact inverse-CDF / Gumbel step. The sorted order is taken from a top-K
        (rows sharing prefix and penalty share it); rows whose nucleus or target
        lies beyond the top-K fall back to a full float64 sort, so the result is
        identical to a full sort (up to cumsum association, ~1e-16)."""
        B, V, K = logits.shape[0], self.V, self.K
        dev = logits.device
        lsm32 = torch.log_softmax(logits, -1)
        H_model = -(lsm32.exp() * lsm32).sum(-1)
        z = logits.double()
        if self.use_pen:
            pen = rho[:, None]
            z = torch.where(seen, torch.where(z > 0, z / pen, z * pen), z)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        rep = torch.zeros(uniq.shape[0], dtype=torch.long, device=dev).scatter_(
            0, inv, torch.arange(B, device=dev))
        vals, ix = torch.topk(z[rep], K, -1, sorted=True)
        zT = z / T[:, None]
        lse = torch.logsumexp(zT, -1)
        zs = vals[inv] / T[:, None]
        idx = ix[inv]
        q = torch.exp(zs - lse[:, None])
        C = torch.cumsum(q, -1)
        full = p >= 1.0
        n = ((C - q) < p[:, None]).sum(-1).clamp_min(1)
        nok = full | (C[:, -1] >= p)
        Z = torch.where(full, torch.ones_like(p), C.gather(1, (n - 1).clamp_max(K - 1)[:, None])[:, 0])
        thr = torch.where(full, torch.full_like(p, -torch.inf), zs.gather(1, (n - 1).clamp_max(K - 1)[:, None])[:, 0])
        target = self.u[t] * Z
        ok = nok & (~full | (target <= C[:, -1]))
        if self.rule == "gumbel":
            ok = nok
        token = torch.zeros(B, dtype=torch.long, device=dev)
        if self.rule == "icdf":
            r = torch.searchsorted(C, target[:, None])[:, 0].clamp_max(K - 1)
            r = torch.where(full, r, torch.minimum(r, n - 1))
            token = idx.gather(1, r[:, None])[:, 0]
        bad = (~ok).nonzero()[:, 0]
        self.n_fallback += len(bad)
        if len(bad):
            zb = zT[bad]
            zsb, ixb = torch.sort(zb, -1, descending=True)
            qb = torch.exp(zsb - lse[bad][:, None])
            Cb = torch.cumsum(qb, -1)
            pb = p[bad]
            nb = ((Cb - qb) < pb[:, None]).sum(-1).clamp(1, V)
            fb = pb >= 1.0
            Zb = torch.where(fb, torch.ones_like(pb), Cb.gather(1, (nb - 1)[:, None])[:, 0])
            thb = torch.where(fb, torch.full_like(pb, -torch.inf), zsb.gather(1, (nb - 1)[:, None])[:, 0])
            Z[bad], thr[bad] = Zb, thb
            if self.rule == "icdf":
                rb = torch.searchsorted(Cb, (self.u[t] * Zb)[:, None])[:, 0]
                rb = torch.minimum(rb, nb - 1)
                token[bad] = ixb.gather(1, rb[:, None])[:, 0]
            del zb, zsb, ixb, qb, Cb
        kept = zT >= thr[:, None]
        logq = (zT - lse[:, None])
        if self.rule == "gumbel":
            y = torch.where(kept, zT + self.gum[t][None], torch.full_like(zT, -torch.inf))
            token = y.argmax(-1)
        lq32 = logq.float()
        H_samp = -(torch.where(kept, lq32.exp() * lq32, torch.zeros((), device=dev))).sum(-1) / Z.float() \
            + torch.log(Z.float())
        logp = lsm32.gather(1, token[:, None])[:, 0]
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
            key = torch.cat([rho.view(torch.int64)[:, None], toks[:, :t]], 1)
            tk, a, b, c = self.step_sample(logits, T, p, rho, seen, t, key)
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
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--fp32", action="store_true", help="fp32 transformer body (default bf16 body, fp32 head)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.y is None:
        args.y = [0.0, 1.0] if args.grid == "tp" else [1.0, 2.0]

    torch.cuda.set_per_process_memory_fraction(0.10)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    pids = chat_ids(tok, PROMPTS[args.prompt])
    model = Qwen3(dtype=torch.float32 if args.fp32 else torch.bfloat16)
    S = Sampler(model, pids, args.L, args.batch, args.rule, args.seed, args.K, use_pen=args.grid == "tr")

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
            print(f"chunk {c + 1}/{nchunks}  fallback rows {S.n_fallback}  {el:.0f}s  eta {el / (c + 1) * (nchunks - c - 1):.0f}s", flush=True)

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
        u=S.u.cpu().numpy(), fp32_body=args.fp32, K=args.K, n_fallback=S.n_fallback, p_fixed=args.p_fixed, seed=args.seed, L=args.L,
        wall=time.time() - t0, batch=args.batch)
    print("saved", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
