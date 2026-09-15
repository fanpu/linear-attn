"""
Day 3: transformer baselines on FineWeb-Edu, one run per (size, seed). Attention only today.

Ownership (RESEARCH_PROGRAM.md §1, loud vs silent):
  Claude (loud):  model plumbing, loader wiring, optimizer/param groups, logging, checkpoint/resume,
                  the loop itself. Wrong -> crash, NaN, or loss that visibly does not fall.
  Fan Pu (silent): lr_at, evaluate, check_disjoint (stubs below). Wrong -> runs fine, wrong number.

Usage:
  python train.py lr-test                                   # sanity-check your lr_at against the spec
  python train.py check-data --data data/fineweb_edu        # your train/val disjointness check
  python train.py run --size 30M --seed 0 --steps 300 --out runs/smoke_s0     # smoke test
  python train.py run --size 30M --seed 0 --tokens 300000000 --out runs/30M_s0  # real run
  python train.py run ... --resume                          # continue from the latest checkpoint in --out

Outputs in --out: log.jsonl (train/val/lr/tok_s rows), ckpt_XX.pt (every 10%), summary.json (at end).
Config (fixed for the baseline; D6 revisits it): AdamW betas (0.9, 0.95), wd 0.1 on >=2-D params,
peak LR 6e-4, warmup 3% linear, cosine to 0.1 x peak, clip 1.0, B=32, T=1024, bf16 autocast,
fp32 master weights, no torch.compile (same as Day 2 so tok/s is comparable), SDPA FLASH_ATTENTION.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from bench_throughput import build_config, install_sdpa_shim, param_buckets
except ImportError as e:
    sys.exit(f"train.py imports shapes from bench_throughput.py (Day 2); put both in the same directory. ({e})")
from data import TrainShards, make_order, train_batches, val_windows

VOCAB = 50304  # GPT-2's 50257, padded up to a multiple of 64 (nanoGPT / llm.c convention)
DEV = "cuda"


# ======================= FAN PU OWNS THESE (silent-failure code) =======================
def lr_at(step, total_steps, peak_lr=6e-4, warmup_frac=0.03, final_frac=0.1):
    """
    Learning rate at `step` (0-indexed; the LR used for the update that consumes batch `step`).
    Spec:
      warmup_steps = max(1, round(warmup_frac * total_steps))
      step < warmup_steps:  linear from 0 at step 0 ... to peak_lr at step == warmup_steps
                            (i.e. lr = peak_lr * (step + 1) / warmup_steps)
      otherwise:            cosine from peak_lr at step == warmup_steps down to final_frac * peak_lr
                            at step == total_steps - 1 (the last update), never below that.
    `python train.py lr-test` checks the endpoints and monotonicity.
    """
    raise NotImplementedError("Fan Pu writes lr_at (RESEARCH_PROGRAM.md §1: silent-failure code)")


@torch.no_grad()
def evaluate(model, val, B, sdpa_ctx):
    """
    Token-weighted mean NLL in nats over the fixed validation windows `val` (int64 [N, T]).
    Spec:
      - model.eval() during, model.train() after (RMSNorm has no dropout; this is for hygiene).
      - same forward path as training: sdpa_ctx() + bf16 autocast, model(input_ids=x, labels=x,
        use_cache=False).loss. That loss is fla's mean over the T-1 predicted positions of each window
        (fla shifts labels left and pads the last position with ignore_index; verify this yourself in
        fla/models/transformer/modeling_transformer.py, `labels = torch.cat((labels[..., 1:], ...`).
      - all windows have the same length, so a mean of per-batch means equals the token-weighted mean
        ONLY if every batch has the same number of windows. Handle N % B != 0 correctly (drop the
        ragged tail, or weight it) and say which in a comment.
      - return a Python float.
    Check: at step 0, with fla's N(0, 0.02^2) init, expect ln(50304) = 10.83 plus a small positive
    term, roughly 10.8 to 11.1. Far outside that band = your evaluate() or the label shift is wrong.
    """
    raise NotImplementedError("Fan Pu writes evaluate")


def check_disjoint(data_dir, T=1024):
    """
    Confirm no training window appears verbatim in validation. Suggested: hash every T-token window
    of the val shard (bytes of the uint16 row -> hash()) into a set, then stream every T-token window
    of every train shard and count set hits. Expected hits: 0 (a handful is possible from near-duplicate
    web pages; >0.01% of windows means the val shard was written twice, or into train). Print the count
    and the total windows checked. Should run in well under a minute over 1.6B tokens.
    """
    raise NotImplementedError("Fan Pu writes check_disjoint")


# =======================================================================================


def lr_test():
    T = 10_000
    w = max(1, round(0.03 * T))
    l0, lw, lend = lr_at(0, T), lr_at(w, T), lr_at(T - 1, T)
    print(f"step 0: {l0:.3e}   step warmup({w}): {lw:.3e}   last step: {lend:.3e}")
    assert abs(lw - 6e-4) < 1e-9, "peak not reached exactly at warmup_steps"
    assert abs(lend - 6e-5) < 1e-7, "final LR should be 0.1 x peak at the last step"
    assert 0 < l0 <= 6e-4 / w + 1e-12, "step 0 should be one warmup increment, not 0 and not peak"
    ls = [lr_at(s, T) for s in range(T)]
    assert all(b >= a for a, b in zip(ls[:w], ls[1 : w + 1])), "warmup not non-decreasing"
    assert all(b <= a for a, b in zip(ls[w:], ls[w + 1 :])), "decay not non-increasing"
    print("lr_at: OK")


def param_groups(model, wd):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)]


def latest_ckpt(out):
    cks = sorted(f for f in os.listdir(out) if f.startswith("ckpt_") and f.endswith(".pt"))
    return os.path.join(out, cks[-1]) if cks else None


def run(a):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from fla.models import TransformerForCausalLM

    install_sdpa_shim()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    cfg = build_config("attn", a.size, vocab=VOCAB)
    model = TransformerForCausalLM(cfg).to(DEV)
    model.train()
    pb = param_buckets(model)
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.peak_lr, betas=(0.9, 0.95), fused=True)
    sdpa_ctx = lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION])

    tokens_per_step = a.B * a.T
    total_steps = a.steps if a.steps else a.tokens // tokens_per_step
    eval_every = max(1, total_steps // 20)
    ckpt_every = max(1, total_steps // 10)

    shards = TrainShards(a.data, a.T)
    order = make_order(total_steps * a.B, shards, a.seed)
    val = torch.from_numpy(val_windows(a.data, a.val_tokens, a.T))
    val_final = torch.from_numpy(val_windows(a.data, a.final_val_tokens, a.T))

    start_step = 0
    if a.resume and latest_ckpt(a.out):
        ck = torch.load(latest_ckpt(a.out), map_location=DEV)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from step {start_step}", flush=True)

    meta = dict(size=a.size, seed=a.seed, d=cfg.hidden_size, L=cfg.num_hidden_layers, heads=cfg.num_heads,
                vocab=VOCAB, B=a.B, T=a.T, tokens_per_step=tokens_per_step, total_steps=total_steps,
                tokens_budget=total_steps * tokens_per_step, peak_lr=a.peak_lr, wd=a.wd,
                val_tokens=len(val) * a.T, final_val_tokens=len(val_final) * a.T,
                torch=torch.__version__, **pb)
    print(json.dumps(meta), flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def write(**kw):
        log.write(json.dumps(kw) + "\n")
        log.flush()

    def do_eval(step, v, tag):
        t = time.perf_counter()
        vl = evaluate(model, v, a.B, sdpa_ctx)
        write(step=step, tokens=step * tokens_per_step, **{tag: vl}, eval_s=time.perf_counter() - t)
        print(f"[{a.size}/s{a.seed}] step {step}/{total_steps}  {tag} {vl:.4f}", flush=True)
        return vl

    write(meta=meta)
    if start_step == 0:
        do_eval(0, val, "val_loss")

    t_log, loss_acc, n_acc = time.perf_counter(), 0.0, 0
    for step, xb in train_batches(shards, order, a.B, start_batch=start_step):
        lr = lr_at(step, total_steps, a.peak_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        x = torch.from_numpy(xb).to(DEV, non_blocking=True)
        with sdpa_ctx(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x, use_cache=False).loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_acc += loss.detach()
        n_acc += 1
        s1 = step + 1

        if s1 % a.log_every == 0 or s1 == total_steps:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t_log
            tl = (loss_acc / n_acc).item()
            assert math.isfinite(tl), f"non-finite train loss at step {s1}"
            write(step=s1, tokens=s1 * tokens_per_step, train_loss=tl, lr=lr, grad_norm=float(gn),
                  tok_s=n_acc * tokens_per_step / dt)
            if s1 % (a.log_every * 10) == 0:
                print(f"[{a.size}/s{a.seed}] step {s1}/{total_steps}  train {tl:.4f}  lr {lr:.2e}  "
                      f"{n_acc * tokens_per_step / dt / 1e3:.1f}k tok/s", flush=True)
            t_log, loss_acc, n_acc = time.perf_counter(), 0.0, 0
        if s1 % eval_every == 0 and s1 != total_steps:
            do_eval(s1, val, "val_loss")
            t_log = time.perf_counter()
        if s1 % ckpt_every == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=s1, meta=meta),
                       os.path.join(a.out, f"ckpt_{s1:07d}.pt"))
            t_log = time.perf_counter()

    final = do_eval(total_steps, val_final, "final_val_loss")
    rows = [json.loads(l) for l in open(os.path.join(a.out, "log.jsonl"))]
    tok_s = [r["tok_s"] for r in rows if "tok_s" in r]
    summary = dict(meta, final_val_loss=final, tok_s_median=float(np.median(tok_s)) if tok_s else None,
                   peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "lr-test", "check-data"])
    ap.add_argument("--size", choices=["30M", "60M", "125M", "250M"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/fineweb_edu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokens", type=int, default=None, help="token budget; steps = tokens // (B*T)")
    ap.add_argument("--steps", type=int, default=None, help="override: exact step count (smoke tests)")
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--peak-lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-tokens", type=int, default=2**22, help="~4.2M tokens for intermediate evals")
    ap.add_argument("--final-val-tokens", type=int, default=2**24, help="~16.8M tokens for the final eval")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "lr-test":
        lr_test()
    elif a.cmd == "check-data":
        check_disjoint(a.data, a.T)
    else:
        assert a.size and a.out and (a.tokens or a.steps), "run needs --size, --out, and --tokens or --steps"
        run(a)
