"""Train one run.  python -m alchemy.train --config cfg.toml --out runs/x [--set train.lr=3e-3 ...]

Writes to --out: config.json, log.jsonl (one line per step / eval), result.json (final), ckpt.pt (while running).
Re-running the same command resumes from ckpt.pt; a finished run (result.json present) is skipped."""
import argparse
import json
import math
import os
import sys
import time
import tomllib

import torch
import torch.nn.functional as F

from .data import TrainStream, ValSet
from .model import ModelConfig, Transformer

DEFAULTS = {
    "model": dict(vocab_size=50257, depth=8, width=384, head_dim=64, mlp_ratio=4, context=512),
    "train": dict(
        data_dir="../day3/data/fineweb_edu",
        tokens=300_000_000, batch_seqs=256, micro_batch_seqs=32,
        lr=1e-3, warmup_steps=100,
        schedule="cosine",          # constant | cosine | linear | wsd
        final_frac=0.1,             # LR at the end of the schedule, as a fraction of peak
        sched_total_steps=0,        # 0 = the run length; otherwise the length the schedule assumes
        cooldown_frac=0.2, cooldown_shape="linear",   # wsd only; shape: linear | sqrt
        weight_decay=0.1, wd_independent=False,       # independent: decay does not scale with peak LR
        beta1=0.9, beta2=0.95, eps=1e-8, clip=1.0,    # clip <= 0 disables clipping
        seed=0, seed_data=-1,       # seed_data < 0: same as seed
        eval_every=250, eval_windows=4096, final_eval_windows=16384, eval_batch_seqs=64,
        norms_every=50, ckpt_every=500, compile=True,
        mem_fraction=0.25,          # cap on this process's share of GPU (unified) memory
    ),
}


def load_config(path, overrides):
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    if path:
        with open(path, "rb") as f:
            for sec, vals in tomllib.load(f).items():
                for k, v in vals.items():
                    assert k in cfg[sec], f"unknown key {sec}.{k}"
                    cfg[sec][k] = v
    for ov in overrides:
        key, val = ov.split("=", 1)
        sec, k = key.split(".")
        assert k in cfg[sec], f"unknown key {key}"
        old = cfg[sec][k]
        if isinstance(old, bool):
            cfg[sec][k] = val.lower() in ("1", "true", "yes")
        else:
            cfg[sec][k] = type(old)(float(val)) if isinstance(old, int) else type(old)(val)
    return cfg


def lr_factor(step, total, t):
    """Multiplier on the peak LR at `step` of a schedule that assumes `total` steps."""
    warm = t["warmup_steps"]
    if step < warm:
        return (step + 1) / warm
    ff = t["final_frac"]
    if t["schedule"] == "constant":
        return 1.0
    p = min(1.0, (step - warm) / max(1, total - warm))
    if t["schedule"] == "cosine":
        return ff + (1 - ff) * 0.5 * (1 + math.cos(math.pi * p))
    if t["schedule"] == "linear":
        return ff + (1 - ff) * (1 - p)
    if t["schedule"] == "wsd":
        cool = int(t["cooldown_frac"] * total)
        q = (step - (total - cool)) / max(1, cool)
        if q <= 0:
            return 1.0
        q = min(1.0, q)
        shape = 1 - math.sqrt(q) if t["cooldown_shape"] == "sqrt" else 1 - q
        return ff + (1 - ff) * shape
    raise ValueError(t["schedule"])


@torch.no_grad()
def evaluate(model, val, batch_seqs, limit, device):
    model.eval()
    tot, n = 0.0, 0
    for w in val.batches(batch_seqs, limit):
        w = w.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(w[:, :-1])
        tot += F.cross_entropy(logits.float().flatten(0, 1), w[:, 1:].flatten(), reduction="sum").item()
        n += w[:, 1:].numel()
    model.train()
    return tot / n


def group_norms(groups, before=None):
    """L2 norm per group of the parameters, or of (parameters - before). One sync for all groups."""
    keys = list(groups)
    sq = torch.stack([
        sum(((p.detach() - b) if before else p.detach()).float().pow(2).sum()
            for p, b in zip(groups[k], before[k] if before else groups[k])) for k in keys])
    return dict(zip(keys, sq.sqrt().tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    result_path = os.path.join(args.out, "result.json")
    if os.path.exists(result_path):
        print("already finished:", args.out)
        return
    cfg = load_config(args.config, args.set)
    m, t = cfg["model"], cfg["train"]
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(t["mem_fraction"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed_data = t["seed"] if t["seed_data"] < 0 else t["seed_data"]
    torch.manual_seed(t["seed"])
    model = Transformer(ModelConfig(**m)).to(device)
    acct = model.accounting()
    groups = model.param_groups()

    tokens_per_step = t["batch_seqs"] * m["context"]
    steps = t["tokens"] // tokens_per_step
    sched_total = t["sched_total_steps"] or steps
    assert t["batch_seqs"] % t["micro_batch_seqs"] == 0 or t["batch_seqs"] < t["micro_batch_seqs"]
    micro = min(t["micro_batch_seqs"], t["batch_seqs"])
    accum = t["batch_seqs"] // micro

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    wd = t["weight_decay"] / t["lr"] if t["wd_independent"] else t["weight_decay"]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=t["lr"], betas=(t["beta1"], t["beta2"]), eps=t["eps"], fused=True)

    start = 0
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed from step {start}")

    fwd = torch.compile(model) if t["compile"] else model
    stream = TrainStream(t["data_dir"], m["context"], t["batch_seqs"], seed_data)
    val = ValSet(t["data_dir"], m["context"], t["final_eval_windows"])
    log = open(os.path.join(args.out, "log.jsonl"), "a")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    diverged_at = None
    n_clipped = 0
    t0 = time.time()
    timed_tokens, timed_from = 0, None
    pending = []   # (record, loss tensor, gnorm tensor); fetched in groups, because every .item() is a sync and
    SYNC_EVERY = 10  # on a time-sliced shared GPU each sync waits for the other process's slice

    def flush():
        nonlocal diverged_at, n_clipped
        if not pending:
            return
        vals = torch.stack([torch.stack((l, g.to(l.dtype))) for _, l, g in pending]).tolist()   # one sync
        for (rec, _, _), (loss_v, gnorm_v) in zip(pending, vals):
            rec["loss"], rec["gnorm"] = loss_v, gnorm_v
            if not math.isfinite(rec["loss"]) and diverged_at is None:
                diverged_at = rec["step"]
                rec["diverged"] = True
            n_clipped += t["clip"] > 0 and rec["gnorm"] > t["clip"]
            emit(rec)
        pending.clear()

    for step in range(start, steps):
        lr = t["lr"] * lr_factor(step, sched_total, t)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = stream.batch(step).pin_memory().to(device, non_blocking=True)
        loss_t = torch.zeros((), device=device)
        for i in range(accum):
            w = batch[i * micro:(i + 1) * micro]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(w[:, :-1])
            loss = F.cross_entropy(logits.float().flatten(0, 1), w[:, 1:].flatten()) / accum
            loss.backward()
            loss_t += loss.detach()
        gnorm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"] if t["clip"] > 0 else float("inf"))
        rec = {"step": step, "tokens": (step + 1) * tokens_per_step, "lr": lr}
        want_norms = t["norms_every"] > 0 and step % t["norms_every"] == 0
        if want_norms:
            before = {k: [p.detach().clone() for p in ps] for k, ps in groups.items()}
        opt.step()
        opt.zero_grad(set_to_none=True)
        if want_norms:
            rec["param_norm"] = group_norms(groups)
            rec["update_norm"] = group_norms(groups, before)
            del before
        if step == start + 20:      # skip compile and warm-up steps when timing
            torch.cuda.synchronize()
            timed_from, timed_tokens = time.time(), 0
        elif timed_from is not None:
            timed_tokens += tokens_per_step
        last = step == steps - 1
        if (step + 1) % t["eval_every"] == 0 and not last:
            rec["val_loss"] = evaluate(fwd, val, t["eval_batch_seqs"], t["eval_windows"], device)
        pending.append((rec, loss_t, gnorm_t))
        do_ckpt = (step + 1) % t["ckpt_every"] == 0 and not last
        if len(pending) >= SYNC_EVERY or do_ckpt or last:
            flush()
            if diverged_at is not None:
                break
        if do_ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
        if step % 100 == 0:
            print(f"step {step}/{steps} lr {lr:.2e}", flush=True)

    torch.cuda.synchronize()
    final_val = None if diverged_at is not None else evaluate(fwd, val, t["eval_batch_seqs"], None, device)
    toks_per_s = timed_tokens / (time.time() - timed_from) if timed_from and timed_tokens else None
    result = {
        "final_val_loss": final_val, "diverged_at": diverged_at, "steps": steps,
        "tokens": steps * tokens_per_step, "clip_rate": n_clipped / max(1, steps - start),
        "tokens_per_s": toks_per_s, "wall_s": time.time() - t0,
        "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9,
        "flops_total": acct["flops_per_token_total"] * steps * tokens_per_step, **acct,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=1)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("done", args.out, "diverged" if diverged_at is not None else "ok", file=sys.stderr)


if __name__ == "__main__":
    main()
