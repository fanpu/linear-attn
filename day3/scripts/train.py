#!/usr/bin/env python
"""Train one run from a TOML config.  [AI-owned]
# Shape after karpathy/nanoGPT train.py (one flat file, the whole step visible);
# config-per-run after pytorch/torchtitan (train_configs/*.toml); run directory
# contents after KellerJordan/modded-nanogpt (config + environment recorded
# with every log so any number is reproducible from the run directory alone).

    python scripts/train.py --config experiments/day3_seed_variance/attn_30M.toml --seed 0
    python scripts/train.py --config experiments/day3_seed_variance/attn_30M_smoke.toml --seed 0 --out runs/smoke_a

Everything except the seed and the output directory comes from the config
file; the flags of a run are its file, never the command line. The seed sets
the initial weights and the order of the training windows; it does not change
which windows are used (a budget of D tokens always uses the first D/T aligned
windows of the training shards, in file order).

Outputs, in the run directory (default <job.dump_folder>/<mixer>_<size>_s<seed>):
    config.toml    a verbatim copy of the config file the run was launched with.
    env.json       torch, triton and fla versions, the GPU name, and a hash of
                   the testbed/ and scripts/ sources at launch.
    metrics.jsonl  one JSON object per line. First line kind="meta". Then
                   kind="train" rows every `log_every` steps with the mean
                   training loss over the interval, the learning rate, the mean
                   gradient norm, tok/s measured over training steps only
                   (evaluation and checkpoint time excluded), peak allocated
                   GPU memory in GB, and wall-clock seconds since launch; and
                   kind="eval" rows with the validation loss.
    ckpt_latest.pt full state (weights, optimizer, step) for resuming; overwritten.
    ckpt_XX.pt     weights only, bf16, at every 10% of the run (XX = percent).
    summary.json   final numbers: final_val_loss, tokens, steps, median tok/s,
                   peak_mem_GB, wall_h, and the meta block.

If summary.json exists the run is skipped; if ckpt_latest.pt exists the run
resumes from it (the data order is a function of the seed, so the replay is
exact up to GPU non-determinism).

A non-finite loss raises immediately: that is a broken run, not a result.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed import model as models  # noqa: E402
from testbed.evals.val_loss import evaluate  # noqa: E402  [you]
from testbed.schedule import lr_at  # noqa: E402  [you]
from testbed.data import WindowIndex, batch_iter, batch_order, load_windows, read_shard, shard_paths  # noqa: E402
from testbed.config import load_config, source_hash  # noqa: E402


def log_line(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def param_groups(model, weight_decay):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="TOML run config, e.g. experiments/day3_seed_variance/attn_30M.toml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None, help="run directory; default <job.dump_folder>/<mixer>_<size>_s<seed>")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    size, vocab = cfg["model"]["size"], cfg["model"]["vocab_size"]
    B, T = cfg["training"]["local_batch_size"], cfg["training"]["seq_len"]
    budget = cfg["training"]["budget_tokens"]
    S_full = budget // (B * T)
    S = S_full if cfg["training"]["steps"] == 0 else min(cfg["training"]["steps"], S_full)
    out = args.out or os.path.join(cfg["job"]["dump_folder"], f'{cfg["model"]["mixer"]}_{size}_s{args.seed}')
    os.makedirs(out, exist_ok=True)
    log_path, ckpt_path, summ_path = (os.path.join(out, n) for n in ("metrics.jsonl", "ckpt_latest.pt", "summary.json"))
    if os.path.exists(summ_path):
        print(f"{summ_path} exists; skipping")
        return
    dev = args.device

    # ---- model and optimizer (seeded) ----
    models.install_sdpa_shim()
    torch.manual_seed(args.seed)
    if dev == "cuda":
        torch.cuda.manual_seed(args.seed)
    model, mcfg = models.build_model(size, vocab, dev)
    model.train()
    counts = models.count_params(model)
    ocfg = cfg["optimizer"]
    opt = torch.optim.AdamW(param_groups(model, ocfg["weight_decay"]), lr=ocfg["lr"],
                            betas=tuple(ocfg["betas"]), fused=(dev == "cuda"))

    # ---- data ----
    train_shards = [read_shard(p) for p in shard_paths(cfg["training"]["dataset_path"], "train")]
    val_shards = [read_shard(p) for p in shard_paths(cfg["training"]["dataset_path"], "val")]
    index = WindowIndex(train_shards, T, S_full * B)
    order = batch_order(args.seed, S_full * B)
    val = load_windows(val_shards, T, cfg["validation"]["windows"]).to(dev)
    ctx = models.sdpa_ctx() if dev == "cuda" else __import__("contextlib").nullcontext()

    eval_every = max(1, round(cfg["validation"]["every_frac"] * S_full))
    ckpt_every = max(1, round(cfg["checkpoint"]["every_frac"] * S_full))

    # ---- resume or start ----
    start_step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from {ckpt_path} at step {start_step}")
    else:
        d, L = models.SHAPES[size]
        shutil.copy(args.config, os.path.join(out, "config.toml"))  # the run's flags are its file
        env = {"torch": torch.__version__,
               "triton": getattr(__import__("triton"), "__version__", "?") if dev == "cuda" else "n/a",
               "fla": getattr(__import__("fla"), "__version__", "?") if dev == "cuda" else "n/a",
               "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
               "source_hash": source_hash(), "config": os.path.abspath(args.config),
               "launched": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv}
        json.dump(env, open(os.path.join(out, "env.json"), "w"), indent=1)
        meta = {"kind": "meta", "mixer": cfg["model"]["mixer"], "size": size, "seed": args.seed, "d": d, "L": L,
                "heads": d // 64, "vocab": vocab, "params_M": counts, "budget_tokens": budget, "steps_full": S_full,
                "steps_this_run": S, "B": B, "T": T, "tokens_per_step": B * T, "eval_every": eval_every,
                "ckpt_every": ckpt_every, "lr": ocfg["lr"], "config_file": os.path.basename(args.config),
                "launched": env["launched"]}
        if os.path.exists(log_path):
            os.remove(log_path)  # a fresh start (no checkpoint) replaces any log from a crashed attempt
        log_line(log_path, meta)
        print(json.dumps(meta))

    def do_eval(step, windows):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        v = evaluate(model, val if windows == cfg["validation"]["windows"] else load_windows(val_shards, T, windows).to(dev), B, ctx)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        log_line(log_path, {"kind": "eval", "step": step, "tokens": step * B * T, "val_loss": v,
                            "val_windows": windows, "eval_s": dt})
        print(f"eval step {step:6d} tokens {step*B*T/1e6:8.1f}M val_loss {v:.4f} ({dt:.0f}s)", flush=True)
        return v, dt

    if start_step == 0:
        do_eval(0, cfg["validation"]["windows"])

    # ---- training loop ----
    t_launch = time.perf_counter()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loss_acc = torch.zeros((), device=dev)
    gn_acc = torch.zeros((), device=dev)
    n_acc = 0
    excluded = 0.0  # eval + checkpoint seconds inside the current throughput window
    if dev == "cuda":
        torch.cuda.synchronize()
    t_window = time.perf_counter()
    tok_s_hist = []
    last_step = start_step
    for step, x in batch_iter(index, order, B, start_step, dev):
        lr = lr_at(step, S_full, ocfg["lr"], cfg["lr_scheduler"]["warmup_frac"], cfg["lr_scheduler"]["min_lr_factor"])
        for g in opt.param_groups:
            g["lr"] = lr
        with ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            loss = model(input_ids=x, labels=x, use_cache=False).loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_acc += loss.detach().float()
        gn_acc += gn.detach().float()
        n_acc += 1
        s1 = step + 1
        last_step = s1

        if s1 % cfg["metrics"]["log_freq"] == 0 or s1 == S:
            if dev == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            mean_loss = (loss_acc / n_acc).item()
            if not math.isfinite(mean_loss):
                raise RuntimeError(f"non-finite training loss at step {s1}: {mean_loss}")
            tok_s = n_acc * B * T / max(1e-9, (now - t_window) - excluded)
            tok_s_hist.append(tok_s)
            peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
            log_line(log_path, {"kind": "train", "step": s1, "tokens": s1 * B * T, "loss": mean_loss, "lr": lr,
                                "grad_norm": (gn_acc / n_acc).item(), "tok_s": tok_s, "peak_mem_GB": peak,
                                "wall_s": now - t_launch})
            print(f"step {s1:6d}/{S} loss {mean_loss:.4f} lr {lr:.2e} gnorm {(gn_acc/n_acc).item():.3f} "
                  f"{tok_s:8.0f} tok/s peak {peak:.1f} GB", flush=True)
            loss_acc.zero_(); gn_acc.zero_(); n_acc = 0; excluded = 0.0; t_window = now

        if s1 % eval_every == 0 and s1 < S:
            _, dt = do_eval(s1, cfg["validation"]["windows"])
            excluded += dt
        if s1 % ckpt_every == 0 and s1 < S_full:
            t0 = time.perf_counter()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s1}, ckpt_path)
            pct = round(100 * s1 / S_full)
            torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()},
                       os.path.join(out, f"ckpt_{pct:02d}.pt"))
            excluded += time.perf_counter() - t0
        if s1 >= S:
            break

    # ---- final evaluation and summary ----
    final_windows = cfg["validation"]["final_windows"] if S == S_full else cfg["validation"]["windows"]
    final_val, _ = do_eval(last_step, final_windows)
    wall = time.perf_counter() - t_launch
    tok_s_hist_sorted = sorted(tok_s_hist)
    summary = {"mixer": cfg["model"]["mixer"], "size": size, "seed": args.seed, "steps": last_step, "tokens": last_step * B * T,
               "final_val_loss": final_val, "final_val_windows": final_windows,
               "train_tok_s_median": tok_s_hist_sorted[len(tok_s_hist_sorted) // 2] if tok_s_hist else None,
               "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0,
               "wall_h_this_process": wall / 3600, "complete": S == S_full, "params_M": counts}
    json.dump(summary, open(summ_path, "w"), indent=1)
    print(json.dumps(summary))
    if S == S_full and os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # the 100% weights are in ckpt_100.pt below; the optimizer state is not needed
    if S == S_full:
        torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, os.path.join(out, "ckpt_100.pt"))


if __name__ == "__main__":
    main()
