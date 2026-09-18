"""Train one model on MQAR and evaluate it every epoch. [AI]

after zoology/train.py (the protocol: AdamW, cosine over epochs, one test
pass per epoch, early stopping on test accuracy) in the shape of
karpathy/nanoGPT's train.py (one flat file, the whole step on one screen).

    python scripts/train_mqar.py --config experiments/day4_mqar/attn_d64.toml \
        --seed 0 [--out runs/<name>]

Every value but the seed and the output directory comes from the TOML file.
The seed sets the model initialization and the batch order. The data seed
(`[data].data_seed`) is a separate key so that every run of the day sees the
same examples. The run directory holds `config.toml` (a verbatim copy),
`env.json`, `metrics.jsonl`, `eval_preds.npz` (the last test pass), `model.pt`
(the final weights), and `summary.json` when the run completes (best_acc, final_acc, best_epoch,
epochs_run, early_stopped, steps, tok_s, peak_mem_GB, wall_s, params,
state_elements); a run whose summary exists is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from testbed.evals.mqar_accuracy import mqar_accuracy  # noqa: E402
from testbed.models.mqar_lm import MQARLM  # noqa: E402
from testbed.tasks.mqar import IGNORE_INDEX, multiquery_ar  # noqa: E402


def source_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(list((REPO / "testbed").rglob("*.py")) + list((REPO / "scripts").glob("*.py"))):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def env_info() -> dict:
    info = {"torch": torch.__version__, "python": sys.version.split()[0], "source_hash": source_hash()}
    try:
        import fla  # noqa: WPS433
        info["fla"] = fla.__version__
    except Exception:  # pragma: no cover
        info["fla"] = None
    try:
        import triton  # noqa: WPS433
        info["triton"] = triton.__version__
    except Exception:  # pragma: no cover
        info["triton"] = None
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    try:
        info["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git"] = None
    return info


def run_name(cfg: dict, seed: int) -> str:
    return f"{cfg['run']['mixer']}_d{cfg['model']['d_model']}_lr{cfg['training']['lr']:.1e}_s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text())
    run, mcfg, dcfg, tcfg = cfg["run"], cfg["model"], cfg["data"], cfg["training"]
    out = Path(args.out) if args.out else REPO / "runs" / run_name(cfg, args.seed)
    if (out / "summary.json").exists():
        print(f"skip {out}: summary.json exists")
        return
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.toml")
    (out / "env.json").write_text(json.dumps(env_info(), indent=2))
    log = open(out / "metrics.jsonl", "w")

    def emit(rec: dict) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and tcfg.get("precision", "bf16") == "bf16"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data: the same examples for every run of the day (data_seed), shuffled per seed
    t0 = time.time()
    x_tr, y_tr = multiquery_ar(dcfg["vocab_size"], dcfg["num_train"], dcfg["input_seq_len"], dcfg["data_seed"],
                               power_a=dcfg["power_a"], num_kv_pairs=dcfg["num_kv_pairs"],
                               random_non_queries=dcfg["random_non_queries"])
    x_te, y_te = multiquery_ar(dcfg["vocab_size"], dcfg["num_test"], dcfg["input_seq_len"], dcfg["data_seed"] + 1,
                               power_a=dcfg["power_a"], num_kv_pairs=dcfg["num_kv_pairs"],
                               random_non_queries=dcfg["random_non_queries"])
    x_tr, y_tr, x_te, y_te = (t.to(device) for t in (x_tr, y_tr, x_te, y_te))
    data_s = time.time() - t0

    # ---- model
    model = MQARLM(
        mixer=run["mixer"], d_model=mcfg["d_model"], n_layers=mcfg["n_layers"], num_heads=mcfg["num_heads"],
        vocab_size=dcfg["vocab_size"], max_position_embeddings=dcfg["input_seq_len"],
        embed_dropout=mcfg["embed_dropout"], attn_dropout=mcfg["attn_dropout"],
        impl=mcfg.get("impl", "chunk" if device == "cuda" else "naive"),
    ).to(device)
    B, Bte = tcfg["batch_size"], tcfg["test_batch_size"]
    steps_per_epoch = dcfg["num_train"] // B
    max_epochs = tcfg["max_epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=0.0)
    emit({"kind": "meta", "run": out.name, "mixer": run["mixer"], "d_model": mcfg["d_model"], "n_layers": mcfg["n_layers"],
          "num_heads": mcfg["num_heads"], "seed": args.seed, "params": model.num_params(),
          "state_elements": model.state_elements(dcfg["input_seq_len"]), "input_seq_len": dcfg["input_seq_len"],
          "num_kv_pairs": dcfg["num_kv_pairs"], "num_train": dcfg["num_train"], "num_test": dcfg["num_test"],
          "lr": tcfg["lr"], "batch_size": B, "steps_per_epoch": steps_per_epoch, "max_epochs": max_epochs,
          "data_gen_s": round(data_s, 2), "device": device, "bf16": use_bf16})

    def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # cross-entropy over the query positions only; identical to CE over all
        # positions with ignore_index=-100, but without materializing the fp32
        # logits of the T - num_kv_pairs positions that carry no label.
        mask = labels != IGNORE_INDEX
        return F.cross_entropy(logits[mask].float(), labels[mask])

    @torch.no_grad()
    def evaluate() -> tuple[float, float, np.ndarray]:
        model.eval()
        accs, losses, preds = [], [], []
        for i in range(0, x_te.shape[0], Bte):
            xb, yb = x_te[i:i + Bte], y_te[i:i + Bte]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(xb)
            losses.append(loss_fn(logits, yb).item() * xb.shape[0])
            accs.append(mqar_accuracy(logits, yb).float().cpu())
            preds.append(logits.argmax(-1).to(torch.int32).cpu())
        model.train()
        acc = torch.cat(accs).mean().item()
        return acc, sum(losses) / x_te.shape[0], torch.cat(preds).numpy()

    # ---- the loop
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    step, train_tokens, train_s = 0, 0, 0.0
    best_acc, best_epoch, early_stopped = 0.0, -1, False
    acc, te_loss, preds = evaluate()
    emit({"kind": "eval", "epoch": -1, "step": 0, "acc": acc, "test_loss": te_loss})
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    wall0 = time.time()
    for epoch in range(max_epochs):
        order = torch.randperm(x_tr.shape[0], generator=g).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t_ep = time.time()
        losses = []
        for i in range(steps_per_epoch):
            idx = order[i * B:(i + 1) * B]
            xb, yb = x_tr[idx], y_tr[idx]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            losses.append(loss.item())
            step += 1
            train_tokens += xb.numel()
        if device == "cuda":
            torch.cuda.synchronize()
        train_s += time.time() - t_ep
        acc, te_loss, preds = evaluate()
        best_acc, best_epoch = (acc, epoch) if acc > best_acc else (best_acc, best_epoch)
        mixers = [layer.mixer for layer in model.layers]
        emit({"kind": "train", "epoch": epoch, "step": step, "train_loss": float(np.mean(losses)),
              "lr": sched.get_last_lr()[0], "tok_s": train_tokens / max(train_s, 1e-9),
              "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
              # mean write gate beta (and decay alpha for the gated rule) per layer over the last test batch
              "beta_mean": [None if getattr(m, "last_beta_mean", None) is None else float(m.last_beta_mean) for m in mixers],
              "decay_mean": [None if getattr(m, "last_decay_mean", None) is None else float(m.last_decay_mean) for m in mixers]})
        emit({"kind": "eval", "epoch": epoch, "step": step, "acc": acc, "test_loss": te_loss})
        print(f"epoch {epoch:3d} step {step:6d} train_loss {np.mean(losses):.4f} test_loss {te_loss:.4f} acc {acc:.4f}", flush=True)
        if acc > tcfg["early_stop_acc"]:
            early_stopped = True
            break
        sched.step()

    np.savez_compressed(out / "eval_preds.npz", preds=preds, labels=y_te.cpu().numpy().astype(np.int32))
    torch.save(model.state_dict(), out / "model.pt")  # final weights, a few MB; for evaluation-only follow-ups
    summary = {
        "run": out.name, "mixer": run["mixer"], "d_model": mcfg["d_model"], "num_heads": mcfg["num_heads"],
        "n_layers": mcfg["n_layers"], "lr": tcfg["lr"], "seed": args.seed, "input_seq_len": dcfg["input_seq_len"],
        "num_kv_pairs": dcfg["num_kv_pairs"], "state_elements": model.state_elements(dcfg["input_seq_len"]),
        "final_acc": acc, "best_acc": best_acc, "best_epoch": best_epoch, "epochs_run": epoch + 1,
        "early_stopped": early_stopped, "steps": step, "tok_s": train_tokens / max(train_s, 1e-9),
        "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "wall_s": time.time() - wall0, "params": model.num_params(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
