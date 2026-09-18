"""The day's runs, as a list of configs built in Python. [AI]

after zoology/experiments/paper_configs/iclr24_zoology_figure2/configs.py:
a sweep is a Python file that expands base configs over the values that
vary, and the launcher runs the list. Here the expansion writes one TOML
per run into `generated/` (torchtitan's one-file-per-run form) and prints
the list that `run.sh` executes.

    python experiments/day4_mqar/sweep.py <stage>

Stages, in the order the handout runs them:
  smoke   three mixers on the (64 tokens, 4 pairs) preset, one learning rate
  lr      attention at d=64, additive and delta at d=128, four learning rates
          each, on (512 tokens, 64 pairs); seed 0
  scan    additive and delta at d in {64, 256, 512} at the best learning rate
          each mixer found in `lr` (read from runs/); seed 0
  scan512 the two d = 512 runs of `scan` alone (the reduced day of the handout)
  seeds   a second seed (1) of the three best-learning-rate runs of `lr`
  gate    OPTIONAL: the gated delta rule at d=128, four learning rates
  pivot   attention at d=64 with 100,000 training examples at the best
          learning rate of `lr` (the pivot of the handout, run only if
          attention fails on the main setting)

Prints one line per run: `<generated toml> <seed> <run dir>`.
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import tomli_w

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEN = HERE / "generated"
RUNS = REPO / "runs"

LRS = [float(f"{x:.3g}") for x in np.logspace(-4, -2, 4)]   # Zoology's grid: 1e-4, 4.64e-4, 2.15e-3, 1e-2
SCAN_D = [64, 256, 512]
SEEDS_EXTRA = [1]


def load(name: str) -> dict:
    return tomllib.loads((HERE / f"{name}.toml").read_text())


def run_name(cfg: dict, seed: int) -> str:
    return f"{cfg['run']['mixer']}_d{cfg['model']['d_model']}_lr{cfg['training']['lr']:.1e}_s{seed}"


def write(cfg: dict, seed: int, prefix: str = "") -> tuple[Path, int, Path]:
    GEN.mkdir(exist_ok=True)
    name = prefix + run_name(cfg, seed)
    path = GEN / f"{name}.toml"
    path.write_text(tomli_w.dumps(cfg))
    return path, seed, RUNS / name


def best_lr(mixer: str, d_model: int) -> float:
    """The learning rate whose seed-0 run reached the highest best_acc (ties: the lower lr)."""
    cands = []
    for s in RUNS.glob(f"{mixer}_d{d_model}_lr*_s0/summary.json"):
        j = json.loads(s.read_text())
        cands.append((j["best_acc"], -j["lr"], j["lr"]))
    if not cands:
        raise SystemExit(f"no finished lr-stage runs for {mixer} at d={d_model}; run the lr stage first")
    cands.sort(reverse=True)
    return cands[0][2]


def stage(name: str) -> list[tuple[Path, int, Path]]:
    out = []
    if name == "smoke":
        for m in ("attn", "linattn", "deltanet"):
            out.append(write(load(f"smoke_{m}_d64"), 0, prefix="smoke_"))
    elif name == "lr":
        for base in ("attn_d64", "linattn_d128", "deltanet_d128"):
            for lr in LRS:
                cfg = load(base)
                cfg["training"]["lr"] = lr
                out.append(write(cfg, 0))
    elif name in ("scan", "scan512"):
        for base in ("linattn_d128", "deltanet_d128"):
            cfg0 = load(base)
            lr = best_lr(cfg0["run"]["mixer"], cfg0["model"]["d_model"])
            for d in (SCAN_D if name == "scan" else [512]):
                cfg = load(base)
                cfg["model"]["d_model"] = d
                cfg["training"]["lr"] = lr
                out.append(write(cfg, 0))
    elif name == "seeds":
        for base in ("attn_d64", "linattn_d128", "deltanet_d128"):
            cfg = load(base)
            cfg["training"]["lr"] = best_lr(cfg["run"]["mixer"], cfg["model"]["d_model"])
            for seed in SEEDS_EXTRA:
                out.append(write(cfg, seed))
    elif name == "gate":
        for lr in LRS:
            cfg = load("gdn_d128")
            cfg["training"]["lr"] = lr
            out.append(write(cfg, 0))
    elif name == "pivot":
        cfg = load("attn_d64")
        cfg["training"]["lr"] = best_lr("attn", 64)
        cfg["data"]["num_train"] = 100_000
        out.append(write(cfg, 0, prefix="pivot100k_"))
    else:
        raise SystemExit(f"unknown stage {name}; one of smoke, lr, scan, scan512, seeds, gate, pivot")
    return out


if __name__ == "__main__":
    for path, seed, run_dir in stage(sys.argv[1]):
        print(path.relative_to(REPO), seed, run_dir.relative_to(REPO))
