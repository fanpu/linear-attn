"""Orchestrate the GB10 inference survey.

For each model: cost every cell with budget.py, drop the ones that do not fit
under the memory ceiling, size a single engine for the cells that survive, and run
them in one guarded subprocess. Results append to results/results.jsonl.

The sweep is resumable. Cells already present in the results file are skipped, so
a run killed by the watchdog (or by you) can be restarted and will only do the
work that remains.

Skipped cells are *recorded*, not silently dropped: "we could not fit batch 256 on
14B" is a finding about the machine, and belongs in the survey.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import budget
import guard

RESULTS = Path("results/results.jsonl")
OUT_LEN = 128

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
]

# (input length, batch sizes). Batch depth is trimmed as context grows so a single
# cell's prefill stays bounded -- batch 256 at 8k context would be a 2M-token
# prefill, which costs minutes per repeat and tells us nothing extra.
GRID = [
    (128, [1, 2, 4, 8, 16, 32, 64, 128, 256]),
    (1024, [1, 4, 16, 64]),
    (8192, [1, 4, 8]),
]

# Per-model wall-clock ceiling for the guarded child.
TIMEOUT_S = 5400


def repeats_for(batch: int, in_len: int) -> int:
    """Cheap cells get 3 repeats for a stable median; expensive ones get 2."""
    return 3 if batch * in_len <= 8192 else 2


def load_done() -> set:
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r.get("model"), r.get("batch"), r.get("in_len"), r.get("out_len")))
    return done


def append(row: dict) -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


def plan_model(spec: budget.ModelSpec, done: set) -> tuple[list, list]:
    """Split this model's grid into runnable cells and recorded skips."""
    cells, skips = [], []
    for in_len, batches in GRID:
        for b in batches:
            key = (spec.name, b, in_len, OUT_LEN)
            if key in done:
                continue
            plan = budget.plan_cell(spec, batch=b, total_len=in_len + OUT_LEN)
            if plan.fits:
                cells.append({"batch": b, "in_len": in_len, "out_len": OUT_LEN,
                              "repeats": repeats_for(b, in_len)})
            else:
                skips.append({
                    "model": spec.name, "batch": b, "in_len": in_len,
                    "out_len": OUT_LEN, "status": "skipped:budget",
                    "reason": plan.reason,
                    "predicted_total_gib": plan.total_bytes / budget.GIB,
                    "predicted_kv_gib": plan.kv_bytes / budget.GIB,
                    "weights_gib": spec.weight_bytes / budget.GIB,
                })
    return cells, skips


def run_model(model: str, done: set, dry_run: bool) -> None:
    try:
        spec = budget.ModelSpec.from_hf_cache(model)
    except Exception as e:
        print(f"  [{model}] unavailable: {type(e).__name__}: {e}")
        return

    cells, skips = plan_model(spec, done)
    for s in skips:
        if not dry_run:
            append(s)
    print(f"  [{model}] weights={spec.weight_bytes/budget.GIB:.1f} GiB  "
          f"kv={budget.kv_bytes_per_token(spec.config)/1024:.0f} KiB/tok  "
          f"{len(cells)} cells to run, {len(skips)} skipped")

    if not cells:
        return

    # One engine for all of this model's cells: size it for the largest.
    max_total_tokens = max(c["batch"] * (c["in_len"] + c["out_len"]) for c in cells)
    max_model_len = max(c["in_len"] + c["out_len"] for c in cells)
    max_num_seqs = max(c["batch"] for c in cells)
    engine_plan = budget.plan_cell(spec, batch=1, total_len=max_total_tokens)
    util = engine_plan.gpu_memory_utilization

    print(f"      engine: util={util:.3f} max_model_len={max_model_len} "
          f"max_num_seqs={max_num_seqs} ({engine_plan.summary()})")
    if dry_run:
        for c in cells:
            print(f"        {c}")
        return

    argv = [
        sys.executable, "bench.py",
        "--model", model,
        "--cells", json.dumps(cells),
        "--gpu-memory-utilization", f"{util:.4f}",
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
    ]
    # MAX_JOBS caps FlashInfer's JIT build parallelism. Left unset, its ninja
    # invocation defaults to nproc+2 -- 22 concurrent nvcc processes here, several
    # GiB each. That is compiler memory, invisible to a budget model that counts
    # only weights and KV, and it is what tripped the watchdog on the 30B MoE:
    # FlashInfer JIT-compiles fused MoE kernels, which dense models never trigger.
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING", MAX_JOBS="4")

    t0 = time.time()
    res = guard.run_guarded(argv, timeout_s=TIMEOUT_S, env=env)
    print(f"      -> {res.status()} in {res.elapsed_s/60:.1f} min, "
          f"min available {res.min_available_bytes/budget.GIB:.1f} GiB")

    meta = {}
    n_rows = 0
    for line in res.output.splitlines():
        if line.startswith("ROWMETA "):
            meta = json.loads(line[len("ROWMETA "):])
        elif line.startswith("ROW "):
            row = json.loads(line[len("ROW "):])
            row["load_s"] = meta.get("load_s")
            row["guard_min_available_gib"] = res.min_available_bytes / budget.GIB
            append(row)
            n_rows += 1

    if n_rows < len(cells):
        # The child died partway. Record why, so the gap in the grid is explained.
        append({
            "model": model, "status": f"incomplete:{res.status()}",
            "cells_requested": len(cells), "cells_returned": n_rows,
            "guard_min_available_gib": res.min_available_bytes / budget.GIB,
            "tail": res.output[-1500:],
        })
        print(f"      !! only {n_rows}/{len(cells)} cells returned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan and the memory budget without running")
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    total = budget.total_memory_bytes()
    print(f"GB10 sweep: total {total/budget.GIB:.1f} GiB, "
          f"ceiling {budget.CEILING_FRAC:.0%} = {budget.CEILING_FRAC*total/budget.GIB:.1f} GiB, "
          f"available now {budget.available_memory_bytes()/budget.GIB:.1f} GiB")
    print(f"watchdog floor {guard.DEFAULT_FLOOR_BYTES/budget.GIB:.0f} GiB\n")

    done = load_done()
    if done:
        print(f"resuming: {len(done)} cells already recorded\n")

    for model in args.models:
        run_model(model, done, args.dry_run)

    print("\nsweep complete")


if __name__ == "__main__":
    main()
