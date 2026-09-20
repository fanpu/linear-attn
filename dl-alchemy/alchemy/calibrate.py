"""Unit 1 calibration: six-point LR grid on the baseline, then extra seeds at the best LR.

    setsid nohup ../.venv/bin/python -m alchemy.calibrate unit1-basics/baseline.toml unit1-basics/quiz/sealed/calibration \
        > logs/calibrate.log 2>&1 < /dev/null &

Resumable (finished runs are skipped). Writes <out>/calibration.json. The full LR bowl is sealed (it is the
student's lr-bowl problem); only best LR, noise floor and timing are public."""
import json
import os
import statistics
import sys

from . import queue

LR_GRID = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2"]
EXTRA_SEEDS = [1, 2, 3, 4]


def run_jobs(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    sys.argv = ["queue", path]
    queue.main()


def result(d):
    p = os.path.join(d, "result.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    config, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    grid = {lr: os.path.join(out, f"lr{lr}_s0") for lr in LR_GRID}
    run_jobs(os.path.join(out, "jobs_grid.txt"), [f"{d} {config} train.lr={lr} train.seed=0" for lr, d in grid.items()])
    losses = {lr: (result(d) or {}).get("final_val_loss") for lr, d in grid.items()}
    ok = {lr: v for lr, v in losses.items() if v is not None}
    assert ok, "every grid run failed or diverged"
    best = min(ok, key=ok.get)
    seeds = {s: os.path.join(out, f"lr{best}_s{s}") for s in EXTRA_SEEDS}
    run_jobs(os.path.join(out, "jobs_seeds.txt"), [f"{d} {config} train.lr={best} train.seed={s}" for s, d in seeds.items()])
    finals = [ok[best]] + [r["final_val_loss"] for r in map(result, seeds.values()) if r and r["final_val_loss"]]
    rs = [r for r in map(result, list(grid.values()) + list(seeds.values())) if r]
    summary = {
        "public": {
            "best_lr": best, "n_seeds": len(finals),
            "noise_floor_std": statistics.stdev(finals) if len(finals) > 1 else None,
            "noise_floor_range": max(finals) - min(finals),
            "baseline_val_loss_mean": statistics.mean(finals),
            "tokens_per_s_median": statistics.median(r["tokens_per_s"] for r in rs if r["tokens_per_s"]),
            "wall_min_median": statistics.median(r["wall_s"] for r in rs) / 60,
            "peak_mem_GB": max(r["peak_mem_GB"] for r in rs),
        },
        "sealed": {"lr_grid_final_val_loss": losses, "seed_final_val_loss": finals},
    }
    with open(os.path.join(out, "calibration.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("calibration finished", flush=True)


if __name__ == "__main__":
    main()
