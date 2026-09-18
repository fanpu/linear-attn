"""Summarize the day's runs and draw the recall-versus-state-size plot. [AI]

after zoology/analysis/ (a plotting script that reads the sweep's results
and draws the paper figure) and Day 3's summarize.py (one table, one png).

    python experiments/day4_mqar/plot.py [--runs runs] [--out experiments/day4_mqar]

Reads every `<runs>/*/summary.json`, prints one row per run, prints the
best-over-learning-rate accuracy per (mixer, d_model), and writes
`mqar_recall.png`: test accuracy against the state size in numbers, one
marker per run (seeds as separate markers at low opacity), the best run
per (mixer, d_model) joined by a line per mixer, attention drawn at its KV
cache size for the 512-token sequence.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from testbed.analysis.state_elements import state_elements  # noqa: E402  [you]: the x-axis

LABEL = {"attn": "attention", "linattn": "linear attention (additive)", "deltanet": "DeltaNet (delta rule)", "gdn": "gated DeltaNet"}
COLOR = {"attn": "black", "linattn": "tab:orange", "deltanet": "tab:blue", "gdn": "tab:green"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted(Path(a.runs).glob("*/summary.json"))]
    rows = [r for r in rows if not r["run"].startswith("smoke_")]
    if not rows:
        raise SystemExit(f"no summaries under {a.runs}")
    # the x-axis comes from your state_elements; the layers' own counters, recorded
    # by train_mqar.py in summary.json, must agree with it
    for r in rows:
        recorded = r["state_elements"]
        r["state_elements"] = state_elements(r["mixer"], r["d_model"], r["num_heads"], r["n_layers"], r["input_seq_len"])
        if recorded != r["state_elements"]:
            print(f"WARNING {r['run']}: state_elements gives {r['state_elements']}, the run recorded {recorded}")

    print(f"{'run':34s} {'mixer':9s} {'d':>4s} {'lr':>8s} {'seed':>4s} {'state':>7s} {'best_acc':>8s} {'final':>6s} {'epochs':>6s} {'stop':>5s} {'tok/s':>8s} {'GB':>5s} {'min':>6s}")
    for r in rows:
        print(f"{r['run']:34s} {r['mixer']:9s} {r['d_model']:4d} {r['lr']:8.1e} {r['seed']:4d} {r['state_elements']:7d} "
              f"{r['best_acc']:8.4f} {r['final_acc']:6.3f} {r['epochs_run']:6d} {'yes' if r['early_stopped'] else 'no':>5s} "
              f"{r['tok_s']:8.0f} {(r['peak_mem_GB'] or 0):5.1f} {r['wall_s'] / 60:6.1f}")

    # best over learning rates, per (mixer, d_model), seed 0 only; other seeds listed beside it
    best: dict[tuple[str, int], dict] = {}
    for r in rows:
        key = (r["mixer"], r["d_model"])
        if r["seed"] == 0 and (key not in best or r["best_acc"] > best[key]["best_acc"]):
            best[key] = r
    print("\nbest over learning rates (seed 0), with the same-lr runs at other seeds:")
    print(f"{'mixer':9s} {'d':>4s} {'state':>7s} {'best lr':>8s} {'acc s0':>7s} {'other seeds':>20s}")
    for (m, d), r in sorted(best.items()):
        others = [f"s{x['seed']}={x['best_acc']:.3f}" for x in rows if x["mixer"] == m and x["d_model"] == d and x["lr"] == r["lr"] and x["seed"] != 0]
        print(f"{m:9s} {d:4d} {r['state_elements']:7d} {r['lr']:8.1e} {r['best_acc']:7.4f} {' '.join(others):>20s}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ("linattn", "deltanet", "gdn", "attn"):
        pts = [r for r in rows if r["mixer"] == m]
        if not pts:
            continue
        ax.scatter([r["state_elements"] for r in pts], [r["best_acc"] for r in pts], color=COLOR[m], alpha=0.35, s=18)
        line = sorted([(d, r) for (mm, d), r in best.items() if mm == m])
        ax.plot([r["state_elements"] for _, r in line], [r["best_acc"] for _, r in line], "-o", color=COLOR[m], label=LABEL[m])
        for d, r in line:
            ax.annotate(f"d={d}", (r["state_elements"], r["best_acc"]), textcoords="offset points", xytext=(4, -10), fontsize=7, color=COLOR[m])
    ax.set_xscale("log")
    ax.set_xlabel("state size at generation, numbers held (2 layers, T = 512 for attention)")
    ax.set_ylabel("MQAR test accuracy, best over learning rates")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("512 tokens, 64 key-value pairs; faint markers = every run, incl. other seeds", fontsize=9)
    out = Path(a.out) / "mqar_recall.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
