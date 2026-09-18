"""Look at MQAR examples, or at the failures of a finished run. [AI]

    python scripts/look_at_mqar.py --config experiments/day4_mqar/attn_d64.toml
        prints two examples in full and ten random query positions with their gaps
    python scripts/look_at_mqar.py --run runs/<name> [--n 10]
        prints ten random failed queries of that run's last test pass, with the
        key position, the query position, the gap, the answer, and the prediction

Zoology keeps no such script; this is the "look at the data before, and at
the outputs after" step of the handout.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from testbed.tasks.mqar import IGNORE_INDEX, multiquery_ar, query_gaps  # noqa: E402


def show_examples(cfg: dict) -> None:
    d = cfg["data"]
    x, y = multiquery_ar(d["vocab_size"], 10, d["input_seq_len"], d["data_seed"], power_a=d["power_a"],
                         num_kv_pairs=d["num_kv_pairs"], random_non_queries=d["random_non_queries"])
    print(f"inputs {tuple(x.shape)} int64, labels {tuple(y.shape)} int64; vocab {d['vocab_size']}: "
          f"keys in [1, {d['vocab_size'] // 2}), values in [{d['vocab_size'] // 2}, {d['vocab_size']})")
    c = 2 * d["num_kv_pairs"]
    for i in range(2):
        pairs = [(int(x[i, 2 * j]), int(x[i, 2 * j + 1])) for j in range(d["num_kv_pairs"])]
        print(f"\nexample {i}: context = {d['num_kv_pairs']} (key, value) pairs at positions 0..{c - 1}; first four: {pairs[:4]}")
        print(f"positions {c}..{c + 15} (pos: input -> label; -100 = no target):")
        print("  " + "  ".join(f"{t}: {int(x[i, t])} -> {int(y[i, t])}" for t in range(c, c + 16)))
    rng = np.random.default_rng(0)
    gaps = query_gaps(x, y, d["num_kv_pairs"])
    print("\nten random queries (example, key position, query position, gap):")
    for _ in range(10):
        i = int(rng.integers(len(gaps)))
        kp, qp, gap = gaps[i][int(rng.integers(len(gaps[i])))]
        print(f"  example {i:2d}  key at {kp:4d}  query at {qp:4d}  gap {gap:4d}")
    all_gaps = np.array([g for rows in gaps for (_, _, g) in rows])
    print(f"\ngap over 10 examples: min {all_gaps.min()}, median {int(np.median(all_gaps))}, max {all_gaps.max()}")


def show_failures(run: Path, n: int) -> None:
    cfg = tomllib.loads((run / "config.toml").read_text())
    d = cfg["data"]
    z = np.load(run / "eval_preds.npz")
    preds, labels = z["preds"], z["labels"]
    x, y = multiquery_ar(d["vocab_size"], d["num_test"], d["input_seq_len"], d["data_seed"] + 1, power_a=d["power_a"],
                         num_kv_pairs=d["num_kv_pairs"], random_non_queries=d["random_non_queries"])
    assert np.array_equal(y.numpy(), labels), "regenerated test set differs from the run's"
    gaps = query_gaps(x, y, d["num_kv_pairs"])
    fails = [(i, kp, qp, gap) for i, rows in enumerate(gaps) for (kp, qp, gap) in rows if preds[i, qp] != labels[i, qp]]
    total = sum(len(r) for r in gaps)
    print(f"{run.name}: {len(fails)} of {total} queries wrong ({1 - len(fails) / total:.4f} accuracy)")
    if not fails:
        return
    rng = np.random.default_rng(0)
    print(f"{n} random failures (example, key pos, query pos, gap, answer, predicted, predicted-is-some-value):")
    for j in rng.choice(len(fails), size=min(n, len(fails)), replace=False):
        i, kp, qp, gap = fails[j]
        p = int(preds[i, qp])
        print(f"  ex {i:4d}  key {kp:4d}  query {qp:4d}  gap {gap:4d}  answer {int(labels[i, qp]):5d}  "
              f"pred {p:5d}  {'value' if p >= d['vocab_size'] // 2 else 'not a value'}")
    fg = np.array([f[3] for f in fails])
    ag = np.array([g for rows in gaps for (_, _, g) in rows])
    print(f"gap of failed queries: median {int(np.median(fg))} (all queries: median {int(np.median(ag))})")
    in_ctx = np.array([f[1] for f in fails]) // 2
    print(f"key index of failed queries (0 = first pair written): median {int(np.median(in_ctx))} of {d['num_kv_pairs'] - 1}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--run")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    if a.run:
        show_failures(Path(a.run), a.n)
    else:
        show_examples(tomllib.loads(Path(a.config).read_text()))
