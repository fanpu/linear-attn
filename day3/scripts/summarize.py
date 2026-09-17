#!/usr/bin/env python
"""Table and plot for every completed or in-progress run.  [AI-owned]
# after HazyResearch/zoology zoology/analysis/ (read every run directory, one
# table and one figure); the report role of karpathy/nanochat.

    python scripts/summarize.py                          # reads runs/
    python scripts/summarize.py --runs experiments/day3_seed_variance/sample_runs  # the shipped synthetic runs (warm-up)

Prints one row per run (size, seed, steps, tokens, final validation loss,
median training tok/s, peak GB, hours), then per-size two-seed estimates and,
via your seed_stats, the pooled seed standard deviation with its 80% range
and the minimum detectable difference. Writes experiments/day3_seed_variance/baselines.png: left,
validation loss against training tokens (log x) with one curve per run; right,
training loss against tokens. Runs still in progress appear in the plot from
their metrics.jsonl but not in the statistics.
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.analysis.seed_stats import seed_stats  # noqa: E402  [you]

SIZES = ["30M", "60M", "125M"]


def read_run(d):
    # An unclean kill can leave NUL bytes where the last page was never written,
    # and a resume from ckpt_latest.pt appends replayed steps; drop the NULs and
    # keep the last row per step.
    text = open(os.path.join(d, "metrics.jsonl"), "rb").read().replace(b"\x00", b"").decode()
    rows, bad = [], 0
    for l in text.splitlines():
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"warning: {d}: skipped {bad} unparseable line(s)", file=sys.stderr)
    meta = next((r for r in rows if r.get("kind") == "meta"), {})
    by_step = lambda kind: sorted({r["step"]: r for r in rows if r.get("kind") == kind}.values(), key=lambda r: r["step"])
    train, ev = by_step("train"), by_step("eval")
    summ_path = os.path.join(d, "summary.json")
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else None
    return meta, train, ev, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="experiments/day3_seed_variance/baselines.png")
    args = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(args.runs, "*")) if os.path.exists(os.path.join(d, "metrics.jsonl")))
    if not dirs:
        print(f"no runs under {args.runs}")
        return
    runs = []
    for d in dirs:
        meta, train, ev, summ = read_run(d)
        runs.append((os.path.basename(d), meta, train, ev, summ))

    print(f"{'run':13s} {'size':>5s} {'seed':>4s} {'steps':>7s} {'tokens':>9s} {'val@0':>7s} {'final val':>9s} {'tok/s':>8s} {'peak GB':>7s} {'hours':>6s}")
    losses = {}
    for name, meta, train, ev, summ in runs:
        size, seed = meta.get("size", "?"), meta.get("seed", "?")
        v0 = next((r["val_loss"] for r in ev if r["step"] == 0), float("nan"))
        if summ and summ.get("complete"):
            print(f"{name:13s} {size:>5s} {seed:>4} {summ['steps']:7d} {summ['tokens']/1e6:8.0f}M {v0:7.3f} {summ['final_val_loss']:9.4f} "
                  f"{summ['train_tok_s_median'] or 0:8.0f} {summ['peak_mem_GB']:7.1f} {summ['wall_h_this_process']:6.2f}")
            losses.setdefault(size, []).append(summ["final_val_loss"])
        else:
            last = train[-1] if train else {}
            tag = "(smoke)" if summ else "(running)"
            print(f"{name:13s} {size:>5s} {seed:>4} {last.get('step', 0):7d} {last.get('tokens', 0)/1e6:8.0f}M {v0:7.3f} {tag:>9s} "
                  f"{last.get('tok_s', 0):8.0f} {last.get('peak_mem_GB', 0):7.1f} {last.get('wall_s', 0)/3600:6.2f}")

    print()
    for size in SIZES:
        xs = losses.get(size, [])
        if len(xs) >= 2:
            s = seed_stats({size: xs})
            print(f"{size:>5s}: losses {', '.join(f'{x:.4f}' for x in xs)}  s_hat = {s['s_pooled']:.4f}  "
                  f"80% range [{s['lo']:.4f}, {s['hi']:.4f}]  (nu = {s['nu']})")
    pooled_in = {k: v for k, v in losses.items() if len(v) >= 2}
    if pooled_in:
        s = seed_stats(pooled_in)
        print(f"pooled over {list(pooled_in)}: s_pooled = {s['s_pooled']:.4f}  nu = {s['nu']}  "
              f"80% range [{s['lo']:.4f}, {s['hi']:.4f}]  MDD (2 seeds per arm) = {s['mdd']:.4f} nats")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"30M": "C0", "60M": "C1", "125M": "C2"}
    for name, meta, train, ev, summ in runs:
        size = meta.get("size", "?")
        c = colors.get(size, "C3")
        ls = "-" if meta.get("seed", 0) % 2 == 0 else "--"
        if ev:
            pts = [r for r in ev if r["step"] > 0]  # step 0 has no place on a log axis; it is in the table
            a1.plot([r["tokens"] for r in pts], [r["val_loss"] for r in pts], ls, color=c, alpha=0.85, label=name)
        if train:
            a2.plot([r["tokens"] for r in train], [r["loss"] for r in train], ls, color=c, alpha=0.6, lw=0.8, label=name)
    for a, t in ((a1, "validation loss (nats)"), (a2, "training loss (nats, mean over log interval)")):
        a.set_xscale("log"); a.set_xlabel("training tokens"); a.set_ylabel(t); a.grid(alpha=0.3); a.legend(fontsize=8)
    a1.set_title("validation loss, one curve per seed (solid = even seed, dashed = odd)")
    a2.set_title("training loss")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
