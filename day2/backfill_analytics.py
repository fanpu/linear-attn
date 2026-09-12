"""Recompute the analytic (non-measured) fields of results.jsonl in place.

flops_per_token, bytes_per_token and state_size_per_layer are pure functions of
(cfg, mixer, T, tokens_per_step) -- all of which each row already records -- so a change to
those models does not require re-running the GPU sweep. Everything the device actually
measured (step times, tok_per_s, peak_mem, clocks, param counts) is copied through untouched.

Usage: python backfill_analytics.py [--dry-run] [--file results.jsonl]
"""

import argparse, json

from bench_throughput import (
    PEAK_BF16_TFLOPS,
    PEAK_BW_GBPS,
    build_config,
    bytes_per_token,
    flops_per_token,
    load_roofline,
    state_size_per_layer,
)

# recomputed here; anything else in a row is measured and passes through verbatim
DERIVED = [
    "state_per_layer",
    "flops_per_token",
    "flops_breakdown",
    "bytes_per_token",
    "bytes_breakdown",
    "achieved_TFLOPs",
    "mfu_vs_peak",
    "mfu_vs_measured",
    "implied_GBps",
    "bw_util_vs_peak",
    "bw_util_vs_measured",
]


def rebuild(row, roof):
    mixer, size, T = row["mixer"], row["size"], row["T"]
    cfg = build_config(mixer, size)
    assert cfg.hidden_size == row["d"], f"{size}: d {cfg.hidden_size} != logged {row['d']}"
    assert cfg.num_hidden_layers == row["L"], f"{size}: L mismatch vs logged row"

    tokens = row["B"] * T
    tps = row["tok_per_s"]  # measured; never recomputed
    fpt_parts = flops_per_token(cfg, mixer, T)
    bpt_parts = bytes_per_token(cfg, mixer, T, tokens)
    fpt, bpt = sum(fpt_parts.values()), sum(bpt_parts.values())

    new = dict(row)
    new["state_per_layer"] = state_size_per_layer(cfg, mixer, T)
    new["flops_per_token"] = fpt
    new["flops_breakdown"] = fpt_parts
    new["bytes_per_token"] = bpt
    new["bytes_breakdown"] = bpt_parts

    achieved = fpt * tps
    new["achieved_TFLOPs"] = achieved / 1e12
    new["mfu_vs_peak"] = achieved / (PEAK_BF16_TFLOPS * 1e12)
    bw = bpt * tps
    new["implied_GBps"] = bw / 1e9
    new["bw_util_vs_peak"] = bw / (PEAK_BW_GBPS * 1e9)
    if roof:
        new["mfu_vs_measured"] = achieved / (roof["tflops"] * 1e12)
        new["bw_util_vs_measured"] = bw / (roof["GBps"] * 1e9)
    else:  # no roofline on disk: drop stale ratios rather than keep ones tied to old numbers
        for k in ("mfu_vs_measured", "bw_util_vs_measured"):
            new.pop(k, None)

    warn = []
    if roof and achieved > roof["tflops"] * 1e12:
        warn.append("implied FLOP rate exceeds measured matmul ceiling")
    if roof and bw > roof["GBps"] * 1e9:
        warn.append("implied bandwidth exceeds measured copy bandwidth")
    return new, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.file) if l.strip()]
    roof = load_roofline()
    print(f"roofline: {roof}\n")

    out, changed = [], 0
    for row in rows:
        new, warn = rebuild(row, roof)
        out.append(new)
        deltas = [k for k in DERIVED if row.get(k) != new.get(k)]
        changed += bool(deltas)
        tag = f"{new['mixer']:4s} {new['size']:5s}"
        print(
            f"{tag}  fpt {row.get('flops_per_token')} -> {new['flops_per_token']}"
            f"   bpt {row.get('bytes_per_token')} -> {new['bytes_per_token']:.0f}"
            f"   state {row.get('state_per_layer')} -> {new['state_per_layer']}"
        )
        print(
            f"{'':10s}  MFU(peak) {row.get('mfu_vs_peak', 0):.4f} -> {new['mfu_vs_peak']:.4f}"
            f"   BW(peak) {row.get('bw_util_vs_peak', 0):.4f} -> {new['bw_util_vs_peak']:.4f}"
        )
        for w in warn:
            print(f"{'':10s}  !! {w}")

    print(f"\n{changed}/{len(rows)} rows changed")
    if a.dry_run:
        print("dry run: nothing written")
        return
    with open(a.file, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {a.file}")


if __name__ == "__main__":
    main()
