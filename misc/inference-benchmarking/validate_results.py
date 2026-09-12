"""Check measured results against physical limits.

The prefix-caching bug produced prefill throughputs ~10x above what the hardware
can compute, and it survived until someone compared a number to the roofline by
hand. This makes that comparison automatic: any row claiming performance the
machine cannot deliver is a measurement error, not a result.

Run after every sweep:
    python3 validate_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import budget
import perf_model as pm

RESULTS = Path("results/results.jsonl")
TOLERANCE = 1.05  # a few percent over the roof is measurement noise, not a bug


def check(rows, roof) -> list[str]:
    problems = []
    specs = {}
    for r in rows:
        if not str(r.get("status", "")).startswith("ok"):
            continue
        model = r["model"]
        if model not in specs:
            try:
                specs[model] = budget.ModelSpec.from_hf_cache(model)
            except Exception as e:
                problems.append(f"{model}: cannot load spec ({type(e).__name__})")
                specs[model] = None
        spec = specs[model]
        if spec is None:
            continue

        act = pm.active_weight_bytes(spec.config, spec.weight_bytes)
        params = act / 2

        # 1. Prefill cannot exceed the compute roof: it must do 2 FLOP per
        #    parameter per token, and the machine has a finite FLOP/s.
        if r.get("prefill_tok_s"):
            achieved = 2 * params * r["prefill_tok_s"]
            if achieved > roof.flops() * TOLERANCE:
                problems.append(
                    f"{model} B={r['batch']} in={r['in_len']}: prefill "
                    f"{r['prefill_tok_s']:,.0f} tok/s implies "
                    f"{achieved/1e12:.0f} TFLOP/s > {roof.tflops:.0f} roof "
                    f"({achieved/roof.flops():.1f}x) — likely served from cache"
                )

        # 2. Decode cannot exceed the bandwidth roof: every step must read at
        #    least the active weights once.
        if r.get("decode_tok_s"):
            eff = pm.effective_bytes_per_step(r["decode_tok_s"], r["batch"], roof)
            if eff < act / TOLERANCE:
                problems.append(
                    f"{model} B={r['batch']} in={r['in_len']}: decode "
                    f"{r['decode_tok_s']:,.0f} tok/s implies reading "
                    f"{eff/1e9:.1f} GB/step < {act/1e9:.1f} GB of active weights"
                )

        # 3. A cell labelled batch N must actually have run N concurrently.
        cap = r.get("kv_capacity_tokens")
        need = r.get("kv_tokens_needed")
        if cap and need and need > cap:
            problems.append(
                f"{model} B={r['batch']} in={r['in_len']}: needed {need:,} KV "
                f"tokens > {cap:,} capacity — ran in waves, not concurrently"
            )
    return problems


def main() -> int:
    if not RESULTS.exists():
        print("no results yet")
        return 0
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    roof = pm.Roofline.from_json("results/roofline.json")
    problems = check(rows, roof)
    ok = sum(1 for r in rows if str(r.get("status", "")).startswith("ok"))
    print(f"checked {ok} usable rows against the measured roofline "
          f"({roof.tflops:.1f} TFLOP/s, {roof.bw_gbps:.1f} GB/s)")
    if problems:
        print(f"\n{len(problems)} PHYSICALLY IMPOSSIBLE RESULT(S):\n")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("all rows are within physical limits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
