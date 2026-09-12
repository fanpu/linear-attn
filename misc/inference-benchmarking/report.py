"""Generate the results tables for README.md from results/results.jsonl.

Kept separate from plot.py so the numeric findings can be regenerated and checked
without rendering anything.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import budget
import perf_model as pm

RESULTS = Path("results/results.jsonl")
ORDER = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
         "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-30B-A3B"]


def short(m):
    return m.split("/")[-1].replace("Qwen3-", "")


def rows():
    out = []
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def ok(rs):
    return [r for r in rs if str(r.get("status", "")).startswith("ok") and r.get("decode_tok_s")]


def table_headline(rs):
    """Single-stream vs best-batch throughput: the practical summary."""
    single, best = {}, {}
    for r in ok(rs):
        m = r["model"]
        if r["in_len"] == 128:
            if r["batch"] == 1:
                single[m] = r
            if r["decode_tok_s"] > best.get(m, (0,))[0]:
                best[m] = (r["decode_tok_s"], r["batch"])
    lines = ["| model | batch-1 tok/s | batch-1 TTFT | best tok/s | at batch | speedup |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in ORDER:
        if m not in single or m not in best:
            continue
        s, b = single[m], best[m]
        lines.append(f"| {short(m)} | {s['decode_tok_s']:.0f} | {s['ttft_ms']:.0f} ms | "
                     f"{b[0]:.0f} | {b[1]} | {b[0]/s['decode_tok_s']:.0f}× |")
    return "\n".join(lines)


def table_vs_predicted(rs):
    """Measured / predicted ratio, the check on the roofline model."""
    roof = pm.Roofline.from_json("results/roofline.json")
    by = defaultdict(dict)
    for r in ok(rs):
        if r["in_len"] == 128:
            by[r["model"]][r["batch"]] = r["decode_tok_s"]
    batches = [1, 8, 64, 256]
    lines = ["| model | " + " | ".join(f"B={b}" for b in batches) + " |",
             "|---" * (len(batches) + 1) + "|"]
    for m in ORDER:
        if m not in by:
            continue
        try:
            spec = budget.ModelSpec.from_hf_cache(m)
        except Exception:
            continue
        cells = []
        for b in batches:
            if b not in by[m]:
                cells.append("—")
                continue
            p = pm.predict_decode(spec.config, spec.weight_bytes, b, 256, roof)
            cells.append(f"{by[m][b]/p.tok_s:.2f}")
        lines.append(f"| {short(m)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_context(rs, model="Qwen/Qwen3-8B"):
    by = defaultdict(dict)
    for r in ok(rs):
        if r["model"] == model:
            by[r["in_len"]][r["batch"]] = r["decode_tok_s"]
    ctxs = sorted(by)
    batches = sorted({b for c in by.values() for b in c})
    lines = ["| prompt | " + " | ".join(f"B={b}" for b in batches) + " |",
             "|---" * (len(batches) + 1) + "|"]
    for c in ctxs:
        cells = [f"{by[c][b]:.0f}" if b in by[c] else "—" for b in batches]
        lines.append(f"| {c} tok | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_prefill(rs):
    by = {}
    for r in ok(rs):
        if r.get("prefill_tok_s") and r["in_len"] == 8192:
            by.setdefault(r["model"], []).append((r["batch"], r["prefill_tok_s"]))
    lines = ["| model | best prefill tok/s | at batch | % of 95.9 TFLOP/s roof |",
             "|---|---:|---:|---:|"]
    for m in ORDER:
        if m not in by:
            continue
        try:
            spec = budget.ModelSpec.from_hf_cache(m)
        except Exception:
            continue
        b, tps = max(by[m], key=lambda x: x[1])
        params = spec.weight_bytes / 2
        achieved = 2 * params * tps / 1e12
        lines.append(f"| {short(m)} | {tps:,.0f} | {b} | {achieved/95.9*100:.0f}% |")
    return "\n".join(lines)


def anomalies(rs):
    out = []
    for r in rs:
        st = str(r.get("status", ""))
        if st.startswith("error") or "wave" in st or st.startswith("incomplete"):
            out.append(f"- `{short(r.get('model','?'))}` B={r.get('batch')} "
                       f"in={r.get('in_len')}: {st}")
        if r.get("status") == "skipped:budget":
            out.append(f"- `{short(r['model'])}` B={r['batch']} in={r['in_len']}: "
                       f"skipped, needed {r['predicted_total_gib']:.0f} GiB")
    return "\n".join(out) if out else "_none — every cell ran concurrently as labelled_"


if __name__ == "__main__":
    rs = rows()
    print(f"# rows: {len(rs)} total, {len(ok(rs))} usable\n")
    print("## Headline\n");           print(table_headline(rs))
    print("\n## Measured / predicted\n"); print(table_vs_predicted(rs))
    print("\n## Context effect (8B)\n");  print(table_context(rs))
    print("\n## Prefill\n");          print(table_prefill(rs))
    print("\n## Anomalies\n");        print(anomalies(rs))
