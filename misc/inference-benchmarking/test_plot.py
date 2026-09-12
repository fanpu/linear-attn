"""Tests for plot.py, driven by synthetic results so no GPU or real run is needed.

Verifies the figures actually render, the summary tables are correct, and the
palette mapping is stable (colour follows the model, not its position in a
filtered list -- the recolour-on-filter anti-pattern).
"""

import json
import os
import random
import tempfile
from pathlib import Path

import plot


def synth_rows():
    """Plausible rows: decode throughput rising sub-linearly with batch."""
    rows = []
    models = plot.DENSE_ORDER + [plot.MOE]
    base = {"Qwen/Qwen3-0.6B": 120, "Qwen/Qwen3-1.7B": 70, "Qwen/Qwen3-4B": 40,
            "Qwen/Qwen3-8B": 22, "Qwen/Qwen3-14B": 13, plot.MOE: 30}
    for m in models:
        for in_len in (128, 1024, 8192):
            for b in (1, 2, 4, 8, 16, 32, 64):
                scale = b ** 0.85 / (1 + in_len / 4096 * b / 32)
                tok_s = base[m] * scale
                rows.append({
                    "model": m, "batch": b, "in_len": in_len, "out_len": 128,
                    "status": "ok", "decode_tok_s": tok_s,
                    "prefill_tok_s": tok_s * 40,
                    "ttft_ms": 20 + in_len * 0.01 * b,
                    "tpot_ms": 1000 * b / tok_s,
                })
    rows.append({"model": plot.MOE, "batch": 256, "in_len": 8192, "out_len": 128,
                 "status": "skipped:budget", "predicted_total_gib": 210.0,
                 "predicted_kv_gib": 150.0, "weights_gib": 57.0,
                 "reason": "exceeds ceiling"})
    return rows


def _setup(tmp):
    Path(tmp, "results").mkdir()
    with open(Path(tmp, "results/results.jsonl"), "w") as f:
        for r in synth_rows():
            f.write(json.dumps(r) + "\n")


def test_figures_render():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            rows = plot.load_rows()
            assert len(rows) > 100
            made = [
                plot.fig_decode_vs_batch(rows),
                plot.fig_context_effect(rows),
                plot.fig_latency(rows),
            ]
            for p in made:
                assert p is not None and Path(p).exists(), f"{p} not written"
                assert Path(p).stat().st_size > 5000, f"{p} suspiciously small"
        finally:
            os.chdir(cwd)


def test_summary_table_picks_the_best_cell():
    rows = synth_rows()
    t = plot.summary_table(rows)
    assert "0.6B" in t and "30B-A3B" in t
    assert t.count("\n") >= 6


def test_skipped_cells_are_reported_not_hidden():
    t = plot.skipped_table(synth_rows())
    assert "30B-A3B" in t and "210" in t


def test_colour_follows_the_model_not_the_filtered_order():
    """Filtering out a model must not repaint the survivors."""
    full = {m: plot.color_for(m) for m in plot.DENSE_ORDER}
    subset = [m for m in plot.DENSE_ORDER if m != "Qwen/Qwen3-1.7B"]
    for m in subset:
        assert plot.color_for(m) == full[m]


def test_moe_does_not_share_a_hue_with_the_dense_ramp():
    assert plot.color_for(plot.MOE) not in plot.DENSE_RAMP


def test_ramp_has_one_step_per_dense_model():
    assert len(plot.DENSE_RAMP) == len(plot.DENSE_ORDER)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
            except Exception as e:
                fails += 1; print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'FAILED' if fails else 'OK'} ({fails} failures)")
    raise SystemExit(1 if fails else 0)
