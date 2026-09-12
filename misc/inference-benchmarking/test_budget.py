"""Tests for budget.py — the pre-flight memory model.

Runs under pytest if available, otherwise `python3 test_budget.py`.
These are pure arithmetic; no GPU and no model weights required.
"""

import budget

GIB = 1 << 30

# Qwen3-0.6B shape, from its published config. GQA: 16 query heads, 8 KV heads,
# explicit head_dim=128 (note 16*128 != hidden_size, so head_dim must be read
# from the config rather than derived from hidden_size // num_attention_heads).
QWEN3_06B = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}

# A config with no explicit head_dim, to exercise the fallback path.
NO_HEAD_DIM = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
}


def test_head_dim_prefers_explicit_field():
    assert budget.head_dim(QWEN3_06B) == 128


def test_head_dim_falls_back_to_hidden_over_heads():
    assert budget.head_dim(NO_HEAD_DIM) == 128


def test_kv_bytes_per_token_uses_kv_heads_not_query_heads():
    # 2 (K and V) * 28 layers * 8 kv_heads * 128 head_dim * 2 bytes (bf16)
    assert budget.kv_bytes_per_token(QWEN3_06B, dtype_bytes=2) == 2 * 28 * 8 * 128 * 2


def test_kv_bytes_per_token_scales_with_dtype():
    bf16 = budget.kv_bytes_per_token(QWEN3_06B, dtype_bytes=2)
    fp8 = budget.kv_bytes_per_token(QWEN3_06B, dtype_bytes=1)
    assert bf16 == 2 * fp8


def test_kv_bytes_scales_linearly_in_batch_and_length():
    per_tok = budget.kv_bytes_per_token(QWEN3_06B)
    assert budget.kv_bytes(QWEN3_06B, batch=4, total_len=1024) == per_tok * 4 * 1024


def test_small_model_small_batch_fits():
    spec = budget.ModelSpec("Qwen3-0.6B", QWEN3_06B, weight_bytes=2 * GIB)
    plan = budget.plan_cell(spec, batch=1, total_len=1024, total_mem_bytes=121 * GIB)
    assert plan.fits
    assert plan.required_util < 0.60


def test_cell_exceeding_ceiling_is_rejected_not_clamped():
    """A cell that needs more than the ceiling must be refused outright.

    This is the core safety property: the pre-flight gate must never quietly
    shrink a request to make it fit, because that would silently produce a
    benchmark row labelled with a batch size that was never actually run.
    """
    spec = budget.ModelSpec("huge", QWEN3_06B, weight_bytes=100 * GIB)
    plan = budget.plan_cell(spec, batch=1, total_len=1024, total_mem_bytes=121 * GIB)
    assert not plan.fits
    assert "ceiling" in plan.reason.lower()


def test_kv_growth_can_push_a_fitting_model_over_the_ceiling():
    spec = budget.ModelSpec("Qwen3-0.6B", QWEN3_06B, weight_bytes=2 * GIB)
    ok = budget.plan_cell(spec, batch=1, total_len=1024, total_mem_bytes=121 * GIB)
    too_big = budget.plan_cell(
        spec, batch=1024, total_len=32768, total_mem_bytes=121 * GIB
    )
    assert ok.fits and not too_big.fits


def test_requested_util_never_exceeds_ceiling():
    spec = budget.ModelSpec("Qwen3-0.6B", QWEN3_06B, weight_bytes=2 * GIB)
    for batch in (1, 8, 64, 256):
        plan = budget.plan_cell(
            spec, batch=batch, total_len=4096, total_mem_bytes=121 * GIB
        )
        if plan.fits:
            assert plan.gpu_memory_utilization <= budget.CEILING_FRAC


def test_requested_util_leaves_room_for_weights_and_activation():
    """vLLM errors out if gpu_memory_utilization cannot cover weights+activation,
    so the requested fraction must always exceed the weight fraction."""
    spec = budget.ModelSpec("Qwen3-14B", QWEN3_06B, weight_bytes=28 * GIB)
    plan = budget.plan_cell(spec, batch=1, total_len=1024, total_mem_bytes=121 * GIB)
    assert plan.fits
    assert plan.gpu_memory_utilization > 28 * GIB / (121 * GIB)


def test_overhead_is_counted_against_the_budget():
    spec = budget.ModelSpec("m", QWEN3_06B, weight_bytes=1 * GIB)
    lean = budget.plan_cell(
        spec, batch=1, total_len=128, total_mem_bytes=121 * GIB, overhead_bytes=0
    )
    fat = budget.plan_cell(
        spec, batch=1, total_len=128, total_mem_bytes=121 * GIB, overhead_bytes=8 * GIB
    )
    assert fat.total_bytes - lean.total_bytes == 8 * GIB


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:
                fails += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'FAILED' if fails else 'OK'} ({fails} failures)")
    raise SystemExit(1 if fails else 0)
