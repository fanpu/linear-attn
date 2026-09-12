"""Tests for perf_model.py — the first-principles throughput prediction.

Pure arithmetic against the measured roofline; no GPU required.
"""

import perf_model as pm

GIB = 1 << 30

QWEN3_8B = {
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}

# Measured on this machine by hw_probe.py.
ROOF = pm.Roofline(tflops=95.9, bw_gbps=236.4)


def test_ridge_is_flops_over_bandwidth():
    assert abs(ROOF.ridge_flop_per_byte() - 95.9e12 / 236.4e9) < 1e-6


def test_decode_at_batch_one_is_memory_bound():
    r = pm.predict_decode(QWEN3_8B, weight_bytes=15 * GIB, batch=1, ctx=1024, roof=ROOF)
    assert r.bound == "memory"


def test_decode_becomes_compute_bound_only_past_the_ridge():
    """Decode arithmetic intensity is ~batch, so the crossover must sit near the
    ridge (~406 FLOP/byte), not at some small batch."""
    below = pm.predict_decode(QWEN3_8B, weight_bytes=15 * GIB, batch=64, ctx=128, roof=ROOF)
    assert below.bound == "memory"
    way_above = pm.predict_decode(QWEN3_8B, weight_bytes=15 * GIB, batch=4096, ctx=128, roof=ROOF)
    assert way_above.bound == "compute"


def test_memory_bound_throughput_scales_linearly_with_batch():
    """While weights dominate the read, doubling batch doubles tokens/s: the same
    weight bytes are amortised over twice the tokens."""
    a = pm.predict_decode(QWEN3_8B, weight_bytes=15 * GIB, batch=8, ctx=128, roof=ROOF)
    b = pm.predict_decode(QWEN3_8B, weight_bytes=15 * GIB, batch=16, ctx=128, roof=ROOF)
    assert 1.9 < b.tok_s / a.tok_s <= 2.0


def test_long_context_erodes_the_batching_benefit():
    """KV traffic grows with batch*ctx, so at long context the weight-amortisation
    win is cancelled and scaling falls below linear."""
    short_a = pm.predict_decode(QWEN3_8B, weight_bytes=15 * GIB, batch=8, ctx=128, roof=ROOF)
    short_b = pm.predict_decode(QWEN3_8B, weight_bytes=15 * GIB, batch=16, ctx=128, roof=ROOF)
    long_a = pm.predict_decode(QWEN3_8B, weight_bytes=15 * GIB, batch=8, ctx=32768, roof=ROOF)
    long_b = pm.predict_decode(QWEN3_8B, weight_bytes=15 * GIB, batch=16, ctx=32768, roof=ROOF)
    assert (long_b.tok_s / long_a.tok_s) < (short_b.tok_s / short_a.tok_s)


def test_single_stream_decode_is_bandwidth_over_weights():
    """The classic result: at batch 1 tok/s is just bandwidth / bytes-read, and
    KV at short context is a rounding error."""
    w = 15 * GIB
    r = pm.predict_decode(QWEN3_8B, weight_bytes=w, batch=1, ctx=128, roof=ROOF)
    naive = 236.4e9 / w
    assert abs(r.tok_s - naive) / naive < 0.05


def test_prefill_is_compute_bound_at_realistic_lengths():
    r = pm.predict_prefill(QWEN3_8B, weight_bytes=15 * GIB, batch=1, in_len=8192,
                           active_param_bytes=None, roof=ROOF)
    assert r.bound == "compute"


def test_moe_reads_only_active_experts_at_low_batch():
    """An MoE's decode speed is set by active, not total, parameters -- until
    batch grows enough that the union of routed experts covers the whole model."""
    total = 60 * GIB
    active = 6 * GIB
    dense = pm.predict_decode(QWEN3_8B, weight_bytes=total, batch=1, ctx=128, roof=ROOF)
    moe = pm.predict_decode(QWEN3_8B, weight_bytes=total, batch=1, ctx=128, roof=ROOF,
                            active_weight_bytes=active)
    assert moe.tok_s > dense.tok_s * 5


def test_predictions_are_positive_and_finite():
    for batch in (1, 8, 256):
        for ctx in (128, 8192):
            r = pm.predict_decode(QWEN3_8B, weight_bytes=15 * GIB, batch=batch,
                                  ctx=ctx, roof=ROOF)
            assert r.tok_s > 0 and r.tok_s < 1e9


QWEN3_MOE = {
    "hidden_size": 2048, "num_hidden_layers": 48, "moe_intermediate_size": 768,
    "intermediate_size": 6144, "num_experts": 128, "num_experts_per_tok": 8,
    "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
    "vocab_size": 151936, "tie_word_embeddings": False,
}


def test_active_weight_bytes_matches_published_active_params():
    """Qwen3-30B-A3B is published as 3.3B active of 30.5B total."""
    total = int(56.9 * (1 << 30))
    active = pm.active_weight_bytes(QWEN3_MOE, total)
    assert 3.0e9 < active / 2 < 3.6e9, f"implied {active/2/1e9:.2f}B active"


def test_dense_config_is_unchanged_by_active_weight_bytes():
    total = 15 * GIB
    assert pm.active_weight_bytes(QWEN3_8B, total) == total


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
