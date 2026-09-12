"""Run a set of benchmark cells for ONE model, inside one process.

Engine construction on GB10 costs ~100 s, so a process-per-cell sweep would spend
most of its wall clock in startup. Instead every cell for a model shares one
engine: the KV cache is sized once for the largest cell, and batch size is varied
by how many requests are submitted.

Protocol per cell:

  1. warmup at this exact shape, discarded -- the first generation of an unseen
     shape triggers Triton JIT compilation, which would otherwise land inside the
     measurement.
  2. a prefill-only pass (max_tokens=1) to get time-to-first-token.
  3. a full pass (max_tokens=out_len); decode throughput is the remaining
     out_len-1 tokens over the remaining time.

Prompts are synthetic random token ids so input length is exact, and sampling uses
ignore_eos so every sequence performs identical work.

Emits one `ROW {json}` line per cell on stdout, for the orchestrator to collect.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def _now() -> float:
    return time.perf_counter()


def make_prompts(n: int, length: int, vocab: int, seed: int = 0) -> list:
    from vllm.inputs import TokensPrompt

    rng = random.Random(seed)
    # Avoid low ids, which are often special/control tokens.
    return [
        TokensPrompt(prompt_token_ids=[rng.randrange(10, vocab - 100) for _ in range(length)])
        for _ in range(n)
    ]


def run_cell(llm, sampling_cls, batch, in_len, out_len, vocab, repeats) -> dict:
    prompts = make_prompts(batch, in_len, vocab, seed=batch * 31 + in_len)

    prefill_only = sampling_cls(max_tokens=1, temperature=0.0, ignore_eos=True)
    full = sampling_cls(max_tokens=out_len, temperature=0.0, ignore_eos=True)

    # Warmup at this exact shape; discarded.
    llm.generate(prompts, full, use_tqdm=False)

    prefill_times, total_times = [], []
    for _ in range(repeats):
        t0 = _now()
        llm.generate(prompts, prefill_only, use_tqdm=False)
        prefill_times.append(_now() - t0)

        t0 = _now()
        outs = llm.generate(prompts, full, use_tqdm=False)
        total_times.append(_now() - t0)

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    t_prefill = median(prefill_times)
    t_total = median(total_times)
    t_decode = max(t_total - t_prefill, 1e-9)

    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    decode_tokens = batch * (out_len - 1)

    return {
        "batch": batch,
        "in_len": in_len,
        "out_len": out_len,
        "repeats": repeats,
        "t_prefill_s": t_prefill,
        "t_total_s": t_total,
        "t_decode_s": t_decode,
        "prefill_tok_s": batch * in_len / t_prefill,
        "decode_tok_s": decode_tokens / t_decode,
        "output_tok_s": batch * out_len / t_total,
        "ttft_ms": t_prefill * 1e3,
        "tpot_ms": t_decode / max(out_len - 1, 1) * 1e3,
        "generated_tokens": gen,
        "prefill_times_s": prefill_times,
        "total_times_s": total_times,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", required=True,
                    help="JSON list of {batch, in_len, out_len, repeats}")
    ap.add_argument("--gpu-memory-utilization", type=float, required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--max-num-seqs", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    cells = json.loads(args.cells)

    from vllm import LLM, SamplingParams

    t0 = _now()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=False,
        disable_log_stats=True,
        # Prefix caching MUST be off. The warmup pass primes the cache with the
        # very prompts the measured passes reuse, so prefill is served from cache
        # and TTFT/prefill throughput come out roughly 10x above the machine's
        # physical ceiling -- 624k tok/s measured on a 0.6B whose compute-bound
        # limit is 64k tok/s. Decode is unaffected, but prefill must recompute.
        enable_prefix_caching=False,
        trust_remote_code=False,
    )
    load_s = _now() - t0

    vocab = llm.get_tokenizer().vocab_size

    # How many tokens of KV the engine actually allocated. If a cell needs more
    # than this, vLLM will silently run it in waves rather than concurrently,
    # which would make the row's batch label a lie -- so such cells are marked.
    kv_capacity_tokens = None
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        if cc.num_gpu_blocks:
            kv_capacity_tokens = cc.num_gpu_blocks * cc.block_size
    except Exception as e:
        print(f"WARN could not read kv capacity: {type(e).__name__}: {e}", flush=True)

    print(f"ROWMETA {json.dumps({'model': args.model, 'load_s': load_s, 'vocab': vocab, 'kv_capacity_tokens': kv_capacity_tokens})}", flush=True)

    for cell in cells:
        batch, in_len, out_len = cell["batch"], cell["in_len"], cell["out_len"]
        repeats = cell.get("repeats", args.repeats)
        need = batch * (in_len + out_len)
        concurrent = kv_capacity_tokens is None or need <= kv_capacity_tokens
        try:
            row = run_cell(llm, SamplingParams, batch, in_len, out_len, vocab, repeats)
            row.update(
                model=args.model,
                dtype=args.dtype,
                status="ok" if concurrent else "ok_but_batched_in_waves",
                kv_capacity_tokens=kv_capacity_tokens,
                kv_tokens_needed=need,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        except Exception as e:
            row = {
                "model": args.model, "batch": batch, "in_len": in_len,
                "out_len": out_len, "status": f"error:{type(e).__name__}",
                "error": str(e)[:400], "kv_capacity_tokens": kv_capacity_tokens,
                "kv_tokens_needed": need,
            }
        print(f"ROW {json.dumps(row)}", flush=True)

    print("CELLS_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
