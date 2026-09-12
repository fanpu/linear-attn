# GB10 Inference Capability Survey — Design

**Date:** 2026-09-12
**Location:** `misc/inference-benchmarking/`
**Status:** approved, in implementation

## Purpose

Measure what a single NVIDIA GB10 (DGX Spark) can actually serve for LLM inference,
across model size, batch size, and context length. The output is a reference table
plus a writeup: the kind of survey people publish when new hardware lands, but with
the numbers grounded against a measured roofline rather than quoted spec sheets.

Explicit non-goal: this is a capability survey, not a research claim about scaling
laws or MoE behaviour. The roofline appears as a reference line for interpreting
measurements, not as the thesis.

## Hardware under test

| | |
|---|---|
| GPU | NVIDIA GB10, Blackwell `sm_121`, 48 SMs |
| Memory | 121.7 GiB **unified** — `torch.cuda` reports the whole system pool |
| CPU | 20-core ARM (10x Cortex-X925 + 10x Cortex-A725) |
| Stack | vLLM 0.23.0, FlashInfer 0.6.12, torch 2.11.0+cu130, transformers 5.12.1 |

Two properties drive the whole design:

1. **Unified memory.** There is no separate framebuffer. `nvidia-smi` reports
   `memory.used = N/A` (verified). GPU over-allocation starves the host and can
   trigger an OOM-kill hang, so memory safety is a first-class requirement, and
   the watchdog must read `/proc/meminfo MemAvailable` rather than NVML.
2. **Bandwidth-to-compute ratio.** ~273 GB/s against ~125 dense BF16 TFLOP/s puts
   the roofline ridge near 460 FLOP/byte. Dense decode has arithmetic intensity
   approximately equal to batch size, so single-stream decode sits far into the
   memory-bound regime. Measurements are reported as achieved fraction of the
   *measured* roofline, not the spec one.

## Architecture

Seven modules, each independently testable.

| File | Responsibility | Depends on |
|---|---|---|
| `hw_probe.py` | Sustained BF16/FP8 GEMM TFLOP/s, achievable copy bandwidth, clocks under sustained load. Writes `results/roofline.json` | torch |
| `budget.py` | **Pure function, no GPU.** Model config -> predicted bytes for weights, KV cache, activation headroom. Drives the pre-flight gate and the analytic reference line | none |
| `guard.py` | Runs a child process under a memory watchdog: samples `MemAvailable` at 1 Hz, SIGKILLs on floor breach or timeout | none |
| `bench.py` | Sweeps batch/length cells for ONE model inside one process: build `LLM`, warm up, measure, emit JSON rows | vllm |
| `serve_bench.py` | Phase 2: launch `vllm serve`, drive at target request rates, record TTFT/TPOT/e2e p50/p99 | vllm, httpx |
| `sweep.py` | Orchestrator: enumerate grid -> pre-flight -> guard -> append `results/results.jsonl`. Resumable | budget, guard |
| `plot.py` | Figures and summary tables for the writeup | matplotlib |

Data flow: `sweep.py` reads the grid, asks `budget.py` whether each cell fits,
launches `bench.py` per *model* through `guard.py`, and appends returned rows to
`results/results.jsonl`. `plot.py` and the writeup read only that JSONL plus
`roofline.json`.

### Why one subprocess per model, not per cell

Measured vLLM engine startup is **101 s**. A ~150-cell grid at one process per cell
would spend 4+ hours in startup alone. Cells for a given model therefore share one
process, with the engine constructed once. Isolation is preserved at the model
boundary, which is where the memory-footprint changes anyway. Because
`results.jsonl` is append-only and `sweep.py` skips cells already present, a child
killed by the watchdog costs only the remaining cells of that one model.

## Experiment grid

Models — Qwen3 family in BF16, one architecture across a 50x size range:
0.6B, 1.7B, 4B, 8B, 14B, and 30B-A3B (MoE, 3B active). The MoE falls back to the
FP8 checkpoint if BF16 weights leave too little room for meaningful KV depth.

Axes:

- **batch / concurrency:** 1, 2, 4, 8, 16, 32, 64, 128, 256 (pre-flight gated)
- **input length:** 128, 1024, 8192; plus 32768 on models small enough to allow it
- **output length:** 128 for throughput cells; 1 for prefill-only TTFT cells
- **mode:** prefill-only, decode-heavy
- **dtype:** BF16 throughout; FP8 on a subset if `sm_121` supports it

Per-cell metrics: prefill tok/s, decode tok/s, end-to-end tok/s, TTFT, TPOT,
peak memory, and achieved fraction of the measured roofline.

## Memory safety

Four independent layers. The requirement is that no run may disturb the host.

1. **Hard ceiling of 73 GiB** (60% of 121.7). `gpu_memory_utilization` is computed
   per cell from the budget and never exceeds 0.60.
2. **Pre-flight gate.** A cell whose predicted weights + KV + headroom exceeds the
   ceiling is *skipped and recorded as skipped*, never attempted.
3. **Watchdog.** The child is SIGKILLed if system `MemAvailable` drops below a
   12 GiB floor, or if it exceeds its wall-clock timeout. The trip is recorded in
   the results row so it is visible in the writeup rather than silently lost.
4. **Serial execution.** Exactly one benchmark child at a time; fresh process per
   model, so no allocator state leaks across models.

## Measurement protocol

- **Warmup must cover every measured shape.** The smoke test showed a Triton JIT
  compilation spike on the first generation of an unseen shape. Each cell runs
  discarded warmup iterations at its own shape before timing.
- Timing by wall clock over a fixed duration where practical, so small and large
  models get equal measurement windows and comparable thermal state.
- Repeats per cell with median reported and spread recorded.

## Testing

- `budget.py` is pure arithmetic and gets real unit tests against known model
  configs, written before the implementation (TDD).
- `guard.py` gets a test that spawns a deliberate memory hog and asserts the
  watchdog kills it.
- A single smoke cell (smallest model, batch 1) runs before the full sweep.

## Deliverable

`misc/inference-benchmarking/README.md`: hardware, method, measured roofline,
result tables and figures, and a direct answer to "what can this box actually
serve" — including the configurations that were skipped as unsafe and why.
