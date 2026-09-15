# §3 Crystal — M1 report (rates, determinism, calibration)

**Status: DONE_WITH_CONCERNS.** All M1 acceptance items are met. There are two concerns:
1. Fingerprint classes are **not** a usable kernel identity at k ≤ 256, so the crystal must be coloured by profiler kernel names. The names are cheap, and M2 records both.
2. No moment with another process on the GPU occurred during the job, so the busy-vs-idle load comparison is still open.

Commit: `7aaecd2` (hardware/crystal: code, NOTES.md, .gitignore). Stack: GB10 (sm_121a), driver 580.173.02, CUDA 13.0, torch 2.14.0+cu130, cuBLAS 13.1.1 (cublasLt 130101), bf16, TF32 off, 2026-09-15.

## What was done

- `hardware/crystal/common.py` reuses lattice's fingerprint method exactly (copied, with source comments):
  - `masters`, `mats` with fresh contiguous A and B and `out=C`;
  - the 6-value fingerprint: raw bits of C[0,0], C[m//2,n//2] and C[-1,-1] for two probes;
  - lattice's profiler kernel-name pass, generalised to (m, k, n).

  Lattice's `common.kernel_label` is imported via importlib. `hardware/lattice/` was not modified.
- `hardware/crystal/m1_calib.py` is one GPU job with checkpointed phases:
  - load0: 2000 shapes, fingerprints, GPU state at job start;
  - rate1 and rate2: the 10⁴ random shapes, fingerprinted twice in independent random orders;
  - slice: k = 4096 check;
  - kern: 2000 profiled shapes;
  - six dense (m, n) slabs at k ∈ {3, 16, 64, 127, 128, 256};
  - two dense (m, k) planes at n ∈ {127, 128};
  - loadS: 2000 shapes under a self-made load;
  - load1: 2000 shapes, GPU state at job end.
- `hardware/crystal/m1_analyze.py` (CPU) writes `cache/m1_stats.json`, `cache/m1_mapping.md` (the full class→kernel tables) and the previews.
- The job was queued via `gpu1.sh` (share mode) at 03:55, ran 04:38:34 to 04:40:40 (2 min of GPU) and exited 0. Log: `hardware/crystal/logs/m1_calib.log`.

```bash
cd /home/fzeng/ml/research/hardware/crystal
setsid nohup /home/fzeng/ml/research/art/_shared/gpu1.sh env OMP_NUM_THREADS=4 /home/fzeng/ml/research/art/.venv/bin/python -u m1_calib.py > logs/m1_calib.log 2>&1 < /dev/null &
OMP_NUM_THREADS=4 /home/fzeng/ml/research/art/.venv/bin/python m1_analyze.py
```

## Fingerprint rate and projected cube time

The job ran in share mode, so contention was possible but not controlled. In fact, nvidia-smi listed no other compute process at any phase. These rates are throughput estimates for planning, not performance measurements.

| measurement | µs / shape |
|---|---|
| fingerprint, 10⁴ random shapes in [1,256]³ (rate1 / rate2) | **63.6 / 62.1** (0.64 s / 0.62 s total) |
| fingerprint, dense slabs and planes (65 536 shapes each) | 74 to 86 |
| profiler kernel names, dense slabs and planes | 115 to 147 (median ≈ 130) |
| profiler kernel names, 2000 random shapes | 475 (per-chunk trace overhead dominates small runs) |

Projected wall time over the 16.7 M-shape cube:

| content | dense [1,256]³ | stride 2 |
|---|---|---|
| fingerprints only | 0.30 h | 0.04 h |
| fingerprints + kernel names | **0.91 h** | 0.11 h |

The spec guessed 4.6 h at 1 ms per shape. The measured cost is about 15× lower.

## Determinism

- **Rerun:** the same 10⁴ shapes in an independent random order gave **10 000 / 10 000 bitwise identical fingerprints**.
- **Repeated sample:** the 2000-shape sample fingerprinted four times (load0, kern, loadS, load1) was 100 % identical each time.
- **Profiler labels:** 66 of the 2000 sample shapes also lie in a dense slab or plane that was profiled separately. Their labels and fingerprints agree 66/66.

## k = 4096 slice check against lattice g256

- **Sub-grid:** m, n ∈ {1 + 4j + (j mod 4) : j = 0..63}. It covers every residue mod 4 and runs from 1 to 256.
- **Reference:** `hardware/lattice/cache/time_bf16_g256_k4096.npz`, measured 2026-09-13 without expandable_segments.
- **Result: 4096 / 4096 bitwise equal**, with 52 fingerprint classes on both sides.
- **Preview:** `hardware/crystal/cache/preview/slice_k4096_check.png`.

## Class → kernel-name mapping

Fingerprint values depend on k, because the probe vectors are drawn per k. Classes are therefore (k, 6-tuple). In the 2000-shape sample, 68 identical 6-tuples occur at different k, often all-zero results, and they mean nothing. The full per-class tables are in `hardware/crystal/cache/m1_mapping.md`.

**2000 random shapes in [1,256]³:**
- 27 kernel labels and 563 (k, fingerprint) classes (410 of them hit more than once).
- **104 classes hold more than one kernel label.** They contain 584 shapes; 196 of those are minority members.
- Most frequent merges:

| times seen | kernel label | merged with |
|---|---|---|
| 40 | cutlass_80 wmma s161616gemm 32x32_128x1 align2 | 32x32_128x2 align2 |
| 37 | cutlass_75 wmma s161616gemm 32x32_32x1 align1 | cutlass_80 32x32_128x2 align2 |
| 18 | cutlass_75 align1 +splitK | cutlass_80 align2 |
| 14 | cutlass_75 align1 | cutlass_75 align1 +splitK |

The merges are the parity lamellae themselves: odd-n and even-n kernels share a fingerprint.

**Sample label counts:**

| kernel label | shapes |
|---|---|
| cutlass_75 wmma s161616gemm 32x32_32x1 align1 +splitK | 956 |
| cutlass_75 wmma s161616gemm 32x32_32x1 align1 | 442 |
| cutlass_80 wmma s161616gemm 32x32_128x2 align2 | 432 |
| cutlass_80 wmma s161616gemm 32x32_128x1 align2 | 50 |
| gemmSN_NN<256,4,2,8,{2..7},4> | 44 |
| nvjet tst_mma, 8 variants | 40 |
| gemvx, gemvNSP, gemmk1 variants | 23 |
| other cutlass_80 align2 / align8 | 13 |

**Dense runs (65 536 shapes each):**

| run | fingerprint classes | kernel labels | purity fp→kernel | purity kernel→fp | classes with > 1 label | minority shapes |
|---|---|---|---|---|---|---|
| k = 3 | 1 | 5 | 0.934 | 1.000 | 1 | 4 336 |
| k = 16 | 4 | 15 | 0.822 | 1.000 | 2 | 11 506 |
| k = 64 | **1** | 18 | 0.465 | 1.000 | 1 | 35 056 |
| k = 127 | 6 | 11 | 0.961 | 1.000 | 1 | 2 550 |
| k = 128 | 10 | 25 | 0.641 | 0.990 | 5 | 23 549 |
| k = 256 | 12 | 28 | **0.358** | 0.982 | 4 | 42 093 |
| n = 127, (m, k) plane | 885 (k, fp) | 11 | 0.961 | (n/a across k) | 213 | 2 562 |
| n = 128, (m, k) plane | 822 (k, fp) | 28 | 0.914 | (n/a across k) | 252 | 5 622 |

- There are 37 distinct labels over all dense runs.
- At k = 256, fingerprint class 0 covers 61 854 shapes and holds cutlass_75 align1 (with and without +splitK), cutlass_80 align2 128x2 and 128x1, and several nvjet kernels.
- At k = 64 there is a single class. Lattice's probe places 64 cancelling big terms, so at k = 64 every term is big and the sum is exactly 0 in any order.
- For comparison, lattice's k = 4096 g256 purity was 0.87 fp→kernel.

**Structure seen along k** (first look at the third axis):
- **Odd k has no parity stripes.** At k = 3 and 127 every n runs align1 kernels. Even k (16, 64, 128, 256) shows lattice's odd-n/even-n stripes. So k's parity gates the lamellae.
- **A hyperbola-like surface** separates align1 +splitK from plain align1. At n = 127 it crosses k = 86 at m = 255, k = 101 at m = 200 and k = 166 at m = 100 (m·k is not constant). The same curve appears in the (m, n) slabs.
- **Label changes along m** at m = 17, 33, 82, 95/97, 115/117, 129 and 161 (k = 128 and 256; n = 200 and 201).
- **Small-m rows:** m ≤ 16 runs gemmSN/gemv kernels, with a boundary near k = 96.

## Load dependence

| run | GPU state | vs load0 |
|---|---|---|
| load0 (04:38:35) | no other compute process, util 0 % | reference |
| kern (04:38:37) | no other compute process, util 57 % (own job) | 100 % identical |
| loadS (04:40:38) | self-made load: a thread on a second CUDA stream ran 11 427 matmuls of 2048² bf16 during the pass, util 93 % | 100 % identical |
| load1 (04:40:39) | no other compute process, util 93 % (own job) | 100 % identical |

**No moment with another process on the GPU occurred during M1.** The art lock excluded other art jobs, and autonomous/ had nothing running. The required contrast against another process's load is therefore still open, and the controller should arrange a busy window in M2. Under in-process concurrent GPU load, kernel choice and output bits did not change at any of the 2000 shapes.

## Decision: dense vs stride

**Dense [1, 256]³.** The projected fingerprint + kernel-name time is 0.91 h, well under the 6 h threshold. Logged in NOTES.md.

## Previews (all viewed)

All previews are in `hardware/crystal/cache/preview/`:
- `slice_k4096_check.png`: lattice sub-grid, crystal, equality mask.
- `slab3.png`, `slab16.png`, `slab64.png`, `slab127.png`, `slab128.png`, `slab256.png`: fingerprint classes beside kernel labels, glasbey palette ordered by area.
- `plane127.png`, `plane128.png`: fingerprint changes along m beside kernel labels over (m, k).

Previews are matplotlib, nearest-neighbour. `cache/` is gitignored, so previews and stats are on disk only.

## Decisions (all in NOTES.md)

- **Probe for k < 64:** masters() is verbatim for k ≥ 64. For k < 64 it uses nbig = 2·(k//4), because lattice's construction cannot place 64 terms.
- **Probe fix for M2:** use nbig = min(64, 2·(k//4)) for every k. This is identical to lattice for k ≥ 128, so the slice check is unaffected. It fixes the degenerate k ∈ [64, 127] range.
- **Fingerprint:** the 6-tuple only, without lattice's `nuniq`.
- **Classes:** defined per k.
- **Slice check sub-grid:** as above.
- **Dense purity runs:** dense slabs and planes were added, because 2000 uniform samples give about 8 shapes per k, too few to measure purity.
- **Load check design:** load0, load1, kern and loadS as described.
- **Colouring:** the crystal is coloured by profiler kernel label. Fingerprints are a secondary layer plus the slice check.
- **Label parsing:** labels are parsed with lattice's `kernel_label` after stripping `std::enable_if<true, void>::type `. Without this, a gemvx variant became `enable_if<T>`.
- **Cube size:** dense cube.

## Risks

- **Fingerprint ≠ kernel.** A crystal coloured by fingerprint would erase the parity lamellae. M2 must record names for every shape; the cost is included in the 0.91 h estimate.
- **+splitK labels.** `+splitK` marks an extra `splitKreduce` kernel launched alongside the main kernel. It splits align1 into two regions along a hyperbola-like surface. This is treated as a distinct dispatch; that it is a real selection difference rather than a profiler artefact is supported by determinism (66/66), not proven.
- **Load dependence against another process:** untested (see above).
- **Small-k probe weakness.** At k ≤ 3 there are no cancelling terms, so fingerprints carry almost no information. That matters only for the secondary fingerprint layer.
- **fp32:** fingerprint rate and purity were not measured for fp32. Expect similar rates.
- **Queue waits** dominate wall time: 43 min queued against 2 min of GPU.
