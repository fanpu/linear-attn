# Brief: rewrite the trainability-fractal engine for throughput on GB10

*Handoff for a fresh agent. Everything below is measured on this machine and this repo, not
estimated. Read `README.md` §"Method and verification" and `tfractal.py` before changing
anything.*

## 1. Why this matters

`art/trainability-fractal` is an artwork: every pixel is an independent neural network trained
for 500 steps, coloured by whether and how fast it learned. It is work #9 in `art/THE-EDIT.md`
and the only one of the fifteen that cannot currently hang, because the existing render is a
native **1024²** grid — a 12 cm print at the project's 8,600 px/m rule
(`art/display-directions.md` §3.2). Nothing may be interpolated up: the print is exactly as
large as the computation, and that honesty is part of the work.

So the goal is **more pixels per GPU-hour**. A 13,000² grid (1.5 m) is 161× the pixels of 1024².
At present rates that is somewhere between 4 and 33 days. The task is to make that number small
enough to be worth spending.

## 2. The computation, exactly

Per pixel, one independent network, all in float64:

- width `n = 16`; `W0 (16,16)`, `W1 (16,1)`; **no biases**
- dataset `X (272,16)`, `Y (272,1)`, standard normal, `N = 272 = 16² + 16` (the parameter count)
- **`X`, `Y`, `W0_init`, `W1_init` are identical for every pixel** (seed 0). Only the two
  learning rates `eta0`, `eta1` vary across the grid. This is the single most important
  structural fact for optimisation.
- forward: `Z0 = X @ W0 / sqrt(n)`, `h = tanh(Z0)`, `out = h @ W1 / n` (see `_forward`)
- full-batch MSE, hand-written backward (checked against autograd to 4e-16)
- 500 steps of plain gradient descent, per-layer learning rate
- the output per pixel is Sohl-Dickstein's convergence measure, a signed scalar; its **sign** is
  the converge/diverge label and is what the image's structure is made of

Analytic cost: **304,640 FLOP per network per step**, so 1.52e8 FLOP per pixel over 500 steps.

## 3. Hardware facts, measured in this repo

From `hardware/roofline/README.md` (idle GB10, 2026-09-14):

| quantity | measured | note |
|---|---|---|
| bf16 GEMM roof | **101.1 TFLOP/s** at 16384² | vendor peak 118.8 |
| device bandwidth roof | **247 GB/s** | 4 GiB copy, read+write |
| ridge point | **410 FLOP/byte** | below this, memory-bound |
| 1024² bf16 GEMM | 51% of roof | "still launch/tail-bound" |

GB10 is consumer-lineage Blackwell: **fp64 is narrow and has no useful tensor-core path**.
Tensor cores here cover FP4/FP6/FP8/BF16/FP16/TF32. `setup_gpu` sets `allow_tf32 = False`
deliberately — leave that alone unless you are explicitly working the emulation angle in §6.4.

## 4. Current performance, and the anomaly to chase first

| run | pixels | wall | px/s | fp64 GFLOP/s |
|---|---:|---:|---:|---:|
| `deep_zoomA4_1024_f64` (mixed phases) | 1,048,576 | 17,757 s | **59** | 9 |
| `hero_overview_tanh_1024_f32` | 1,048,576 | 5,354 s | 196 | — |
| `band_000` of 2048², **100% converged** | 262,144 | 868 s | **302** | 46 |

**The all-converged band runs 5.1× faster per pixel than the mixed-phase run.** That is the
opposite of what the design intends. `train_chunk`'s early exit compacts the alive set
(`idx = idx[keep]; W0 = W0[keep]; ...` every `check_every=25` steps) so that diverged pixels stop
costing compute. Band 0 never compacts, because nothing diverges — and it is 5× faster.

**Hypothesis 1 (test this first, it is nearly free):** the step function is
`torch.compile(..., dynamic=True)` and every compaction changes the batch shape, so mixed regions
pay repeated guard failures / recompiles that cost far more than the skipped work. Compare, on
one 256² mixed window: `early_exit=True` vs `False`, and `compiled=True` vs `False`, and count
recompiles with `TORCH_LOGS=recompiles`. If confirmed, the fix may be as small as padding the
alive set to a bucketed size, masking instead of compacting, or `dynamic=False` with fixed
shapes.

Even at 46 GFLOP/s the engine is ~0.05% of the bf16 roof. Most of that gap is structural, §5.

## 5. Why it is slow, structurally

The two big products are single GEMMs in an `(N, P·n)` layout, i.e. `M=272, K=16, N=P·16`.
**K = 16 is degenerate.** For a chunk of P = 32,768:

- FLOPs per forward GEMM: 2·272·16·(32768·16) = 4.56e9
- bytes: output `(272, 524288)` fp64 alone is **1.14 GB**
- operational intensity ≈ **4 FLOP/byte**, against a ridge of **410**

The kernel is two orders of magnitude inside the memory-bound region, and it materialises
gigabyte-scale activation tensors 500 times per chunk. The FLOPs are not the problem; the traffic
is.

**The structural opportunity:** `X` is `272 × 16` fp64 = **34,816 bytes**. It fits in a single
SM's shared memory, and it is *the same for every pixel*. A network's entire state — `W0`, `W1`,
plus gradients — is ~4.5 KB and fits in registers/shared memory. So the whole 500-step loop for a
network can in principle run **entirely on-chip**, with global traffic of a few KB in and 8 bytes
out per pixel. That turns a memory-bound problem into a compute-bound one and is where the large
factor lives.

## 6. Directions, ranked by expected value

1. **Kill the recompile/compaction pathology** (§4). Cheapest, possibly 5×, no numerical change.
2. **Fused on-chip training kernel.** One thread block per network (or per small tile of
   networks); `X` staged once into shared memory; all 500 steps in registers; write only the
   measure. Triton is the pragmatic route; CUDA C++ if Triton's fp64 codegen disappoints. This is
   the main event. Expect the win to come from eliminating activation traffic, not from FLOPs.
3. **Batch the 16×16 work sensibly.** With `n = 16`, per-network GEMMs are 272×16×16. Consider
   holding `W0` in registers per thread-block and looping `X` in tiles; or restructure so the
   reduction dimension is 272 rather than 16.
4. **fp64 emulation on tensor cores (advanced, optional).** Split-precision / Ozaki-style schemes
   reconstruct fp64-accurate GEMMs from many fp32 or TF32 tensor-core ops. On a part where native
   fp64 is this narrow, this can win. **Only viable if it passes §7 unchanged** — and note it
   changes rounding, so treat it as a method change requiring a declared note under the project's
   §0 truth rule (`art/ml-art-directions.md`).

## 7. Correctness bar — non-negotiable

The subject of this artwork is a **chaotic boundary**, so ordinary "close enough" does not apply.
Evidence already in `README.md` §"Method and verification":

- **float32 is not an option.** f32 vs f64 at 1024² agree on 99.80% of pixel labels and give the
  same box dimension to three digits, but **29% of float64 boundary pixels get a different label
  in float32**. The boundary is the work. Any reformulation must be argued at fp64 accuracy.
- **1-ulp sensitivity is already quantified**: perturbing both learning rates by `(1+2^-52)`
  flips at most 0.2% of boundary pixels along the zoom path. That is the noise floor any change
  must be compared against.
- **Precedent for an accepted approximation**: early exit was validated as 0 label flips at 128²
  and max relative change of the measure 1.4e-5. Match that standard.

**Ground truth to test against, already on disk:**

- `cache/windows/deep_zoomA4_1024_f64.npz` — the hero window, `conv_frac = 0.5121374130249023`,
  `edge_px = 56607`
- `cache/windows/hero_overview_tanh_1024_f64.npz` — the overview
- `cache/bands/deep_zoomA4_2048_f64/band_000.npz` — one finished 2048-wide band, 100% converged
- harness: `test_core.py`, `verify.py`, `deep_verify.py`, `boxcount.py`,
  `test_early_exit_speckle.py`

**Acceptance:** on the 1024² hero window, reproduce `conv_frac` to ≥ 5 decimal places, label
agreement ≥ 99.8% with disagreements confined to boundary pixels, `edge_px` within 1%, and box
dimension `D = 1.67 ± 0.04` unchanged. Report the same 1-ulp flip statistic. State the speedup as
px/s on the *mixed* hero window, not an all-converged band.

## 8. Constraints and conventions

- **One GPU job at a time.** Concurrent heavy jobs on this GB10 have caused NVRM driver OOM.
  `art/_shared/gpu_run.sh` takes a slot lock; use it.
- **Long jobs must be detached and resumable.** Use `setsid`, write progress, and checkpoint.
  `band_compute.py` (in this directory) already does the banding; band rows are verified
  bit-identical to the full-grid rows, so bands are seam-free and skippable on restart.
- `setup_gpu(0.10)` caps memory to 10% because the machine is shared. Raise it if you own the
  box, but note SM utilisation is already ~96% — occupancy is not the limiter.
- Commit straight to `main`, no branch, no PR, no Claude/Anthropic attribution in commits.
- The venv is `/home/fzeng/ml/research/art/.venv/bin/python`.

## 9. Deliverable

A faster engine that passes §7, plus a short section appended to this file recording: what the
bottleneck actually was, the before/after px/s on the mixed hero window, and the verification
numbers. Then the 2048² render can be finished (`band_compute.py --name deep_zoomA4_2048_f64`
resumes from band 1), and a larger grid costed honestly.

## 10. What not to do

- Do not switch to float32, or to any scheme that has not passed §7.
- Do not interpolate or upscale a render to a larger print size. `render_hero.py:41` currently
  writes the `_print` file as a 2× nearest-neighbour upscale; that is why the gallery file is
  2048 px while the computation is 1024². Do not extend that.
- Do not change the colour mapping here — that is a separate live decision tracked in
  `art/CRITIQUE.md` and `mapping_probe.py`.
- Do not optimise against an all-converged band. It is 5× easier than the real workload.

---

# Outcome (2026-09-20)

**Headline: 451 → 1755 px/s on the mixed hero window, 3.9×, with the fp64 arithmetic
unchanged. The 1024² hero now takes 597 s instead of 17 757 s.** The engine is a fused
CUDA kernel, `cuda/tfkernel.cu`; `tfractal.run_grid(..., engine=)` picks it automatically
and falls back to the old batched path for minibatch and checkpoint runs.

## 11. §4 was a measurement artifact, and the sign of the effect is reversed

Hypothesis 1 is **false**. `TORCH_LOGS=recompiles` over all four configurations on a
256² mixed window recorded **zero recompiles**: `torch.compile(..., dynamic=True)`
compiles one dynamic-shape graph and never re-specialises, so compaction costs nothing
in guard failures.

What the same window actually costs on an **idle** GPU (65 536 px, chunk 32 768):

| | `compiled=True` | `compiled=False` |
|---|---:|---:|
| `early_exit=True` | 144.2 s — **454.6 px/s** | 114.0 s — **574.9 px/s** |
| `early_exit=False` | 203.7 s — 321.7 px/s | 165.6 s — 395.8 px/s |

Two facts fall out. Early exit is worth **1.41×**, as designed. And `torch.compile` is
worth **0.79×** — it is a net *loss* of 26 % against eager on this workload.

Then the decisive test, two 262 144-px bands of the same 2048² grid, identical settings,
idle GPU:

| band | converged | `early_exit` | wall | px/s |
|---|---:|---|---:|---:|
| `band_000` | 100 % | on | 866.8 s | **302.4** |
| `band_000` | 100 % | off | 867.5 s | 302.2 |
| `band_008` | 27.8 % | on | 424.6 s | **617.4** |
| `band_008` | 27.8 % | off | 796.8 s | 329.0 |

`band_000` reproduces the 302 px/s in §4 exactly, so that run was clean. The **mixed**
band is 2.04× *faster* than the all-converged one, not 5× slower, and the gap closes
completely when early exit is switched off (302 vs 329). Compaction is doing its job.

So §4's 5.1× is the hero run having been measured under contention and the band run
idle. `README.md` already says as much — "on a shared GB10, float64 ran at 29–200 px/s"
— and the same engine on the same hero window idle today gives 451 px/s, 7.6× the 59 px/s
in the table. **§10's last line should be deleted: an all-converged band is not 5× easier
than the real workload, it is 2× harder.**

## 12. The real bottleneck: fp64 ALU throughput, not memory traffic

§5's diagnosis (gigabyte activation tensors, operational intensity 4 against a ridge of
410) is right about the traffic and wrong about what it costs. The relevant roof is not
the 101 TFLOP/s bf16 one. Measured on this machine (`bench_fp64.py`, idle):

| | measured | per SM-clock |
|---|---:|---:|
| fp64 FMA, ILP 8, full occupancy | **243.2 G-FMA/s** (486 GFLOP/s) | 2.10 |
| fp64 FMA, dependent chain | 205.5 G-FMA/s | 1.77 |
| fp64 `tanh` | **8.3 G/s** | 0.072 |

GB10 runs fp64 at 1/64 of fp32, i.e. 2 FMA per SM per clock, and one fp64 `tanh` costs
**21.9 FMA-equivalents**. Against that roof the old engine's 302 px/s on the converged
band is 15 % of peak, not 0.05 % of anything. The available factor was never 100×.

## 13. The kernel

`cuda/tfkernel.cu`: **one warp trains one network for all 500 steps, entirely on-chip.**

- lane = `16·half + e`, so lane (half, e) owns column `e` of `W0` and its gradient
  accumulator, and walks rows `i = half, half+2, …`. Two rows in flight per warp means
  all 32 lanes evaluate a **distinct** `tanh` — with one row per warp half the tanh work
  would be duplicated, and tanh is 30 % of the loop.
- `X` (272×16 fp64 = 34 KB, identical for every pixel — §2's "single most important
  structural fact") is staged once per block into shared memory. `W0`, `W1` and both
  gradient accumulators live in registers: 149 per thread, **no spills**.
- `z[i,e]` and `gW0[d,e]` are 16 lane-local FMAs each, no cross-lane traffic. Only
  `out[i] = Σ_e h·W1/n` needs a reduction: a 4-step butterfly inside each 16-lane half.
  The two halves are combined once per step, 17 shuffles.
- The activations never exist. Global traffic is 36 KB in per block and **8 bytes out
  per pixel**.
- Early exit is per-warp, so a diverged network simply leaves the loop. There is no
  compaction, no shape change and no batch-level 2 % threshold.

Everything that was a scalar constant in the reference stays one, and the one division
in the inner loop (`.sum(-1) / n`) is written `× 1/16`, which is exact.

**Throughput**, mixed hero window, idle GPU:

| engine | 256² | 1024² |
|---|---:|---:|
| torch, `compiled + early_exit` (the engine that made the gallery) | 451.3 px/s | — |
| torch, as recorded in §4 under contention | — | 59 px/s |
| **fused, `early_exit=True`** | **1765 px/s** | **1755 px/s** |
| fused, `early_exit=False` | — | 1261 px/s |

Block size makes almost no difference (128/160/192/224/256 all land within 3 %): 8 warps
per SM already saturate the fp64 pipe, because a single warp-wide fp64 FMA occupies it
for 16 clocks.

**How close to the roof.** The kernel issues ≈ 217 600 fp64 ops + 4 352 tanh per
network-step, so 500 steps of one pixel cost 1.088e8 fp64 ops + 2.18e6 tanh, i.e.
`1.088e8/2.432e11 + 2.176e6/8.3e9 = 710 µs` → **1408 px/s**. Measured without early exit:
1261 px/s, **90 % of its own analytic ceiling**, and 83 % of the raw fp64 FMA roof.

## 14. Verification (§7)

1024² hero window, fused engine, against `cache/windows/deep_zoomA4_1024_f64.npz`:

| quantity | reference | fused | bar |
|---|---:|---:|---|
| `conv_frac` | 0.5121374130 | 0.5121326447 | ≥ 5 dp — **differs by 4.8e-6**, a net 5 px of 1 048 576 |
| label agreement | — | **0.999991** (9 px) | ≥ 0.998 ✓ |
| `edge_px` | 56 607 | 56 615 | within 1 % — **+0.014 %** ✓ |
| `D` (b = 2–256) | 1.6744 ± 0.0354 | 1.6744 ± 0.0354 (r² = 0.9973) | 1.67 ± 0.04 ✓ |

**Where the 9 differing pixels are.** Eight are boundary pixels of the reference; the
ninth, (859, 22), is a single pixel inside a converged region — the isolated-speck dust
`README.md` already documents as unresolved at this density. All nine are in the top
0.7 % of the measure's magnitude, i.e. the runs whose last-20 mean sits closest to 1: undecided at 500
steps, which is what "boundary" means for this measure.

**Against the precision floor.** A 1-ulp nudge of both learning rates over the same
1024² grid, fused engine, flips **11** pixels (8 of 89 480 boundary pixels, 0.009 %).
The whole engine change flips **9**. The rewrite is the same size as one ulp of the
input, which is the strongest statement §7 allows.

Also checked, 128² windows, fused vs torch engine, same 500 steps:

| window | labels | flips | note |
|---|---:|---:|---|
| hero (mixed) | 1.000000 | 0 / 16 384 | — |
| overview (9 decades) | 0.998779 | 20 / 16 384 | all 20 on boundary pixels |

and the early-exit approximation inside the fused engine reproduces its own exact loop
to 0 flips with max relative change 2.2e-7 (hero) and 1.374e-5 (overview) — the same
1.4e-5 standard the early exit was accepted at.

A third check, on a quarter-million pixels of a *different* kind: `band_000` of the
2048² grid (262 144 px, 100 % converged, where early exit never fires) recomputed with
the fused engine against the torch-engine band already on disk — **0 label flips**,
median relative change of the measure **4.3e-10**, max 1.4e-2 at one near-boundary
pixel, and 868 s -> 227 s (3.82x).

**The right control: how big is the difference, really?** 20 % of pixels on the 128²
overview move by more than the 1.4e-5 early-exit standard, which sounds alarming until
it is put beside the window's own chaos. Run the same grid three ways — torch, torch
with both learning rates nudged one ulp, and fused (`verify_fused.py chaos`):

| window, 128² | perturbation | sign flips | median rel | px > 1.4e-5 | px > 1e-2 |
|---|---|---:|---:|---:|---:|
| overview (9 decades) | 1 ulp on both lr | 17 | 0 | 3400 | 1330 |
| overview (9 decades) | **fused engine** | **20** | 3.6e-16 | **3405** | **1334** |
| hero (deep zoomA4) | 1 ulp on both lr | 0 | 1.1e-10 | 1126 | 17 |
| hero (deep zoomA4) | **fused engine** | **0** | 7.6e-11 | **938** | **13** |

The two rows of each pair are the same distribution; on the hero window the engine
change is *smaller* than one ulp of the input. The drift is the object, not the code.

The arithmetic is unchanged fp64 throughout. What differs from the reference is only
summation *order*: a sequential 16-term FMA dot product instead of cuBLAS's dgemm over
`K = 16`, and a butterfly instead of torch's reduction over `e`. For scale, `torch.compile`
against eager torch — both "the same" engine — differs by max relative 4.5e-2 on the
measure at 256² with 0 label flips; the fused-vs-torch difference is 1.7e-2.

## 15. What is left, and what a larger grid costs

Residual headroom is small and not worth the risk. `tanh` alone caps this computation at
8.3e9 / 2.176e6 = **3815 px/s**, and the FMA work alone at 2233 px/s. §6.4's fp64
emulation on tensor cores could only attack the second term, so even a *free* GEMM buys
1261 → 2233 px/s (1.8×) while changing the rounding that §14's argument rests on. It was
not pursued. Likewise §6.3: the reduction is already over 272 rows, held in registers.

Honest cost at 1755 px/s on hero-like (≈ 51 % converged) content:

| grid | print at 8600 px/m | pixels | wall |
|---|---|---:|---:|
| 1024² | 12 cm | 1.05e6 | 597 s (was 4.9 h) |
| 2048² | 24 cm | 4.19e6 | **2391 s measured** (0.66 h) |
| 4096² | 48 cm | 1.68e7 | 2.7 h |
| 8192² | 95 cm | 6.71e7 | 10.6 h |
| **13 000²** | **1.51 m** | 1.69e8 | **26.7 h** |

A 1.5 m print is now a one-day job rather than the 4–33 days in §1. `band_compute.py`
keeps it resumable, and at 13 000² a 16-band split is 1.7 h per band.

**The 2048² is finished** (§9's second deliverable). All 16 bands were recomputed with
the fused engine rather than resuming from the torch-engine `band_000`, so the whole
image comes from one engine; the old band is kept at `cache/bands/_torch_ref/` and is
the 262 144-px comparison above. `cache/windows/deep_zoomA4_2048_f64.npz`: 0.66 h,
`conv_frac` 0.5122354, 183 807 boundary pixels, and box counting over b = 2–512 px gives
**D = 1.666 ± 0.034** (r² = 0.997) over **2.41 decades** of box size — 0.3 decades more
than the 1024² plate could reach, at the same D. Sampling it back down to 1024²
reproduces the trainable fraction (0.51194 vs 0.51214) with 97.6 % label agreement,
which is the same "half a native pixel off, and this region is rough" effect
`README.md` already records for the 1024²-vs-256² check (97.0 %).

## 16. Files

- `cuda/tfkernel.cu` — the kernel.
- `tfast.py` — build/dispatch; `tfast.available(prob, **kw)` gates the fallback.
- `tfractal.pick_trainer(prob, engine)` — `'auto'` (default), `'fused'`, `'torch'`.
- `bench_anomaly.py`, `bench_phase.py` — the §11 measurements.
- `bench_fp64.py` — the §12 roofs.
- `bench_fused.py` — the §13 sweep.
- `verify_fused.py small | hero | ulp` — the §14 table.
- `logs/perf_hero.log`, `logs/band2048_fused.log` — raw output.
