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
