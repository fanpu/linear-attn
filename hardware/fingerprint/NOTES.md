# Fingerprint NOTES (living handoff)

## Setup
- Stack: GB10 sm_121, driver 580.173.02, CUDA 13.0, torch 2.14.0+cu130, cuBLAS 13.1.1.3, cuDNN 92400, tf32 off. Model: decode-map's
  hand-written Qwen3-0.6B (`art/decode-map/qwen.py`, bf16 body, fp32 head, static KV, eager).
- `invariance.py`: one fixed row inside batches B = 1..512 at positions first/middle/last, 25 ops (TM torch.mm example, Qwen L13
  linears, RMSNorm variants, softmax, SDPA x 5 backends x decode/prefill, one decoder layer, full-model prefill). Stores raw output
  bits (first 4096 elements), ndiff, max|diff|, run-to-run + filler-swap checks, torch.profiler kernel signature per B.
  Per-op checkpoints in `cache/inv_parts/`; rerunning skips finished ops.
- `divergence.py`: greedy decode of one chat prompt at 84 batch sizes (1..64, 72..128 by 8, 160..512 by 32), L = 320 tokens.
  Row 0's tokens, top-2 margin, max/median |Δlogit| vs B = 1, within-batch disagreement, kernel signatures. Per-B checkpoint
  `cache/div_<p>.npz.partial.*`; rerunning resumes.
- **Memory pitfall (fixed):** every B allocates a slightly different size; the CUDA caching allocator kept every size cached and
  the first attempt OOM'd the whole 130 GB unified pool. Both scripts `torch.cuda.empty_cache()` per B.
- **Concurrency pitfall:** running invariance + divergence at the same time (2026-09-14 05:28) hit an NVIDIA *driver* OOM
  (`journalctl -k | grep NVRM`: NV_ERR_NO_MEMORY) and both died silently. Run GPU jobs one at a time: `hardware/run_part2.sh`.
- **Session pitfall:** a session restart kills children. Launch long runs with `setsid nohup ... < /dev/null &`.
- Renderers: `render_invariance.py` (bitmaps night/spectral/paper/riso/oslo + print versions, atlas, bands, position triptychs for
  all heroes, `cache/inv_summary.json`), `render_divergence.py` (raster paper/night/riso, drift, texts, firstdiv, triptych, film).
  Render on CPU pinned away from the GPU job: `OMP_NUM_THREADS=3 taskset -c 0-3 nice python render_*.py`.

## Results (full runs, 2026-09-14 09:26-09:40, idle GPU)
- `cache/inv.npz/json` (25 ops, 512 B, 12 min). `cache/inv_distinct.json` = distinct outputs per op over all B and whether the
  kernel signature determines the output. Headline: bf16 GEMMs 1-3 distinct outputs, fp32 GEMMs 8, full model 5 classes
  (B = 1 / 2-3 / 4 / 5-16 / 17-512); signature -> output holds for every bf16 op; B = 1 (gemv) is the odd one out;
  RMSNorm/softmax/math+eff SDPA/prefill SDPA invariant; flash/cuDNN decode SDPA switch at B = 7 / 4. TM snippet = 1641.
  Position in batch matters only for the full model. All repeat + filler-swap checks pass.
- `cache/div_feynman.npz/json`: 83/84 batch sizes depart at token 69 ("Lectures on/in Physics"), B = 1 margin there 0.013,
  6 distinct completions, classes match piece-1 classes (B = 17-30, 31-32).
- story and sky prompts: running in `run_part2.sh` (then roofline, staircase, pulse).

## Renderer fixes this session
- bitmap plate: axes raised so the caption no longer overlaps the x label; legend only when <= 6 signatures, else a one-line
  count; kernel-switch rules only when <= 40 switches. Atlas subtitles shortened. Bands: labels only for bands >= 24 B wide,
  alternating heights, labelled by the GEMM/attention kernel in the signature (`salient()`), not the alphabetically first copy kernel.
  Texts plate: block spacing fixed.

## Status
README.md written with piece 1 + feynman; link check passed. **Remaining:** render story + sky (`render_divergence.py` default
renders all three prompts, triptych and film), add their numbers to the README's piece-2 section, add the Fingerprint row to
`art/README.md`, commit.
