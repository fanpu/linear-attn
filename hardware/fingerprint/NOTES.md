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
  the first attempt OOM'd the whole 130 GB unified pool (in the SDPA math op). Both scripts now `torch.cuda.empty_cache()` per B.
- **Session pitfall:** a session restart kills children. Launch long runs with `setsid nohup ... < /dev/null &`.
- Renderers: `render_invariance.py` (bitmaps night/spectral/paper/riso/oslo + print versions, atlas, bands, position triptych,
  `cache/inv_summary.json`), `render_divergence.py` (raster paper/night/riso, drift, texts, firstdiv, triptych, film).

## Findings so far (toy bmax=32 run and partial full runs)
- TM snippet `torch.mm(a[:1], b) - torch.mm(a, b)[:1]` max diff = 1641 (blog: 1642).
- Batch-VARIANT (differ at ~all B >= 2): torch.mm fp32/bf16, down_proj bf16, up_proj fp32, lm_head fp32, up_proj prefill,
  SDPA decode flash/cuDNN/default (506-509/512), full-model prefill logits (all 512, and position in batch matters).
- Batch-INVARIANT: q_proj bf16 (0/512), all RMSNorm variants, softmax, SDPA math (both), SDPA mem-efficient, all prefill SDPA.
- up_proj bf16 differs at only 16/512 B; layer_decode at 8/24 (toy).
- Run-to-run repeats and filler-row swaps: identical everywhere checked (dispatch is deterministic; differences are per-shape).
- Divergence (feynman, first attempt, lost to a restart): first differing token = 69 for every B >= 2 up to 480 (B=2,3 identical to
  B=1 within 320 tokens? check), rows within one batch disagree at several B (rows_disagree up to 22).

## Running (launched 05:12, setsid; logs/inv_full.log, logs/div_{feynman,story,sky}.log, flag logs/div_done.flag)
- `invariance.py --bmax 512 --out cache/inv.npz` (~15 min) and the 3-prompt divergence chain (~25 min each).

## Next
1. `render_invariance.py` then `render_divergence.py` (both CPU, ~5 min). View plates, pick heroes.
2. README.md (phenomenon, heroes, gallery, what was computed, verification: TM numbers, run-to-run, filler swap; caveats).
3. Add Fingerprint and Lattice rows to `art/README.md` hardware table.
