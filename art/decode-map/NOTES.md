# decode-map: NOTES (living handoff)

## State (2026-09-13, agent 3)
- `qwen.py`: hand-written Qwen3-0.6B (bf16 body, fp32 head), static KV. NEW `StepGraphs`: CUDA-graph
  single-token step per window offset (masked full-length attention, fixed Fb rows).
- `decode.py`: prefix-trie engine. NEW certified sampler `sample_fast` (default for icdf):
  per node row, fp32 top-K (K=2048) + histogram of all logits (bins 0.04 below max; count, fp64 sum).
  Tail mass per (row,T) bracketed by Jensen (lower) and convexity chord between bin edges (upper);
  a pixel is accepted only if the top-p cutoff / icdf token agree at W_lo and W_hi and lie inside K'.
  Tiers K'=256 -> 2048 -> 32768 -> full float64 sort. Flags: `--nograph`, `--slow` (old sampler).
  `DM_TIMING=1` env prints per-stage seconds in stats.
- Exactness: `test_engine.py` (fp32, vs naive batch-1 full sort) unchanged by the new path:
  tp 1/144, tr 9/144, same pixels/ranks as before (all near-tied tail tokens, ranks 1.6k-88k).
- Speed (128^2, L=12, contended GPU): old 34-49 s -> 33 s; sampler 17 s -> ~12 s, fallbacks ~0.2% pixel-steps.
  Forward is compute bound (~1.6 ms/row at Fb=128; CUDA graphs barely help under contention).
  Memory: KV = 73 MB per (Ncap+Fb) row-block of 1280... i.e. 28*rows*8*128*2B per position:
  Ncap+Fb=512, P+L~85 -> 5.8 GB. Ncap 1024 OOMs at the 0.10 fraction. Use Ncap 384, Fb 128.
- Toy L-as-time animation: `gallery/toy/refine_toy_glass.{mp4,gif}` + stills (looks good).

## Running (launched ~16:20 via gpu_run, logs in logs/c*.log)
- c1: toy_story_tp64_fast (64^2 L32 x0-2, new engine), story_tp256, fact_tp256 (L64, T 0-1.5)
- c2: list_tp256, story_tr192 (T x repetition penalty 1-2, p=1)
- c3: placement set: place_story64_{shuffle,eager,slow_eager,fp32}, story_tp128 (L48),
  place_story128_shuffle, story_tp128_gumbel
- Resume if killed: `nohup ../_shared/gpu_run.sh bash logs/cN.sh > logs/cN.log 2>&1 &` (finished
  outputs are NOT skipped automatically; delete finished lines first).

## Next
1. verify.py placement pairs: add (toy_story_tp64_trie vs toy_story_tp64_fast) = old eager/slow vs graph/fast;
   eager vs graph; slow vs fast (same eager) should be 0.
2. Heroes from story_tp256: mosaic, glass, ink boundary lines, Spectral split, riso; anim.py on 256^2.
3. Box counting over refinement range (verify.py boxcount_global), README.
