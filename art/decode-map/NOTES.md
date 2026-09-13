# decode-map: NOTES (living handoff)

## State (2026-09-13, agent 4) -- README written, heroes rendered
- README.md complete draft: gallery, placement table, horizon, box counting verdict, commands, caveats.
- heroes.py (CPU): `maps` (glass/ink/mosaic/age for story/list/fact 256; story also firstdiv/coherence Spectral,
  aurora, riso), `metrics` (sheet, rep Spectral, entropy magma), `diptych` (icdf vs gumbel 128), `table`
  (cache/story_tp256_transects.html; README embeds rows with shared<48), `redo` (story plates only).
- verify.py: NEW `horizon` (prefix agreement + boundary Jaccard vs l). Run: `python verify.py placement cells horizon box`.
- Films: anim2 job (logs/anim2.log) re-rendering gallery/film/refine_story256_{glass,ink} slower (hold 22, fade 8, ~80 s).
  If ink files missing: rerun the python -c line in README Commands with mode='ink'.
- decode.py patch: UB capped at 192 when repetition penalty is used (fp64 penalised logits OOMed at UB=1024).
- c4 (logs/c4.log, via gpu_run): story_tr192 relaunched with Ncap 256 (first attempt c2b OOMed). Not in README yet.

## Key numbers
- Placement table (64², L32): same kernels 0%; shuffled Fb16 1.95%; eager vs graph 68.5%; bf16 vs fp32 57.9%;
  128² Fb32 vs Fb128 78.5%. Prefix agreement at l=8: 100/99.6/82/81/95%; l=16: 100/99.2/67/65/57%.
- Cells story256: 152 (l1), 955 (l4), 1604 (l8), 2056 (l16), 8376 (l64); singletons 79% (128²: 78%).
- icdf vs gumbel 128 L48: 71 vs 2 first tokens, 1945 vs 269 texts, singletons 78% vs 12%.
- Box counting: whole plane slopes 1.3-1.8, no plateau, resolution-consistent; nulls smooth 0.6-0.97, speckle 1.84-1.99.
  Resolved half T<0.75 at l<=16: 0.9-1.1 -> piecewise smooth curves; not fractal.
- EOS: story/list never end within horizon; fact ends at token 10 in >95%.

## Weak spots / next
1. When c4 finishes: `python heroes.py` style plates for story_tr192 (axes label tr handled via LAB), add a README section.
2. firstdiv Spectral plate is flat (few fd values); age plate dim at downscale (consider w=3). Mosaic muddier than glass.
3. Check the slower film frames and GIF sizes (<15 MB), then commit.
4. Careful: `pkill -f` matches your own shell; kill by PID.

## Earlier findings (agent 3)

- Speed reality: toy 64^2 L32 T0-2: old engine 309 s -> new 207 s (25k node-steps, ~8 ms/node-step incl.
  1.6 ms forward). High-T pixels with huge nuclei (T>1.2, p->1) dominate sampler cost (tiers 32768/full).
  Projection: 256^2 L64 T0-1.5 may take several hours on the contended GPU. If too slow: cap T at 1.2,
  or L48 (list/tr already L48). Watch `grep ^states logs/c1.log | tail -1` (pending px).
- NUMERICS FINDING (README-worthy): old toy (eager) vs new toy (CUDA graph) differ in 68% of pixels at L=32
  (67% in coherent region, median first differing token 16). Not a bug: scratch/graph_vs_eager.py shows eager
  bf16, graph bf16 and graph with fp32 attention are all equally far from the fp32 model
  (median |dlogit| ~0.031, ~1.5% argmax flips on random-token inputs). The bf16 body itself carries ~0.03 logit
  noise; any kernel change re-rolls near-margin token decisions and the texts diverge. Brute(B=256) vs trie
  (Fb=128) = 0/4096 only because they used identical kernels. So: cell geometry at small L is robust, fine
  structure past ~16 tokens is precision/kernel dependent. Quantify with c3 results (eager vs slow_eager
  should be ~0 = sampler exact on bf16; fp32 vs graph = precision effect; shuffle = placement effect).
- Placement test (new engine, same kernels): toy fast vs shuffled pixel order with Ncap=32, Fb=16 gives
  1.95% of pixels differing (80/4096) at L=32. Changing the window size Fb changes the batched kernel.
- `StepGraphs(attn_fp32=True)` option exists but is NOT used by running jobs (keep maps consistent).

## Running (launched ~16:20 via gpu_run, logs in logs/c*.log)
- c1: toy_story_tp64_fast (64^2 L32 x0-2, new engine), story_tp256, fact_tp256 (L64, T 0-1.5)
- c2b (logs/c2b.log): list_tp256 L48, story_tr192 L48 (T x repetition penalty 1-2, p=1). (c2 L64 was killed.)
- c3: placement set: place_story64_{shuffle,eager,slow_eager,fp32}, story_tp128 (L48),
  place_story128_shuffle, story_tp128_gumbel
- Resume if killed: `nohup ../_shared/gpu_run.sh bash logs/cN.sh > logs/cN.log 2>&1 &` (finished
  outputs are NOT skipped automatically; delete finished lines first).

## Next
0. Careful: `pkill -f`/`pgrep -f` patterns match your own shell command line; kill by PID.
1. `python verify.py` placement() (pairs already updated) once c3 finishes; put the table in README.
1b. verify.py placement pairs: add (toy_story_tp64_trie vs toy_story_tp64_fast) = old eager/slow vs graph/fast;
   eager vs graph; slow vs fast (same eager) should be 0.
2. Heroes from story_tp256: mosaic, glass, ink boundary lines, Spectral split, riso; anim.py on 256^2.
3. Box counting over refinement range (verify.py boxcount_global), README.
