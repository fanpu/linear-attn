# decode-map: paused checkpoint (2)

## Done
- `qwen.py`: hand-written Qwen3-0.6B forward pass. bf16 body, fp32 final norm and unembedding. Static KV cache, no padding, row-offset windows.
- `sample.py`: brute-force fixed-batch sampler. Inverse-CDF on the prob-sorted vocabulary with shared u_t, or Gumbel-max. float64 CDF.
- `decode.py`: **prefix-trie engine**. Each distinct prefix is forwarded once, with CPU offload when the trie exceeds Ncap and fixed forward windows of Fb rows. About 5x fewer forward rows than brute force.
- Tests:
  - `test_sampler.py`: 0 mismatches.
  - `test_engine.py` (fp32, against a naive batch-1 full-sort loop): tp/icdf 1/144, tp/gumbel 0/144, tr/icdf 9/144. Every mismatch happens at a sorted rank of 1,650–88,000, where tail tokens are near-tied and their order is set by ~1e-6 logit noise. That is a numerical limit, not a logic bug.
- Placement test: brute force (B=256) vs trie (Fb=128) on the 64², L=32 story map gives **0/4096 pixels differ**.
- `analysis.py`: hashes, cells, stable colouring, first divergence, metrics, box counting.
- `render.py` styles: mosaic, glass, ink, boundary-age, riso, Spectral split, continuous.
- `anim.py`: length-as-time MP4/GIF (untested on real data).
- `tree.py`: rose-window trie sunburst, looks good on the toy.
- `hover.py`: HTML hover map and transect table (works on the toy).
- `zoom.py`: zoom driver. `verify.py`: placement / cells / box counting / zoom.
- Toy caches: `cache/toy_story_tp64.npz` (brute) and `cache/toy_story_tp64_trie.npz`, both T in [0,2], p in [0,1], L=32. The toy shows fan-shaped top-p cells, 1142 distinct outputs, and pixel-scale speckle for T > ~1.2.

## Next (nothing ran at scale: GPU slots were starved for ~1.5 h, then the chains were killed)
1. Run `logs/chainA.sh` and `logs/chainB.sh` via gpu_run. They produce 256², L=64 maps for story/list/fact (T in [0,1.5]), the story T-repetition-penalty map at 192², a 128² Gumbel map, and the placement/fp32 runs.
2. Look at story_tp256 and pick zoom centres, then run `python zoom.py --name zA --T0 .. --p0 .. --w0 1.0 --levels 12`.
3. `python verify.py`. Then render: `render.py` styles at tile 8, `anim.py cache/story_tp256.npz gallery/refine_story 4`, `tree.py`, `hover.py`.
4. README (hover transect table, verification numbers, ideas-not-pursued list: seed atlas, divergence relief, text weave, flip-margin veins), link check, commit.

## Resume
    cd /home/fzeng/ml/research/art/decode-map
    nohup ../_shared/gpu_run.sh bash logs/chainA.sh > logs/chainA.log 2>&1 &
    nohup ../_shared/gpu_run.sh bash logs/chainB.sh > logs/chainB.log 2>&1 &
