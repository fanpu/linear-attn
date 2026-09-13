# decode-map: paused checkpoint

## Done
- `qwen.py`: hand-written Qwen3-0.6B forward, static KV cache, no padding (shared prompt prefilled once, KV copied to all rows). Now bf16 body + fp32 final norm/unembedding (fp32 logits); `--fp32` gives a full fp32 body. The fp32 version matched HF to 4.6e-5 (`test_model.py`, run before the bf16 switch).
- `sample.py`: inverse-CDF sampling on the prob-sorted vocab with shared per-position uniforms u_t (`--rule gumbel` is the alternative), float64 softmax/CDF, HF top-p and repetition penalty, fixed batch. Fast top-K path plus exact full-sort fallback. `test_sampler.py`: 0 token mismatches against the naive full-sort reference over 5120 rows.
- Toy run: `cache/toy_story_tp64.npz` (64², L=32, T∈[0,2], p∈[0,1]) took 469 s with the GPU contended (~0.9 s/step at B=256; high-T rows hit the fallback sort often).

## Next
1. Render the toy (hash mosaic) and check it. Speed up: larger K (4096) to cut fallbacks, B=512, maybe dedupe the forward pass.
2. Batch-invariance test (same pixels in shuffled batch placements, bf16 vs fp32 mismatch rate).
3. Heroes at 256²/512² with L=64–128 via gpu_run.sh; T-rep grid; list/fact prompts; zoom sequence; L-as-time animation.
4. Render styles (mosaic, stained glass, ink boundaries, Spectral split, dark continuous, HTML hover), §11 box counting plus null model, README, link check, commits.

## Resume commands
    cd /home/fzeng/ml/research/art/decode-map
    PYTHONPATH=. ../.venv/bin/python test_sampler.py
    ../_shared/gpu_run.sh env PYTHONPATH=. ../.venv/bin/python sample.py --prompt story --res 64 --L 32 --out cache/toy_story_tp64.npz
