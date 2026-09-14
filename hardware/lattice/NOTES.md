# Lattice NOTES (living handoff)

## Setup
- Stack: GB10 sm_121a (capability 12.1), driver 580.173.02, CUDA 13.0, torch 2.14.0+cu130, **cuBLAS 13.1.1 (cublasLt 130101)**,
  BLAS backend cuBLAS, CUBLAS_WORKSPACE_CONFIG unset, date 2026-09-13.
- C = torch.mm(A, B, out=C), A m x k, B k x n, **k = 4096 fixed**; sweep (m, n). All rows of A identical, all columns of B identical
  (cancellation probes, `sweep.py:masters`: 64 products of +-2^E that cancel exactly + 4032 small ones), so output bits are a
  shape-independent fingerprint of accumulation order/precision. Two probes per dtype (E from PROBE_E).
- Timing per shape: 3 warmup, 1 estimate, N calls per window so window >= 1.5 ms, median of 5 windows. Randomized order (seed 0).
  Every 256 shapes: reference 200x200 and 1x1 re-timed. nvidia-smi + compute-apps + loadavg logged before/after (in npz info + logs/campaign.log).
- Kernel names: `--mode kernels`, torch.profiler chrome trace, kernels mapped to shapes via correlation id -> cuda_runtime/cuda_driver launch ts -> record_function range.
- Files: `sweep.py` (GPU), `run_all.sh` (sequential jobs; waits for GPU <= 50 C and util <= 5 % before each), `common.py`, `quicklook.py TAG`,
  `render.py` (per-run plates), `render_compare.py` (triptych, tf32, retest, alignment close-up, writes `cache/stats.json`).
  cache/ is gitignored (npz ~ few MB each).
- Rendering during sweeps: `OMP_NUM_THREADS=1 taskset -c 0-4 nice` (A725 cores 0-4,10-14; X925 = 5-9,15-19).

## Done (all clean: no other compute apps; start temp <= 50 C, util 0-3 %)
| run | wall | kernels | fp classes | IQR/med median, p99 | ref std | odd/even-n time | n%8==0 time vs rest |
|---|---|---|---|---|---|---|---|
| bf16 g256 | 6.3 min | 30 | 52 | 0.15 %, 24 % | 3.2 % | 1.29 | 0.70 |
| fp16 g256 | 6.2 | 36 | 55 | 0.17 %, 26 % | 3.2 % | 1.24 | 0.72 |
| fp32 g256 | 7.1 | 18 | 14 | 0.12 %, 28 % | 3.0 % | 1.00 | 1.00 (n%64: 0.95) |
| fp32 tf32 g256 | 6.6 | 16 | 13 | 0.14 %, 31 % | 3.4 % | 1.02 | 0.85 |
| bf16 g128 retest (seed 1, 9x3 ms) | ~5 | - | 100 % identical fp to first run | 0.11 % | 0.29 % | - | n%8==0: 0.64x row median |
- Test-retest (bf16 [1,128]^2, independent order): median |ln t2/t1| = 0.28 %, 9.8 % of shapes differ > 5 % (isolated speckle, no structure);
  fingerprints bitwise identical at 100 % of shapes -> dispatch is deterministic, timing speckle is noise.
- TF32: same kernel at only 0.8 % of shapes; median speed-up x1.50 (range 0.54-5.2).
- 1x1 overhead probe 6.1-9.7 us; ramps up ~20 % in the first ~30 s of each run then flat (CPU-side warm state), ref 200x200 flat with spikes to 1.2.
- Temperature rises 41 -> ~69 C during a 6-min sweep (GPU at ~78 % util, 48 W).
- Structure: bf16/fp16 = vertical stripes by n parity (cutlass_75 s1688 align1 for odd n, cutlass_80 s16816 align2 for even n, align8/wmma
  on multiples of 8), horizontal bands at m = 16/32/64/128/192/224, hyperbolic (m*n = const) boundaries at small sizes, nvjet islands.
  fp32 = no alignment stripes, large polygonal cells of cutlass simt sgemm tilings (128x32, 64x64, 128x64...) with smooth internal gradients.
  Only n (the number of columns of C in torch) carries the parity/multiple-of-8 alignment; m shows only band boundaries, no alternation.

## Gallery (gallery/)
- Per run TAG (bf16/fp16/fp32/fp32_tf32 _g256_k4096): lattice_spectral_TAG (hero, SD Spectral split of residual), lattice_aurora_ember/
  cyanotype_vandyke (palettes.py pairings), lattice_fire_TAG (dark, cet_fire of rank residual), throughput_dark_TAG (lajolla rank GFLOPS),
  dispatch_mosaic_TAG, fingerprint_mosaic_TAG, dispatch_contours_riso_TAG (pink = odd-n sub-lattice region contours, blue = even-n),
  dispatch_contours_ink_TAG, riso_closeup_TAG_1-128, plate_TAG (scientific plate with stack caption).
- dtype_triptych_spectral.png, tf32_diptych_plate.png, tf32_ratio_spectral.png, alignment_closeup_plate.png, alignment_closeup_spectral.png.
- Viewed: fp32 dispatch mosaic (RdPu cells) is excellent; bf16/fp16 mosaics read grey when downscaled (parity stripes average) - fine at full res.
  alignment_closeup_plate: retest panel is small, could be enlarged.
- Strongest so far: lattice_spectral_fp32_g256_k4096.png, lattice_spectral_bf16_g256_k4096.png, dtype_triptych_spectral.png.

## Stride-13 [13,2041]^2 campaign (done 2026-09-13 23:12-23:30, idle GPU; smi dumps logs/smi_{before,after}_s2048.txt)
| run | wall | kernels | fp | odd/even-n | n%8 | boundary/interior jump | >20% jumps on boundary | ref range/std | rho(resid, order) | temp |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 3.3 min | 70 | 45 | 1.85 | 0.71 | 15.1/2.1 % | 79 % | 15.8/2.6 % | 0.03 | 42->78 C |
| fp16 | 3.3 | 77 | 52 | 1.83 | 0.72 | 14.9/2.1 % | 79 % | 19.8/3.0 % | 0.03 | 51->82 |
| fp32 | 4.1 | 11 | 22 | 1.01 | 1.00 | 12.4/2.7 % | 58 % | 14.6/2.1 % | 0.05 | 51->82 |
- Residual saw-tooth period ~128 (m and n; fp32 n: ~256). fp32 sgemm_256x128 only where n mod 256 >= 128 (plaid). Biggest cliff: m=26, n%8==0,
  n>=1560 -> nvjet 192x8x64, x2.15 slower. Square s1688_128x256 islands (odd n, windows ~897-1014, 1417-1521), same speed. No memory cliff.
- New code: render_large.py (cliffs_fire_*, lattice_spectral_split_*, profiles_plate_*, large_triptych_spectral, cache/stats_large.json);
  render.py adds dispatch_mosaic_split_* (fix for grey mud of interleaved orange/blue parity stripes) + pad fix for odd G;
  render_compare.py closeup plate: retest panel enlarged.

## Status: COMPLETE
README.md written (stack, findings, noise/drift, caveats, gallery), link check passed (all src paths exist), committed. No background processes.
Strongest: gallery/lattice_spectral_split_bf16_s2048_k4096.png, gallery/lattice_spectral_fp32_s2048_k4096.png, gallery/cliffs_fire_fp32_s2048_k4096.png.
Possible extras (not required): retest at s2048; transposed-call check of why only n carries alignment; other k.
