# Lattice NOTES (living handoff)

## State (2026-09-13 ~22:30)
- Stack: GB10 sm_121a (cap 12.1), driver 580.173.02, CUDA 13.0, torch 2.14.0+cu130, cuBLAS 13.1.1 (cublasLt 130101), BLAS backend cuBLAS, no CUBLAS_WORKSPACE_CONFIG.
- Setup: C = torch.mm(A, B), A m x k, B k x n, **k = 4096 fixed**; sweep (m, n). Rows of A identical, columns of B identical
  (cancellation probes, `sweep.py:masters`) so output bits = shape-independent algorithm fingerprint.
- `sweep.py` (compute, GPU) -> `cache/{time,kernels}_<dtype>[_tf32]_<grid>_k4096.npz`; `run_all.sh` runs jobs sequentially,
  waits for GPU <=50 C & idle before each; logs in `logs/` (`campaign.log` has nvidia-smi after each job).
- `common.py` (load/kernel names), `quicklook.py TAG` (stats + preview), `render.py` (all single-dtype plates).
- Rendering during sweeps is pinned: `OMP_NUM_THREADS=1 taskset -c 0-4` (A725 efficiency cores).

## Findings so far (bf16, [1,256]^2, 6.3 min)
- 30 kernel labels (profiler): cutlass_75 s1688 align1 (odd n), cutlass_80 s16816 align2 (even n), wmma 32x32, nvjet tst_mma (several tilings),
  gemvx for m=1 or n=1, dot for 1x1. Vertical stripes = parity of n; horizontal bands at m = 16, 32, 64, 128, 192, 224; hyperbolic boundaries (m*n const) at small sizes.
- 52 fingerprint classes; purity kernel->fp 0.79, fp->kernel 0.87 (fingerprints split nvjet by tiling, merge some cutlass).
- Timing: t 4.2-46 us/call; median IQR/median 0.15 %, p99 24 %; ref (200x200) std 3.2 %; 1x1 overhead probe ramps 0.8->1.0 in first 30 s then flat.

## Campaign queued (nohup run_all.sh, started 22:21): fp16 g256, fp32 g256, fp32 tf32 g256 (+kernels each),
  bf16 g128 retest (seed 1, reps 9, 3 ms windows, tag _retest), s2048 (stride 13) bf16/fp16/fp32 (+kernels).
  Check `logs/campaign.log`; it ends with "=== ... done".

## Next
- render_compare.py: dtype triptych, tf32 ratio diptych (Spectral split at ratio 1), retest noise map, alignment close-up plate with mod-8 grid.
- README with gallery; commit via `art/_shared/commit.sh hardware/lattice "..."`.
