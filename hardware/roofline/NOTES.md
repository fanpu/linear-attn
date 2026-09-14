# Roofline NOTES (handoff)

- `measure.py --out cache/roofline.npz` writes `cache/roofline.json` (stars + meta). ~1 min on an idle GPU. Needs decode-map's
  model and fingerprint's `common.py` (smi/stack helpers) on sys.path (hard-coded).
- `render_roofline.py [--tag roofline|test_roofline]` -> `gallery/<tag>_{night,paper}.png`, `cache/<tag>_stars.md`.
  Compute roof = best measured kernel of any kind (was: best square GEMM; the skinny linear at B=512 beat it on the contaminated run).
  GEMM star labels below the marker to avoid the roof label.
- Runs: 05:13 `test_roofline.json` on a GPU shared with two fingerprint jobs (90 % util): 60.1 TFLOP/s, 91 GB/s, tiny kernels 1-3 ms.
  KEPT as the contention exhibit. 10:06 `roofline.json` idle: 101.1 TFLOP/s (GEMM 16384²), 246.8 GB/s (copy 4 GiB), ridge 410.
- Findings: decode linear B<=16 above the memory roof (6.3 MB weight L2-resident, L2 = 24 MB); Qwen3 decode 41-80 % of the
  memory roof B=1..512 (12.4 -> 160 ms/step); prefill flat at 34 % of the compute roof (unfused eager model); flash SDPA 82 TFLOP/s
  at T=8192; GEMM 1024² only 51 %.
- Status: COMPLETE (README, gallery, index row in art/README.md). Possible extras: fp8/fp4 roofs, a sustained (30 s) roof with the
  thermal drop, stars for the day1/day2 linear-attention kernels.
