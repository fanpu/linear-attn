# Lattice: the matmul performance map of one GB10 and one cuBLAS

*Time `torch.mm` at every shape and colour the grid. What you get is not a smooth surface but a woven textile: stripes by the parity of one dimension, a lattice every 8, tiles every 128 and 256, and mosaic cells where the library's heuristic switches kernels.*

<img src="gallery/lattice_spectral_split_bf16_s2048_k4096.png" width="100%">

<sub>Hero: bf16, stride-13 grid m, n ∈ [13, 2041], k = 4096. Left panel is odd n, right panel is even n (each column doubled). Colour is the residual of ln t against a robust overhead+linear size fit, Sohl-Dickstein Spectral split at 0 and rank-normalised per side (purple = faster than the size trend, red = slower). The residual is measured; the detrending and the colour map are declared choices.</sub>

## The phenomenon

`C = torch.mm(A, B)` with A ∈ R^(m×k) and B ∈ R^(k×n) (k = 4096 fixed) costs 2·m·k·n FLOPs. If hardware were a smooth machine, the time would follow

  t(m, n) ≈ a + b·m + c·n + d·m·n

(fixed dispatch overhead plus linear compute), and GFLOPS = 2mkn/t would be a smooth ramp. In practice cuBLAS/cuBLASLt picks among dozens of kernels (cutlass tensor-op GEMMs with different tile sizes and alignment, nvjet, small-matrix gemmSN) using heuristics over shape, dtype and alignment. Each kernel tiles the matrix into blocks (128×128, 256×128, ...), so time within a kernel's region saw-tooths with the tile period, and the region boundaries are discontinuities. The piece is the map of those decisions: it shows **software (one library version's heuristics) running on hardware**, not the silicon alone.

## Stack

| | |
|---|---|
| GPU | NVIDIA GB10 (sm_121a, capability 12.1), ~120 GB unified memory |
| Driver | 580.173.02 |
| CUDA (torch build) | 13.0 |
| torch | 2.14.0+cu130, BLAS backend cuBLAS |
| cuBLAS | **13.1.1**, cublasLt **130101** |
| CUBLAS_WORKSPACE_CONFIG | unset (default workspace) |
| Date | 2026-09-13; machine idle, no other compute processes during any timing run |

## Hero plates

<table>
<tr><td width="50%"><img src="gallery/lattice_spectral_fp32_s2048_k4096.png"></td><td width="50%"><img src="gallery/cliffs_fire_fp32_s2048_k4096.png"></td></tr>
<tr><td><sub>fp32, [13, 2041]²: residual, Spectral split at 0. A plaid of 128- and 256-periodic tiles, each with its own saw-tooth gradient.</sub></td>
<td><sub>fp32 cliff map: kernel boundaries (profiler) lit by the residual jump across them (cet_fire, log scale 2 % to 65 %), over faint log GFLOPS (oslo).</sub></td></tr>
</table>

<img src="gallery/large_triptych_spectral.png" width="100%">

<sub>Large-scale dtype triptych (bf16 / fp16 / fp32), same residual and Spectral mapping. The half-precision maps are woven by n parity; fp32 has no parity stripes but shows large polygonal kernel cells.</sub>

## Findings

All ratios are medians over rows m of t / (row median), comparing columns in one class of n against the rest.

**[1, 256]², every integer (65 536 shapes per run)**

| run | kernels | fingerprint classes | odd-n / even-n time | n%8==0 vs rest | noise: IQR/median (median, p99) | 200×200 reference std |
|---|---|---|---|---|---|---|
| bf16 | 30 | 52 | **×1.29** | **×0.70** | 0.15 %, 24 % | 3.2 % |
| fp16 | 36 | 55 | ×1.24 | ×0.72 | 0.17 %, 26 % | 3.2 % |
| fp32 | 18 | 14 | ×1.00 | ×1.00 (n%64: ×0.95) | 0.12 %, 28 % | 3.0 % |
| fp32 + TF32 | 16 | 13 | ×1.02 | ×0.85 | 0.14 %, 31 % | 3.4 % |

**[13, 2041]², stride 13 (24 649 shapes per run)**

| run | wall | kernels | fp classes | odd-n / even-n | n%8==0 vs rest | median jump across kernel boundary / inside a cell | jumps > 20 % lying on a boundary | GFLOPS max |
|---|---|---|---|---|---|---|---|---|
| bf16 | 3.3 min | 70 | 45 | **×1.85** | ×0.71 | 15.1 % / 2.1 % | 79 % | 105 k |
| fp16 | 3.3 min | 77 | 52 | ×1.83 | ×0.72 | 14.9 % / 2.1 % | 79 % | 105 k |
| fp32 | 4.1 min | 11 | 22 | ×1.01 | ×1.00 | 12.4 % / 2.7 % | 58 % | 20 k |

1. **Odd-n align1 slowdown (bf16/fp16).** Odd n runs `cutlass_75 tensorop s1688gemm ... align1`; even n runs `cutlass_80 tensorop s16816gemm ... align2` (align8 / wmma variants on multiples of 8). Odd n is ×1.29 slower on [1, 256]² and **×1.85** slower on the large grid: at m, n ≈ 1000 an odd-n matmul delivers ~40 k GFLOPS and a neighbouring even n ~80 to 100 k (`profiles_plate_bf16_s2048_k4096.png`). Only n (columns of C) carries the alignment; m%2, m%8, m%16 are all ×1.00.
2. **Multiple-of-8 lattice at ×0.70.** Columns with n%8 == 0 are ×0.70 the time of the rest (bf16 [1,256]²; ×0.64 of the row median in the [1,128]² retest, n%8 = 1..7 all ≈ ×1.00 of it). On the stride-13 grid multiples of 8 fall every 8th column (n = 104j), so the lattice reappears as a 104-periodic comb.
3. **Tile saw-tooths beyond 256.** Inside a kernel region the residual ramps and resets with the kernel's tile size: FFT of the bf16/fp16 residual (odd n, m, n ≥ 260) peaks at period ≈ 128 along both m and n (bins at 122 and 132, harmonic at 63). fp32 peaks at ≈ 128 along m and ≈ 256 along n, matching the `sgemm_256x128` / `128x64` / `128x128` tilings.
4. **fp32 polygonal kernel cells.** fp32 has no alignment stripes. Instead, large polygonal cells of `cutlass_80 sgemm` tilings with smooth internal gradients (g256), and at large scale a plaid: `sgemm_256x128` is chosen almost only when n mod 256 ∈ [128, 256) (3 673 of 3 704 shapes with m, n ≥ 520) and runs ×0.93 of the trend, against ×1.02 to ×1.04 for its neighbours. A diagonal boundary and hyperbolic (m·n ≈ const) boundaries cut across the plaid.
5. **Cliffs are kernel switches, not memory.** Across kernel boundaries the residual jumps a median of 15 % (p90 38 %) against 2 % inside cells, and 79 % of all neighbour jumps above 20 % sit on a profiler-detected boundary (bf16). The largest are ×2: at m = 26, n%8 == 0, n ≥ 1560 the heuristic picks `nvjet tst_mma_192x8x64` over the neighbouring `s16816 align2` kernel and is **×2.15 slower** (n = 1950 → 1976, bf16; ×2.06 in fp16). There is **no memory cliff** out to 2041: the largest C is 2041×2041 (a few MB) on 120 GB of unified memory, and GFLOPS keeps rising or saturates with m. The doc's "cliffs where a matrix stops fitting in fast memory" did not appear in this range.
6. **Square islands.** In the odd-n bf16 panel, square islands of `s1688 128x256` appear inside the `128x128` sea, where m and n both lie in windows near [897, 1014] and [1417, 1521] (period ≈ 520, edges ±13 from the stride). They run at the same speed as the sea (residual ×1.76 vs ×1.81), so this is a visible heuristic decision with no performance consequence.
7. **TF32 ×1.50.** Enabling `allow_tf32` for fp32 changes the kernel at 99.2 % of shapes; median speed-up ×1.50 (range ×0.54 to ×5.2) and brings back an n%8 lattice (×0.85).
8. **Deterministic dispatch.** A bf16 [1,128]² retest (seed 1, independent random order, 9 windows of 3 ms) reproduced the bitwise output fingerprint at **100 %** of shapes, so kernel choice is deterministic and the map is not noise. Timing test-retest: median |ln t₂/t₁| = 0.28 %; 9.8 % of shapes differ by > 5 %, as isolated speckle with no spatial structure. Profiler kernel label and fingerprint class agree well but not one-to-one (bf16 g256: kernel→fp purity 0.79, fp→kernel 0.87; s2048: 0.56 / 0.69; one kernel label can round differently depending on shape, plausibly via tile remainders, which was not verified).

## Gallery

### Large scale, [13, 2041]² stride 13
<table>
<tr><td width="50%"><img src="gallery/dispatch_mosaic_split_bf16_s2048_k4096.png"></td><td width="50%"><img src="gallery/dispatch_mosaic_fp32_s2048_k4096.png"></td></tr>
<tr><td><sub>bf16 dispatch mosaic, odd-n | even-n panels: which kernel ran (torch.profiler). Declared categorical scheme: hue by kernel family (orange cutlass_75 align1, blue cutlass_80 align2/8, green wmma, violet nvjet), lightness interleaved within family.</sub></td>
<td><sub>fp32 dispatch mosaic (RdPu = cutlass sgemm tilings). The 256-periodic plaid and the diagonal cut are the heuristic's decision boundaries.</sub></td></tr>
<tr><td><img src="gallery/cliffs_fire_bf16_s2048_k4096.png"></td><td><img src="gallery/lattice_spectral_split_fp16_s2048_k4096.png"></td></tr>
<tr><td><sub>bf16 cliff map: boundaries lit by the jump across them. Vertical lines = the multiple-of-8 comb.</sub></td><td><sub>fp16 residual, odd-n | even-n (Spectral split). Nearly identical to bf16: same kernels with different names (tst vs hsh nvjet).</sub></td></tr>
<tr><td><img src="gallery/throughput_dark_bf16_s2048_k4096.png"></td><td><img src="gallery/fingerprint_mosaic_fp32_s2048_k4096.png"></td></tr>
<tr><td><sub>bf16 GFLOPS, rank-normalised, lajolla on dark. Measured throughput, no detrending.</sub></td><td><sub>fp32 bitwise fingerprint classes, coloured by each class's majority kernel family.</sub></td></tr>
<tr><td><img src="gallery/dispatch_contours_ink_fp32_s2048_k4096.png"></td><td><img src="gallery/lattice_aurora_ember_fp32_s2048_k4096.png"></td></tr>
<tr><td><sub>fp32 kernel-region contours, single ink on paper.</sub></td><td><sub>fp32 residual, palettes.py aurora_ember split pairing (declared).</sub></td></tr>
</table>

<img src="gallery/profiles_plate_bf16_s2048_k4096.png" width="100%">
<sub>Survey sheet: GFLOPS along m at four fixed n (ticks = kernel switch), and the per-row alignment penalty by n class.</sub>

<img src="gallery/plate_fp32_s2048_k4096.png" width="100%">
<sub>Scientific plate (fp32, large): time (batlow log), residual (RdBu), kernel mosaic with legend, fingerprint mosaic, drift inset, nvidia-smi before/after.</sub>

Also in `gallery/`: `plate_{bf16,fp16}_s2048_k4096.png`, `lattice_{spectral,fire,aurora_ember,cyanotype_vandyke}_*_s2048_k4096.png`, `dispatch_contours_{riso,ink}_*_s2048_k4096.png`, `cliffs_fire_fp16_s2048_k4096.png`, `profiles_plate_{fp16,fp32}_s2048_k4096.png`.

### [1, 256]², every integer
<table>
<tr><td width="50%"><img src="gallery/lattice_spectral_fp32_g256_k4096.png"></td><td width="50%"><img src="gallery/lattice_spectral_bf16_g256_k4096.png"></td></tr>
<tr><td><sub>fp32 residual, Spectral split: polygonal kernel cells with smooth internal gradients.</sub></td><td><sub>bf16 residual, Spectral split: parity stripes, bands at m = 16/32/64/128, hyperbolic boundaries at small sizes.</sub></td></tr>
<tr><td><img src="gallery/dispatch_mosaic_fp32_g256_k4096.png"></td><td><img src="gallery/dispatch_mosaic_split_bf16_g256_k4096.png"></td></tr>
<tr><td><sub>fp32 dispatch mosaic.</sub></td><td><sub>bf16 dispatch mosaic, odd-n | even-n panels (see palette note below).</sub></td></tr>
<tr><td><img src="gallery/dispatch_contours_riso_bf16_g256_k4096.png"></td><td><img src="gallery/lattice_fire_fp16_g256_k4096.png"></td></tr>
<tr><td><sub>Two-spot riso contours: pink = odd-n sub-lattice region boundaries, blue = even-n, declared misregistration.</sub></td><td><sub>fp16 rank residual, cet_fire on dark.</sub></td></tr>
<tr><td><img src="gallery/lattice_cyanotype_vandyke_fp32_g256_k4096.png"></td><td><img src="gallery/riso_closeup_bf16_g256_k4096_1-128.png"></td></tr>
<tr><td><sub>fp32 residual, cyanotype/vandyke split pairing.</sub></td><td><sub>[1,128]² riso close-up: blue = slower than trend (dithered), pink = kernel boundaries.</sub></td></tr>
</table>

<img src="gallery/dtype_triptych_spectral.png" width="100%">
<sub>[1,256]² dtype triptych, Spectral residual.</sub>

<table>
<tr><td width="50%"><img src="gallery/tf32_diptych_plate.png"></td><td width="50%"><img src="gallery/tf32_ratio_spectral.png"></td></tr>
<tr><td><sub>fp32 vs fp32+TF32 kernel mosaics.</sub></td><td><sub>ln(t_fp32 / t_tf32), Spectral split at 1× (declared).</sub></td></tr>
</table>

<img src="gallery/alignment_closeup_plate.png" width="100%">
<sub>Alignment close-up [1,128]² (bf16 retest): residual with hairlines every 8, bars of time by n mod 8, test-retest map.</sub>

Per-run variants for all four [1,256]² runs (`plate_*`, `throughput_dark_*`, `fingerprint_mosaic_*`, `dispatch_contours_*`, `lattice_*`) and `alignment_closeup_spectral.png` are also in `gallery/`.

**Palette critique.** The interleaved bf16/fp16 dispatch mosaic uses complementary hues (orange align1 vs blue align2) on alternating columns; at any display size below 1:1 they average to a grey-tan mud. The fix is a parity split (`dispatch_mosaic_split_*`, `lattice_spectral_split_*`): odd and even n are two interleaved dispatch lattices, shown side by side. Each panel then reads as clean cells. The interleaved versions remain in the gallery for full-resolution viewing. The fp32 RdPu mosaic has no parity weave and needs no split.

## What was computed

- **Operation:** `torch.mm(A, B, out=C)`, A m×4096, B 4096×n, preallocated outputs.
- **Inputs = cancellation probes** (`sweep.py:masters`): all rows of A identical and all columns of B identical, with 64 products of ±2^E that cancel exactly plus 4 032 small terms. Every entry of C is therefore the same sum, and its bits depend only on the kernel's accumulation order and precision. That makes the fingerprint (bits of C[0,0], C[m/2,n/2], C[-1,-1] for two probes) shape-independent.
- **Timing per shape:** 3 warmup calls, 1 estimate, then N calls per window so each window is ≥ 1.5 ms, median of 5 windows with `torch.cuda.synchronize()`. Shapes in random order (seed 0). A 200×200 reference and the 1×1 overhead probe were re-timed every 256 shapes.
- **Kernel names:** a separate `--mode kernels` pass under `torch.profiler`. Kernels are mapped to shapes via correlation id → launch timestamp → `record_function` range. That this association is valid rests on deterministic dispatch (finding 8).
- **Grids:** g256 = every integer in [1, 256]²; g128 retest; s2048 = {13, 26, ..., 2041}² (157² shapes).
- **Hygiene:** `run_all.sh` runs jobs sequentially and waits for GPU ≤ 50 °C and util ≤ 5 % before each. nvidia-smi, compute apps and loadavg are recorded before/after in every npz and in `logs/campaign.log`, with full `nvidia-smi` dumps in `logs/smi_{before,after}_s2048.txt`. No other compute process was present in any run. Rendering ran pinned to efficiency cores (`taskset -c 0-4`) under `nice`.
- **GPU time:** about 45 min in total (4 × g256 ≈ 26 min, retest ≈ 5 min, 3 × s2048 ≈ 11 min, profiler passes < 2 min).

```bash
cd /home/fzeng/ml/research/hardware/lattice
./run_all.sh "--dtype bf16 --grid g256" "--dtype bf16 --grid g256 --mode kernels"      # likewise fp16, fp32, "fp32 --tf32"
./run_all.sh "--dtype bf16 --grid g128 --seed 1 --reps 9 --window 3e-3 --tag _retest"
./run_all.sh "--dtype bf16 --grid s2048" "--dtype bf16 --grid s2048 --mode kernels" \
             "--dtype fp16 --grid s2048" "--dtype fp16 --grid s2048 --mode kernels" \
             "--dtype fp32 --grid s2048" "--dtype fp32 --grid s2048 --mode kernels"
P=/home/fzeng/ml/research/art/.venv/bin/python
OMP_NUM_THREADS=4 $P render.py            # per-run plates (all tags present in cache/)
OMP_NUM_THREADS=4 $P render_compare.py    # triptych, TF32, retest close-up -> cache/stats.json
OMP_NUM_THREADS=4 $P render_large.py      # s2048 cliffs, profiles, splits, triptych -> cache/stats_large.json
$P quicklook.py bf16_s2048_k4096          # kernel vocabulary, purity, noise
```

## Verification, noise and drift

- **Measurement noise:** the median within-shape IQR/median is 0.12 to 0.17 % (g256) and 0.23 to 0.27 % (s2048). The p99 is 24 to 31 %: a few percent of shapes are noisy, mostly at small sizes where a call takes 4 to 50 µs and CPU overhead dominates.
- **Drift:** the 200×200 reference has std 3.0 to 3.4 % (g256) and 2.1 to 3.0 % (s2048). Its range over a run is 15 to 20 % at s2048, because GPU temperature rises from ≤ 51 °C to 78 to 82 °C (86 to 92 W) during a 3 to 4 min large sweep (g256: 41 → ~69 °C). The median of the first half of reference times over the second half is ×0.99 to ×1.00, and Spearman ρ between residual and measurement order is 0.03 (bf16), 0.03 (fp16) and 0.05 (fp32). Randomized order keeps drift from painting gradients: it shows up as speckle, not structure. The 1×1 overhead probe (6.1 to 9.7 µs) ramps ~20 % during the first ~30 s of a run, then stays flat.
- **Test-retest:** 100 % fingerprint identity, median timing difference 0.28 % (finding 8).
- **Structure vs noise:** kernel boundaries carry 7× the median residual jump of cell interiors, and the 128/256 periods match the kernel names' tile sizes.
- **No fractal claim.** The structure is periodic and nested (8 → 128 → 256 tiles, square islands), not self-similar, and no box-counting dimension is claimed.

## Caveats

- **This maps software on hardware.** It is a portrait of GB10 + cuBLAS 13.1.1 / cublasLt 130101 + driver 580.173.02 + torch 2.14, with the default workspace. A different library version, workspace size (`CUBLAS_WORKSPACE_CONFIG`) or `torch.backends.cuda.preferred_blas_library` would redraw the mosaic. It is not "the GB10's" intrinsic map.
- Only n shows alignment. This is presumably a consequence of how torch maps row-major `mm` onto cuBLAS's column-major GEMM (which operand dimension becomes the kernel's aligned one); not verified with a transposed call.
- The residual plates depend on a declared detrending (robust fit of t to a + bm + cn + dmn in relative error), so the seam at 0 means "as fast as the smooth size trend predicts". Spectral rank-normalisation per side is aesthetic. Use the GFLOPS plates (`throughput_dark_*`, plate panel a) for raw throughput.
- Small shapes (< ~32) are dominated by the ~6 to 10 µs Python/dispatch overhead. Their colours reflect overhead as much as kernels.
- The stride-13 grid aliases: multiples of 8 appear every 104 in n, and boundaries are located only to ±13. The large sweep has no retest of its own; determinism is inferred from the g128 retest plus bitwise fingerprints.
- The kernel pass is a separate run from the timing pass (profiler overhead would distort timing). The label ↔ time association assumes deterministic dispatch, which the fingerprints support.
- One device, one day, one k (4096). Other k values give different maps.

## References

- NVIDIA cuBLAS / cuBLASLt documentation: heuristics (`cublasLtMatmulAlgoGetHeuristic`), workspace, alignment requirements for tensor-op kernels.
- NVIDIA, *Matrix Multiplication Background User's Guide* (Deep Learning Performance docs): dimensions as multiples of 8 for tensor cores.
- Sohl-Dickstein, *The boundary of neural network trainability is fractal* (2024), github.com/Sohl-Dickstein/fractal: Spectral split colouring convention.
- `hardware/hw-art-directions.md` §6 (proposal); `/home/fzeng/ml/research/art/color-research/palettes.py` (split pairings, overprint).
