# Solid Edge — M1 report (toys and the free space-time volume)

**Status: DONE.** Commit `82d1fc7` (`art/solid-edge: M1 engine, toys (sigma and three-lr), null, space-time checks`).
Directory: `/home/fzeng/ml/research/art/solid-edge/`. No gallery or README yet (M3). No packages installed.

## What was done

1. **Engine** `se_engine.py`. It imports `art/trainability-fractal/tfractal.py` read-only and has three model kinds:
   - `net2` is the source network. It reuses `tfractal._make_step`'s hand-written loss and gradient unchanged, and adds an init scale σ on W0 and W1.
   - `net3` has two hidden layers of width 16 with three learning rates: h0 = φ(XW0/√n), h1 = φ(h0W1/√n), ŷ = h1W2/n, with N = #params = 528.
   - `quad3` is the null: the exactly quadratic loss of ŷ = F0a + F1b + F2c. The features F0 = X/√n, F1 = h0/n and F2 = h1/n are frozen at net3's init, and each block gets its own learning rate.

   The convergence measure and the early-exit rule are the source's. The one change is **bucketed compaction**: a batch only shrinks to power-of-two sizes ≥ 1024, padded with frozen rows. That keeps chunk shapes to a fixed small set on the shared memory pool.
2. **Tests (CPU)**: `test_engine.py` (6 tests) and `test_analysis.py` (4 tests) all pass.
   - `net2` matches `tfractal.train_chunk` to rtol 1e-9, including σ, early exit and compaction.
   - The `net3` and `quad3` gradients match autograd.
   - Compaction changes no labels.
   - The η grid equals `tfractal.log_grid` bit for bit.
   - 3D box counting gives exactly 2 on a plane, about 2 on a sphere, and > 2.9 on noise.
3. **Space-time volume**: `spacetime.py` reads the source cache only (no compute).
4. **GPU chain** `run_m1.sh`, queued once through `gpu1.sh`, float32, 64³, 500 steps, one checkpoint per chunk. It ran toy (a), then the null, then a float64 recompute of toy (a)'s σ = 1 plane (4096 nets), then toy (b).
5. **Analysis** `analyze_toys.py`: structure tests, slice agreement and matplotlib previews.

### Commands
```
cd /home/fzeng/ml/research/art/solid-edge
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest -q test_engine.py test_analysis.py
OMP_NUM_THREADS=4 ../.venv/bin/python spacetime.py
setsid nohup /home/fzeng/ml/research/art/_shared/gpu1.sh bash /home/fzeng/ml/research/art/solid-edge/run_m1.sh > logs/m1_toys.log 2>&1 < /dev/null &
OMP_NUM_THREADS=4 ../.venv/bin/python analyze_toys.py
```
The chain waited about 3 min behind ribbon's Stage B, then ran from 03:37:27 to 03:45:35 (8 min wall). Another GPU process was active (62 % utilisation when the chain started).

## Space-time checks against the source README

**Sign convention.** Converged ↔ negative measure. This was verified three ways:
- `train_chunk` returns −Σv when a run converges;
- the signs of `measure_T` at T = 1000 equal the signs of `measure`;
- at T = 1000 the 16 lowest-η1 rows are 100 % negative and the 16 highest are 0 %.

The label volume (100, 384, 384), with 1 = converged, is saved to `cache/spacetime_labels.npz`.

| T | trainable (ours) | README | per-T 2D D, b = 2–32 (ours) | README D |
|---|---|---|---|---|
| 10 | 64.80 % | 64.8 % | 1.046 | 1.05 |
| 30 | 53.81 % | – | 1.270 | 1.27 |
| 100 | 49.29 % | 49.3 % | 1.367 | 1.37 |
| 250 | 49.00 % | – | 1.420 | 1.42 |
| 500 | 48.97 % | – | 1.414 | 1.41 |
| 1000 | 48.94 % | 48.9 % | 1.412 | 1.41 |

- Every row matches the README to its printed precision. We used the source's own `boxcount.dimension_of_measure` with bmax = 48, which gives the same box sizes as the source.
- The volume has 300 990 boundary voxels (6-neighbour label change, space-time).
- No 3D dimension was computed, because T is not a length.

## Hyperparameter toys (64³, float32) and the null

**Axes.**
- η0 and η1 use every 16th pixel centre of the 1024² overview (pixel 16j+8), spanning log10 η ∈ [−2.93, 5.93] at 0.14 decade per voxel.
- Toy (a)'s third axis is log10 σ = (k−26)·6/64 ∈ [−2.44, 3.47], so σ = 1 exactly on plane 26.
- Toy (b) and the null use the same η grid on all three axes.

**Box counting.** 3D box counting runs on 2×2×2 edge cells over b = 1–16. That range is **1.2 decades (< 2): preview only, not a dimension claim.**

| | toy (a) net2 (η0, η1, σ) | toy (b) net3 (η0, η1, η2) | null quad3 (η0, η1, η2) |
|---|---|---|---|
| trainable fraction | 58.22 % | 54.83 % | 15.00 % |
| boundary voxels (6-neighbour) | 11 543 | 10 825 | 6 305 |
| edge cells (2×2×2) | 8 420 | 7 501 | 3 433 |
| **box-count slope, b = 1–16** | **2.122 ± 0.013** | **2.066 ± 0.019** | **1.994 ± 0.019** |
| slope, b = 2–16 | 2.130 | 2.055 | 2.000 |
| local slopes (1→2, 2→4, 4→8, 8→16) | 2.09, 2.10, 2.22, 2.04 | 2.06, 2.17, 2.03, 1.97 | 1.98, 1.93, 2.13, 1.89 |
| median 2D D of axis slices (b = 1–16; along axis 0 / 1 / 2) | 1.23 / 1.34 / 1.20 | 1.76 / 1.16 / 1.25 | 1.01 / 1.01 / 0.98 |
| lines along axis 0 / 1 / 2 whose label changes | 21 % / 99.5 % / 23 % | 100 % / 24 % / 11 % | 26 % / 26 % / 34 % |
| wall time; px/s (excluding the compile chunk) | 175 s; 1494 (1553) | 273 s; 959 (1026) | 13 s; 20 810 |

**Acceptance.** Toy (a) is 0.128 above the null, which clears the 0.1 margin. Toy (b) is 0.072 above, inside it. The rule asks for at least one toy above the margin, so the status is DONE. The margin is thin, though (see Risks).

**What the previews show.**
- The null is a smooth box corner, with 2D slice D ≈ 1.0.
- Toy (a) is mostly a wall at η1 ≈ 10², but σ changes it:
  - the wall steps up and down with σ;
  - at σ < 10⁻¹ there is a tongue of converged runs at η1 = 10²–10^4.7 (low η0);
  - at large σ, diverged pockets appear at tiny η1 and η0 > 10^3.3;
  - speckle appears where a slice lies inside the boundary.
- Toy (b) is essentially one wall normal to η2 at η2 ≈ 10². η0 barely matters. Speckle appears at η1 ≈ 10⁵, and runs diverge at η2 < 10^−1.5 when η1 is large.

## Slice agreement, toy (a) at σ = 1 vs the float64 1024² overview

- **4084/4096 = 99.71 %** of labels agree at overview pixels (16i+8, 16j+8). Trainable fraction is 58.11 % (toy plane) vs 58.25 % (overview at those pixels).
- **Near-boundary disagreement:** 12/214 = **5.6 %** of pixels near the boundary disagree. "Near" means a float64 overview edge cell lies within 16 px (one toy voxel).
- **Far from the boundary:** 0 of 3882 pixels disagree.
- **Float64 recompute of the same 4096 nets:** it agrees with the overview on **100.00 %** of labels, including 0 near-boundary misses. So all 12 misses are float32 flips, not a chart or kernel problem.
- Toy plane vs the float32 overview at the same pixels: 99.93 % agree. The two overviews (float32 vs float64) agree with each other on 99.73 % of these pixels.

## Throughput (GB10, driver 580.173.02, CUDA 13.0, torch 2.14.0+cu130, 2026-09-15)

All rates were measured on a shared GPU; they include early-exit savings and exclude the compile chunk.

| run | dtype | rate | vs source estimate |
|---|---|---|---|
| toy (a) | float32 | 1553 px/s | ~8× the source's ~196 px/s |
| toy (b) | float32 | 1026 px/s | about 1.5× the cost per voxel of toy (a) |
| σ = 1 plane recompute | float64 | 236 px/s | vs 63 px/s; this is one 4096-net chunk including compile, so a lower bound |
| null | float32 | 20 810 px/s | – |

**Implied M2 cost** for toy (a) at 128³ (2.1 M voxels): about 23 min in float32, plus the float64 shell.

## Axis-set decision

**Decision: toy (a), axes (log η0, log η1, log σ), goes to 128³.** Reasons:

1. **Rougher boundary.** It is the only toy whose boundary is rougher than the null by more than 0.1 (2.12 vs 2.07 for (b) and 1.99 for the null).
2. **σ is not an extrusion.** 21 % of σ-lines change label, and 25 % of all label changes cross the σ axis. Toy (b) is essentially one wall normal to η2: its η0-lines change only 11 % of the time.
3. **It contains the σ = 1 plane.** That plane reproduces the float64 overview plate (§0.7), and M3's cutaway needs it.
4. **Cheaper.** About 1.5× fewer seconds per voxel.

This is logged in `NOTES.md`.

## Previews (all opened and inspected)

- `art/solid-edge/cache/preview/spacetime_T_slices.png`: labels at T = 10, 30, 100, 250, 500, 1000.
- `art/solid-edge/cache/preview/spacetime_vertical_and_curves.png`: T × η0 and T × η1 slices, plus trainable fraction and D against T.
- `art/solid-edge/cache/preview/toy_a_slices.png`, `toy_b_slices.png`, `toy_null_slices.png`: six axis slices each (per axis, one reference plane and the plane with the most edge cells).
- `art/solid-edge/cache/preview/toy_a_projections.png`, `toy_b_projections.png`, `toy_null_projections.png`: converged fraction along each axis (a declared mean, not a slice).
- `art/solid-edge/cache/preview/toy_a_sigma1_vs_overview.png`: float64 overview samples, the toy plane, and a disagreement map.
- `art/solid-edge/cache/preview/toys_boxcount.png`: N(b) and local slopes for (a), (b) and the null.

Numbers are in `cache/spacetime_summary.json` and `cache/toys_summary.json`.

## Decisions (all in NOTES.md)

- **Sign convention:** verified as described above.
- **η grid:** overview pixel centres computed in float64 exactly as `log_grid` does, then cast to float32.
- **σ range:** six decades, with σ = 1 on a grid plane.
- **Toy (b) architecture:** net3 with two hidden layers. Three learning rates need three weight matrices.
- **Null:** the source's frozen-feature quadratic extended to three blocks, run through the same trainer, dtype and measure.
- **Chunk sizes:** 32768 for (a) and the null, 16384 for (b), with power-of-two compaction buckets.
- **Box counting:** uses 2×2×2 edge cells, the 3D analogue of his `extract_edges`. The two-sided 6-neighbour set biases a plane's slope to 2.20; it is still reported as the boundary-voxel count.
- **"Near the boundary"** means within one toy voxel (16 px) of a float64 overview edge cell.
- **Extra float64 plane recompute:** added to attribute the slice disagreements. It cost about 17 s.

## Risks

1. **The roughness margin is thin and measured at the wrong scale.**
   - (a) beats the null by 0.13 over only 1.2 decades, with local slopes wandering ±0.1.
   - At the overview window (0.14 decade per voxel) the body is mostly a flat wall. The 2D fractal structure in the source lives at zooms of 10^1.5 and deeper.
   - **Recommendation for M2 (not decided here):** centre the 128³ (η0, η1) window on the seam, e.g. the 10¹ zoom window of `steps_zoomA2` / zoomA kf2. Its σ = 1 plane is still an existing plate, so the slice test survives. At the overview window, the 128³ D table will likely sit only slightly above 2.
2. **Float32 flips.** 5.6 % of near-boundary σ = 1 pixels flip at this sampling. M2's float64 shell is necessary, not optional.
3. **The space-time solid is mostly extrusion.** The body stops changing above T ≈ 250, so the top 75 % of the linear T axis is nearly an extrusion. M3 should declare this, or consider cropping or reparameterising the time axis (the checkpoints are linear in T, so a log-T axis would have few samples at small T).
4. **The null shares one init draw.** It reuses net (b)'s init (seed 0), and its trainable fraction (15 %) differs from the networks'. Its boundary is a smooth corner, as expected for linear GD. It was not run against toy (a)'s σ axis, because σ has no meaning for the frozen-feature quadratic.
5. **Throughput will vary.** The rates were measured while the GPU was lightly shared. They may drop severalfold under heavier contention.
