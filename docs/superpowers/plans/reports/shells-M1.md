# Shells (§2), M1 report: pipeline, slice reproduction, ResNet-20 volumes

**Status:** DONE. Both ResNet-20 volumes are complete, verified and analysed.
**Commits:** `8931ef1` (pipeline), `939b8e5` (volumes, K=1 path, extension job), plus the final M1 commit (NOTES, extension support, analysis).
**Directory:** `/home/fzeng/ml/research/art/shells/`. The source piece `art/loss-landscape/` is imported, never modified.

## What was done
- `lib.py` imports `loss-landscape/common.py`. It reuses:
  - the models;
  - `fixed_subset(1000)` (seed 0);
  - BN in eval mode, fp32 with TF32 off;
  - `random_direction` (filter-normalised, biasbn zeroed).

  It also adds seed 3 and two evaluators:
  - `SeqEvaluator` is the loss-landscape path, one point per pass.
  - `VmapEvaluator` uses `torch.func.functional_call` + `vmap` over K stacked parameter sets, with fixed shapes and padded chunks.
- `pca3.py` (CPU) computes the top-3 PCA directions of ResNet-20's 17 checkpoints (ep000..ep040 minus final; unit norm, biasbn zeroed).
  - Explained variance is **0.667 / 0.164 / 0.057**, so the top 3 capture **0.888**.
  - d1 and d2 match loss-landscape's `resnet20_pca.pt` with |cos| = 1.0000.
- `check.py` compares vmap with the sequential path, evaluates slice points against g51, and sweeps K.
- `volume.py` builds the volume with one checkpoint per z-slab, plus `--extend/--reuse` and `--slab-shard`.
- `analyze.py` computes slice reproduction, anisotropy and an ellipsoid null.
- `preview.py` draws the matplotlib previews.
- GPU chain: `jobs/m1.sh`, log `art/shells/logs/m1.log`, ran 04:09–04:38.

Commands:
```
cd art/shells; PY=../.venv/bin/python; export OMP_NUM_THREADS=4
$PY pca3.py resnet20
$PY check.py --device cpu --n 12 --K 4
setsid nohup ../_shared/gpu1.sh bash jobs/m1.sh > logs/m1.log 2>&1 < /dev/null &      # GPU check, random 27^3, PCA 27^3
for k in 0 1 2; do setsid nohup $PY volume.py --device cpu --dirs pca --K 1 --extend 0 9 0 0 0 5 \
   --reuse cache/vol/resnet20_final_pca_g27.npz --slab-shard $k 3 > logs/pca_ext_cpu$k.log 2>&1 < /dev/null & done
$PY analyze.py cache/vol/<vol>.npz; $PY preview.py cache/vol/<vol>.npz
```

## vmap equivalence and speedup
| path (GB10, fp32, 1000 images) | pts/s | max rel vs sequential (50 pts) | peak alloc |
|---|---|---|---|
| sequential (loss-landscape path) | **31.8** | — | — |
| vmap K=1 | 21.1 | 1.9e-7 | 0.4 GiB |
| vmap K=4 | 15.0 | 1.9e-7 | 3.1 GiB |
| vmap K=8 | 12.6 | 1.9e-7 | 6.1 GiB |
| vmap K=16 (image batch 500) | 12.4 | 1.9e-7 | 4.6 GiB |
| vmap K=32 (250) | 13.3 | 1.9e-7 | 4.6 GiB |
| vmap K=64 (125) | 14.2 | 1.9e-7 | 4.6 GiB |

- vmap passes the equivalence check: at K=16 the max relative difference is 1.85e-7 and the median 6.3e-8, against a threshold of 1e-5.
- vmap is **slower** than the sequential path, at 0.39–0.66× its speed. It batches through grouped convolutions, which are slow on this GPU and driver.
- On CPU the speedup is also absent (0.88× at K=4, 12 points, max rel 1.5e-7).
- **vmap is dropped for production.**
- Wrinkle: the chain's automatic pick chose vmap K=1 for the random volume, which ran at 20.5 pts/s. Before the PCA volume started, `volume.py` was changed so that K=1 means the sequential path, which ran at 30.5 pts/s. Both paths agree to 1.9e-7.

## Slice reproduction
The c = 0 slab of the random-direction volume was compared with `loss-landscape/cache/surf/resnet20_final_g51.npz`. Both use n = 1000, seeds 1 and 2, the final checkpoint, fp32, and a GPU. The comparison covers all **625** coincident points: the interior 25×25 of the slab, which are the odd g51 indices. Reference loss runs from 0.124 to 71.6.
- **max relative difference 2.7e-7, median 4.3e-8** (p99 1.9e-7).
- The b = c = 0 row against `resnet20_final_line` (25 coincident points) gives max 2.0e-7 and median 4.4e-8.
- The earlier checks agree: GPU slice points gave 1.8e-7 (vmap) and 8.7e-8 (sequential) on 50 points; the CPU toy 7³ gave 3.1e-7 on 49 points.
- The minimum sits at the centre vertex, with loss 0.124409, identical to the plate.

## Throughput
- GPU sequential: 30.5 pts/s over a full volume (31.8 in the check). That is 8× the 3.8 pts/s loss-landscape measured under 8-way contention.
- GPU vmap K=1: 20.5 pts/s.
- CPU, 4 threads: 3.3 pts/s idle and 1.5–1.9 pts/s with 3 concurrent processes on a loaded machine.
- Wall time:
  - random 27³: 16.0 min;
  - PCA 27³: 10.7 min;
  - PCA extension: 11,421 points in about 42 min wall on CPU.

## Volume status
| volume | grid (a × b × c) | status | file |
|---|---|---|---|
| ResNet-20 final, random dirs (seeds 1, 2, 3) | 27³, h = 0.08 on [−1.04, 1.04] | done (GPU) | `art/shells/cache/vol/resnet20_final_random_g27.npz` |
| ResNet-20 final, PCA dirs, spanning trajectory + 8% | 27³; h = 1.276 / 1.304 / 0.713 | done (GPU); shells clipped at the +a face | `art/shells/cache/vol/resnet20_final_pca_g27.npz` |
| same, extended +9 a, +5 c | 36 × 27 × 32, a ∈ [−29.4, 15.3], b ∈ [−19.6, 14.3], c ∈ [−10.7, 11.4] | done (base from GPU, 11,421 new points on CPU) | `art/shells/cache/vol/resnet20_final_pca_g27_ext.npz` |

- Loss range: random 0.124–440; PCA extended 0.124–4702.
- No non-finite values.

## Shell anisotropy
**Method (declared):**
- Take the 26-connected component of {loss ≤ L} that contains the minimum.
- Supersample 4× by trilinear interpolation of log loss; native-grid values are within 0.05 of the ratios shown.
- Compute second moments in physical coordinates and take semi-axes r = √(5λ), as for a solid ellipsoid.
- fill = volume / (4/3·π·r₁r₂r₃).
- Levels are declared: 0.5, 1.0, and ln 10 = 2.303 (chance-level cross-entropy).

| volume | L | semi-axes r₁, r₂, r₃ | r₁/r₃ | r₂/r₃ | r₁/r₂ | fill | closed |
|---|---|---|---|---|---|---|---|
| random | 0.5 | 0.332, 0.263, 0.198 | **1.67** | 1.32 | 1.27 | 0.991 | yes |
| random | 1.0 | 0.441, 0.370, 0.280 | **1.58** | 1.32 | 1.19 | 0.988 | yes |
| random | 2.303 | 0.599, 0.524, 0.407 | **1.47** | 1.29 | 1.14 | 0.972 | yes |
| PCA (extended) | 0.5 | 6.16, 4.81, 2.78 | **2.22** | 1.73 | 1.28 | 0.985 | yes |
| PCA (extended) | 1.0 | 8.56, 6.61, 3.81 | **2.25** | 1.74 | 1.30 | 0.981 | yes |
| PCA (extended) | 2.303 | 12.42, 9.35, 5.55 | **2.24** | 1.69 | 1.33 | 0.952 | yes |

**Units:**
- Random-direction units are filter-normalised: each direction has norm 27.24, equal to ‖w*‖ over conv and linear weights. The directions are nearly orthogonal (pairwise cos ≤ 0.0064).
- PCA units are Euclidean weight-space units.
- At L = 2.303 the random shell is about 16 × 14 × 11 in weight norm, and the PCA shell is 12.4 × 9.4 × 5.6.

**Null through the identical function:** a rotated analytic ellipsoid with ratio 3.00 recovers as 2.97 on the random grid and 2.94 on the PCA box, with fill 0.999 / 0.997 and axis |cos| > 0.9999.

**Reading the numbers:**
- As the spec's honest expectation predicts, the random-direction shells are smooth, closed and near-ellipsoidal (fill ≥ 0.97).
- The anisotropy is modest, and it *decreases* outward: r₁/r₃ goes 1.67 → 1.47.
- The PCA shells are flatter (r₁/r₃ ≈ 2.2) at every level, and their long axes mix a and c.
- The trajectory reaches the basin from the −a, −c side.

## Previews (all opened)
All previews are in `/home/fzeng/ml/research/art/shells/cache/preview/`:
- `resnet20_final_random_g27_{slices,atlas,levelsets}.png`. `slices` has three centre slabs, the g51 plate, and a relative-difference map (max 2.7e-7, with no spatial pattern).
- `resnet20_final_pca_g27_ext_{slices,atlas,levelsets}.png`, with the checkpoint trajectory projected in green.
- `resnet20_final_pca_g27_{slices,atlas,levelsets}.png`: the base volume, which shows the +a clipping.
- `resnet20_final_random_g7_cputoy_*`: the CPU toy.

## Decisions (also in NOTES.md)
1. **Grid is 27 points per axis on [−1.04, 1.04], not 26 on [−1, 1].** The 26 points on [−1, 1] are the even g51 indices and exclude 0, so neither w* nor the c = 0 plane would be on the grid. With 27 points, w* is a vertex, the three coordinate planes are slabs, and 625 slab points coincide with g51. Cost: +12%.
2. **Production uses the sequential path.** vmap is equivalent but slower.
3. **PCA recipe:** all 17 checkpoints, unit norm, and signs chosen so ep000 is negative, matching loss-landscape.
4. **PCA grid spacing is per axis,** with 0 on a vertex; the physical axes are stored.
5. **The PCA grid was extended past the trajectory box** because every declared shell was clipped at +a. The extension ran on CPU because a roughly 3 h solid-edge chain held the GPU queue. CPU and GPU both reproduce the GPU-measured g51 plate to ≤ 3e-7.
6. **Anisotropy levels and method** are fixed before looking at the data, as above. The ellipsoid null is scaled to each box.
7. **Previews** use log10 loss on magma with nearest-neighbour pixels. They are not gallery pieces.

## Risks and open items
- **The queued extension job is finished.** `jobs/m1_pca_ext.sh` (`art/shells/logs/m1_pca_ext.log`) got the GPU slot at 06:48, after the CPU shards were done, and exited 0 in 4 s. It only assembled the already finished slabs, and the CPU assembly overwrote its output at 06:49, so the final meta records all 11,421 new points as CPU-evaluated. `volume.py` now exits at once when the output already exists. No jobs of mine are left running or queued.
- **Two evaluator paths.** The random volume used vmap K=1 and the PCA volume the sequential path; they agree to 1.9e-7. The random volume's meta lacks an `evaluator` field; this report is the record.
- **Mixed hardware in the extension.** Its new points are CPU-evaluated, while the base 27³ points are GPU-evaluated. The expected mismatch is ~1e-7 relative.
- **PCA-plane losses are not checkpoint losses.** The volume evaluates w* + coordinates·d with final BN statistics, and ep000 lies 26% outside the 3-plane. The trajectory passes over a plateau at loss ~10 rather than through low loss, so M3 captions must say this.
- **Coarse resolution.** The random shells span only 5–15 voxels across, which is enough for moment ratios (the null recovers 2.97/3.00). Shape detail beyond ellipsoids is not resolvable at h = 0.08.
- **M2 timing.** Under the new ≤ 20-min rule, ResNet-56 26³/27³ at about 13 pts/s (≈ 25 min) must be split into slab segments. `--slab-shard` plus a CPU-side loop is ready for that.
- **Report not committed.** This report is outside `art/shells` and was not committed; `commit.sh shells` only commits the piece directory.
