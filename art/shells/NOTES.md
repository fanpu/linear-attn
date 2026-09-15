# NOTES: shells (§2 of art/ml-art-3d.md) — living handoff

## State
- M1 DONE, gate PASSED (27-point grid accepted). Report docs/superpowers/plans/reports/shells-M1.md.
- M2 IN PROGRESS (started 06:55):
  - `jobs/m2_gpu_loop.sh` (log logs/m2_gpu_loop.log, per-model logs/m2_resnet56*.log): ResNet-56 and ResNet-56-noshort 27^3
    random volumes, 4 slab shards each, one gpu1.sh job per shard, then CPU assemble + analyze + preview.
  - `jobs/m2_film_cpu.sh` (log logs/m2_film_cpu.log, logs/film_ep*_k.log): ResNet-20 17^3 film volumes on CPU.
  - Both loops resume: rerun the same `setsid nohup bash jobs/<loop>.sh > logs/<loop>.log 2>&1 < /dev/null &`.

## PAUSED (2026-09-15 ~08:37, controller/user request)
Loop scripts `jobs/m2_gpu_loop.sh` and `jobs/m2_film_cpu.sh` were killed, and so were the 3 CPU ep016 workers (0 slabs done;
slab writes are atomic, so nothing is partial). The resnet56 GPU shard 1/4 that was already running was left to finish
(expected exit ~08:41, 14/27 slabs). Nothing else is queued.

| volume | slabs done | assembled |
|---|---|---|
| resnet20_final_random_g27 / _pca_g27 / _pca_g27_ext (M1) | 27/27, 27/27, 32/32 | yes |
| resnet20_ep{040,000,001,002,004,008}_random_g17 | 17/17 each | yes (analysed + previews) |
| resnet20_ep016_random_g17 | 0/17 | no |
| resnet56_final_random_g27 | 11/27 at pause (shards 0 done, 1 finishing -> 14/27) | no |
| resnet56_noshort_final_random_g27 | 0/27 | no |

Resume (both loops skip finished slabs, shards and volumes):
```
cd /home/fzeng/ml/research/art/shells
setsid nohup bash jobs/m2_gpu_loop.sh >> logs/m2_gpu_loop.log 2>&1 < /dev/null &   # GPU shards, <= ~10 min each
setsid nohup bash jobs/m2_film_cpu.sh >> logs/m2_film_cpu.log 2>&1 < /dev/null &   # CPU, only ep016 left (~16 min)
```
Then: analyze and preview run automatically. Still to do: write reports/shells-M2.md (anisotropy table for all models and
epochs, slice reproduction vs g51 and g101). Findings so far: ep040 17^3 vs final 27^3 at 2,197 shared points gives max rel
6.5e-7; ep040 vs g51 gives 3.4e-7; ResNet-56 centre loss 0.092708 (= g51). At ep000 the minimum is off-centre and
the relative shells touch the ±1.28 box.

## M3 caption rules
- The PCA-direction volume moves the *final* weights w* along the PCA directions (final BN statistics). The checkpoint
  path drawn through it crosses a loss ~10 plateau of *that* field; this is not the loss of those checkpoints and not
  a level set of them. Captions must say so, and must caption the thread plate and the random onion as different objects. Code: `lib.py` (imports loss-landscape/common.py; SeqEvaluator, VmapEvaluator),
  `pca3.py` (CPU), `check.py` (equivalence + slice points + K sweep), `volume.py` (27^3, per-slab checkpoints),
  `analyze.py` (slice reproduction, anisotropy, ellipsoid null), `preview.py` (matplotlib previews to cache/preview/).
- GPU chain `jobs/m1.sh` (check -> choose K -> random 27^3 -> PCA 27^3), log `logs/m1.log`: DONE 04:38.
  `cache/vol/resnet20_final_random_g27.npz` (16.0 min, 20.5 pts/s), `cache/vol/resnet20_final_pca_g27.npz` (10.7 min, 30.5 pts/s).
- PCA extension (+9 a, +5 c points) -> `cache/vol/resnet20_final_pca_g27_ext.npz` (36 x 27 x 32 on (a, b, c)), DONE on CPU
  (3 slab shards, logs/pca_ext_cpu{0,1,2}.log, 11,421 new points, ~42 min wall, 1.7 pts/s per process) because the GPU
  queue was held by a multi-hour solid-edge chain. `jobs/m1_pca_ext.sh` is still queued in gpu1.sh (logs/m1_pca_ext.log);
  volume.py now exits immediately when the output exists, so it is a no-op when it runs.
- Controller rule (2026-09-15): one gpu1.sh job <= ~20 min; split volumes into slab segments (volume.py --slab-shard)
  with a CPU-side loop queueing the next segment. Applies to M2.

## Resume
```
cd /home/fzeng/ml/research/art/shells; PY=/home/fzeng/ml/research/art/.venv/bin/python; export OMP_NUM_THREADS=4
setsid nohup /home/fzeng/ml/research/art/_shared/gpu1.sh bash jobs/m1.sh > logs/m1.log 2>&1 < /dev/null &   # resumes per slab
$PY analyze.py cache/vol/resnet20_final_random_g27.npz && $PY preview.py cache/vol/resnet20_final_random_g27.npz
setsid nohup /home/fzeng/ml/research/art/_shared/gpu1.sh bash jobs/m1_pca_ext.sh > logs/m1_pca_ext.log 2>&1 < /dev/null &
$PY analyze.py cache/vol/resnet20_final_pca_g27_ext.npz && $PY preview.py cache/vol/resnet20_final_pca_g27_ext.npz
```

## Log
- 2026-09-15 CPU checks (ResNet-20, 1000 images, 4 threads): vmap vs sequential on 12 grid points max rel 1.5e-7
  (K=4; CPU speedup 0.88x, i.e. none on CPU). Sequential CPU 3.3 pts/s. Slice points vs g51: max rel 1.5e-7.
- CPU toy volume 7^3 (h=0.32, `cache/vol/resnet20_final_random_g7_cputoy.npz`, 2.4 pts/s): c=0 slab vs g51 at 49
  coincident points max rel 3.1e-7, median 8.6e-8; line vs resnet20_final_line max 1.5e-7. Centre loss 0.124409.
- PCA3 (ResNet-20, 17 checkpoints): explained 0.667 / 0.164 / 0.057 (top-3 0.888); d1, d2 match
  loss-landscape's resnet20_pca.pt with |cos| = 1.0000. Coordinates a in [-26.3, 1.2], b in [-16.4, 11.7],
  c in [-9.4, 5.9]; the init checkpoint's residual outside the 3-plane is 26% of its distance to w*.

- GPU checks (04:10, `cache/check_cuda.json`): vmap vs sequential on 50 points max rel 1.9e-7 (pass), but vmap is
  SLOWER on the GB10: sequential 31.8 pts/s; vmap K=1 21.1, K=4 15.0, K=8 12.6, K=16 12.4, K=32 13.3, K=64 14.2 pts/s
  (speedup 0.39-0.66x). 50 slice points vs g51: max rel 1.8e-7.
- Random volume: c=0 slab vs g51 at 625 coincident points max rel 2.7e-7, median 4.3e-8; b=c=0 row vs line (25 pts)
  max 2.0e-7. Minimum at the centre vertex, 0.124409. Loss max 440.
- Random shells: r1/r3 = 1.67 / 1.58 / 1.47 at L = 0.5 / 1 / 2.303, fill 0.99-0.97 (near-ellipsoidal), all closed.
- PCA extended volume: shells closed; r1/r3 = 2.22 / 2.25 / 2.24 at L = 0.5 / 1 / 2.303, fill 0.985 / 0.981 / 0.952.
  Ellipsoid null on the PCA box: ratio 3.00 -> 2.94, axis |cos| > 0.99999. On the random grid: 3.00 -> 2.97.
- PCA base volume: all three shells touch the +a face (and the 2.3 shell the +c face): the basin extends past
  w* away from the trajectory, beyond the 8% margin. -> extension job.

## Decisions (M2)
- Decision: M2 final-checkpoint volumes use the M1 27^3 grid (h = 0.08, [-1.04, 1.04]) and seeds 1, 2, 3 normalised to
  each model; slice reproduction against both g51 and g101 (25x25 coincident points each).
- Decision: epoch-film grid 17^3 at h = 0.16 on [-1.28, 1.28] — odd so w_epoch is a vertex; 0.16 is 2x the 27^3
  spacing and 4x g51's, so the ep040 volume shares 13^3 points with the final 27^3 volume (an internal check of the
  per-epoch loading path) and 13x13 points with g51; the wider box leaves room for the broader early-epoch basins.
- Decision: epochs {0, 1, 2, 4, 8, 16, 40} as planned; ep040 first (the check), then chronological.
- Decision: film volumes on CPU (3 slab-shard workers x 4 threads) — the GPU queue had ~9 jobs waiting; the M1
  toy showed CPU reproduces the GPU plate to 3e-7.
- Decision: ResNet-56 volumes in 4 GPU shards (~7 slabs, ~5,100 points each, expected ~7-10 min at ~12 pts/s).
- Decision: film anisotropy uses the absolute levels 0.5, 1, ln 10 plus levels relative to each checkpoint's centre
  loss (x2, x4, x8), because early checkpoints have centre loss near ln 10 and the absolute shells are empty.

## Decisions (M1)
- Decision: grid is 27 points per axis at spacing 0.08 on [-1.04, 1.04], not the plan's 26 on [-1, 1] — 26 points
  on [-1, 1] are the even g51 indices and do not contain 0, so neither w* nor the c = 0 plane would lie on the grid.
  27 points keep spacing 0.08, put w* on a vertex and all three coordinate planes on grid slabs, and 25x25 points of
  the c = 0 slab coincide with g51 (odd indices). Cost +12% (19,683 points).
- Decision: third direction = seed 3 through the identical `random_direction` recipe (filter-normalised, biasbn zeroed).
- Decision: PCA over all 17 cached checkpoints ep000..ep040 minus final (ep040 == final bit for bit, so one zero row),
  as in loss-landscape/pca_dirs.py; unit-norm (not filter-normalised, as Li et al. §7); signs so ep000 is negative
  on each axis (matches loss-landscape's plate signs).
- Decision: PCA grid axes per direction span [min, max] of the trajectory (and 0) plus an 8% margin, 27 points,
  0 on a vertex; spacing differs per axis (stored as physical axes a, b, c; renderers must use the spacing).
- Decision: vmap batch K chosen automatically on the GPU as the fastest K in {1,4,8,16,32,64} whose 50-point
  equivalence is <= 1e-5 relative, with K * image-batch <= 8000 to bound memory; all passes have fixed shapes
  (the last chunk of a slab is padded by repeating its final point).
- Decision: vmap dropped for production — equivalent (1.9e-7) but 0.39-0.66x the sequential speed on the GB10
  (grouped convs are slow here). The chain's auto-pick chose K=1 vmap for the random volume (20.5 pts/s, already
  running when this was seen); volume.py was then changed so K=1 means the sequential loss-landscape path, which the
  PCA volume used (30.5 pts/s). Both paths agree to 1.9e-7, so the volumes are comparable.
- Decision: extend the PCA grid by 9 points on +a (to a = 15.3) and 5 on +c (to c = 11.4), same spacing and origin,
  prefilled from the base 27^3 — the plan's "grid spanning the trajectory" clips every declared shell on the +a
  face, which would make the anisotropy table meaningless. Base volume kept; anisotropy reported on the extension.
- Decision: the ellipsoid null is scaled to each box (semi-axes 0.9/0.5/0.3 x half-extent/1.04, centred), so it is
  identical on the random grid and resolvable on the PCA grid.
- Decision: run the 11k-point PCA extension on CPU (3 processes x 4 threads) instead of waiting behind a ~3 h GPU
  chain — CPU and GPU agree with the GPU-measured g51 plate to 3e-7 (toy volume), far below any shell-level effect.
- Decision: one gpu1.sh chain for check + both volumes (spec §12 chain pattern; avoids re-queueing behind other pieces).
- Decision: anisotropy levels L = 0.5, 1.0, ln 10 = 2.303 (declared before seeing the volume): ln 10 is chance-level
  cross-entropy for 10 classes; 0.5 and 1.0 are ~4x and ~8x the minimum 0.124. Method: 26-connected component of
  {loss <= L} containing the grid minimum, on the grid supersampled 4x by trilinear interpolation of log loss
  (declared; native-grid numbers reported alongside), second moments in physical coordinates, semi-axes sqrt(5 lambda)
  (solid ellipsoid), fill = volume / ellipsoid volume. Null: rotated analytic ellipsoid on the same grid.
- Decision: previews use log10 loss on matplotlib magma, nearest-neighbour pixels; not gallery pieces.
