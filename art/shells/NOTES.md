# NOTES: shells (§2 of art/ml-art-3d.md) — living handoff

## State
- M1 in progress. Code: `lib.py` (imports loss-landscape/common.py; SeqEvaluator, VmapEvaluator),
  `pca3.py` (CPU), `check.py` (equivalence + slice points + K sweep), `volume.py` (27^3, per-slab checkpoints),
  `analyze.py` (slice reproduction, anisotropy, ellipsoid null), `preview.py` (matplotlib previews to cache/preview/).
- GPU chain `jobs/m1.sh` (check -> choose K -> random 27^3 -> PCA 27^3), log `logs/m1.log`: DONE 04:38.
  `cache/vol/resnet20_final_random_g27.npz` (16.0 min, 20.5 pts/s), `cache/vol/resnet20_final_pca_g27.npz` (10.7 min, 30.5 pts/s).
- `jobs/m1_pca_ext.sh` (log `logs/m1_pca_ext.log`): PCA volume extended +9 a, +5 c points -> `resnet20_final_pca_g27_ext.npz`.

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
- PCA base volume: all three shells touch the +a face (and the 2.3 shell the +c face): the basin extends past
  w* away from the trajectory, beyond the 8% margin. -> extension job.

## Decisions
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
- Decision: one gpu1.sh chain for check + both volumes (spec §12 chain pattern; avoids re-queueing behind other pieces).
- Decision: anisotropy levels L = 0.5, 1.0, ln 10 = 2.303 (declared before seeing the volume): ln 10 is chance-level
  cross-entropy for 10 classes; 0.5 and 1.0 are ~4x and ~8x the minimum 0.124. Method: 26-connected component of
  {loss <= L} containing the grid minimum, on the grid supersampled 4x by trilinear interpolation of log loss
  (declared; native-grid numbers reported alongside), second moments in physical coordinates, semi-axes sqrt(5 lambda)
  (solid ellipsoid), fill = volume / ellipsoid volume. Null: rotated analytic ellipsoid on the same grid.
- Decision: previews use log10 loss on matplotlib magma, nearest-neighbour pixels; not gallery pieces.
