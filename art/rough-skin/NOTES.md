# Rough Skin — running notes

Spec: `art/ml-art-3d.md` §7 (+ §0, §11). Plan: `docs/superpowers/plans/2026-09-15-3d-pieces.md` §7.
Source piece (read-only): `art/depth-roughness/` (net definition from `nets.py`, estimator protocol from
`compute_calibration.py`, finite-width cutoff ε* ≈ 4/n from its README §4.4).

## State

- **M1 (fields and dimensions): DONE_WITH_CONCERNS**, report `docs/superpowers/plans/reports/rough-skin-M1.md`.
  Width 4096, 256³, Heaviside L = 1: 3D calibrated D = 2.393 ± 0.058 (theory 2.5, misses the 0.1 gate by 0.007);
  slice-calibrated 2.518 ± 0.037; five width-1024 draws give 2.483 ± 0.071. L = 2: 2.856 ± 0.034 (theory 2.75).
  L ≥ 3 exceed the calibrated range (estimator saturates near 2.97 at 256³). ReLU raw 2.01–2.06 vs smooth null 2.03–2.06.
  float32 vs float64 output sign-flip rate ≤ 1.8e-4 (Heaviside L = 4), 0 at L = 1. Rate 6.3e4 voxels/s at width 4096 (shared GPU).
- Previews (viewed): `cache/preview/slices_w1024_r128.png`, `cache/preview/slices_w4096_r256.png`.
- **M2 (renders): DONE**, report `docs/superpowers/plans/reports/rough-skin-M2.md`. Three width-4096 draws (seeds 7, 8, 9):
  Heaviside L = 1 3D calibrated 2.426 ± 0.031, slices 2.448 ± 0.067; L = 2 3D 2.774 ± 0.076, slices 2.769 ± 0.040 (theory 2.5, 2.75).
  Plate reproduction vs depth-roughness nets_patch (n = 4096): 3 and 27 of 1,048,576 pixels differ in sign (L = 1, 2).
- **M3 (film, objects, README): in progress.** Diptych re-rendered with measured D beside theory (ruling); STL made
  (Heaviside L1 96³ native, ReLU L1 128³ cache subsample, both watertight, 80 mm); film queued via gpu1.sh
  (`render_film.py all --size 1080 --device cuda`); README written. See M3 log below for the resume state if
  interrupted, and `docs/superpowers/plans/reports/rough-skin-M3.md` for the final report.

## Files

| file | what |
|---|---|
| `common.py` | chart (exp map), weights, nested forward pass, boundary-voxel box counting, oblique slices |
| `test_common.py` | analytic checks: exp map on S³ and its singular values; plane → D = 2, noise → 3; line → 1; slices inside the cube; trilinear exact on linear fields |
| `compute_chart.py` | chart distortion → `cache/chart.json` |
| `compute_calibration.py` | periodic-box protocol (comparison) → `cache/calibration3d.json` |
| `compute_calibration_window.py` | window protocol (primary): windows of periodic 1024³ power-law fields → `cache/calibration3d_window.json` |
| `run_seeds.sh` | extra width-1024 128³ draws, seeds 8–11 (GPU, via gpu1.sh) |
| `compute_fields.py` | GPU nets (queued via `gpu1.sh`), per-z-plane checkpoints → `cache/field_w<n>_r<R>.npy`, float64 slab, exact oblique slices, timing |
| `measure_dims.py` | D tables → `cache/dims.json`, `cache/dims_tables.md` |
| `render_preview.py` | matplotlib check slices → `cache/preview/` |

## Resume commands (from this directory)

```bash
PY=/home/fzeng/ml/research/art/.venv/bin/python
OMP_NUM_THREADS=4 $PY test_common.py && $PY compute_chart.py
OMP_NUM_THREADS=4 $PY compute_calibration.py 6          # ~3 min, resumable (comparison)
OMP_NUM_THREADS=4 $PY compute_calibration_window.py 4   # ~10 min, ~15 GB RAM peak, resumable (primary)
setsid nohup ../_shared/gpu1.sh $PWD/run_seeds.sh > logs/seeds_w1024.log 2>&1 < /dev/null &
setsid nohup ../_shared/gpu1.sh $PY compute_fields.py 1024 2 > logs/fields_w1024_r128.log 2>&1 < /dev/null &
setsid nohup ../_shared/gpu1.sh $PY compute_fields.py 4096 1 > logs/fields_w4096_r256.log 2>&1 < /dev/null &   # resumable per z-plane
OMP_NUM_THREADS=4 $PY measure_dims.py
$PY render_preview.py 1024 128 ; $PY render_preview.py 4096 256
```

## Derivation: β = 1/2 for Heaviside (ours, not the paper's statement)

Di Lillo, Marinucci, Salvi & Vigogna (arXiv:2504.06250) give dim_H T_L⁻¹(u) = d − β^L (with positive
probability) for activations whose kernel has covariance regularity index β ∈ (0, 1), i.e. κ(1 − t) = 1 − c t^β + …
The Heaviside value is our derivation: with Γ_W = 2, κ(u) = 1 − arccos(u)/π, and arccos(1 − t) = √(2t)(1 + O(t)),
so κ(1 − t) = 1 − (√2/π) t^{1/2} + O(t^{3/2}): β = 1/2. Composition gives 1 − κ_L(1 − t) ∝ t^{2^{−L}}, i.e. a
variogram ∝ θ^{2H} in geodesic angle with H = 2^{−L}. On S³ (d = 3) the prediction is 3 − 2^{−L}. It agrees with
`depth-roughness/`'s measured 2 − 2^{−L} on S² (e.g. L = 2: 1.618 vs 1.619 for an exact-dimension field).

## Decisions

- Decision: base point p = (0.3, −0.5, 0.8, 0.1)/‖·‖, tangent frame from QR of [p | I₄] — arbitrary; the fields are isotropic so any p is equivalent in law.
- Decision: grid nodes t_i = (i − 127.5)·h, h = 0.5/256 rad (256³); the 128³ grid is the even-index subsample (spacing 2h), so the width-4096 128³ field is exactly a subsample of the 256³ one — clean resolution doubling on one net.
- Decision: one seed (7) for both widths; weights drawn in float64 on CPU and cast, so the float64 slab uses the identical draw.
- Decision: nets as in `depth-roughness/nets.py` lifted to ℝ⁴ input: T_L = √(2/n) v·σ(h_L), h₁ = W₀x, h_{l+1} = √(2/n) W_l σ(h_l), N(0,1) entries, no biases; readout v shared across depths; ReLU uses literally the same tensors.
- Decision: level = median of each volume (midpoint of the two central order statistics) — the theorem holds for every u, and the zero level can miss a 0.5 rad patch because of the kernel's constant component (κ(−1) = 0 but κ > 0 over the patch). Same choice as `depth-roughness/` tiles.
- Decision: boundary voxel = voxel with a 6-neighbour on the other side (T > u vs T ≤ u), both sides marked; boxes aligned to the grid, partial boxes dropped (none for powers of 2).
- Decision: 3D fit b = 2–64 voxels on 256³ (per plan) and b = 2–32 on 128³ (same physical range 7.8e-3–0.125 rad at the top). Neither is ≥ 2 decades (1.5 and 1.2); §11's two-decade requirement is not met by a single volume and is flagged as a risk.
- Decision: calibration fields are generated on periodic M³ boxes and subsampled to the target grid, keeping k = log2(M/T) octaves of aliased sub-voxel roughness, because a finite net keeps roughness down to ≈ 4/n: k = 1 for width 4096 on 256³, k = 2 for width 4096 on 128³, k = 0 for width 1024 on 128³. Other k are reported as a systematic. First run showed k matters by up to 0.08 at D = 2.75.
- Decision: smooth null for the calibration table is a Gaussian-spectrum field with correlation length 48/256 of the box — a first try at 8/256 measured 2.61 because its surface is dense at b ≥ 16 (a smooth null must be smooth on the fitted scales).
- Decision: slices = 12 random oblique planes (normals ≥ 15° from every axis, offsets ±0.1 half-side, square of side 0.68 of the cube), sampled from the volume by linear interpolation at the volume's own spacing (176² on 256³, 88² on 128³), level = the volume median, 4-neighbour boundary pixels, fit b = 2–32 / 2–16; the same code runs on the calibration fields. The exact net on the same planes (GPU, 176² at 256³ spacing) is a cross-check for the interpolation.
- Decision: calibration inverts the mean estimator response over power-law fields (H = 0.9 … 0.0625, 6 common-random-number seeds) by linear interpolation; ± is the seed sd divided by the local slope. ReLU (D = 2) lies below the power-law range and is compared with the smooth null instead.
- Decision: the **window protocol** is the primary calibration — calibration volumes are central, non-periodic windows (1/8–1/2 of the box side) of periodic 1024³ power-law fields. Why: the width-4096 L = 1 field matched H = 1/2 periodic-box fields in local slope at b ≤ 8 but fell below them at b ≥ 16; a 0.5 rad cube is a window of a field on all of S³ (≈ 1/6 of a great-circle half-turn), so it carries trends longer than the cube, which a periodic box of grid size does not. The periodic numbers stay in the table as a systematic (they move calibrated D by −0.04 to −0.08).
- Decision: extra width-1024 draws (seeds 8–11) to measure draw-to-draw scatter of D, since the width-4096 field is a single draw (≈ 15 s GPU each).
- Decision: report L ≥ 3 3D values as "above calibrated range" rather than extrapolating; the estimator on exact D = 2.969 fields reads 2.94–2.96, and the nets read 2.96–2.99.

## M2 (renders) — log

Rulings from the M1 gate: 2 more width-4096 draws (seeds 8, 9) and draw statistics for L = 1, 2; the README D table (M3) must use them, state the calibration systematics and the 1.5-decade fit range, and make no D claim for L ≥ 3; 3D renders only for Heaviside L = 1, 2 and ReLU; L = 3/4 only in the slice atlas (and M3 film frames).

Commands:
```bash
setsid nohup ../_shared/gpu1.sh $PWD/run_draws4096.sh > logs/draws_w4096.log 2>&1 < /dev/null &   # seeds 8, 9 (no float64 slab)
setsid nohup ../_shared/gpu1.sh $PWD/run_m2.sh > logs/m2_render.log 2>&1 < /dev/null &           # verify_plate, casts, fog at 2048²
OMP_NUM_THREADS=4 $PY render_atlas.py                                                            # CPU, slice atlases
$PY render_diptych.py exterior ; $PY render_diptych.py cutaway
OMP_NUM_THREADS=4 $PY measure_dims.py                                                            # adds width-4096 draw statistics
```

- Decision: solid = {T ≤ u}, seen from above (el 32°, az −35°, orthographic) — for this draw both Heaviside L = 1 and ReLU put {T > u} mostly in the upper half (centroid z ≈ +0.4 of the half side), so the lower side is a block whose top is the skin; the other side would hide the skin inside a closed box.
- Decision: quadrant cutaway (x > 0, y < 0, full height) instead of a single octant — the top octant misses most of the L = 1 skin near z ≈ 0; two vertical section planes show the skin's profile. Section faces (cube boundary and cut planes) are a darker plaster.
- Decision: one raking light (0.45, −1, 0.75), hard shadow, 32-ray AO radius 0.12 (15 voxels); plaster albedo 0.95/0.93/0.89, section 0.70/0.68/0.65. Light shows form only.
- Decision: ReLU is rendered twice — crisp voxels through the identical Heaviside path (the null, used in the diptych) and a linear isosurface (allowed for ReLU only; the gradient-normalised field makes the cutaway a CSG min with the quadrant's signed distance).
- Decision: Spectral fog on paper ground only; a dark ground made low-opacity emission read muddy. TF declared in render_fog.py: per-side rank normalisation over the slab, seam colours #5e4fa2 | #9e0142, opacity 0.10 + 0.90(1 − |q|)^24, density 25, nearest-voxel sampling; slab z-voxels 116–139.
- Decision: slice atlas cuts are z = const planes of the chart (z-voxels round(linspace(8, 247, 12))), coastline along voxel edges at the whole-volume median, 4 px per voxel, depth-roughness paper/ink/C059 helpers imported from its render_common.py.
- Decision: plate reproduction (§0.7): the great S² {x4 = 0} ⊂ S³; `verify_plate.py` feeds depth-roughness's own n = 4096, seed 11 weights through this piece's forward code and compares with `depth-roughness/cache/nets_patch.npz`.
- Decision: the casts and fog were rendered on CPU (4 threads, 50–140 s per 2048² cast, 6 s per fog) rather than through gpu1 — the art GPU queue was held by solid-edge's multi-hour B256 volume with four other waiters, and r3d on CPU at this size is minutes. The GPU chain `run_m2.sh` still ran (plate reproduction; the cast/fog stages skipped the existing outputs).

## M3 (film, objects, README) — log

Ruling: diptych and cast captions print the measured D (3-draw mean ± sd from the M2 gate) beside theory, and the diptych
is re-rendered with that caption. README D table uses the 3-draw statistics, states the calibration systematics
(window vs periodic −0.07, k = 0 vs 1 +0.04) and the 1.5-decade fit range, and makes no D claim for L ≥ 3. Film: L = 1→4
depth dial (measured D per frame where it exists, "saturated" for L ≥ 3), hard cuts only, plus an L = 1 turntable,
≤ 1080², 20–30 s, MP4 + GIF, MP4 ≤ 20 MB. STL: Heaviside L1 and ReLU zero-set solids at 128³, watertight, 80 mm.

- Decision: diptych re-render (`render_diptych.py`) — Heaviside caption now reads "D = 2.426 ± 0.031 (theory 2.5)"
  with the slice value and the 3-draw/calibration protocol in the italic subtitle; shortened from the original wording
  so the line fits within its half of the canvas (checked with `ImageDraw.textlength`, which the original caption did
  not need since it was shorter).
- Decision: STL grid — the 128³ mesh of Heaviside L = 1 (from the cached 256³ field's even-node subsample) is 584k
  faces / ~29.2 MB, over the 20 MB budget; no mesh-decimation library is installed (`trimesh`/`open3d`/`pyvista` all
  absent from `art/.venv`). Per the ruling's fallback, Heaviside L1 is instead a **native 96³ evaluation** (same
  weights, seed 7, freshly forward-passed on a native 96-node grid — not a decimation of the 128³ mesh), 291,782
  faces, 14.6 MB, watertight. ReLU L1 stays at 128³ (cache subsample), 153,872 faces, 7.7 MB, watertight — no
  fallback needed since ReLU is smooth. Both scaled to 80 mm. `make_stl.py`, 6.1 s total (CPU + a GPU forward pass
  for the 96³ re-evaluation). Level = volume median at each mesh's own resolution (same per-volume-median rule as
  every other measurement in this piece); "zero-set" in the spec's language means this declared level, not literal
  T = 0 (T = 0 can miss the patch, per the M1 level decision above).
- Decision: film composition (`render_film.py`) — one film, two hard-cut parts (no interpolation between depths,
  per the ruling's preference): a depth dial (L = 1, 2 as the exact 3D crisp-voxel cast, camera/level/light
  identical to the M2 casts, captioned with the measured D beside theory; L = 3, 4 switch to the plotter idiom —
  one of the 12 atlas cuts enlarged, same `coast_tile` code as `render_atlas.py` — captioned "saturated", since the
  3D box-count estimator has no measured value there and the isosurface itself is foam), then a 240-frame, 360°
  turntable of the L = 1 Heaviside cast (fixed world-space light, so faces rotate through it — a real property of
  a turntable under one raking light, not a bug). FPS 24, hold 3 s/depth (72 frames, repeated via hardlink of one
  render — no re-render needed since the four dial sub-clips are static), target length ~22 s (12 s dial + 10 s
  turntable), within the 20–30 s window. Captions use `ImageDraw.textlength`-based auto-fit (`fit_font`) after the
  first attempt at 1080² overflowed the frame at a fixed font size; theory 2^-L is written as a fraction (1/2, 1/4,
  …) rather than a unicode superscript, which the C059 font did not render correctly.
- Decision: turntable and both dial 3D frames render through `gpu1.sh` (`--device cuda`); timed at 320²/CPU (17s)
  vs 1080²/GPU (2.5–3.7 s) before committing to GPU — CPU would have been 60–90 min for 240 turntable frames alone,
  over the ~30 min CPU budget and over gpu1's ≤20 min per-segment rule; GPU fits in one ~15 min segment.
- Decision: MP4 CRF — `r3d.write_film` hardcodes CRF 18; re-encoded to CRF ≤ 27 after the fact if the first pass is
  over the 20 MB budget (see M3 report for the final size and whether a re-encode was needed).

**M3 result:** film job ran 08:04:32–08:12:57 (queued ~45 min behind solid-edge's B256 volume before that), 8.4 min
GPU wall time for 4 dial frames + 240 turntable frames at 1080². First-pass MP4 (CRF 18, `r3d.write_film` default)
was 21.34 MB, over the 20 MB budget; re-encoded with `ffmpeg -c:v libx264 -crf 23 -pix_fmt yuv420p` to 13.13 MB
(replaces the CRF-18 file; the GIF, 11.66 MB, was left as `write_film` made it). Final: 1080×1080, 24 fps, 528
frames, 22.0 s. Checked frames: dial L1/L4 and a mid-turntable frame (viewed) — captions legible, hard cut at
frame 288 (dial → turntable) lands on the same camera angle as the dial's L1 frame, by construction.
**M3: DONE.** README written and link-checked clean (`grep -o 'gallery/[^")> ]*' README.md | ...`, no missing files
after rewording four inline `gallery/` backtick references that were false-positiving on the trailing backtick).
Report: `docs/superpowers/plans/reports/rough-skin-M3.md`.
