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
- M2 (renders, needs r3d) and M3 (film, STL, README) not started.

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
