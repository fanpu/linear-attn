# Rough Skin (§7) — M1 report: fields and dimensions

**Status: DONE_WITH_CONCERNS.** The primary number, the width-4096 256³ Heaviside L = 1 3D calibrated D, is 2.393 ± 0.058. That is 0.107 below 2.5, so it misses the 0.1 gate by 0.007. The other estimates of the same quantity pass:
- the calibrated oblique-slice estimate, 2.518 ± 0.037;
- calibration assuming no sub-voxel roughness (k = 0), 2.428;
- the mean of five width-1024 draws, 2.483 ± 0.071.

The miss is about one draw-to-draw sd. L = 2 reads high: 2.856 ± 0.034 against 2.75. L ≥ 3 saturates the 256³ estimator, so those values sit above the calibrated range. ReLU is consistent with D = 2.

Directory: `/home/fzeng/ml/research/art/rough-skin/` (code, NOTES.md; caches and logs are gitignored).

## What was done (commands, from the piece directory, `PY=art/.venv/bin/python`)

```bash
OMP_NUM_THREADS=4 $PY test_common.py                      # analytic checks: exp map, plane→2, noise→3, line→1, slices
$PY compute_chart.py                                      # cache/chart.json
OMP_NUM_THREADS=4 $PY compute_calibration.py 6            # periodic-box protocol (comparison), 3 min CPU
OMP_NUM_THREADS=4 $PY compute_calibration_window.py 4     # window protocol (primary), 10 min CPU
setsid nohup ../_shared/gpu1.sh $PY compute_fields.py 1024 2 > logs/fields_w1024_r128.log 2>&1 < /dev/null &   # 14 s
setsid nohup ../_shared/gpu1.sh $PY compute_fields.py 4096 1 > logs/fields_w4096_r256.log 2>&1 < /dev/null &   # 22 min total
setsid nohup ../_shared/gpu1.sh $PWD/run_seeds.sh > logs/seeds_w1024.log 2>&1 < /dev/null &                   # 4 × 14 s
OMP_NUM_THREADS=4 $PY measure_dims.py                     # cache/dims.json, cache/dims_tables.md
$PY render_preview.py 1024 128 ; $PY render_preview.py 4096 256
```

No packages were installed; scipy and scikit-image were already present.

## Model and chart

- **Nets:** these are the nets of `depth-roughness/nets.py`, lifted to x ∈ S³ ⊂ ℝ⁴.
  - Definitions: h₁ = W₀x, h_{l+1} = √(2/n) W_l σ(h_l), T_L = √(2/n) v·σ(h_L). All entries are N(0,1), with no biases.
  - Shared weights: depth L uses the first L layers of one draw, and v is shared across depths. ReLU uses the identical tensors.
  - Seed 7 is the main draw, and seeds 8–11 are extra width-1024 draws.
  - Precision: weights are drawn in float64 and cast to float32.
- **Chart (declared):** the exponential map at p = (0.3, −0.5, 0.8, 0.1)/‖·‖ of the tangent cube [−0.25, 0.25]³ rad.
  - Grid nodes are (i − 127.5)·h with h = 0.5/256 = 1.953e-3 rad. The 128³ grid is the even-node subsample.
  - **Measured distortion:**
    - The chart never stretches: the largest singular value is 1 + 1e-10.
    - It compresses by up to **3.10 %** transversally at the cube corners (smallest singular value 0.96904). This matches the analytic 1 − sin(0.433)/0.433 to 1e-10.
    - The maximum anisotropy is 1.032.
    - Voxel edges measure 1.9531e-3 rad at the centre and 1.9130e-3 rad along the corner rows.
- **Level:** the median of each volume (declared). Boundary voxels are those with a 6-neighbour across the level. The 3D fit uses b = 2–64 voxels on 256³ and b = 2–32 on 128³.
- **Slices:** 12 random oblique planes, with normals at least 15° from every axis and a square of side 0.68 of the cube.
  - They are linearly interpolated from the volume at the volume's own spacing, with 4-neighbour boundary pixels and fits over b = 2–32 (256³) or 2–16 (128³).
  - The same code runs on the calibration fields.
  - The exact net on the same planes (176² at the 256³ spacing) is a cross-check.

## β = 1/2 (our derivation, not the paper's statement)

The paper (arXiv:2504.06250) gives dim = d − β^L for covariance regularity index β. For Heaviside, κ(u) = 1 − arccos(u)/π, so κ(1 − t) = 1 − (√2/π)t^{1/2} + O(t^{3/2}), which gives β = 1/2. On S³ the prediction is therefore 3 − 2^{−L}. This is consistent with `depth-roughness`'s measured 2 − 2^{−L} on S².

## Estimator calibration

Synthetic isotropic Gaussian fields with S(k) ∝ k^{−(3+2H)} have level sets of dimension exactly 3 − H. The smooth null has a Gaussian spectrum with correlation length 48/256 of the box, so D = 2.

- **Window protocol (primary).** Each field is a periodic 1024³ FFT box. The calibration volume is a central non-periodic window of it, subsampled to T with k octaves of sub-voxel roughness.
  - k is matched to the finite-width cutoff ≈ 4/n: k = 1 for width 4096 on 256³, k = 2 for width 4096 on 128³, and k = 0 for width 1024 on 128³.
  - The window was adopted after the first width-4096 field matched H = 1/2 fields in local slope at b ≤ 8 but fell below periodic-box fields at b ≥ 16. The cube is a window of a sphere-wide field.
  - Local slopes at b = 1→64, width-4096 L1: 2.18 2.33 2.39 2.42 2.48 2.63.
  - Local slopes of the periodic H = 1/2 fields: 2.17 2.33 2.44 2.56 2.75 2.95.
- **Periodic protocol (comparison, the plan's literal "periodic 256³ box").** The field fills the whole periodic box. It moves calibrated D by −0.04 to −0.08.
- **Bias:**
  - With the window protocol, the 3D estimator is biased by +0.10 to +0.13 at D = 2.1, 0 to +0.03 at D = 2.5, and −0.01 to −0.04 at D ≥ 2.94, where it saturates.
  - The slice estimator reads 0.06–0.15 low for D ≥ 2.5.
  - Both biases are corrected by inverting the mean response curve. The ± values are the seed sd divided by the local slope.

### Estimator calibration (power-law Gaussian fields; mean ± sd over seeds: window 4, periodic 6)

| protocol | grid T | sub-voxel octaves k | estimator | D=2.1 | D=2.25 | D=2.5 | D=2.625 | D=2.75 | D=2.875 | D=2.938 | D=2.969 | smooth null (D=2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| window | 256 | 0 | 3D box | 2.195 ± 0.053 | 2.298 ± 0.054 | 2.501 ± 0.048 | 2.620 ± 0.043 | 2.747 ± 0.032 | 2.859 ± 0.019 | 2.902 ± 0.013 | 2.919 ± 0.010 | 2.027 ± 0.017 |
| window | 256 | 0 | 1 + slice | 2.209 ± 0.021 | 2.281 ± 0.026 | 2.442 ± 0.027 | 2.519 ± 0.020 | 2.614 ± 0.017 | 2.716 ± 0.022 | 2.768 ± 0.020 | 2.792 ± 0.020 | 2.097 ± 0.023 |
| window | 256 | 1 | 3D box | 2.232 ± 0.027 | 2.321 ± 0.034 | 2.534 ± 0.058 | 2.655 ± 0.056 | 2.780 ± 0.042 | 2.888 ± 0.022 | 2.928 ± 0.014 | 2.944 ± 0.011 | 2.049 ± 0.009 |
| window | 256 | 1 | 1 + slice | 2.225 ± 0.022 | 2.290 ± 0.030 | 2.434 ± 0.025 | 2.522 ± 0.023 | 2.619 ± 0.020 | 2.732 ± 0.017 | 2.792 ± 0.014 | 2.818 ± 0.013 | 2.093 ± 0.020 |
| window | 128 | 0 | 3D box | 2.203 ± 0.062 | 2.292 ± 0.059 | 2.478 ± 0.066 | 2.605 ± 0.058 | 2.742 ± 0.038 | 2.854 ± 0.020 | 2.896 ± 0.014 | 2.914 ± 0.012 | 2.034 ± 0.017 |
| window | 128 | 0 | 1 + slice | 2.238 ± 0.028 | 2.294 ± 0.023 | 2.424 ± 0.033 | 2.519 ± 0.034 | 2.605 ± 0.030 | 2.700 ± 0.041 | 2.749 ± 0.038 | 2.772 ± 0.038 | 2.129 ± 0.023 |
| window | 128 | 1 | 3D box | 2.217 ± 0.066 | 2.319 ± 0.067 | 2.528 ± 0.058 | 2.649 ± 0.053 | 2.781 ± 0.037 | 2.891 ± 0.020 | 2.930 ± 0.014 | 2.946 ± 0.011 | 2.034 ± 0.024 |
| window | 128 | 1 | 1 + slice | 2.235 ± 0.024 | 2.308 ± 0.024 | 2.458 ± 0.034 | 2.541 ± 0.019 | 2.627 ± 0.018 | 2.730 ± 0.017 | 2.782 ± 0.023 | 2.810 ± 0.022 | 2.134 ± 0.037 |
| window | 128 | 2 | 3D box | 2.256 ± 0.038 | 2.342 ± 0.046 | 2.554 ± 0.076 | 2.675 ± 0.068 | 2.802 ± 0.047 | 2.907 ± 0.024 | 2.944 ± 0.014 | 2.958 ± 0.011 | 2.062 ± 0.010 |
| window | 128 | 2 | 1 + slice | 2.255 ± 0.025 | 2.319 ± 0.041 | 2.451 ± 0.036 | 2.528 ± 0.033 | 2.625 ± 0.028 | 2.740 ± 0.023 | 2.804 ± 0.019 | 2.834 ± 0.017 | 2.141 ± 0.032 |
| periodic | 256 | 0 | 3D box | 2.264 ± 0.038 | 2.365 ± 0.038 | 2.571 ± 0.028 | 2.683 ± 0.023 | 2.790 ± 0.018 | 2.878 ± 0.013 | 2.913 ± 0.010 | 2.928 ± 0.008 | 2.088 ± 0.011 |
| periodic | 256 | 0 | 1 + slice | 2.260 ± 0.027 | 2.324 ± 0.028 | 2.435 ± 0.017 | 2.507 ± 0.021 | 2.608 ± 0.026 | 2.711 ± 0.025 | 2.762 ± 0.024 | 2.786 ± 0.022 | 2.147 ± 0.033 |
| periodic | 256 | 1 | 3D box | 2.270 ± 0.050 | 2.375 ± 0.046 | 2.598 ± 0.038 | 2.718 ± 0.032 | 2.828 ± 0.023 | 2.914 ± 0.013 | 2.945 ± 0.009 | 2.957 ± 0.007 | 2.089 ± 0.014 |
| periodic | 256 | 1 | 1 + slice | 2.257 ± 0.025 | 2.315 ± 0.028 | 2.466 ± 0.037 | 2.548 ± 0.046 | 2.644 ± 0.052 | 2.751 ± 0.042 | 2.803 ± 0.036 | 2.827 ± 0.033 | 2.126 ± 0.019 |
| periodic | 128 | 0 | 3D box | 2.300 ± 0.056 | 2.393 ± 0.060 | 2.579 ± 0.057 | 2.684 ± 0.047 | 2.786 ± 0.035 | 2.872 ± 0.021 | 2.906 ± 0.015 | 2.921 ± 0.012 | 2.126 ± 0.029 |
| periodic | 128 | 0 | 1 + slice | 2.300 ± 0.046 | 2.361 ± 0.041 | 2.493 ± 0.032 | 2.553 ± 0.052 | 2.633 ± 0.056 | 2.716 ± 0.054 | 2.757 ± 0.056 | 2.777 ± 0.055 | 2.165 ± 0.008 |
| periodic | 128 | 1 | 3D box | 2.301 ± 0.046 | 2.404 ± 0.043 | 2.612 ± 0.033 | 2.725 ± 0.026 | 2.830 ± 0.021 | 2.912 ± 0.014 | 2.942 ± 0.010 | 2.954 ± 0.009 | 2.117 ± 0.014 |
| periodic | 128 | 1 | 1 + slice | 2.313 ± 0.040 | 2.363 ± 0.036 | 2.457 ± 0.016 | 2.531 ± 0.020 | 2.622 ± 0.023 | 2.730 ± 0.026 | 2.780 ± 0.027 | 2.806 ± 0.027 | 2.207 ± 0.052 |
| periodic | 128 | 2 | 3D box | 2.306 ± 0.059 | 2.411 ± 0.056 | 2.633 ± 0.046 | 2.752 ± 0.037 | 2.857 ± 0.026 | 2.936 ± 0.014 | 2.962 ± 0.009 | 2.972 ± 0.007 | 2.120 ± 0.020 |
| periodic | 128 | 2 | 1 + slice | 2.300 ± 0.025 | 2.352 ± 0.028 | 2.491 ± 0.039 | 2.567 ± 0.054 | 2.658 ± 0.062 | 2.765 ± 0.054 | 2.819 ± 0.047 | 2.846 ± 0.043 | 2.172 ± 0.036 |

## D table

In the table below:
- **3D calibrated** uses the window protocol at the matched k.
- **other k** and **periodic** show the systematics.
- **1+slice** is the mean ± sd over 12 oblique slices.
- **exact slice** is the net evaluated directly on the planes.

Theory is 3 − 2^{−L} for Heaviside and 2 for ReLU.

### Dimension table (level = volume median)

| width | grid | act | L | theory | 3D raw | 3D calibrated (window, k) | 3D cal, other k | 3D cal, periodic protocol | 1+slice raw (12 oblique) | 1+slice calibrated | 1+exact slice raw (256 spacing) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 128³ | heaviside | 1 | 2.5000 | 2.546 | 2.567 ± 0.062 (k=0) | k=1: 2.518, k=2: 2.490 | 2.455 | 2.477 ± 0.049 | 2.570 ± 0.048 | 2.460 ± 0.046 |
| 1024 | 128³ | heaviside | 2 | 2.7500 | 2.788 | 2.801 ± 0.035 (k=0) | k=1: 2.758, k=2: 2.736 | 2.754 | 2.629 ± 0.040 | 2.781 ± 0.045 | 2.629 ± 0.037 |
| 1024 | 128³ | heaviside | 3 | 2.8750 | 2.884 | 2.919 ± 0.025 (k=0) | k=1: 2.867, k=2: 2.848 | 2.896 | 2.770 ± 0.079 | 2.965 ± 0.050 | 2.789 ± 0.056 |
| 1024 | 128³ | heaviside | 4 | 2.9375 | 2.967 | above range (>2.9688) (k=0) | k=1: nan, k=2: nan | nan | 2.898 ± 0.041 | above range (>2.9688) | 2.896 ± 0.026 |
| 1024 | 128³ | relu | 1 | 2.0000 | 2.077 | below range (<2.100) (k=0) | k=1: nan, k=2: nan | nan | 2.165 ± 0.066 | below range (<2.100) | 2.105 ± 0.036 |
| 1024 | 128³ | relu | 2 | 2.0000 | 2.044 | below range (<2.100) (k=0) | k=1: nan, k=2: nan | nan | 2.168 ± 0.093 | below range (<2.100) | 2.120 ± 0.065 |
| 1024 | 128³ | relu | 3 | 2.0000 | 2.037 | below range (<2.100) (k=0) | k=1: nan, k=2: nan | nan | 2.134 ± 0.056 | below range (<2.100) | 2.100 ± 0.049 |
| 1024 | 128³ | relu | 4 | 2.0000 | 2.050 | below range (<2.100) (k=0) | k=1: nan, k=2: nan | nan | 2.136 ± 0.038 | below range (<2.100) | 2.097 ± 0.027 |
| 4096 | 256³ | heaviside | 1 | 2.5000 | 2.443 | 2.393 ± 0.058 (k=1) | k=0: 2.428 | 2.326 | 2.447 ± 0.057 | 2.518 ± 0.037 | 2.512 ± 0.068 |
| 4096 | 256³ | heaviside | 2 | 2.7500 | 2.871 | 2.856 ± 0.034 (k=1) | k=0: 2.893 | 2.812 | 2.678 ± 0.051 | 2.815 ± 0.021 | 2.786 ± 0.044 |
| 4096 | 256³ | heaviside | 3 | 2.8750 | 2.960 | above range (>2.9688) (k=1) | k=0: nan | nan | 2.821 ± 0.044 | above range (>2.9688) | 2.939 ± 0.027 |
| 4096 | 256³ | heaviside | 4 | 2.9375 | 2.988 | above range (>2.9688) (k=1) | k=0: nan | nan | 2.907 ± 0.020 | above range (>2.9688) | 3.005 ± 0.009 |
| 4096 | 256³ | relu | 1 | 2.0000 | 2.011 | below range (<2.100) (k=1) | k=0: nan | nan | 2.091 ± 0.043 | below range (<2.100) | 2.091 ± 0.043 |
| 4096 | 256³ | relu | 2 | 2.0000 | 2.034 | below range (<2.100) (k=1) | k=0: nan | nan | 2.094 ± 0.042 | below range (<2.100) | 2.093 ± 0.043 |
| 4096 | 256³ | relu | 3 | 2.0000 | 2.022 | below range (<2.100) (k=1) | k=0: nan | nan | 2.106 ± 0.038 | below range (<2.100) | 2.107 ± 0.038 |
| 4096 | 256³ | relu | 4 | 2.0000 | 2.045 | below range (<2.100) (k=1) | k=0: nan | nan | 2.101 ± 0.035 | below range (<2.100) | 2.102 ± 0.035 |
| 4096 | 128³ | heaviside | 1 | 2.5000 | 2.442 | 2.368 ± 0.076 (k=2) | k=0: 2.451, k=1: 2.397 | 2.284 | 2.456 ± 0.079 | 2.509 ± 0.060 | 2.513 ± 0.065 |
| 4096 | 128³ | heaviside | 2 | 2.7500 | 2.895 | 2.861 ± 0.037 (k=2) | k=0: 2.935, k=1: 2.881 | 2.810 | 2.687 ± 0.066 | 2.817 ± 0.028 | 2.786 ± 0.044 |
| 4096 | 128³ | heaviside | 3 | 2.8750 | 2.971 | above range (>2.9688) (k=2) | k=0: nan, k=1: nan | 2.965 | 2.826 ± 0.052 | 2.961 ± 0.018 | 2.939 ± 0.027 |
| 4096 | 128³ | heaviside | 4 | 2.9375 | 2.992 | above range (>2.9688) (k=2) | k=0: nan, k=1: nan | nan | 2.908 ± 0.026 | above range (>2.9688) | 3.005 ± 0.009 |
| 4096 | 128³ | relu | 1 | 2.0000 | 2.014 | below range (<2.100) (k=2) | k=0: nan, k=1: nan | nan | 2.125 ± 0.058 | below range (<2.100) | 2.091 ± 0.052 |
| 4096 | 128³ | relu | 2 | 2.0000 | 2.045 | below range (<2.100) (k=2) | k=0: nan, k=1: nan | nan | 2.130 ± 0.064 | below range (<2.100) | 2.093 ± 0.052 |
| 4096 | 128³ | relu | 3 | 2.0000 | 2.031 | below range (<2.100) (k=2) | k=0: nan, k=1: nan | nan | 2.129 ± 0.051 | below range (<2.100) | 2.099 ± 0.045 |
| 4096 | 128³ | relu | 4 | 2.0000 | 2.058 | below range (<2.100) (k=2) | k=0: nan, k=1: nan | nan | 2.139 ± 0.061 | below range (<2.100) | 2.101 ± 0.043 |

### float32 vs float64 (32 central z-planes)

| field | voxels | output sign flips | rate | boundary voxels differing | hidden-code flip rate (layer L) | max abs diff |
|---|---|---|---|---|---|---|
| w1024_heaviside_L1 | 262144 | 1 | 3.81e-06 | 1 | 1.49e-08 | 3.76e-02 |
| w1024_heaviside_L2 | 262144 | 2 | 7.63e-06 | 1 | 2.61e-07 | 1.70e-01 |
| w1024_heaviside_L3 | 262144 | 5 | 1.91e-05 | 4 | 1.44e-06 | 3.65e-01 |
| w1024_heaviside_L4 | 262144 | 9 | 3.43e-05 | 31 | 4.69e-06 | 5.59e-01 |
| w1024_relu_L1 | 262144 | 0 | 0.00e+00 | 0 | 1.49e-08 | 7.73e-07 |
| w1024_relu_L2 | 262144 | 0 | 0.00e+00 | 0 | 8.94e-08 | 2.02e-06 |
| w1024_relu_L3 | 262144 | 0 | 0.00e+00 | 0 | 1.68e-07 | 2.56e-06 |
| w1024_relu_L4 | 262144 | 0 | 0.00e+00 | 0 | 1.90e-07 | 3.38e-06 |
| w4096_heaviside_L1 | 2097152 | 0 | 0.00e+00 | 0 | 6.40e-09 | 4.99e-02 |
| w4096_heaviside_L2 | 2097152 | 34 | 1.62e-05 | 59 | 3.21e-07 | 2.21e-01 |
| w4096_heaviside_L3 | 2097152 | 136 | 6.48e-05 | 230 | 5.10e-06 | 7.87e-01 |
| w4096_heaviside_L4 | 2097152 | 381 | 1.82e-04 | 474 | 2.70e-05 | 1.04e+00 |
| w4096_relu_L1 | 2097152 | 3 | 1.43e-06 | 6 | 6.40e-09 | 6.48e-07 |
| w4096_relu_L2 | 2097152 | 5 | 2.38e-06 | 10 | 1.40e-07 | 4.11e-06 |
| w4096_relu_L3 | 2097152 | 3 | 1.43e-06 | 6 | 2.97e-07 | 6.13e-06 |
| w4096_relu_L4 | 2097152 | 3 | 1.43e-06 | 4 | 3.73e-07 | 6.93e-06 |

### Draw-to-draw scatter, width 1024 on 128³ (calibrated with k = 0)

| act | L | seeds | 3D raw per draw | raw mean ± sd | calibrated mean ± sd |
|---|---|---|---|---|---|
| heaviside | 1 | 5 | 2.546, 2.395, 2.539, 2.424, 2.457 | 2.472 ± 0.061 | 2.483 ± 0.071 (5/5 in range) |
| heaviside | 2 | 5 | 2.788, 2.795, 2.811, 2.689, 2.815 | 2.780 ± 0.046 | 2.794 ± 0.048 (5/5 in range) |
| heaviside | 3 | 5 | 2.884, 2.935, 2.946, 2.934, 2.944 | 2.929 ± 0.023 | 2.919 ± 0.000 (1/5 in range) |
| heaviside | 4 | 5 | 2.967, 2.978, 2.980, 2.974, 2.970 | 2.974 ± 0.005 | out of range |
| relu | 1 | 5 | 2.077, 2.071, 2.028, 2.018, 1.962 | 2.031 ± 0.042 | out of range |
| relu | 2 | 5 | 2.044, 2.035, 2.030, 2.048, 2.050 | 2.041 ± 0.008 | out of range |
| relu | 3 | 5 | 2.037, 2.008, 2.061, 2.059, 2.011 | 2.035 ± 0.023 | out of range |
| relu | 4 | 5 | 2.050, 2.041, 2.041, 2.012, 2.058 | 2.040 ± 0.016 | out of range |

### Resolution doubling (width 4096: 256³ vs its 128³ subsample, same net)

- **Calibrated 3D D (256³ / 128³):** L1 2.393 / 2.368, L2 2.856 / 2.861. The two resolutions agree within errors.
- **Slice-calibrated D (256³ / 128³):** L1 2.518 / 2.509, L2 2.815 / 2.817.
- **Boundary box counts at the same physical box size,** as the ratio 256³ / 128³ at ε = 2h, 4h, 8h, 16h, 32h, 64h:
  - Heaviside L1: 1.21 1.10 1.06 1.04 1.02 1.00
  - Heaviside L2: 1.28 1.11 1.04 1.01 1.00 1.00
  - Heaviside L3: 1.26 1.06 1.01 1.00 1.00 1.00
  - Heaviside L4: 1.18 1.02 1.00 1.00 1.00 1.00
  - ReLU (all L): 0.93–0.96 at 2h, rising to 1.00 at 64h.
- **Interpretation:** the Heaviside surfaces gain surface when the grid is refined, which is what real structure does. ReLU converges from slightly above: staircase voxelisation of a smooth surface overcounts at coarse resolution.

### float32 / float64

Output sign-flip rates at the volume median, on 32 central z-planes (2.1M voxels at width 4096):
- Heaviside: **0 (L1), 1.6e-5 (L2), 6.5e-5 (L3), 1.8e-4 (L4)**.
- ReLU: 1.4e-6 to 2.4e-6.
- Hidden-code flips at layer L range from 6.4e-9 to 2.7e-5. Heaviside flips cascade: one flipped unit moves the output by √(2/n)|v_i|.

float32 is adequate for D; boundary voxels that differ are ≤ 474 of ~10⁶. Full table above.

### Throughput (shared GB10, driver 580.173.02, CUDA 13.0, torch 2.14.0+cu130, float32, 2026-09-15)

- **Width 4096, 256³:** 6.3e4 voxels/s. Each voxel gets both activations and all L = 1–4 outputs in one pass: 6 × 4096² products, about 12.7 TFLOP/s. The sweep took 266 s (1.0 s per z-plane).
- **Float64 slab plus float32 codes:** about 18 min.
- **Width 1024, 128³:** 5.9e5 voxels/s, 14 s end to end.
- **Comparison with the spec:** the spec's "≈ 1 h" estimate was pessimistic for the sweep itself.

## Previews (opened and checked)

- `/home/fzeng/ml/research/art/rough-skin/cache/preview/slices_w1024_r128.png`
  - The Heaviside coastlines roughen with L. At the 256³ spacing, oblique exact slices show polygonal great-sphere facets, which is the width-1024 cutoff made visible.
  - ReLU shows one smooth interface at every L.
- `/home/fzeng/ml/research/art/rough-skin/cache/preview/slices_w4096_r256.png`
  - Heaviside L1 shows large islands with crinkled coasts; L2 is foam-like; L3 and L4 are close to noise at the voxel scale.
  - ReLU is smooth. The facets are no longer visible.

## Decisions

All decisions are logged in `art/rough-skin/NOTES.md`. The load-bearing ones:
- **Level:** the volume median.
- **Resolution doubling:** the 128³ grid is the even subsample of the 256³ grid, so doubling is tested on one net.
- **Sub-voxel octaves:** k matched to the ≈ 4/n cutoff.
- **Calibration:** the window protocol is primary and the periodic protocol is a systematic.
- **Fit ranges:** b = 2–64 (256³) and 2–32 (128³).
- **Slices:** linear interpolation from the volume, cross-checked by exact net slices.
- **ReLU:** compared with the smooth null rather than calibrated. ReLU lies below the power-law calibration range, where the smooth null reads 2.03–2.06 in 3D.
- **Out-of-range values:** L ≥ 3 values are reported as out of range rather than extrapolated.
- **Draw scatter:** 4 extra width-1024 draws.

## Risks and open issues

1. **The L1 gate misses narrowly.** The primary 3D value is 2.393 ± 0.058 (calibration) on a single draw. Width-1024 draws scatter by sd 0.07, so one or two more width-4096 draws (≈ 5 min GPU each for the sweep, without the float64 slab) would settle whether this is a bias or the draw.
2. **L = 2 reads about 0.1 high** (2.86 against 2.75) at width 4096; the width-1024 draw mean is 2.79 ± 0.05. Finite nets may have extra sub-voxel roughness from the Heaviside code cascade, or aliasing that the k-matching does not capture.
3. **L ≥ 3 is not resolvable by 3D box counting at 256³.** Exact-D fields of 2.94 and 2.97 read 2.93 and 2.94, while the nets read 2.96–2.99. The dial beyond L = 2 must be carried by slices and the films (the spec's "foam" beauty risk is confirmed visually).
4. **Scale range:** the fits span 1.5 decades (256³, b = 2–64) and 1.2 decades (128³). Spec §11's two decades are not met by a single volume. The 128³ → 256³ doubling adds one octave, and M2/M3 zoom sub-volumes (GP bands) would be needed for more.
5. **Calibration depends on the protocol.** Window versus periodic shifts D by 0.04–0.08, and k shifts it by up to 0.08. These systematics are the same size as the gate.
6. **The slice estimator reads low before calibration,** by 0.06–0.15, and after calibration it agrees with 3D at L1 to within 0.13 (2.52 vs 2.39) and at L2 to within 0.04. Marstrand's theorem is an expectation for typical slices, not a guarantee.
7. **Not done in M1:** no GP-limit volumes and no 3D rendering (r3d is not needed). The §0.7 plate-reproduction check against `depth-roughness` is deferred to M2's slice atlas.
