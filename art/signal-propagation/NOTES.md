# NOTES: signal-propagation (Order and Chaos + Finite Width)

Living handoff (replaces PAUSED.md). `cache/` is gitignored; it survives locally but is not in git.

## Session 3 (agent 3) progress
- analyze_fractal.py now records `adj_corr` (label correlation of adjacent pixels) and saves `cache/null_labels_{label}_r256.npz`; A and B reports rerun (B rescheck 512: 1.16/1.85/1.67 vs 256: 1.08/1.87/1.70). Chain A adj corr 0.07-0.13 past x256; B 0.47 (x256) -> 0.87 (x262k).
- Correction: Apert mismatch is not exactly 0 at every level (max 0.35% at x16, 0 at most deep levels).
- New `render_boxcount_plate.py` -> gallery/verification_boxcount_plate.png (viewed, good). `render_verification.py` now 6 panels (adds neighbour corr + resolution check); viewed.
- render_zoom_plates.py poster captions dynamic. Test spectral poster from Bvideo/Bres2 looked great (x256 is the most intricate level).
- README section 5 written (tokens SYNC_PERT_TBD, RES1024_TBD to fill when Bpert/Bres3/Bplate2a/2b finish).
- Waiting on GPU: bvideo2 (levels to 7), bplate2b (running), bplate2a + batch2 (queued).

### Post-job steps for the successor (session 3 handoff, 12:5x)
GPU jobs left running via gpu_run.sh (do NOT relaunch): bvideo2 (1280 f32, windows x32..x4096, 8 levels, ~330 s/level; level 2 done at 12:40), bplate2b (1024 f64 x65536, running), bplate2a (1024 f64 x1024, queued), batch2 (Bpert 256 f64 2 windows, then Bres3 512 f64 x262144; queued). Check: `tail -n 2 logs/{bvideo2,bplate2a,bplate2b,batch2}.log`. Each writes its npz after every level.
```
cd /home/fzeng/ml/research/art/signal-propagation; PY=/home/fzeng/ml/research/art/.venv/bin/python; export OMP_NUM_THREADS=4
# 1. analysis (Bpert, Bres3, Bplate2a/b auto-detected by glob; ~10 s), then sheets
$PY analyze_fractal.py --tag B --label sync --null_levels 10
$PY render_verification.py; $PY render_boxcount_plate.py
# 2. fill README tokens: SYNC_PERT_TBD (Bpert mismatch_pert at x64 & x262144 from report) and RES1024_TBD (1024 slope_fine at x1024, x65536)
# 3. plates: poster x1,x16,x1024,x65536 (+ raw native PNGs for x256, x4096 from Bvideo2)
$PY render_zoom_plates.py --report cache/fractal_report_B_N100_sync.json --chain cache/zoom_Bvideo_N100_D1000_s0_f32_r1280.npz:0 cache/zoom_Bvideo_N100_D1000_s0_f32_r1280.npz:4 cache/zoom_Bplate2a_N100_D1000_s0_f64_r1024.npz:0 cache/zoom_Bplate2b_N100_D1000_s0_f64_r1024.npz:0
$PY render_zoom_plates.py --styles spectral,aurora,riso --prefix frontier_single --chain cache/zoom_Bvideo2_N100_D1000_s0_f32_r1280.npz:3 cache/zoom_Bvideo2_N100_D1000_s0_f32_r1280.npz:7   # raw x256/x4096; delete its sheet if unwanted
# 4. video (13 keyframes x1..x4096), check a few frames downscaled
$PY merge_chains.py --out cache/zoom_Bvideoall_N100_D1000_s0_f32_r1280.npz cache/zoom_Bvideo_N100_D1000_s0_f32_r1280.npz:0-4 cache/zoom_Bvideo2_N100_D1000_s0_f32_r1280.npz
$PY render_zoom_video.py --chain cache/zoom_Bvideoall_N100_D1000_s0_f32_r1280.npz --styles spectral,magma,ink --name deepzoom
# 5. README: check every gallery/ link exists; fill GPU time line in section 4; commit; final report
```
Test renders of all 5 plate styles with stand-in data were viewed and look good (x256 most intricate; magma deep levels low-contrast by design).
GPU time estimate so far (slot wall-clock, from logs): empirical_tanh 8670 s, trainability 4620 s, meanfield ~800 s (CPU), chains A/B/f32/pert ~7400 s, widths ~4900 s, Bvideo 2700 s, Bres/Bres2 2350 s, lyap 360 s, depth dial 150 s, bvideo2 so far ~1000 s, plus killed partial runs (unknown, ~1-2 h) => ~10-12 GPU-slot-hours before the current jobs.

## Session 2 (agent 2) progress
- Flow videos: colour now log10(1 - c) (was linear c, saturated); Spectral = per-side rank split; fixed a bug where the chi_1 phase mask was not flipped with the image. Small multiples use log10(1-c).
- Width: `compute_mf_thit.py` (CPU, mean-field t_hit) removes cancellation noise in the N = infinity tile; `width_as_time_{spectral,magma,ink}` + `width_seeds_grid_spectral` rendered and viewed. N = 640/1024 skipped (GPU budget).
- `render_zoom_plates.py` generalised: `--chain path:level ...` mixes chains/resolutions; sync label handled.
- GPU jobs launched 12:1x (logs/): `batch2` (Bpert 256 f64 + Bres3 512 f64), `bplate2a` (1024 f64, x1024), `bplate2b` (1024 f64, x65536), `bvideo2` (1280 f32, video levels 5..12 -> merge with zoom_Bvideo levels 0..4).

## Done (the results)

### Paper check
- arXiv:2508.03222 is real. Code: github.com/jon-dong/fractal-deep-info-prop.
- Setup: erf MLP; W~N(0,1)/√N and b~N(0,1), redrawn per layer. The **same random draw is used at every pixel (CRN)**.
- Frontier: L = |x1−x2|² of two independent inputs, binarised at τ.
- Box sizes are 1–49 px on one image, and the dimension reported is the **max over τ ∈ [1e-5, 1]**, which biases it upward. Paper's MLP value: 1.85.

### Part A (tanh)
- **Analytic map** (`cache/meanfield_tanh.npz`, 2400×1200, 800 s): the critical point σ_b²=0.05 → σ_w²=1.7610, which matches Schoenholz et al.
- **Measured map** (`cache/empirical_tanh.npz`, 480×240, N=1000, D=400, K=6, 8670 s), measured/analytic median ratios:

  | quantity | ordered side | chaotic side |
  |---|---|---|
  | q* | 0.997 | 0.997 |
  | c* | 1.000 | 1.003 |
  | χ₁ | 1.002 | 0.999 |
  | ξ_c | 0.99 | 0.83 |
  | ξ_q | 0.67 (under 1 layer, crude fit) | 0.64 |

  The ridge of maximum measured ξ_c sits a median 0.026 in σ_w² from the analytic line (3 px). The measured χ₁ sign agrees with theory on 99.7% of pixels.
- **Trainability** (`cache/trainability.npz`: MNIST, width 300, σ_b²=0.05, 1000 SGD steps): the largest trainable depth follows 6ξ_c at σ_w² = 1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0. Near criticality (1.76) every net deeper than 100 fails; this is probably the step budget.

### Part B (erf, N=100, D=1000)
Reports are `cache/fractal_report_{A_N100_tau,B_N100_sync}.json`.

**Chain A** (paper threshold L_avg>1e-5; f64 256², ×4 per level, centred at maximal mixing):
- Local slope by zoom: 1.14 (×1), 1.49 (×4), 1.70 (×16), 1.92 (×64), then 1.98–1.99 from ×256 to ×65536.
- Past ×256, adjacent pixels are nearly uncorrelated (corr 0.07–0.2). The frontier fills an area; it is not a curve with a stable non-integer dimension.
- f32 vs f64 label mismatch rises from 0.1% to 43%. f64 with inputs perturbed by 1e-13 gives **0 mismatch** at every level, so the structure is deterministic and not roundoff.

**Chain B** (sync label: the pair never reached L<1e-10 within D; f64 256²):
- Laminated filaments at deep zoom.
- Local slope by zoom: 1.08 (×1), 1.39, 1.60, 1.79, 1.87 (×256, the peak), 1.82, 1.70, 1.65, 1.52, 1.48 (×262,000). The slope falls as the stripes become resolved (adjacent corr 0.58 → 0.87), consistent with a smooth analytic boundary below ~1e-8 at finite depth.
- f32 mismatch 0.02% → 17%, but f32 slopes agree with f64 to within 0.04.

**Also:**
- Null model (mean field, same pipeline, own zoom chain): slope 0.98–1.03 at every level.
- Stitched N(ε) over 6.9 decades: global slope 1.97. This is upper-biased because the windows are chosen at maximal mixing.
- Dimension vs τ: at ×4, the slope over τ from 1e-12 to 1 spans 1.33–1.75.
- Width maps: `cache/widthmap_widths_N{8..512}_s0` plus seeds 1,2 for N=20,100,320; the listing shows 25 of the 29 planned (N=640 and 1024 not done).
- Extras:
  - Depth dial: `cache/depthdial_N100_r512.npz`.
  - Lyapunov map: `cache/lyap_full_N100...r1024.npz`; 47.1% of pixels have λ>0. The λ field is smooth, so the frontier's roughness is not in λ.

### Gallery rendered and viewed
- `phase_spectral_analytic` (hero), `phase_spectral_measured`, `phase_ridge_magma_measured` (+ `_with_analytic_line`), `phase_riso_measured`, `phase_split_{aurora_ember,indigo_madder}_measured`, `phase_contours_paper`, `phase_atlas_analytic_vs_measured`
- `trainability_check`
- `lyapunov_{spectral,cyanotype_vandyke,vik_linear}`
- `depth_dial_{spectral,magma,ink}.{mp4,gif}` + `depth_dial_small_multiples_spectral`
- `flow_correlation_{oslo,spectral,cyanotype}.{mp4,gif}` + small multiples (rendered, **not yet viewed**)
- `verification_sheet` (draft; missing the Bpert squares and the resolution-check markers)

## Next
1. Resume the killed GPU jobs; each writes after every level, and partial files exist:
   - `Bplate1` / `Bplate2`: 1024² f64, windows `cache/win_Bplate{1,2}.json`.
   - `Bvideo`: f32 1280², 13 windows, 5 done. Add `--centers` for the rest, or rerun all.
   - `Bres2`: 512² f64, 2 of 3 done.
   - `Bpert`.
   - `widths2`: N=640 and 1024 left.
2. `analyze_fractal.py --tag B --label sync --null_levels 10`, then `render_verification.py`.
3. Renders:
   - `render_zoom_plates.py --chain <1024 plates merged> --panels ...` (Spectral, aurora, magma, ink, riso)
   - `render_zoom_video.py --chain cache/zoom_Bvideo_...npz`
   - `render_width.py`
4. View the flow videos; write README.md (all numbers above), with an "Ideas explored / not pursued" section:
   - built: Lyapunov map, depth dial
   - not pursued: backprop version, CNN/FDF, uncertainty-exponent estimator, 3D width stack, sonification
5. README link check; final commit.

## Resume commands
```
cd /home/fzeng/ml/research/art/signal-propagation; PY=/home/fzeng/ml/research/art/.venv/bin/python; G=../_shared/gpu_run.sh
$G $PY compute_zoom_chain.py --N 100 --res 1024 --dtype f64 --tag Bplate1 --label sync --centers cache/win_Bplate1.json &
$G $PY compute_zoom_chain.py --N 100 --res 1024 --dtype f64 --tag Bplate2 --label sync --centers cache/win_Bplate2.json &
$G $PY compute_zoom_chain.py --N 100 --res 1280 --dtype f32 --tag Bvideo --label sync --centers cache/win_Bvideo.json &
$G $PY compute_zoom_chain.py --N 100 --res 512 --dtype f64 --tag Bres2 --label sync --centers cache/win_Bres2.json &
$G $PY compute_zoom_chain.py --N 100 --res 256 --dtype f64 --tag Bpert --label sync --perturb 1e-13 --centers cache/win_Bpert.json &
$G ./logs/widths2.sh &     # skips finished widths
```
