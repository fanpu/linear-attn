# Filter-Normalized: the loss landscape, surveyed honestly

*Four CIFAR-10 ResNets, their loss measured on a grid of weight perturbations and drawn the way a survey office draws a mountain: contour sheets, hachures, raking light and hypsometric tints, with no rainbow 3-D surfaces.*

> **Read this before looking at any picture.** Every map here is a **2-D slice** through a loss surface with 269,722 to 855,770 dimensions. When a slice is non-convex, the full loss is non-convex too. When a slice looks smooth, that says very little about the other ~10⁶ directions. Loss "sharpness" is also not reparametrization-invariant (Dinh et al. 2017). Filter normalization only removes the per-filter scale symmetry, so a wide basin in these units does not by itself mean good generalization. Li et al. (2018) warn that 1-D interpolation plots in particular can mislead.

**Status:** complete for this round: 51² surveys of all four nets, 101² heroes for ResNet-56 with and without shortcuts, zoom test, PCA trails, 1-D checkpoint slices and film, STL heightfields. **Headline: our ResNet-56-noshort slices are smooth at every scale we measured, from spacing 0.04 down to 5·10⁻⁶. This does not reproduce Li et al.'s chaotic no-shortcut landscape** (see *Why ours differs*).

## The phenomenon

Li et al. (NeurIPS 2018) choose two random Gaussian directions δ, η in weight space. Each filter of each direction is then rescaled to the norm of the matching filter of the trained weights w\*:

$$\delta_{i,j} \leftarrow \frac{\delta_{i,j}}{\lVert\delta_{i,j}\rVert}\,\lVert w^*_{i,j}\rVert, \qquad f(\alpha,\beta) = \mathcal L\left(w^* + \alpha\,\delta + \beta\,\eta\right).$$

BN and bias entries of the directions are set to zero. This removes the ReLU/BN scale ambiguity that otherwise makes apparent sharpness meaningless. Their well-known result is that deep networks *without* skip connections (ResNet-56-noshort) have chaotic, crumpled slices, while with shortcuts the slices become smooth, nearly convex bowls.

## Hero: ResNet-56 with and without shortcuts, 101² (spacing 0.02)

<img src="gallery/survey_resnet56_g101.png" width="49%"> <img src="gallery/survey_resnet56_noshort_g101.png" width="49%">

*Contour survey sheets. Contours every 1/10 decade of training loss on the shared clip [0.08, 150], bold labelled index contours, red dashed line at chance level (ln 10), graticule and title block. The heights are measured; ink, paper and layout are declared.*

<img src="gallery/hachure_dark_resnet56_g101.png" width="49%"> <img src="gallery/hachure_dark_resnet56_noshort_g101.png" width="49%">

*Lehmann hachures on a dark ground. The strokes follow the fall line of log10 loss between 1/10-decade contours, and stroke weight grows with slope.*

<img src="gallery/spectral_resnet56_g101.png" width="49%"> <img src="gallery/spectral_resnet56_noshort_g101.png" width="49%">

*Sohl-Dickstein Spectral split at the chance-level contour. The basin (L < ln 10) runs purple→pale and everything above chance runs deep red→pale. Each side is rank-normalized separately, so the dark seam is exactly the L = ln 10 contour. The colour mapping is a declared aesthetic choice.*

## Gallery: 101² pair

| style | files | measured | aesthetic (declared) |
|---|---|---|---|
| contour survey | `survey_resnet56{,_noshort}_g101.png` | log10 train loss | one ink + red chance line, title block |
| hachures (paper / dark) | `hachure{,_dark}_*_g101.png` | fall line and slope | Lehmann strokes, weight ∝ slope |
| raking-light hillshade | `hillshade_*_g101.png` | aspect and slope of log10 loss | NW light, aspect shading with slope saturated at its median, Tanaka illuminated contours every 1/10 decade |
| hypsometric tint | `hypsometric_*_g101.png` | log10 loss in 24 bands | Imhof-like ramp × hillshade |
| Spectral / palettes.py | `spectral_*_g101.png`, `hubble_*_g101.png` | sign of log(L / ln 10) | rank-normalized split palettes (`sd_spectral`, `hubble_sho`) |
| curvature | `curvature_split_g101.png` | λ_min of the Hessian of L on the slice | Spectral split at λ_min = 0 |
| ridgelines | `ridge_{dark,paper}_*_g101.png` | 41 rows β = const of the 101² grid | inverted (low loss up), Joy-Division stacking |
| STL heightfields | `gallery/stl/resnet56{,_noshort}_g101.stl`, `stl_preview_g101.png` | log10 loss, 241² cubic upsample | 150 mm tile, 40 mm relief, 6 mm base; 232k triangles, 11.6 MB each |
| light-sweep film | `sweep_g101.mp4` / `.gif` | the same two surfaces | light azimuth circles 360° over 12 s |

<img src="gallery/hillshade_resnet56_g101.png" width="49%"> <img src="gallery/hillshade_resnet56_noshort_g101.png" width="49%">

<img src="gallery/hypsometric_resnet56_g101.png" width="32%"> <img src="gallery/hubble_resnet56_noshort_g101.png" width="32%"> <img src="gallery/hachure_resnet56_noshort_g101.png" width="32%">

<img src="gallery/ridge_dark_resnet56_g101.png" width="32%"> <img src="gallery/ridge_dark_resnet56_noshort_g101.png" width="32%"> <img src="gallery/ridge_paper_resnet56_noshort_g101.png" width="32%">

<video src="gallery/sweep_g101.mp4" autoplay loop muted playsinline width="100%"></video>

<img src="gallery/stl_preview_g101.png" width="100%">

<img src="gallery/curvature_split_g101.png" width="100%">

*Principal-curvature map. The red side is locally convex (λ_min > 0); the purple side has negative curvature. Inside the chance contour only 0.3% (ResNet-56) and 5.3% (noshort) of points are non-convex. Outside it, most of the slice curves downward, because the loss saturates toward a plateau.*

## Training trail and checkpoints

<img src="gallery/pca_trail_resnet56.png" width="49%"> <img src="gallery/pca_trail_resnet56_noshort.png" width="49%">

*The training trajectory drawn on the plane of the top-2 PCs of its own checkpoints (41² grid, final BN statistics). Grey marks loss above 150, with dashed decade contours up to 10³⁰. The early checkpoints lie far outside the final basin because the weight norm shrinks by a large factor during training (see Verification). The noshort plane starts at epoch 10.*

<video src="gallery/ckpt_slices.mp4" autoplay loop muted playsinline width="100%"></video>

<img src="gallery/ckpt_ridge.png" width="100%">

*One filter-normalized 1-D slice (101 points) through the weights at 16 saved epochs. Film frames between saved epochs are linear blends in log loss (declared). With shortcuts, a valley exists from epoch 1 and deepens steadily. Without shortcuts, the slice stays a near-flat plateau for the first ~10 epochs, with a wavy, lumpy profile, and only forms a clear valley after the lr decays at epochs 20 and 30.*

## Gallery: 51² surveys, all four models (spacing 0.04)

| style | measured | aesthetic (declared) |
|---|---|---|
| `survey_*` contour sheets | log10 train loss, shared clip [0.08, 150] | one ink + red chance line, title block, graticule |
| `hachure_*`, `hachure_dark_*` | fall line and slope of log10 loss | Lehmann-style strokes between 1/10-decade contours, weight ∝ slope |
| `hillshade_*` | log10 loss | NW raking light (aspect shading) + Tanaka illuminated contours |
| `hypsometric_*` | log10 loss in 24 bands | declared Imhof-like ramp × hillshade |
| `spectral_*`, `hubble_*` | sign of log(L/ln 10) | rank-normalized split palettes (Spectral; palettes.py `hubble_sho`) |
| `curvature_split_g51.png` | smaller principal curvature λ_min of L on the slice | Spectral split at λ_min = 0 |
| `atlas_g51.png` | all four sheets on one shared scale | 2×2 atlas |

<img src="gallery/atlas_g51.png" width="100%">

<img src="gallery/hachure_resnet56_noshort_g51.png" width="32%"> <img src="gallery/hillshade_resnet56_noshort_g51.png" width="32%"> <img src="gallery/hypsometric_resnet56_noshort_g51.png" width="32%">

<img src="gallery/curvature_split_g51.png" width="100%">

*Principal-curvature map. The red side is locally convex (λ_min > 0); the purple side has negative curvature. Inside the chance contour only 3–5% of grid points are non-convex. Outside it, most of the slice curves downward, because the loss is saturating toward a plateau.*

<img src="gallery/zoom_test.png" width="49%"> <img src="gallery/slices_1d.png" width="49%">

## What was computed

- **Models:** the Li et al. CIFAR ResNets (16-32-64 channels, BasicBlocks, 1×1 projection shortcuts). The `noshort` variants drop every shortcut. Code: `common.py`.
- **Training** (`train.py`): SGD with Nesterov momentum 0.9, lr 0.1, wd 5e-4, batch 128, standard augmentation, bf16 autocast, seed 0. **40 epochs** instead of Li et al.'s 300, with ×0.1 decays at epochs 20/30/37 (the same fractions). Training ran for 17–30 min per model.

| model | params | train acc | test acc | train loss |
|---|---|---|---|---|
| ResNet-20 | 272,474 | 95.6% | 90.4% | 0.131 |
| ResNet-20 noshort | 269,722 | 92.9% | 87.9% | 0.205 |
| ResNet-56 | 855,770 | 96.8% | 90.9% | 0.097 |
| ResNet-56 noshort | 853,018 | **81.8%** | 78.8% | 0.522 |

  In 40 epochs the 56-layer net without shortcuts could not fit the training set, which is the trainability half of Li et al.'s story.
- **Surfaces** (`landscape.py`): training cross-entropy on a fixed subset of **1000** training images (`fixed_subset(1000)`, seed 0), BN in eval mode, fp32 with TF32 disabled. Directions come from seeds 1 and 2, filter-normalized per checkpoint. The grid is α, β ∈ [−1, 1], 51² (spacing 0.04), and the heroes are 101² (spacing 0.02, reusing the 51² points, run as 2 shards merged by `merge_shards.py`). Speed on the contended GB10 was ~1.7 points/s for ResNet-56 and ~4 points/s for ResNet-20. The 51² ResNet-56-noshort surface took 26 min.
  The subset size was cut from the full set to 1000 so the job fits a shared GPU. On a 101-point slice of ResNet-56-noshort, n = 1000 and n = 5000 agree to a median |log10 ratio| of **0.0064** (1.5% in loss), with a maximum of 0.030 (7%). The subset does not create or hide structure at this scale.
- **Zoom test:** 201-point 1-D windows along β = 0 centred on α = 0.5, with half-widths 0.5, 0.05, 0.005 and 0.0005. The two finest are in float64.
- **PCA planes:** 41² grids over the checkpoint PCA plane (`pca_dirs.py`; ResNet-56 all epochs, noshort epochs ≥ 10).
- **GPU time:** training ≈ 1.5 h (4 × 17–30 min). Surfaces, lines, zooms and PCA ≈ 4.2 h of slot wall time on the shared GB10 (sum of `meta.wall_s`).
- **Analysis:** `analyze_surf.py` computes principal curvatures by finite differences, strict local minima and quadratic-fit residuals, and writes `cache/surf_stats_<tag>.json`. `pca_dirs.py` computes the PCA directions of the checkpoint trajectory.
- **Rendering:** `render_maps.py` and `render_extra.py`, CPU only, from the cache.

```bash
PY=/home/fzeng/ml/research/art/.venv/bin/python; G=/home/fzeng/ml/research/art/_shared/gpu_run.sh
$G $PY train.py --depth 56 --noshort --epochs 40
$G $PY landscape.py --n 1000 --model resnet56_noshort --res 51 --tag g51
$PY analyze_surf.py g51 && $PY render_maps.py --tag g51 && $PY render_extra.py curv --tag g51
$PY merge_shards.py resnet56_noshort_final_g101 resnet56_final_g101
$PY analyze_surf.py g101 && $PY render_maps.py --tag g101 --models resnet56 resnet56_noshort
for w in stl ridge sweep curv zoom lines pca ckfilm; do $PY render_extra.py $w --tag g101; done
# full job list: jobs/chainA.sh .. jobs/chainD.sh (chainD reruns the PCA planes)
```

## Verification / honesty

**Slice statistics** (`cache/surf_stats_g51.json`, `_g101.json`):

| model | centre loss | strict local minima (interior) | basin (L < ln 10) area | non-convex fraction inside basin | quad-fit RMS residual (log10) |
|---|---|---|---|---|---|
| ResNet-20 | 0.124 | 2 (1 in basin) | 17.0% | 2.7% | 0.227 |
| ResNet-20 noshort | 0.207 | 1 | 22.2% | 4.2% | 0.187 |
| ResNet-56 | 0.093 | 2 (1 in basin) | 16.8% | 0.0% | 0.234 |
| **101²** ResNet-56 | 0.093 | 2 (1 in basin) | 17.2% | 0.3% | 0.234 |
| **101²** ResNet-56 noshort | 0.528 | 1 | 7.3% | 5.3% | 0.198 |
| ResNet-56 noshort | 0.528 | 1 | 7.1% | 4.9% | 0.198 |

**Negative result against Li et al.:** at 0.04 and 0.02 spacing, our ResNet-56-noshort slice is **not chaotic**. It shows one elongated, tilted valley with a single minimum, wavy non-convex shoulders and a basin much *narrower* (7% vs 17–22% of the square) than the shallower nets. It does not reproduce the crumpled terrain in Li et al.'s Fig. 5. Likely reasons, none tested yet:
1. The schedule is 7.5× shorter. Our noshort-56 stopped at 81.8% train accuracy, far from the near-interpolating minimum Li et al. surveyed.
2. The loss is measured on a 1000-image subset.
3. It is one pair of random directions.

**Resolution check (101² vs 51²):** cubic interpolation of the 51² grid predicts the 7,600 new 101² points with a median error of 4·10⁻⁵ (noshort) / 8·10⁻⁵ (ResNet-56) in log10 loss. The maximum error is 0.008 / 0.012, under 0.5% of the 2.3–2.7-decade range. Halving the spacing reveals no hidden structure.

**Zoom test** (`cache/zoom_test.json`, `zoom_test.png`, `zoom_roughness.png`):

| model | half-width | spacing | dtype | log10-loss range | slope sign changes | RMS Δ² | RMS Δ² / s² |
|---|---|---|---|---|---|---|---|
| ResNet-56 | 0.5 | 5e-3 | fp32 | 2.23 | 0 | 4.4e-4 | 17.6 |
| | 0.05 | 5e-4 | fp32 | 0.141 | 0 | 2.1e-6 | 8.4 |
| | 0.005 | 5e-5 | fp64 | 0.0141 | 0 | 4.6e-8 | 18.5 |
| | 0.0005 | 5e-6 | fp64 | 0.00141 | 0 | 1.3e-9 | 50 |
| ResNet-56 noshort | 0.5 | 5e-3 | fp32 | 1.36 | 1 (the basin floor) | 3.2e-4 | 13 |
| | 0.05 | 5e-4 | fp32 | 0.0952 | 0 | 4.4e-7 | 1.8 |
| | 0.005 | 5e-5 | fp64 | 0.00956 | 0 | 1.3e-8 | 5.3 |
| | 0.0005 | 5e-6 | fp64 | 0.000956 | 0 | 4.6e-10 | 18 |

**Verdict:**
- **No persistent no-shortcut roughness.** In every window below 0.5 both profiles are strictly monotone (zero slope sign changes). The window range shrinks exactly 10× per 10× zoom, which is what a locally linear, differentiable function does.
- **A roughness exponent below 2 at the finest scales.** The RMS second difference falls as s^1.61 (ResNet-56) and s^1.49 (noshort) over the three finest windows. A C² function gives s²; a rough, fractal-like profile would give an exponent near 0–1. So there is a small excess texture at spacings ≤ 5·10⁻⁵, about 10⁻⁹ in log10 loss. That is consistent with ReLU activation-pattern kinks (the loss is only piecewise smooth) or evaluation noise.
- **The excess is not a no-shortcut property.** It is smaller for noshort than for ResNet-56, and it is invisible at any rendered scale.

We make no fractal claim.

### Why ours differs from Li et al.

Li et al.'s Fig. 5 shows ResNet-56-noshort as a crumpled, many-minima landscape at the same filter-normalized scale. Ours is one tilted valley. The candidate explanations, in the order we believe them:

1. **Under-training.** We trained for 40 epochs; Li et al. trained for 300. Our noshort-56 stopped at **81.8% train accuracy** (loss 0.52), on the shoulder of a wide region rather than at a sharp, near-interpolating minimum. The checkpoint film supports this: the noshort slice is a lumpy plateau for the first ~10 epochs and only carves a valley once the lr decays. Longer training would likely sharpen that valley and could expose the chaotic walls. **Not tested** (it needs a ~150–300-epoch retrain).
2. **Direction pair and seed.** We use one pair of random directions (seeds 1, 2) and one training seed. Li et al. also show one slice per net.
3. **Subset loss:** **ruled out** at this scale. n = 1000 and n = 5000 agree to 1.5% (median).
4. **Resolution or precision:** **ruled out.** The 101² grid matches the 51² interpolant, and the float64 zooms are monotone.

What *does* reproduce: without shortcuts the basin is 2.4× smaller in area (7.3% vs 17.2%), its floor is 5.7× higher (0.53 vs 0.093), it is strongly anisotropic, and it has 17× the non-convex fraction inside the basin (5.3% vs 0.3%). The trainability gap (81.8% vs 96.8% train accuracy) is the other half of their story.

**PCA trajectory surprise:** for ResNet-56-noshort, PCA over all checkpoints puts 99.3% of the variance on one component. Epoch-1 weights sit at distance 412 from the final weights: a large early growth of weight norm that weight decay then shrinks, which is harmless for BN nets because of scale invariance. That plane just shows radial shrinkage, so the trail sheet uses epochs ≥ 10 (92.1% / 5.4%). ResNet-56 over all epochs: 76.6% / 9.8%. Because BN running statistics are the final ones, early checkpoints on the PCA plane evaluate to losses up to 10³³ (grey region). That reflects the weight-norm mismatch, not a real barrier crossed during training.

## Caveats

- **Slice, not landscape** (see the box at the top). Non-convexity in a slice implies global non-convexity; smoothness implies little. Dinh et al. (2017) show that sharpness can be changed arbitrarily by reparametrization, so no "flat = generalizes" reading is licensed.
- The height is **log10** training loss with a shared clip [0.08, 150] across models, which exaggerates the bottom of the bowl relative to linear-loss plots.
- Cubic-spline upsampling of the grid (51² or 101² up to 600–1600 px) is a rendering step. Structure finer than the grid spacing is not measured.
- BN is in eval mode with the running statistics of the unperturbed net, as in Li et al.'s code. Perturbed nets would be scored differently if BN statistics were recomputed.
- 40-epoch training, one seed, one direction pair, a 1000-image subset.
- The PCA directions are unit-norm, not filter-normalized, so the PCA-plane axes are not comparable to the α, β axes of the survey sheets.
- The zoom test samples only one line (β = 0, around α = 0.5, on the valley wall). A rough region elsewhere cannot be excluded.
- Hillshade, hachure, ridgeline inversion and STL relief are rendering choices; only the loss values are measured.

## References

- H. Li, Z. Xu, G. Taylor, C. Studer, T. Goldstein. *Visualizing the Loss Landscape of Neural Nets.* NeurIPS 2018. Code: github.com/tomgoldstein/loss-landscape.
- L. Dinh, R. Pascanu, S. Bengio, Y. Bengio. *Sharp Minima Can Generalize for Deep Nets.* ICML 2017.
- I. Goodfellow, O. Vinyals, A. Saxe. *Qualitatively characterizing neural network optimization problems.* ICLR 2015.
- J. Sohl-Dickstein. *The boundary of neural network trainability is fractal.* 2024 (Spectral split style).
- E. Imhof. *Cartographic Relief Presentation.* 1965. J. G. Lehmann's hachure system (1799).
