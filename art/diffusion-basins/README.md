# Which Dog: basins of attraction of diffusion samplers

*Colour every starting noise vector by what the sampler turns it into. The deterministic samplers paint smooth, stretched countries. Iterating the denoiser as a fixed-point map instead paints Newton-style fractal flowers that repeat every 9 zoom levels.*

<p align="center">
<img src="gallery/zooms/zoom_iter_ring8_plate.png" width="49%">
<img src="gallery/mnist/mnist_basin_confidence.png" width="49%">
</p>

<p align="center">
<video src="gallery/zooms/zoom_iter_ring8.mp4" autoplay loop muted playsinline width="49%"></video>
<img src="gallery/mnist/mnist_mosaic_boundary.png" width="49%">
</p>

## 1. The phenomenon

A diffusion model turns noise into data. Fix a sampler and every noise vector **z** has an outcome: which mixture mode it lands in (2D toys), which digit it becomes (MNIST, as judged by a classifier), or which training image it reproduces (a memorising model). Colouring a 2D slice through noise space by that outcome gives a **basin map**. The question is whether the boundaries between basins are smooth curves or fractals.

We use a VP forward process with linear β(t) = 0.1 + 19.9 t, α̅(t) = exp(−∫β), σ(t)² = (1−α̅)/α̅. The three families of maps are:

| map | update | invertible? | expected boundary |
|---|---|---|---|
| **probability-flow ODE** (RK4, 100 steps) | dx/dt = −½β(t)[x + ∇log p_t(x)] | yes: flow of a smooth vector field | smooth, possibly stretched |
| **DDIM, N steps** | x_{k+1} = √α̅_{k+1} x̂₀(x_k) + √(1−α̅_{k+1}) ε̂(x_k) | yes for small steps (near-identity) | smooth |
| **DDPM with frozen noise** | ancestral step + fixed injected noise sequence | yes, but it forgets x_T | start noise irrelevant |
| **iterated (over-relaxed) denoiser** | x ← x + γ (D(x, σ) − x), repeated 2000× | **no**: folds for γ ≳ 1.06 | can be fractal |

Here D(x, σ) = E[x₀ | x_σ = x] is the Tweedie denoiser. For the toys it is exact (Gaussian mixture, component std s = 0.08–0.10).

**Why the samplers cannot be fractal (the sampler is invertible).** The PF-ODE is the flow of a smooth, Lipschitz vector field over a finite time. By Picard–Lindelöf, trajectories never cross, and integrating backwards inverts the map, so z ↦ x₀ is a diffeomorphism. The outcome label is a smooth partition of x₀ space (nearest mode, or classifier argmax) pulled back through that diffeomorphism. The pullback of a smooth curve under a diffeomorphism is a smooth curve, so the boundary has dimension 1. It can be *stretched*: the Jacobian's largest singular value ranges over 10^−1…10^2 here, which bends and thins the countries. But it cannot be *folded*, and fractal basin boundaries (Newton, Julia, Wada) need a many-to-one map iterated without end, so that preimages of the boundary pile up. DDIM is a finite composition of near-identity smooth steps and behaves the same way. The **iterated denoiser** is the exception. With over-relaxation γ > 1 its Jacobian I + γ(∂D/∂x − I) changes sign across curves (folds), and 2000 iterations accumulate preimages. That is exactly Newton's method on a mixture, and it gives Newton's fractals. A mode stays attracting while |1 − γ(1 − s²/(s²+σ²))| < 1, i.e. γ < 2.125 for ring8 at σ = 0.4. The heroes sit at γ = 2.10–2.12, just below that limit.

**Slices through noise must be norm-preserving.** In d = 784, real noise lives on a thin shell of radius √d. MNIST slices therefore use great-sphere coordinates z(a, b) = √d [cos ρ e₀ + sinc ρ (a u + b v)], with ρ = √(a²+b²) and (e₀, u, v) orthonormal (seed 2024). Every point is a typical noise vector.

## 2. Gallery

### 2a. The iterated denoiser: fractal (non-invertible map)

<table>
<tr>
<td><img src="gallery/iterated/iter_ring8_confidence.png"></td>
<td><img src="gallery/iterated/iter_ring8_spectral_orientation.png"></td>
<td><img src="gallery/iterated/iter_ring8_night.png"></td>
</tr>
<tr>
<td>Ring of 8, γ = 2.12, σ = 0.4, 2048² over [−4, 4]². Hue = fixed point reached (declared palette), brightness = rank of log iterations-to-converge (measured).</td>
<td><b>Spectral split</b> of orientation: sign det J of the 2000-fold map picks the half of Spectral, and rank-normalised log|det J| sets position within it. The dark seams are fold curves (measured, declared mapping).</td>
<td>Night style: basin hue with seam darkening by distance to boundary (aesthetic).</td>
</tr>
<tr>
<td><img src="gallery/iterated/iter_ring8_ink.png"></td>
<td><img src="gallery/iterated/iter_ring8_riso.png"></td>
<td><img src="gallery/iterated/iter_ring8_convergence_lajolla.png"></td>
</tr>
<tr>
<td>Single-ink plotter drawing of all basin boundaries.</td>
<td>Two-spot riso with deliberate misregistration (aesthetic).</td>
<td>Iterations to converge, cmcrameri lajolla (measured; boundaries converge slowest).</td>
</tr>
<tr>
<td><img src="gallery/iterated/iter_ring6_confidence.png"></td>
<td><img src="gallery/iterated/iter_ring6_spectral_orientation.png"></td>
<td><img src="gallery/iterated/iter_scatter12_confidence.png"></td>
</tr>
<tr>
<td>Ring of 6, γ = 2.10: flowers only at the 6-fold junctions.</td>
<td>Ring 6, Spectral orientation split.</td>
<td>12 scattered modes, γ = 2.10: nearly Voronoi, with a few tongues. Symmetry matters.</td>
</tr>
</table>

**Zoom, 36 levels ×2 (float64, width 8 → 1.2 × 10⁻¹⁰):**

<p align="center"><img src="gallery/zooms/zoom_iter_ring8_plate.png" width="80%"></p>

MP4 [zoom_iter_ring8.mp4](gallery/zooms/zoom_iter_ring8.mp4) (1080², 54 s) · GIF [zoom_iter_ring8.gif](gallery/zooms/zoom_iter_ring8.gif) (360 px).

**γ sweep film** (1.0 → 2.12): the invertible map (γ ≤ 1.06) has straight ray boundaries, and the flowers grow out of the junctions as γ approaches the stability limit.

<p align="center"><video src="gallery/animations/gamma_sweep_ring8_night.mp4" autoplay loop muted playsinline width="60%"></video><br><a href="gallery/animations/gamma_sweep_ring8_night.gif">GIF</a></p>

### 2b. The samplers: smooth but stretched (invertible maps)

<table>
<tr>
<td><img src="gallery/toy/atlas_analytic_confidence.png"></td>
<td><img src="gallery/toy/atlas_analytic_paper.png"></td>
</tr>
<tr>
<td>Atlas. Rows: scatter12 / ring8 / grid25. Columns: DDIM-10, DDIM-50, DDIM-1000, PF-ODE, DDPM-1000 frozen noise. Exact score, 1024² over the start-noise plane [−3, 3]². Brightness = 1 − d₁/d₂ sample-to-mode margin (measured). The DDPM column is a single colour because the start noise is forgotten.</td>
<td>The same atlas in paper and ink. Ring8 gives exactly straight rays: a symmetry null that is smooth by construction.</td>
</tr>
</table>

<table>
<tr>
<td><img src="gallery/toy/hero_ode_scatter12_confidence.png"></td>
<td><img src="gallery/toy/hero_ode_scatter12_riso.png"></td>
<td><img src="gallery/toy/hero_ode_scatter12_ink.png"></td>
</tr>
<tr><td>PF-ODE hero, scatter12, confidence shading.</td><td>Riso.</td><td>Ink.</td></tr>
<tr>
<td><img src="gallery/toy/stretch_ode_scatter12_fire.png"></td>
<td><img src="gallery/toy/stretch_ddim10_scatter12_fire.png"></td>
<td><img src="gallery/toy/hero_ode_scatter12_night.png"></td>
</tr>
<tr><td>Stretch: log₁₀ σ_max of the ODE sampler Jacobian (colorcet fire, measured). Range −0.97…2.0. det J < 0 on 6 × 10⁻⁵ of pixels (finite-difference noise).</td><td>The same for DDIM-10.</td><td>Night style.</td></tr>
</table>

<table>
<tr>
<td><img src="gallery/diptych/diptych_ode_vs_iterated_spectral.png"></td>
<td><img src="gallery/diptych/diptych_ode_vs_iterated_night.png"></td>
</tr>
<tr><td>Diptych, Spectral orientation split: the ODE (left) has no seams because it is orientation-preserving everywhere, while the iterated denoiser (right) is full of fold seams.</td><td>The same diptych, night basin style.</td></tr>
</table>

**Step-count film** (DDIM N = 1 … 300, scatter12, exact score): the countries rearrange for N < 10, then freeze into their ODE limit.

<p align="center"><video src="gallery/animations/steps_scatter12.mp4" autoplay loop muted playsinline width="55%"></video><br><a href="gallery/animations/steps_scatter12.gif">GIF</a></p>

**DDPM with frozen noise.** Slicing the start noise gives one colour, so we slice through the *injected* noise sequence instead (great sphere in ℝ^{2N}):

<table><tr>
<td><img src="gallery/toy/ddpm30_noise_slice_confidence.png"></td>
<td><img src="gallery/toy/ddpm1000_noise_slice_confidence.png"></td>
<td><img src="gallery/toy/ddpm1000_noise_slice_paper.png"></td>
</tr><tr><td>DDPM-30, injected-noise slice.</td><td>DDPM-1000.</td><td>DDPM-1000, paper.</td></tr></table>

[ddpm_noise_slices.png](gallery/toy/ddpm_noise_slices.png) shows both side by side.

### 2c. MNIST: which digit (DDIM-50 of a real DDPM)

<table>
<tr>
<td><img src="gallery/mnist/mnist_basin_confidence.png"></td>
<td><img src="gallery/mnist/mnist_basin_riso.png"></td>
</tr>
<tr>
<td>192² great-sphere slice (±π/2 rad). Hue = classifier argmax (declared palette), brightness = classifier max-probability (measured), bilinearly interpolated between samples to 2304².</td>
<td>Riso: blue density = exp(−margin/2), where margin is the top-1 vs top-2 log-odds (measured). Pink boundaries are misregistered by 5 px (aesthetic).</td>
</tr>
<tr>
<td><img src="gallery/mnist/mnist_basin_paper.png"></td>
<td><img src="gallery/mnist/mnist_margin_spectral_sequential.png"></td>
</tr>
<tr>
<td>Survey-sheet ink with a faint tint.</td>
<td>Spectral used as a <i>sequential</i> map of the log-odds margin (labelled variant: dark red = boundary).</td>
</tr>
</table>

**Thumbnail mosaics.** Every cell is the actual generated digit at that noise position.

<table><tr>
<td width="50%"><img src="gallery/mnist/mnist_mosaic_boundary.png"></td>
<td width="50%"><img src="gallery/mnist/mnist_mosaic.png"></td>
</tr><tr>
<td>Across a boundary: all 24 × 24 adjacent samples in the window with the most classes (0/3/5/6/8/9). The digits morph continuously while the classifier label flips, often between near-identical 6s. The boundary belongs to the classifier; the sample itself varies smoothly.</td>
<td>The whole slice at stride 2 (96 × 96 digits, 2688²).</td>
</tr></table>

**Zoom and step count:**

<p align="center"><img src="gallery/mnist/mnist_zoom_plate.png" width="90%"></p>
<p align="center"><video src="gallery/mnist/mnist_steps.mp4" autoplay loop muted playsinline width="45%"></video><br>DDIM steps 1 → 45 on a 128² slice (<a href="gallery/mnist/mnist_steps.gif">GIF</a>).</p>

### 2d. Memorisation cells (40 training images)

<p align="center"><img src="gallery/mnist/memo_cells.png" width="90%"></p>

A DDPM trained on only 40 MNIST images memorises them: the median nearest-neighbour distance ratio d₁/d₂ is 0.04, and 98% of samples have ratio < 1/3. Each noise vector is coloured by *which* training image it returns. Hue = digit class and lightness = instance (declared); brightness = 1 − d₁/d₂ (measured). Left: the network. Right: DDIM-50 with the **exact empirical score** of the 40 images, which returns training images exactly (median d₁ = 0).

<p align="center"><img src="gallery/mnist/memo_mosaic.png" width="60%"><br>Memorisation mosaic: the returned images, tinted by cell.</p>

## 3. What was computed

- **Toy data:** 2D Gaussian mixtures ring8 and ring6 (radius 2, s = 0.10), grid25, scatter12 (seeded, s = 0.08). The exact Tweedie denoiser and its analytic Jacobian were checked against autograd (error 4 × 10⁻¹⁵). A learned ε-MLP (width 256, depth 3, 16 Fourier features, AdamW 2e-3 OneCycle, 12k steps, batch 4096, EMA) was trained for ring8, grid25 and scatter12.
- **Samplers:** DDIM (Euler in σ) with N ∈ {10, 50, 1000}; PF-ODE by RK4 with 100 steps (label mismatch vs 400 steps: 7.6 × 10⁻⁶); DDPM-1000 with frozen noise; t_min = 10⁻³. The iterated denoiser runs 2000 iterations in float64, with fixed points from γ = 1 and active-set compaction.
- **MNIST:** PatchUNet DDPM (1.28M params, 15k steps, batch 128, AdamW 4e-4, EMA), a CNN classifier (test acc 99.3%), and a memorisation DDPM on 40 images (12k steps, batch 64). DDIM-50, float32. Maps are 192² over ±π/2 rad of a great sphere with radius 28. Step sweep: N ∈ {1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45} at 128². Zoom: 4 levels ×4 at 128², re-centred on the multi-class point nearest the centre.
- **Wall time:** toy CPU jobs (2 threads) took ~2 h in total. MNIST GPU (GB10, shared): training ~40 min, compute chain ~1.8 h (hero map 20 min, memo 25 min, step sweep 32 min, zoom 27 min).

Commands (run from this directory with `/home/fzeng/ml/research/art/.venv/bin/python`; long GPU jobs go through `_shared/gpu_run.sh`; prefix `DB_DEVICE=cpu DB_THREADS=2` for CPU):

```
python toy.py train ring8 ; python toy.py train grid25 ; python toy.py train scatter12
python toy_compute.py iter hero ; python toy_compute.py verify ; python toy_compute.py iter gamma
python toy_compute.py maps analytic ; python toy_compute.py steps analytic ; python toy_compute.py ddpm
python toy_compute.py zoom ode_scatter12 ; python toy_compute.py zoom iter_ring8 ; python toy_compute.py zoom iter_ring6
python toy_compute.py maps learned ; python toy_compute.py zoom ddim50_learned_scatter12
python mnist.py memo --steps 12000 --bs 64 ; python mnist.py clf+ddpm --steps 15000 --bs 128
sh run_mnist_chain.sh            # mnist_compute.py map/stepsweep/zoom/uncert/f64check
python mnist_compute.py uncert 16384
python render_toy.py atlas hero iter diptych steps gamma verify ddpm zoom:ode_scatter12 zoom:iter_ring8 zoom:iter_ring6
python render_mnist.py basin boundary mosaic memo steps zoom
```

## 4. Verification

<p align="center"><img src="gallery/verify/toy_dimension.png" width="95%"></p>

**Verdicts**

| map | box-count D (2048², fit 2–256 px) | uncertainty exponent D = 2 − α | zoom | verdict |
|---|---|---|---|---|
| PF-ODE, scatter12 (exact score) | 1.094 ± 0.015 | **0.995** (M = 2¹⁸, ε 10⁻¹…10⁻⁵) | 32 levels ×2: boundary = one straight line from width 9 × 10⁻⁵ on; windowed D → 0.98–1.02 | **smooth** |
| DDIM-50, scatter12 | 1.093 | **0.983** | – | **smooth** |
| DDIM-10, scatter12 | 1.087 | – | – | smooth |
| DDPM frozen noise, injected-noise slice | D[2,128] = 1.22 (N = 30), 1.19 (N = 1000) at 512² | – | – | smooth organic lobes (finite-resolution D) |
| null: circle | 1.061 | – | – | smooth reference |
| null: ring8 exact rays / iterated γ = 1 | 1.070 / 1.070 | – | – | smooth reference |
| **iterated denoiser, ring8, γ = 2.12** | **1.571 ± 0.021** | **1.270** | 36 levels ×2, float64: D repeats with **period 9 levels** (×512), mean 1.46 | **fractal** |
| iterated denoiser, ring6, γ = 2.10 | 1.408 ± 0.033 | 1.177 | 30 levels, period 9, D 1.18–1.38 | fractal |
| positive control: Newton z³ − 1 | 1.470 | 1.441 | – | fractal reference |
| **MNIST DDPM, DDIM-50** | – | 1.17 (M = 2048, ε 0.1…0.003, weak; local slope in last decade 0.97) | 4 levels ×4: 5 → 2 classes, one straight boundary at width 0.012 rad | **smooth** |

Box counting on smooth curves over-reads 1.06–1.09 at this resolution (see the circle null), so the samplers match the nulls exactly. The uncertainty exponent is the sharper test: 0.98–1.00 for the samplers versus 1.27 for the iterated denoiser and 1.44 for Newton.

**Resolution check** (boundary pixels at 256/512/1024/2048 over the same window): ODE 1157 → 2349 → 4747 → 9578 (×2.0 per doubling, as a smooth curve should); Newton 3708 → 75877 (×2.7). In the global 2048² resolution window (centre (2.2, 1.1), half-width 0.3), the iterated ring8 map scales ×2.03, so it is locally smooth there. **The fractal set is concentrated on the flower accumulation points, not spread everywhere.** The zoom supplies the evidence at those points: at 768² the boundary never decays towards a single line (16k–81k boundary px per window at every level), and the windowed D sequence 1.56 1.59 1.35 1.34 1.49 1.60 1.50 1.25 1.42 repeats exactly three times. That is discrete self-similarity with a scale factor of 2⁹.

**Precision.** The toy iterated map and its zoom are float64, and the deepest window (1.2 × 10⁻¹⁰ wide at 768²) is still ~10⁵ × above machine ε relative to coordinate magnitude 2.8. MNIST float32 vs float64 at the deepest zoom window (48², half-width 6.1 × 10⁻³ rad): **label mismatch 0 / 2304**.

**MNIST step sweep.** Agreement with N = 45 is 2% (N = 1), 55% (2), 81% (4), 92% (8), 98% (23), 99.2% (32). Boundary edge count rises from 358 (N = 1) and saturates at ~1170 by N = 8, so more steps do not add boundary.

**Memorisation.** The network and the exact empirical score each reach 16–17 of the 40 images on this slice, but they assign the same image to only **44.5%** of noise vectors. The net memorises the images, not the exact-score partition of noise space.

**Negative / surprising results**
- DDPM with frozen noise ignores the start noise: the x_T coefficient in the final sample is ~0.015, so the z-plane map is a single colour for N = 10…1000.
- ring8 ODE/DDIM basins are exactly straight rays by symmetry, and grid25 gives plain rectangles. Only asymmetric scatter12 shows stretched curved boundaries.
- For scatter12 the iterated denoiser is nearly Voronoi. The flowers need a symmetric junction where several modes meet.
- The iterated map is invertible (smooth, ray boundaries) for γ ≤ ~1.06. Folding and fractality are tied to over-relaxation, not to denoising itself.
- The MNIST uncertainty exponent at M = 2048 is statistically weak (4 flips at ε = 10⁻³). A larger run is described in NOTES.md, and its number is added below if it finished.

## 5. Caveats

- The deterministic samplers are **not** fractal. Where the fractals doc implies fractal DDIM basins, that is wrong for exact or smooth scores. What looks intricate in sampler maps is *stretching* (σ_max up to 100×), not folding.
- The iterated over-relaxed denoiser is **not a sampler**. It is a Newton-like fixed-point map built from the same denoiser, and it is labelled as such everywhere.
- MNIST labels come from a classifier, so the digit boundary is partly the classifier's opinion (see the mosaic, where the label flips between near-identical 6s). The generated image itself varies smoothly.
- A 2D slice of 784-dimensional noise shows one great sphere through one random orthonormal triple. Other slices differ in layout, not in smoothness (by the invertibility argument).
- Colours of categorical basins are declared palettes. Brightness and seams are measured quantities (margin, max-prob, iteration count, det J) or declared distance-to-boundary shading, as captioned.
- Box-count D on finite images is biased upward for smooth curves (≈1.06–1.09 here). Only compare against the nulls.

## 6. References

- Song et al. 2021, *Score-Based Generative Modeling through SDEs* (probability-flow ODE).
- Song, Meng, Ermon 2021, *DDIM*.
- Ho, Jain, Abbeel 2020, *DDPM*.
- Efron 2011, *Tweedie's formula and selection bias*.
- McDonald, Grebogi, Ott, Yorke 1985, *Fractal basin boundaries* (uncertainty exponent).
- Carlini et al. 2023 and Somepalli et al. 2023, on memorisation in diffusion models. Kadkhodaie et al. 2024, *Generalization in diffusion models arises from geometry-adaptive harmonic representations* (memorisation vs number of training images).
- Sohl-Dickstein 2024, *The boundary of neural network trainability is fractal* (Spectral split style).
- Files over 20 MB are not committed: none in gallery (the zoom MP4 was re-encoded to 16.6 MB at CRF 21, GIF 13.8 MB).
