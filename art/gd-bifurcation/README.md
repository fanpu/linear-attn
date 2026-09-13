# Cascade: the bifurcation diagram of gradient descent

*Turn up the learning rate on a four-parameter "network" and gradient descent stops converging. First it
flips between two values, then four, eight, and so on into chaos. Inside the chaos, every periodic window
holds a small, complete copy of the whole picture.*

<img src="gallery/hero_dark_fire.png" width="100%">

<sub>**hero_dark_fire.png** (8000×2000). GD on f(x) = ½(x₁x₂x₃x₄ − 1)² with all four coordinates, float64.
16 000 step sizes η ∈ [0.48, 0.99], 20 000 burn-in steps, then 8 192 iterates per η. Horizontal: η.
Vertical: the network output P = x₁x₂x₃x₄. Brightness is the log count of visited iterates per pixel.
**Declared choices:** the tone ceiling is the 99.7th percentile of the chaotic band, so only caustic folds
reach white; strokes in periodic columns are widened to 5 px so the period-1/2/4 branches survive at print
scale; the palette is colorcet `fire`.</sub>

---

## 1. The phenomenon

Take gradient descent with a fixed step size η on a product of scalars (a depth-k scalar "linear network"):

$$f(x)=\tfrac12\,(x_1x_2\cdots x_k-1)^2,\qquad x\leftarrow x-\eta\nabla f(x).$$

At any global minimum the Hessian is $gg^\top$ with $g_i=1/x_i$, so its sharpness is $s=\sum_i 1/x_i^2$.
Under the constraint $\prod x_i=1$, AM-GM gives $s\ge k$, with equality only at the **balanced** minima $|x_i|=1$.

**The first period doubling is not "2/λ_max of the minimum you are at".** Minima form a (k−1)-dimensional
manifold. When η is too large for the current minimum, GD oscillates, and the oscillation itself shrinks
the imbalance. The exact identity is

$$x_i'^2-x_j'^2=(x_i^2-x_j^2)\Big(1-\eta^2 r^2\frac{P^2}{x_i^2x_j^2}\Big),\qquad r=P-1 .$$

GD therefore walks toward flatter minima until the sharpness is just under 2/η (the "sharpness adaptivity"
of Zhu et al.). A stable fixed point exists only while some minimum has s < 2/η. The rule is

$$\eta^\ast=\frac{2}{s_{\min}},\qquad s_{\min}=k=\text{sharpness of the balanced (flattest) global minimum},$$

so η* = 1 for k = 2 and η* = 0.5 for k = 4. Past η*, every orbit that survives ends on the balanced line
$x_i=u$, where GD is **exactly** the one-dimensional map

$$u\leftarrow u-\eta\,(u^k-1)\,u^{k-1}.$$

That map has a single quadratic critical point in the relevant interval, which makes it a unimodal map. Unimodal
maps undergo the Feigenbaum period-doubling cascade. For a deep linear network with aligned singular vectors,
each singular mode reduces to the same scalar problem with k = depth. Ghosh et al. (2025) and Liang & Montúfar
(2025) prove this decoupling.

## 2. Hero images

<img src="gallery/hero_riso_lyapunov.png" width="100%">

<sub>**hero_riso_lyapunov.png**. The same measurement in two spot inks. Blue is the cascade (density). Fluorescent
pink is the Lyapunov exponent λ of the oscillating mode, measured over 8 192 steps per η. Above the
baseline λ > 0 (chaos). Below it, in a 40% tint, λ < 0, with the superstable dips toward −∞ clipped at −1.6.
**Declared:** inks, multiply overprint, 6/−5 px misregistration, paper grain.</sub>

<img src="gallery/braid_crop.png" width="100%">

<sub>**braid_crop.png** (6000×3000), η ∈ [0.645, 0.68], separate 16 000-η sweep. Hue is the circular mean of
the *bit-reversed iterate phase* (t − t₀) mod 16, where t₀ is the iterate with the largest P among the first 16.
It is a cyclic quantity, so a cyclic map (colorcet `cyclic_mygbm`) is used. With this encoding the first
doubling splits the hue circle in half, the next doubling splits each half again, and so on, so the branch
tree becomes coloured ribbons. The colour survives into the 4-band and 2-band chaotic regimes (the attractor
still visits bands in order) and turns grey exactly where the bands merge. Brightness is density.</sub>

## 3. Self-similarity, the centrepiece

### 3a. Window atlas: every window is a copy of the whole

<img src="gallery/atlas_dark.png" width="100%">

<sub>**atlas_dark.png**. An automatic detector scanned 400 000 η values in [η∞, 0.99] (balanced-line map, period
≤ 256 at tolerance 1e−7). For each base period 3 … 16 it picked the widest window, then **recomputed every
plate with the full 4-coordinate GD** (4 800 η per plate, 30 000 burn-in, up to 60 000 iterates). Each
plate zooms into the branch nearest the map's critical point. Zoom factors run from ×21 (p = 3) to ×28 409
(p = 13) in η. Each plate shows a saddle-node birth, then 2, 4, 8, …, chaos, internal windows, and a crisis.
Copies on orientation-reversing branches appear upside down (p = 6, 8, 9, 10, 14, 16). Strip: λ of the
oscillating mode with sign-preserving √ compression (declared). Variants: `atlas_dark_byzoom.png` (ordered by
zoom factor), `atlas_paper.png` (single ink), `gallery/atlas/pXX.png` (full-resolution individual plates with
coordinates).</sub>

### 3b. Nested zoom: period 3 → 9 → 27 → 81 → 243

<video src="gallery/zoom_tower.mp4" autoplay loop muted playsinline width="100%"></video>

<sub>**zoom_tower.mp4** (1920×1080, 30 fps, H.264) and **zoom_tower.gif**. A continuous camera path from
the full cascade into the period-3 window's central branch. Inside that branch it finds the miniature's own
period-3 window (period 9 overall), then period 27, 81 and 243. **Every frame is an independent full-GD computation**
(1 920 η per frame, 30 000 burn-in, up to 300 000 iterates), not a magnified image. The on-screen text gives η range,
P range and cumulative zoom factor; the deepest frame is ≈ ×2·10⁸ in η. Strip: λ of the oscillating mode.
**Declared:** ease-in/out camera path, log tone with ±15-frame temporally smoothed percentiles, fire palette.</sub>

<img src="gallery/tower_plates.png" width="100%">

<sub>**tower_plates.png**. The keyframes of the film as a plate sequence (full → p3 → p9 → p27 → p81 → p243).</sub>

**The self-similarity is measured, not only seen.** Superstable parameters $s_m$ of the nested period-$3^m$
windows were computed in IEEE quad precision (Floquet multiplier = 0, found by bisection with Newton continuation).
The ratios of successive gaps are

| | $(s_2-s_1)/(s_3-s_2)$ | $(s_3-s_2)/(s_4-s_3)$ | $(s_4-s_3)/(s_5-s_4)$ |
|---|---|---|---|
| GD, k = 4 (balanced-line map) | 59.50 | 54.99 | **55.265** |
| logistic map, same code | 55.14 | **55.268** | |

Both converge to the universal period-tripling scaling constant for quadratic maps (≈ 55.25). The nested
windows really are renormalised copies.

### 3c. Universality overprint

<img src="gallery/universality_riso.png" width="100%">

<sub>**universality_riso.png**. Blue ink is the logistic map and pink ink is GD on ½(x₁x₂x₃x₄ − 1)², on a shared
logarithmic axis $s=\log_{10}[A/(\eta_\infty-\eta)]$. A is fixed from each system's measured η_n, so the
n-th doubling lands at $s=n\log_{10}\delta$ if δ is universal. Vertical: (value − critical point)/max|value −
critical point| per column. 8 000 η, 120 000 burn-in, 2 048 iterates per η. The doubling positions agree from
n = 2 on. The branch heights differ at small n and approach each other to the right. **Declared:** both axis
transforms, 7-px minimum stroke, inks, misregistration.</sub>

## 4. Gallery

### Cascade, other styles

| | |
|---|---|
| <img src="gallery/hero_dark_fire_full.png" width="100%"> | **hero_dark_fire_full.png**. Honest full range η ∈ [0.45, 0.99], including the converged regime and the first doubling at η = 0.5. |
| <img src="gallery/hero_paper_ink.png" width="100%"> | **hero_paper_ink.png**. Single ink on paper: ink coverage is the log density. Plotter/survey-sheet idiom. |
| <img src="gallery/hero_phase_braid.png" width="100%"> | **hero_phase_braid.png**. Full range with bit-reversed phase hue. Colour is confined to the pre-merging region because phase carries no meaning in fully mixed chaos. |
| <img src="gallery/plate_scientific_prod4.png" width="100%"> | **plate_scientific_prod4.png**. Scientific plate over the full range η ∈ [0.45, 1.21]. Rule line η* = 2/s_min; dashed line at the *wrong* rule 2/s_GF(x₀); Newton–Floquet η₂, η₃, η∞; window periods; the δ_n table; the Lyapunov strip (grey fill: oscillating mode; red: max exponent of the full 4-D map). Past η ≈ 0.99 the band touches P = 0, the degenerate stationary point x = 0, and the attractor becomes intermittent (sparse bursts). |

### One pipeline, four systems

<img src="gallery/systems_dark.png" width="100%">

<sub>**systems_dark.png**. Identical code and tone rule for (1) the logistic map (reference/null), (2) GD on
½(x₁x₂ − 1)², (3) GD on ½(x₁x₂x₃x₄ − 1)², and (4) a deep linear network ½‖W₃W₂W₁ − M‖² with W_l ∈ ℝ^{5×5} and
target singular values 10, 6, 3 (Ghosh et al. App. B.1, with the ½). The DLN is shown from their aligned init and
from a random Gaussian init. DLN vertical: $c_i=u_i^{*\top}W_3W_2W_1v_i^{*}$ (orange: mode 1, the dim line at
6: mode 2). Red hairlines mark the onset rule. The DLN's black gap from η ≈ 0.0435 to 0.057 is a real
result, not missing data (see §6).</sub>

<img src="gallery/overlay_inits_dark.png" width="100%">

<sub>**overlay_inits_dark.png**. 48 log-normal initialisations (scalar products) and 17 random seeds (DLN)
overlaid. The attractor does not depend on the init. Whether a given init reaches it does: the orange strip
shows the fraction of inits that stay bounded.</sub>

<img src="gallery/plate_threshold_rule.png" width="100%">

<sub>**plate_threshold_rule.png**. Left: the sharpness of the minimum GD converges to, for 49 inits. It hugs 2/η
(sharpness adaptivity) and reaches the floor s_min = k exactly at η = 2/k. Right: the first η where GD from x₀
has not converged after T steps, divided by 2/k. At finite T it depends on the init (median 0.84 at T = 10² for
k = 4), and it converges to 1 for every init as T grows (0.9989 at 10⁵; the residual is critical slowing down).
Neither alternative rule matches: 2/λ_max(x₀) (grey ticks) and 2/sharpness of the minimum gradient flow would
reach from x₀ (red ticks).</sub>

### Initialisation basin maps (Zhu et al.'s fractal boundary)

GD on ½(1 − xyzw)² with z = x, w = y, at η = 0.2, which is exactly the reduced map of Zhu et al. eq. (2).
Each pixel is one initialisation (x₀, y₀), run for 10 000 steps in float64.

| | | |
|---|---|---|
| <img src="gallery/basin_wide_dark.png" width="100%"> | <img src="gallery/basin_wide_paper.png" width="100%"> | <img src="gallery/basin_wide_riso.png" width="100%"> |
| **basin_wide_dark**, (x₀, y₀) ∈ [0, 3.8]², 4096². Blue ramp: log steps to loss < 10⁻¹² (converged). Fire: smooth escape value ν = t_esc − log(log r / log 10³)/log 7 (diverged). | **basin_wide_paper**. Single ink: the converge/diverge boundary only, over a faint tint of the converged set. | **basin_wide_riso**. Two inks, stochastic screens: blue density ∝ convergence time, pink density ∝ escape value. |
| <img src="gallery/basin_z1_dark.png" width="100%"> | <img src="gallery/basin_z1_paper.png" width="100%"> | <img src="gallery/basin_z2_paper.png" width="100%"> |
| **basin_z1_dark**, [3.0, 3.4] × [0.1, 0.5] (Zhu et al. Fig. 6a), ×9.5. | **basin_z1_paper**. Boundary line drawing. | **basin_z2_paper**, [3.29375, 3.30] × [0.30625, 0.3125], ×608: the boundary is a stack of parallel stripes, locally a Cantor set × a curve. |

<img src="gallery/plate_basin_verification.png" width="100%">

<sub>**plate_basin_verification.png**. Fractals-doc §11 panel: zoom sequence, box counting, resolution
check, null model, positive control, and 1-D transects.</sub>

### Extra pieces: cyclic learning-rate Lyapunov planes

| | |
|---|---|
| <img src="gallery/lyapplane_AABAB_dark.png" width="100%"> | **lyapplane_AABAB_dark.png**. GD on ½(x₁x₂x₃x₄ − 1)² with the cyclic step-size schedule A, A, B, A, B, … Each pixel (η_A, η_B) ∈ [0.45, 1]² gives λ of the oscillating mode (2048², 1 500 burn-in + 3 000 steps, balanced-line sub-dynamics). A random sample of 4 000 pixels re-run with full 4-D GD: sign of λ agrees on 98.9%, and |Δλ| < 0.05 on 97.4% of chaotic pixels. **Declared:** gold λ<0 / black 0 / blue λ>0 (the Markus–Lyapunov convention). The diagonal η_A = η_B is the constant-step Lyapunov strip. |
| <img src="gallery/lyapplane_AABAB_paper.png" width="100%"> | **lyapplane_AABAB_paper.png**. Single ink, coverage tanh|λ| for λ<0, so superstable curves (λ → −∞) draw themselves as dark lines. |
| <img src="gallery/lyapplane_AB_dark.png" width="100%"> | **lyapplane_AB_dark.png**. Schedule A, B, A, B. Full-GD check: sign agrees on 92.5%. Not symmetric under A↔B, because the fixed start u₀ = 1.0001 chooses between coexisting attractors differently depending on which step comes first. |
| <img src="gallery/lyapplane_AABAB_zoom_dark.png" width="100%"> | **lyapplane_AABAB_zoom_dark.png**. Zoom ×23 on an isolated periodic island inside the chaotic sea: a "swallow/shrimp" with its own satellites. |

## 5. What was computed

All dynamics are float64 unless marked quad. Scalar systems ran on CPU (numpy); DLN on CPU (torch); basin maps
and transects on the GPU (torch).

| piece | system | grid | steps | init / seeds |
|---|---|---|---|---|
| cascade sweeps `compute_bifurcation.py` | prod2, prod4, logistic | 16 000 η | 20 000 burn + 2 048 rec (Lyapunov) and + 8 192 rec (`_hi`, density) | x₀ = (1.1, 0.9); (1.1, 0.9, 1.05, 0.95); logistic x₀ = 0.3 |
| many inits `… _multi` | prod2, prod4 | 4 000 η × 48 inits | 20 000 + 256 | x₀ ~ exp(0.35·N(0,1)), numpy seed 1 |
| DLN `compute_dln.py` | L = 3, d = 5, M from `dln_target()` (QR of seeded Gaussians, singular values 10, 6, 3) | 16 000 η ∈ [0.028, 0.0675] | 20 000 + 2 048 | aligned (W₃ = 0, W₁ = W₂ = 0.1 I); random seed 0 (N(0, 0.5²/5)); 16 more seeds at 4 000 η |
| Feigenbaum A `compute_feigenbaum.py` | logistic, bal2, bal3, bal4 (1-D, float64 and quad), prod2, prod4 (full 2-D/4-D) | bisection to 1e−16 | Newton on G^p(x) = x | — |
| Feigenbaum B | logistic, prod2, prod4, DLN (`compute_dln_feig.py`) | 64 η per round × 3 rounds | T = 2·10⁴ and 2·10⁵ | declared fixed inits |
| threshold `compute_threshold.py` | prod2, prod4 | 1 700 η × 49 inits | T = 10², 10³, 10⁴, 10⁵ | fixed + 48 log-normal |
| windows `compute_windows.py` | bal4 detection, 400 000 η | | 20 000 + ≤256·… | |
| atlas / tower / film `compute_plates.py` | full prod4 | 2 400 cols (atlas/tower), 1 920 cols × 601 frames (film) | 30 000 burn + ≤ 60 000 / 400 000 / 300 000 rec | fixed x₀ |
| tripling `compute_tripling.py` | bal4, logistic, quad | — | — | — |
| universality `compute_universality.py` | logistic, full prod4 | 8 000 log-spaced η | 120 000 + 2 048 | fixed |
| basins `compute_basins.py` | zhu4 (η = 0.2), prod2 (η = 0.2), lmreg (η = 1) | 1024² … 8192², nested zooms | 10 000 (lmreg 3 000); converged if loss < 1e−12; diverged if max|x| > 10³ | — |
| transects `compute_transect.py` | zhu4 | 2²² points × 3 nested segments | 10 000 | — |
| Lyapunov planes `compute_lyapplane.py` | bal4 with cyclic η, 4 000-pixel full-GD check | 2048² | 1 500 + 3 000 | u₀ = 1.0001 |
| DLN transverse `compute_dln_transverse.py` | full DLN on the exact reduced orbit | 185 η | 20 000 + 4 000 tangent | — |

Reproduce, from `art/gd-bifurcation/`, with `PY=/home/fzeng/ml/research/art/.venv/bin/python`:

```bash
export OMP_NUM_THREADS=4
for s in prod2 prod4 logistic; do $PY compute_bifurcation.py $s; $PY compute_bifurcation.py $s --rec 8192 --onlyP --tag _hi; done
$PY compute_bifurcation.py prod4braid --rec 8192 --onlyP --tag _hi
for s in prod2_multi prod4_multi; do $PY compute_bifurcation.py $s --n 4000 --rec 256 --ninit 48; done
$PY compute_dln.py --init aligned --cpu; $PY compute_dln.py --init random --cpu
for s in $(seq 1 16); do $PY compute_dln.py --init random --seed $s --n 4000 --rec 256 --lyap_tail 1 --tag _seed$s --cpu; done
$PY compute_feigenbaum.py; $PY compute_dln_feig.py random; $PY compute_threshold.py
$PY compute_windows.py; $PY compute_tripling.py; $PY compute_universality.py
$PY compute_plates.py atlas --workers 4; $PY compute_plates.py tower --workers 5
$PY compute_plates.py zoom --nframes 601 --levels 5 --workers 5
G=../_shared/gpu_run.sh
$G $PY compute_basins.py zhu4 --x0 0 --x1 3.8 --y0 0 --y1 3.8 --res 4096 --T 10000 --tag wideT
for r in 1024 2048 4096 8192; do
  $G $PY compute_basins.py zhu4 --x0 3.0 --x1 3.4 --y0 0.1 --y1 0.5 --res $r --T 10000 --tag z1
  $G $PY compute_basins.py zhu4 --x0 3.29375 --x1 3.30000 --y0 0.30625 --y1 0.31250 --res $r --T 10000 --tag z2
  $G $PY compute_basins.py prod2 --x0 -4.5 --x1 4.5 --y0 -4.5 --y1 4.5 --res $r --T 10000 --tag null
  $G $PY compute_basins.py lmreg --x0 -4 --x1 4 --y0 -4 --y1 4 --res $r --T 3000 --tol 1e-10 --tag pos
done
$G $PY compute_basins.py zhu4 --x0 3.0 --x1 3.4 --y0 0.1 --y1 0.5 --res 4096 --T 10000 --tag z1T
$G $PY compute_basins.py zhu4 --x0 3.29375 --x1 3.30000 --y0 0.30625 --y1 0.31250 --res 4096 --T 10000 --tag z2T
$G $PY compute_transect.py; $PY analyze_boxcount.py
$PY compute_lyapplane.py --res 2048 --pattern AABAB --lo 0.45 --hi 1.0
$PY compute_lyapplane.py --res 2048 --pattern AB --lo 0.45 --hi 1.0
$PY compute_lyapplane.py --res 2048 --pattern AABAB --lo 0.733 --hi 0.757 --blo 0.748 --bhi 0.772 --tag _zoom
$PY compute_dln_transverse.py
# renders (cache -> gallery)
$PY render_hero.py; $PY render_hero.py braidcrop; $PY render_plate.py; $PY render_atlas.py; $PY render_zoom.py
$PY render_universality.py; $PY render_systems.py; $PY render_threshold.py
$PY render_basins.py wideT z1T z2T; $PY render_basin_plate.py; $PY render_lyapplane.py AABAB AB AABAB_zoom
```

Shared code lives in `common.py` (torch maps, DLN), `npmaps.py` (numpy maps, analytic Jacobians and tangent maps)
and `render_lib.py`. Gradients and Hessians were unit-tested against autograd (max error 3e−15).
Wall time was roughly 3.5 h on a heavily shared machine (load average 30–50 on 20 cores). **GPU: ≈ 1.3 GPU-hours**
(basin maps 57 min, transects TBD); everything else ran on CPU.

## 6. Verification and honesty

### Did the cascade appear, and where does it start?
Yes, in all three systems. **k = 4:** the Floquet multiplier of the balanced fixed point reaches −1 at
η = 0.5000000000 (bisection on the numerical Jacobian), which is 2/s_min. Simulation bisection from the
declared init gives 0.49965 after 2·10⁵ steps; the offset is critical slowing down, and it shrinks with T. **k = 2:** 1.0000000000.
**DLN:** simulation bisection from a random init gives 0.030944 (T = 2·10⁵) against the rule
2/(Lσ₁^{2−2/L}) = 0.0309439.

### Feigenbaum δ (Newton–Floquet bisection; ± is |float64 − quad| propagated)

| n | logistic | GD k = 2 | GD k = 3 (DLN mode) | GD k = 4 (full 4-D) |
|---|---|---|---|---|
| δ₂ | 4.75145 | 4.54293 | 4.38293 | 4.33777 |
| δ₃ | 4.65625 | 4.64120 | 4.62564 | 4.62195 |
| δ₄ | 4.66824 | 4.66318 | 4.65820 | 4.65674 |
| δ₆ | 4.66913 | 4.66893 | 4.66872 | 4.66866 |
| δ₈ | 4.66920 | 4.66919 | 4.66918 | 4.66918 |
| δ₁₂ | 4.66920 ± 1e−6 | 4.66921 ± 3e−6 | 4.66920 ± 5e−6 | 4.66921 ± 6e−6 (1-D), 4.66920 ± 7e−8 (δ₁₀, full 4-D) |

- The first ratios are **not** universal. They depend on k and differ from the logistic map's, and δ_n converges from below
  for GD but from above for the logistic map. By n ≈ 8 all four agree with δ = 4.669201.
- The full 2-D/4-D GD bifurcation points match the 1-D balanced-map points to ≤ 6·10⁻¹⁵. This is the test of "effectively 1-D":
  the periodic orbits lie on the balanced line and their transverse multipliers stay inside the unit circle. The
  transverse exponent (exact imbalance multiplier) is ≤ −0.01 throughout η ∈ [0.66, 0.99] for k = 4, and ≤ −0.11 for k = 2.
- A black-box training-loop measurement, **method B** (simulation bisection, 64 η × 3 rounds, orbit declared
  p-periodic at tol 1e−9), is biased low by critical slowing down: η_n − η_n^A ≈ −3.5e−4 … −1.1e−7 at T = 2·10⁵.
  Its δ estimates are 4.349, 4.621, 4.655, 4.664, 4.663 for k = 4, and 4.389, 4.625, 4.657, 4.664, 4.663 for the DLN.
  They are good to about 1e−3 only.
- **DLN vs scalar reduction:** DLN doublings from a *random, unaligned* init sit at η_n(bal3)/σ₁^{4/3} to a
  relative 1.6e−7 at n = 7 (T = 2·10⁵), the same bias as method B on the scalars. So the DLN's period-doubling
  cascade *is* the k = 3 scalar cascade, rescaled.

### Surprises and negative results
1. **The DLN loses the chaotic band.** The scalar reduction predicts bounded chaos up to η ≈ 0.0668, but
   GD on the DLN diverges for η ∈ [0.0435, 0.057] from every init tried (aligned, 17 random seeds), and even
   from a state placed *exactly* on the reduced chaotic attractor. `compute_dln_transverse.py` propagates a
   tangent vector in the off-diagonal (singular-vector-mixing) subspace along the exact reduced orbit. The
   transverse exponent is ≈ 0 through the periodic regime and becomes **positive (+0.01 at η = 0.0428, +0.04
   … +0.13 across the band)**, and the onset coincides with where divergence starts. This is a blowout
   bifurcation. The Feigenbaum cascade survives in the network; its chaotic band does not.
2. **k = 4 intermittency past η ≈ 0.99.** The chaotic band touches P = 0, the degenerate stationary point x = 0
   (the 1-D map has derivative exactly 1 there). Orbits linger near P = 0 for long laminar phases, the
   diagram turns into sparse streaks, and both Lyapunov exponents collapse toward 0. Near η ≈ 1.158 the full 4-D GD is
   chaotic (λ ≈ 0.4–0.5) where the balanced-line orbit is periodic, so the 1-D reduction fails there.
3. **"The onset depends on the init"** holds only at finite time. At T = 10² the measured onset for k = 4 ranges over
   0.31 … 0.91 × (2/k). At T = 10⁵ it is 0.9984 … 0.9989 for all 49 inits. What does depend on the init is *whether* the
   attractor is reached (the overlay's bounded-fraction strip).
4. Among the other Lyapunov-plane patterns I looked at, AB is less intricate than AABAB.

### Basin boundary: §11 protocol
| test | Zhu degree-4, z1 region (×9.5) | Zhu degree-4, z2 region (×608) | null: ½(xy−1)² | analytic ellipse (render null) | positive control: L&M reg. |
|---|---|---|---|---|---|
| box counting at 8192², ε ∈ [2.4e−4, 0.12] of the side (2.7 decades) | **1.728 ± 0.006** | 1.80 ± 0.01 (local slopes 1.62→1.98: the stripes fill the box at large ε, so use the small-ε slopes ≈ 1.7) | **1.038 ± 0.005** | 1.038 ± 0.005 | 1.185 ± 0.003 |
| resolution scaling, boundary pixels ∝ R^D over R = 1024…8192 | **1.741** | **1.715** | **1.000** | — | 1.284 |
| labels that flip under 2× refinement | 4.6% → 3.8% → 3.2% (∝ R^−0.27, so D = 2 − 0.27 = **1.73**) | 14.1% → 11.4% → 9.4% (D ≈ 1.71) | 0.04% → 0.01% (∝ R^−1) | — | 0.11% → 0.05% |

- Three independent estimates on the Zhu boundary agree: **D ≈ 1.72–1.74**. The null model (Liang & Montúfar prove its
  convergence region is a smooth ellipse up to measure zero) gives D = 1.00–1.04 through the identical pipeline. It is even
  *pixel-identical* to the analytic ellipse, so the pipeline is not manufacturing fractality. The positive control
  gives 1.19–1.28, against the authors' 1.249, which they measured on a symmetry-reduced projection over only 2 decades.
- **Resolution check:** the fraction of flipped labels keeps falling as a power law, and new stripes keep appearing
  at every refinement down to 8192² of a 0.00625-wide box (pixel 7.6e−7). Float64 is nowhere near its floor at that scale.
- **Structure:** at ×608 the boundary is a stack of nearly parallel stripes, locally (Cantor set) × (curve). That is the
  classic geometry of a basin boundary formed by the stable manifold of a chaotic saddle. The 1-D transects measure the
  Cantor factor directly: TBD.
- **Honest caveat:** the fractal fringe occupies only small parts of the plane (the lobes near (3.2, 0.3) and
  (0.3, 3.2)). Most of the converge/diverge boundary in the wide view is smooth. "Fractal" applies to the fringe.

### Precision floors
- Nested zoom: the deepest frame spans 2.5·10⁻⁹ in η (column spacing 1.3·10⁻¹²), four decades above float64
  resolution at η ≈ 0.76 (1.1·10⁻¹⁶). The floor would be hit around the period-2187 window (width ≈ 2·10⁻¹³). I stopped at 243.
- Feigenbaum points: float64 and quad differ by ≤ 8·10⁻¹⁵, which caps useful n at about 13 (η₁₃ − η₁₂ ≈ 1e−11).
- Basins: float64 throughout. The smallest boundary pixel is 7.6·10⁻⁷, and the transects reach 1e−11.

## 7. Caveats
- **This is iteration near an instability, not a deep-learning secret.** The cascade here is the Feigenbaum cascade of a
  1-D unimodal map that GD on a product of scalars happens to *be* on its invariant balanced line. That is exactly
  why δ is universal. Blind spot 1 of the fractals doc applies literally: "this is what iteration near an instability
  always looks like, and training is iteration near an instability."
- Universality needs a 1-D unimodal effective map. It holds for the scalar products (balanced line transversally
  attracting, bifurcation points identical to 1e−15) and for the DLN's periodic regime. It **fails** for the DLN's
  chaotic band (blowout) and for k = 4 near η ≈ 1.16.
- Every diagram uses one declared init. Plates at other inits show the same attractor but different holes
  (divergence) where the init is outside its basin.
- The y-axes are P = x₁⋯x_k or $u^{*\top}W v^{*}$, chosen because they are gauge-invariant network outputs. Plotting
  a single coordinate x₁ gives the same bifurcation points, with branches rescaled by the init-dependent imbalance.
- Lyapunov strips show the exponent of the oscillating mode. The full map's maximal exponent is pinned at 0 in the
  converged regime, because the manifold of minima is neutral. The scientific plate draws both.
- The 2-D basin map is a slice (z = x, w = y) of a 4-D init space. A fractal boundary in a slice is strong evidence;
  the reverse inference would not be.
- The Lyapunov planes use the balanced-line sub-dynamics (checked against full GD on 4 000 random pixels). They are not
  full-GD rasters.

## 8. Ideas explored / not pursued
Brainstormed (★ = built):
1. ★ **Window atlas.** Auto-detect windows and plate each one (§3a). This became the centrepiece.
2. ★ **Cyclic step-size Lyapunov plane** (η_A × η_B, Markus–Lyapunov construction). A learning-rate schedule
   with two values gives a 2-D parameter plane with swallows and shrimps (§4).
3. ★ **Universality overprint** of the logistic map and GD in two inks (§3c).
4. ★ **Phase braid.** Colour each iterate by its (bit-reversed) phase (orchestrator suggestion).
5. ★ **Nested tower zoom** with measured tripling constant (§3b).
6. *Not pursued:* stacked plotter drawing of the attractor in (x₁, x₂) for successive η. On the balanced line
   x₁ = x₂, so every orbit lies on the diagonal and the drawing would be a set of dots on one line.
7. *Not pursued:* cobweb of an effective return map sharpness_t → sharpness_{t+1}. The exact return map is the
   balanced-line polynomial, so a cobweb would re-draw a textbook figure with no new measurement.
8. *Not pursued:* η × (second objective parameter, e.g. target y or depth k treated as continuous) plane coloured by
   period. The Lyapunov planes cover the "Mandelbrot-cousin" idea with a more ML-meaningful second axis.
9. *Not pursued:* η × init-imbalance plane. The attractor is init-independent past η*, so the plane would mainly
   show basin holes, which the overlays already show.

## 9. Files over 20 MB (not committed)
Listed by `commit.sh` at commit time: none of the final gallery files after the grain change (TBD: recheck).
`cache/` is git-ignored (≈ 5 GB of raw arrays).

## 10. References
- X. Zhu, Z. Wang, X. Wang, M. Zhou, R. Ge. *Understanding Edge-of-Stability Training Dynamics with a Minimalist Example.* ICLR 2023, arXiv:2210.03294.
  **Checked:** the objective is ½(1 − xyzw)² with symmetric init z = x, w = y. §5 / Fig. 6a shows a fractal-looking converge/diverge
  boundary at η = 0.2 in [3.0, 3.4] × [0.1, 0.5], and trajectories near it oscillate chaotically before settling onto the two-step
  parabola. This matches the doc's claim. The paper shows no dimension measurement; that is done here.
- A. Ghosh, S. M. Kwon, R. Wang, S. Ravishankar, Q. Qu. *Learning Dynamics of Deep Linear Networks Beyond the Edge of Stability.* ICLR 2025, arXiv:2502.20531.
  **Checked:** Fig. 1 shows a period-doubling route to chaos in singular values and Hessian eigenvalues of a 3-layer DLN.
  Lemma 1 gives the balanced-minimum sharpness Lσ₁^{2−2/L}, and Lemma 4 shows it is the flattest minimum. This matches the doc and the
  corrected onset rule. The paper does not measure δ, and its Fig. 1 range stops before the chaotic band, where this project finds divergence.
- S. Liang, G. Montúfar. *Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region.* arXiv:2509.25351 (v3, Feb 2026).
  **Checked:** for unregularised scalar factorisation the convergence region is proven to be a smooth ellipsoid almost everywhere (used here
  as the null model). With ℓ₂ regularisation the boundary is self-similar, box dimension 1.249 (used here as the positive control).
  The doc cites it only as "chaos and fractal convergence region"; note that it proves the *unregularised* degree-2 boundary is **not** fractal.
  Minor inconsistency in the paper: the Fig. 3 caption says y = 0.5, App. H says y = 1.
- M. J. Feigenbaum, *Quantitative universality for a class of nonlinear transformations*, J. Stat. Phys. 19 (1978).
- M. Markus, B. Hess, *Lyapunov exponents of the logistic map with periodic forcing*, Computers & Graphics 13 (1989).
- J. Sohl-Dickstein, *The boundary of neural network trainability is fractal*, arXiv:2402.06184 (context, fractals doc §1).
- H.-O. Peitgen, H. Jürgens, D. Saupe, *Chaos and Fractals*, for box counting and the uncertainty exponent (D = 2 − α).
