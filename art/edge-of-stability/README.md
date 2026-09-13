# Edge of Stability: the braid

*Full-batch gradient descent pushes the curvature of the loss up until it reaches 2/η, the exact value at which a quadratic model says GD should blow up. Then it stays there, wobbling back and forth between two sheets of weight space, and still trains.*

<p align="center"><img src="gallery/print_eta80_detail_night.png" width="100%"></p>
<p align="center"><video src="gallery/film_eta80.mp4" autoplay loop muted playsinline width="100%"></video><br>
<sub>film_eta80.mp4 (42 s, 1080p; <a href="gallery/film_eta80.gif">GIF</a>): the braid being drawn, one GD step per zig. Bottom: the last 160 steps magnified.</sub></p>

## 1. The phenomenon

On a quadratic loss with curvature λ, gradient descent with step size η multiplies the displacement along that direction by (1 − ηλ) at every step:

  x<sub>t+1</sub> = (1 − ηλ) x<sub>t</sub>,

so it converges only while λ < **2/η**. At λ > 2/η the factor is below −1 and the iterate flips sign every step with growing amplitude. The top Hessian eigenvalue λ₁ ("sharpness") of a neural network's loss is not fixed, though. Cohen et al. (2021) found that under full-batch GD:

1. **Progressive sharpening.** λ₁ rises during training.
2. **Edge of stability (EoS).** Once λ₁ reaches 2/η, it stops rising and hovers just above 2/η for the rest of training. The loss is no longer monotone, but it still falls over long horizons.

Along the top eigen-direction the iterate oscillates with period 2 (even steps on one side, odd steps on the other). Drawing the even steps and the odd steps as two separate strands gives a **braid**. A crossing, where the strands swap, is a phase slip of the oscillation. The loop that holds λ₁ at the edge is *self-stabilisation* (Damian et al. 2022): the oscillation grows while λ₁ > 2/η, its cubic interaction with the gradient of the sharpness pushes λ₁ back down, the oscillation then decays, and progressive sharpening pushes λ₁ up again.

The threshold 2/η applies to **full-batch GD only**. SGD, momentum, and Adam have their own, different stability thresholds. Nothing in this project uses them.

## 2. Gallery

Everything shown is measured, one value per GD step, except where a caption says *declared*. η = 2/80 is the hero run.

### 2.1 The three-layer print (2/η hairline, λ stroke, braid ribbon)

Layers: the hairline is 2/η. The strokes are λ₁ (bright), λ₂, and λ₃ (dim) of the loss Hessian. The braid is the windowed-PCA oscillation coordinate (§4.3), with even and odd steps as two strands and a thin zigzag joining consecutive steps.

| | |
|---|---|
| <img src="gallery/print_eta80_detail_night.png"> **night, detail** (steps 2000–2599). Strand colours are declared. Each braid node, where the strands meet and swap, is a genuine phase slip. | <img src="gallery/print_eta80_detail_paper.png"> **paper, detail.** Single ink plus a red hairline. |
| <img src="gallery/print_eta80_detail_spectral.png"> **Spectral, detail.** Zigzag colour is λ₁ − 2/η, split at 0 (declared Sohl-Dickstein split; each side rank-normalised onto half of `Spectral`). | <img src="gallery/print_eta80_detail_riso.png"> **riso.** Two spot inks (fluorescent pink and blue); the odd plate is deliberately misregistered (declared). |
| <img src="gallery/print_eta80_night.png"> **night, whole run** (6000 steps). At this scale the braid reads as a seismograph envelope: the catapult at step ≈ 405, then bursts. | <img src="gallery/print_eta80_paper.png"> **paper, whole run.** |
| <img src="gallery/print_eta80_spectral.png"> **Spectral, whole run.** Above-edge steps dominate, so the warm half of the map dominates. | <img src="gallery/print_eta80_riso.png"> **riso, whole run.** |

### 2.2 Four step sizes: small multiples

| | |
|---|---|
| <img src="gallery/multiples_5400_night.png"> **night.** η = 2/50, 2/80, 2/120, 2/200 over the same window (steps 5400–5999). The λ strips show (λ − 2/η)/(2/η) on one shared scale. The braid gain is per row (declared, printed). | <img src="gallery/multiples_5400_paper.png"> **paper.** |
| <img src="gallery/multiples_5400_spectral.png"> **Spectral.** Zigzag colour is λ₁ − 2/η, split at 0 and rank-normalised per row (declared). | <img src="gallery/ladder_spectral.png"> **ladder.** Absolute λ₁ for all four runs against their hairlines: sharpening, catapult, clinging. Stroke colours are declared. |
| <img src="gallery/ladder_paper.png"> ladder, paper | <img src="gallery/ladder_night.png"> ladder, night |

### 2.3 Phase portraits and return maps

| | |
|---|---|
| <img src="gallery/orbit_main4_night.png"> **Orbit plane, night.** Horizontal: r = (λ₁ − 2/η)/(2/η). Vertical: log₁₀ amplitude. Both use a 9-step centred mean (declared), and colour encodes the step (declared). Steps after t_edge + 100 only. | <img src="gallery/orbit_main4_spectral.png"> **Orbit plane, Spectral.** Colour is the amplitude growth d log A/dt, split at 0 (declared split). Warm and cool mix on both sides of the edge line; see §5.4. |
| <img src="gallery/orbit_main4_paper.png"> orbit, single ink | <img src="gallery/growth_main4_paper.png"> **Growth plate.** Measured one-step growth log(A<sub>t+1</sub>/A<sub>t</sub>) against r for all four η, with binned medians. The dashed line is the quadratic-model prediction log\|1 + 2r\|, which is a single curve in these units. The measured zero crossing sits at r ≈ +0.05 to +0.08, not at 0. |
| <img src="gallery/return_main4_paper.png"> **Burst return maps** (A<sub>n</sub>, A<sub>n+1</sub>) and (τ<sub>n</sub>, τ<sub>n+1</sub>), cobweb idiom. These are clouds, not curves: a negative result (§5.4). | <img src="gallery/return_main4_night.png"> return maps, night |

### 2.4 Plotter braid (one continuous line)

| | |
|---|---|
| <img src="gallery/plotter_eta80_oneink.png"> **One ink.** η = 2/80, steps 40–5999, 200 steps per row, boustrophedon, one vertex per GD step, one polyline. A single global gain (declared); big bursts cross into neighbouring rows, as on a drum recorder. Plotter file: [`plotter_eta80.svg`](gallery/plotter_eta80.svg) (A2, mm, Inkscape layers). | <img src="gallery/plotter_eta80_twopen.png"> **Two pens.** The red pen marks the rare steps after the edge is first reached where λ₁ < 2/η, i.e. where GD is locally stable on the quadratic model (declared choice of which event gets the second pen). |

### 2.5 The braid-drawing film

[`film_eta80.mp4`](gallery/film_eta80.mp4) (1920×1080, 30 fps, 5 GD steps per frame, H.264 yuv420p, silent) and [`film_eta80.gif`](gallery/film_eta80.gif) (640 px, 12 fps, 9.7 MB). The magnifier's vertical scale follows the running 99th percentile of |c| over its window (declared auto-gain; the gain is printed).

### 2.6 Honesty plate: rotating frame vs data frame

<img src="gallery/diptych_frames_eta80.png" width="100%">

The same 400 steps (η = 2/80) in two coordinates. **Top:** the displacement along the *current* top eigenvector u₁(t), the naive choice. The ticks mark steps where the refreshed eigenvector turned by more than acos 0.95. λ₁ and λ₂ are nearly degenerate at the edge, so the eigenvector labels swap, and the braid shows jumps and false crossings. **Bottom:** the windowed-PCA coordinate (§4.3) is smooth, and it swaps strands only at amplitude nodes.

<!-- SWEEP -->

## 3. What was computed

- **Model and data** (checked against the text of arXiv:2103.00065): the first 5000 CIFAR-10 training images, standardised per channel with full-CIFAR statistics. The network is the fc-tanh MLP 3072-200-200-10 (656,810 parameters) with PyTorch default init, the same init for every η (seed 0). Loss: MSE ½‖f(x) − onehot‖² averaged over examples.
- **Optimiser:** full-batch GD with constant η, no momentum, no weight decay.
- **Batched training** (`eos_batch.py`): M networks, one per η, are trained side by side in one process. The summed objective has a block-diagonal Hessian, so one reverse-over-reverse HVP gives all M per-model HVPs at once.
- **Sharpness:** top-3 eigenpairs by warm-started block subspace iteration, 1 iteration per step, refreshed every step (main4) or every 5 steps (sweep16), with Rayleigh–Ritz residuals. An independent cold start (60 iterations from random) runs every 500 steps as an audit.
- **Oscillation coordinates:** x<sub>t</sub> = ⟨θ<sub>t</sub> − θ̄<sub>t</sub>, u₁(t)⟩, where θ̄ is a 21-step centred mean. A 1024-bin count-sketch of θ<sub>t</sub> − θ₀ and of u₁(t) is also logged each step and feeds the windowed-PCA coordinate.
- **Precision:** float32. On the GB10, float64 HVPs cost 1.38 s vs 0.066 s (17×). The float32 floor on oscillation amplitude is about 1e-5 in sketch units; amplitudes below that are treated as "no oscillation".
- **Runs:**
  - `scout.npz`: 2/η ∈ {50, 80, 120, 200, 300}, eigenvalues every 10 steps, killed at t ≈ 3200, 1173 s. It showed visible kinks at refreshes, so the main runs refresh every step.
  - `main4.npz`: 2/η ∈ {50, 80, 120, 200}, 6000 steps, eigenvalues every step, 7617 s wall on a contended shared GPU.
  - `sweep16.npz`: 16 step sizes, 5000 steps, eigenvalues every 5 steps.
  - 2/η = 20 diverges at step 31 (init sharpness is 88.0).
- **GPU time:** about 4.5 slot-hours in total (scout + main4 + sweep16, each one slot).

Reproduce (from `art/edge-of-stability/`, `P=../.venv/bin/python`):

```bash
../_shared/gpu_run.sh $P eos_batch.py --invs 50,80,120,200 --steps 6000 --eig-every 1 --sketch 1024 \
    --check-every 500 --check-iters 60 --snap-every 100 --save-every 250 --out cache/main4.npz
../_shared/gpu_run.sh $P eos_batch.py --invs 30,40,50,60,70,80,90,100,115,130,145,160,180,200,225,250 \
    --steps 5000 --eig-every 5 --sketch 512 --check-every 1000 --check-iters 40 --snap-every 250 \
    --save-every 500 --out cache/sweep16.npz
$P verify.py main4.npz                              # -> cache/main4_verify.json
$P render_print.py main4.npz 1 eta80                # whole run, 4 styles
$P render_print.py main4.npz 1 eta80_detail 2000 2600
$P render_multiples.py main4.npz 5400 6000          # multiples + ladder
$P render_phase.py main4.npz main4                  # orbit, growth, return maps
$P render_plotter.py main4.npz 1 eta80 200          # SVG + proofs
$P render_film.py main4.npz 1 eta80 5               # MP4 + GIF (frames in cache/frames, delete after)
$P render_diptych.py main4.npz 1 eta80 2000 2400
$P render_sweep.py sweep16.npz sweep16
EOS_COORD=u1 $P render_print.py ...                 # any renderer in the rotating-frame coordinate
```

Rendering is CPU-only from the cache.

## 4. Verification and honesty

All numbers come from `cache/main4_verify.json` (6000 steps).

### 4.1 Did EoS happen? Yes, at all four step sizes

| 2/η | t_edge | median (λ₁ − 2/η)/(2/η) after t_edge+200 | 5–95 % | λ₂ median | final loss | steps with loss ↑ after edge |
|---|---|---|---|---|---|---|
| 50 | 8 | **+7.6 %** | +2.2 … +12.5 % | +2.2 % | 0.072 | 47 % |
| 80 | 405 | **+6.2 %** | +1.6 … +9.8 % | +1.7 % | 0.094 | 47 % |
| 120 | 1274 | **+4.9 %** | +0.5 … +7.7 % | +1.3 % | 0.133 | 45 % |
| 200 | 3688 | **+2.1 %** | −1.5 … +4.3 % | −0.3 % | 0.172 | 36 % |

Initial sharpness is 88.0, which is above 2/η for 50 and 80. Those runs first catapult down (loss spike, λ₁ drops) and then sharpen back to the edge. The loss goes *up* on nearly half the steps at the edge, yet it falls overall. λ₂ also sits at the edge for 2/η ≤ 120: two directions are critical at once.

### 4.2 Is the sharpness estimate right?

The warm-started estimate (1 subspace iteration per step) was compared with cold starts (60 iterations) every 500 steps. The warm estimate is **never above** the cold one. Maximum relative underestimate: 7.6 % (2/η = 50), 5.8 % (80), 1.9 % (120), 0.4 % (200). The median Rayleigh–Ritz residual ‖Hu − λu‖/λ is 18 %, 14 %, 10 %, and 0.03 %. The residual is large when the top three eigenvalues are nearly degenerate, where subspace iteration mixes them. The bias therefore runs *against* the finding: the true λ₁ sits even further above 2/η than the tabled hover. The λ strokes in the prints are lower bounds, by up to about 8 % at the largest step size.

### 4.3 The coordinate problem and the choice made

The task correction warned that top eigenvectors rotate. They do: |⟨u₁(t), u₁(t−1)⟩| < 0.95 on 35 / 28 / 25 / 10 % of steps (2/η = 50/80/120/200). The braid along the current u₁ therefore has **1073 / 1052 / 779 / 373 crossings** after the edge. They occur at a median 25–50 % of the local peak amplitude, which is not where a genuine phase slip happens (a slip needs the amplitude to pass near zero).

Audit: 75 / 68 / 69 / 49 % of those crossings lie within 2 steps of an eigenvector swap. That sounds damning, but a swap is so frequent that a *random* post-edge step is within 2 steps of one 55 / 46 / 43 / 30 % of the time. The enrichment is about 1.4–1.6×. This is real, but the earlier handoff's "69–76 % are artefacts" overstated it. The stronger evidence is visual (§2.6): the u₁ braid jumps discontinuously.

**Declared coordinate for all main pieces: windowed PCA.**
1. Take the count-sketch of θ<sub>t</sub> − θ₀ and subtract a 21-step centred moving average.
2. In 64-step windows with a 32-step hop, take the top principal direction.
3. Sign-align it to the previous window, project, and blend overlapping windows with triangular weights.

In this coordinate, crossings after t_edge+100 drop to **225 / 82 / 23 / 0**. They occur at amplitude **nodes** (median 3.6 / 2.3 / 2.7 % of the local peak), and their coincidence with eigenvector swaps (35 / 33 / 57 %) is at or below the base rate. These are phase slips at beat nodes, not artefacts. The principal direction agrees with the top eigenvector (median |cos(PC, u₁)| after the edge 0.77 / 0.79 / 0.76 / 0.80), so it is the same oscillation seen in a frame that does not relabel itself.

Caveats of this choice: the PCA frame is a sketch-space estimate (1024 bins, so inner products are preserved only approximately), and its triangular blending smooths amplitude changes faster than about 32 steps.

### 4.4 Repeating structure

The oscillation comes in **bursts**. Burst peaks were detected with a declared detector (peaks of the 7-step running max of amplitude, prominence ≥ 0.3× median, ≥ 15 steps apart). Median spacing is **34 / 42 / 50 / 70 steps** (152 / 121 / 88 / 24 bursts), so bursts get slower as η shrinks. They are quasi-periodic, not periodic: the burst return maps (A<sub>n</sub> → A<sub>n+1</sub>, τ<sub>n</sub> → τ<sub>n+1</sub>) are diffuse clouds with no low-dimensional curve. **No fractal or self-similar claim is made in this project**, and none was tested for, because nothing here suggested one.

## 5. Negative results and surprises

1. **The quadratic model predicts the wrong equilibrium.** In the growth plate (§2.3) the binned median growth crosses zero at r ≈ +5–8 % above the edge, not at r = 0. Below that point, the oscillation *shrinks* even though λ₁ > 2/η. This is consistent with the hover values in §4.1 and with a cubic damping term (self-stabilisation). Part of the offset may come from the PCA coordinate's smoothing and from the sharpness underestimate (§4.2), which moves points left, i.e. the true offset is larger.
2. **The naive braid is mostly artefact** (§4.3), and even the audit statistic needed a base-rate null to be read correctly.
3. **Return maps have no structure** (§4.4). The self-stabilisation loop does not reduce to a 1D map of burst peaks.
4. **Orbit plane:** loops exist (clearly spiral-shaped at 2/η = 200), but for large η they are a tangle, because λ₂ and λ₃ join the edge and the dynamics is not two-dimensional.
5. **float64 was not used** (17× slower). **Eigenvalue refreshes every 10 steps** (scout) put visible kinks into x, so the main run refreshes every step.
6. **Whole-run prints** cannot show individual strands at 6000 steps; the 600-step detail prints are the ones that read as a braid.

## 6. Caveats

- One architecture, one dataset subset, one seed, MSE loss. Cohen et al. report the effect for cross-entropy and other architectures as well, but that was not checked here.
- 2/η is the full-batch GD threshold. Minibatch SGD, momentum (threshold (2 + 2β)/η), and Adam (preconditioned sharpness) differ.
- λ values are warm-start estimates and lower bounds (§4.2). λ₁ and λ₂ are near-degenerate at the edge, so "the" top eigenvector is ill-defined; hence the PCA coordinate.
- "Braid" is a declared visual metaphor: even and odd steps drawn as strands. Nothing is braided in weight space.
- Colour mappings (strand colours, Spectral splits, riso inks, lajolla step colouring) are aesthetic declarations. Spectral splits are rank-normalised per side, which equalises histograms and exaggerates small differences near the seam.
- File sizes: every gallery file is under 20 MB (largest: `film_eta80.mp4`, 12 MB).

## 7. References

- Cohen, Kaur, Li, Kolter, Talwalkar. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR 2021, arXiv:2103.00065.
- Damian, Nichani, Lee. *Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability.* ICLR 2023, arXiv:2209.15594.
- Lewkowycz, Bahri, Dyer, Sohl-Dickstein, Gilmer. *The large learning rate phase of deep learning: the catapult mechanism.* arXiv:2003.02218.
- Arora, Li, Panigrahi. *Understanding Gradient Descent on the Edge of Stability in Deep Learning.* ICML 2022, arXiv:2205.09745.
- Sohl-Dickstein. *The boundary of neural network trainability is fractal.* arXiv:2402.06184 (source of the Spectral split idiom).
- Palettes: `art/color-research/palettes.py`; cmcrameri (Crameri 2018); ColorBrewer Spectral.
