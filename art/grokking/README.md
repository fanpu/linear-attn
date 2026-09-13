# Circuit Formation: grokking as crystallization

*A one-layer transformer learning (a + b) mod 113 memorizes first. Tens of thousands of steps later, its embedding snaps into star polygons {113/k}.*

<video src="gallery/film_grid_seed0_nocturne.mp4" autoplay loop muted playsinline width="720"></video>

<img src="gallery/specimen_plotter.png" width="720">

## The phenomenon

The task is to predict c = (a + b) mod p with p = 113, given the input sequence "a b =". The training set is a random 30% of all 113² pairs, and weight decay is strong (λ = 1). The network fits the training set within about 200 steps while test accuracy stays at chance. Much later, test accuracy jumps to 100%. That delayed jump is "grokking".

Nanda et al. (2023) reverse-engineered the solution. The embedding sends each token a to cosines and sines at a few **key frequencies** w_k = 2πk/113:

  W_E[a] ≈ Σ_k ( cos(w_k a) u_k + sin(w_k a) v_k )

The MLP forms products like cos(w_k a)cos(w_k b). The logits then read out Σ_k cos(w_k (a + b − c)), which peaks at the correct c.

In the 2D plane of frequency k, token a sits at angle 2π·k·a/113. That means integer order around the circle looks scrambled: ring neighbours differ by k⁻¹ mod 113. Drawing a line from a to a+1 traces the **star polygon {113/k}**. Each key frequency has its own star.

Nanda et al. describe three phases: **memorization**, **circuit formation** (the Fourier circuit grows while test loss stays flat), and **cleanup** (weight decay removes the memorizing solution, and test loss collapses).

## Gallery

All data comes from real training runs. Every choice about colour, line weight or normalization is marked **declared** below.

### Crystallization film (seed 0, key frequencies 5, 9, 36, 53)
- `gallery/film_grid_seed0_nocturne.mp4` (1080², 24 fps, 69 s) and `.gif` (480 px, 8 fps, 9.7 MB).
- **Measured:** W_E(t) at 1,645 checkpoints, projected onto the *final* (cos, sin) plane of each key frequency. The camera is fixed, so any motion is the weights moving.
- **Declared:**
  - each frame is centred and scaled to unit RMS radius, because weight decay shrinks everything;
  - edges are coloured by residue a with the cyclic map CET-C6;
  - time warp: every checkpoint up to step 1k, every 20 steps up to step 26.3k, then every 100 steps.
- The bottom strip shows the measured train and test loss. R is the ring order parameter |Σ_a z_a e^{-i w_k a}| / √(P Σ|z_a|²).

### Star polygons, final checkpoint (seed 0)
| nocturne | plotter | plate |
|---|---|---|
| <img src="gallery/stars_smallmultiples_seed0_nocturne.png" width="300"> | <img src="gallery/stars_smallmultiples_seed0_plotter.png" width="300"> | <img src="gallery/stars_smallmultiples_seed0_plate.png" width="300"> |
| <img src="gallery/stars_overlay_seed0_nocturne.png" width="300"> | <img src="gallery/stars_overlay_seed0_plotter.png" width="300"> | <img src="gallery/stars_overlay_seed0_plate.png" width="300"> |
| <img src="gallery/stars_labelled_seed0_nocturne.png" width="300"> | <img src="gallery/stars_labelled_seed0_plotter.png" width="300"> | <img src="gallery/stars_labelled_seed0_plate.png" width="300"> |

- Rows are: one star per key frequency; all key stars superposed; and the k = 53 ring with every integer labelled (neighbours differ by 32 = 53⁻¹ mod 113).
- **Declared:** line weights, the palette, and a fixed rotation/reflection that puts token 0 at angle 0.

### Seed wall and specimen sheet (12 seeds)
| nocturne | plotter | plate |
|---|---|---|
| <img src="gallery/seedwall_nocturne.png" width="300"> | <img src="gallery/seedwall_plotter.png" width="300"> | <img src="gallery/seedwall_plate.png" width="300"> |
| <img src="gallery/specimen_nocturne.png" width="300"> | <img src="gallery/specimen_plotter.png" width="300"> | <img src="gallery/specimen_plate.png" width="300"> |

- **Seed wall:** all key stars of each seed, superposed.
- **Specimen sheet:** all 46 key-frequency stars from the 12 seeds, ordered by k. As k grows, the winding number goes from a near-circle ({113/1}) to a dense sunburst ({113/56}).

### Spectral pieces
| nocturne | plotter | plate |
|---|---|---|
| <img src="gallery/spectrogram_seed0_nocturne.png" width="300"> | <img src="gallery/spectrogram_seed0_plotter.png" width="300"> | <img src="gallery/spectrogram_seed0_plate.png" width="300"> |
| <img src="gallery/spectrogram_12seeds_nocturne.png" width="300"> | <img src="gallery/spectrogram_12seeds_plotter.png" width="300"> | <img src="gallery/spectrogram_12seeds_plate.png" width="300"> |
| <img src="gallery/spectral_diptych_seed0_nocturne.png" width="200"> | <img src="gallery/spectral_diptych_seed0_plotter.png" width="200"> | <img src="gallery/spectral_diptych_seed0_plate.png" width="200"> |

- **Measured:** ‖DFT_a W_E‖₂ over d_model, for each frequency k = 1…56 and at every checkpoint.
- **Declared:**
  - each column is normalized to the share of ‖W_E‖² (the DC term is excluded);
  - sqrt tone map, saturating at the final peak;
  - checkpoints are resampled onto a 20-step grid.
- The **Before/After diptych** compares step 20,430 (the end of seed 0's plateau: memorized, test accuracy below 10%) with step 40,000. Both panels use the same scale.

### Training curves
| nocturne | plotter | plate |
|---|---|---|
| <img src="gallery/curves_seed0_nocturne.png" width="300"> | <img src="gallery/curves_seed0_plotter.png" width="300"> | <img src="gallery/curves_seed0_plate.png" width="300"> |

- **Measured:** train loss (every step), test loss (every 10 steps), and restricted and excluded loss (every 250 steps).
- **Declared phase boundaries:**
  - circuit formation starts at the minimum of excluded loss;
  - cleanup starts at the last step where test accuracy is below 10% before test accuracy reaches 50%;
  - "stable" starts when test accuracy reaches 99.9%.

## What was computed

- **Model:** follows Nanda et al. §3 and the TransformerLens `Grokking_Demo`.
  - 1 layer, d_model 128, 4 heads × 32, d_mlp 512, ReLU, no LayerNorm, learned positional embeddings, untied embed/unembed;
  - biases frozen at 0; init N(0, 0.8/√128);
  - logits are read at "="; float64 log-softmax.
- **Training:**
  - full-batch AdamW, lr 1e-3, weight decay 1.0, betas (0.9, 0.98), no warmup;
  - **40,000 steps** (same as the paper);
  - 30% train fraction (3,830 pairs); the test set is the other 8,939.
- **Seeds:**
  - init seeds 0–7 share the TL demo's data split (seed 598);
  - init seeds 8–11 use data-split seeds 1–4.
  - The 12 models were trained as a vectorized ensemble. The loss is a sum of per-model means and AdamW is elementwise, so each model trains exactly as it would alone.
- **Checkpoints:**
  - W_E at every step ≤ 50, every 10 steps ≤ 1k, then every 20 steps (2,096 checkpoints);
  - full parameters at powers of 2 and every 250 steps (177 checkpoints).
  - Model weights are float32; spectra and projections are computed in float64.
- **Analysis:** `analyze.py` does the following.
  - DFT spectra and ring projections.
  - Key frequencies: every k whose Fourier norm in the neuron-logit map W_L = W_out·W_U exceeds 20% of the maximum (a declared threshold).
  - Restricted and excluded loss, following the paper's 2D-DFT definition: keep, or drop, the constant plus the 4 (cos/sin a × cos/sin b) terms for each key k.
  - Final neuron activations over the full grid.
- **Wall time:**
  - training took 2 × 2.2 h on a GPU shared with about 5 other agents (the ensemble was split into two 6-model processes);
  - analysis took 4 min;
  - rendering took about 10 min on the CPU.

```
cd art/grokking
../_shared/gpu_run.sh ../.venv/bin/python train.py --steps 40000 --full_every 250 --init_seeds 0,1,2,3,4,5 --data_seeds 598,598,598,598,598,598 --out cache/runA
../_shared/gpu_run.sh ../.venv/bin/python train.py --steps 40000 --full_every 250 --init_seeds 6,7,8,9,10,11 --data_seeds 598,598,1,2,3,4 --out cache/runB
../_shared/gpu_run.sh ../.venv/bin/python analyze.py
python render_stars.py --seed 0; python render_curves.py --seed 0; python render_spectral.py --seed 0
python render_seedwall.py; python render_film.py --seed 0 --styles nocturne
```

## Verification

- **Grokking appeared in all 12 runs.**
  - Train accuracy reaches 100% by step 160–220.
  - Test loss first *rises* to about 20–33 nats.
  - Test accuracy reaches 99.9% at steps 5,760 / 6,910 / 8,190 / 9,360 / 9,680 / 10,140 / 10,260 / 10,830 / 11,080 / 12,040 / 15,160 / 23,330 (median 10.2k). The paper reports about 10k.
  - At step 40k, every model has test accuracy 1.0 and test loss 1.4–3.4 × 10⁻⁷.
  - Seeds 1 and 2 generalize *gradually* from about step 400 instead of in one late jump. Seed 0 is the latest and sharpest grokker, which is why it is the film's subject.
- **The three phases show up (seed 0, plate II):**
  - Excluded loss bottoms out at step 1.5k (5 × 10⁻⁶), then rises steadily to about 20 by step 23k while train loss stays flat. That is circuit formation.
  - Restricted loss falls before test loss does.
  - Both collapse together at 20.4k–23.3k. That is cleanup.
  - The Gini coefficient of the W_E Fourier norms goes from 0.20 on the plateau to 0.75–0.80 at the end.
- **Fourier circuit:**
  - Rings are near-perfect at the end: R = 0.989–1.000 for all 46 key frequencies, and singular-value ratio 0.88–1.00.
  - Each key plane holds 13–42% of ‖W_E‖².
  - Excluded loss at the end is 5.4–14.8 nats, against a restricted loss of about 10⁻⁷.
  - For seed 0, 85% of live MLP neurons have more than 85% of their non-DC power in a single key frequency (84.6% in the paper).
- **Two independent implementations agree:** the training forward pass (batched bmm) and the analysis forward pass (per-model einsum) give the same final train loss to 7 significant digits.
- **Key frequencies by seed:**

  | seed | keys |
  |---|---|
  | 0 | 5, 9, 36, 53 |
  | 1 | 9, 18, 53, 55 |
  | 2 | 31, 53, 55 |
  | 3 | 10, 36, 38, 52 |
  | 4 | 7, 8, 21, 38 |
  | 5 | 22, 34, 36, 40 |
  | 6 | 23, 41, 53 |
  | 7 | 9, 30, 36, 53, 55 |
  | 8 | 1, 16, 31, 44, 56 |
  | 9 | 31, 39, 40, 53 |
  | 10 | 33, 43, 44 |
  | 11 | 8, 42, 49 |

  Each seed uses 3–5 key frequencies, and no two seeds share a set.
- **Surprise:** the choice is *not* uniform. k = 53 appears in 6 of 12 seeds (5 of the 8 that share one data split, plus one of the others), k = 36 in 4, and k = 9, 31, 55 in 3 each. With 56 possible frequencies this is unlikely to be chance. Whether it comes from the shared split, the init scale, or something about p = 113 is untested. So "the frequencies are arbitrary" holds only in the weaker sense that the set varies by seed.
- **W_E vs W_L:** the W_E spectrum sometimes has one or two extra, weaker peaks that are not in W_L (for example seed 6 has k = 7 in W_E only). The paper saw the same thing (6 embedding frequencies, 5 key).
- **Fractal claims:** none. Star polygons are exactly periodic, not fractal, so no box counting was done.
- **The film's jaggedness is real:** even at R ≈ 0.999, the angular spacing around each ring is not perfectly even (the labelled k = 53 plate shows points clustered in pairs). Nothing is smoothed.

## Caveats

- **Grokking is a setting, not a law.** It needs restricted data (30% here) *and* regularization. The paper finds no grokking without weight decay, and no train/test gap with enough data. None of this implies that networks in general grok.
- **The mechanism is contested at the level of "why".** Circuit formation followed by cleanup is one account. Others explain it by the efficiency of the generalizing circuit under weight decay, or by slingshot/optimizer effects. The pictures show *what* the network ends up computing, not which account is right.
- **The pictures depend on how they were framed:**
  - Each star is a 2D projection of a 128-D embedding onto a plane *chosen from the final checkpoint*, so earlier tangles are that fixed plane's view of the weights.
  - Per-frame RMS normalization hides the large change in weight norm.
  - With a per-frame plane, early frames would look different.

## Ideas explored / not pursued

Brainstorm, with the chosen ideas in bold:

1. **Specimen sheet** of every {113/k}, ordered by k. Made; it is the strongest still.
2. **Seed-wall overlays.** Made.
3. Neuron "textile": 113×113 activation plaids grouped by frequency, with an "unscrambled" companion re-indexed by k·a mod 113. `render_textile.py` is written and activations are cached, but it was **not rendered** because the scope was reduced.
4. Answer-table film: argmax over (a, b) moving from memorization noise to anti-diagonal bands, on a cyclic map. `render_tables.py` is written and tables are cached, but it was not rendered.
5. Lissajous figures from pairs of learned rings. Dropped: the curve is mostly determined by the two chosen frequencies rather than the data.
6. Torus-knot view of two rings jointly, (cos k₁a, sin k₁a, cos k₂a, sin k₂a). Not pursued, because 4D→2D would add more declared choices than it shows data.
7. An unnormalized film in which weight decay visibly shrinks the tangle. Not pursued.
8. The riso two-ink style was written for the stars, seed wall and spectrogram, then dropped to save budget.

Iteration notes, from looking at the renders:
- **Round 1:**
  - Seed-wall overlays coloured by residue turned to grey mush, so they were switched to a single warm ink at low alpha.
  - Stars were too small in their panels.
  - The "before" spectrum was taken at step 180, which is uninformative, so it was moved to the end of the plateau (step 20,430).
  - Spectrogram phase labels collided.
  - The first film had star panels overlapping the loss strip, a 40 MB MP4 and a 19.6 MB GIF. Its layout and encoding were fixed.

## References

- Nanda, Chan, Lieberum, Smith, Steinhardt. *Progress Measures for Grokking via Mechanistic Interpretability*, ICLR 2023, arXiv:2301.05217. Setup, phases, and restricted/excluded loss were checked against the PDF; the appendix confirms that "the specific frequencies differ by seed".
- TransformerLens `demos/Grokking_Demo.ipynb`: hyperparameters, frozen biases, data seed 598.
- Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, 2022.
- Liu et al., *Omnigrok*, 2022; Varma et al., *Explaining grokking through circuit efficiency*, 2023; Thilak et al., *The Slingshot Mechanism*, 2022. These are the competing explanations mentioned in the caveats.
