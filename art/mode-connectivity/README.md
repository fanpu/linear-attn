# One Basin

*Two networks trained from different seeds look like two separate valleys. List one network's hidden units in a different order and the mountain between them disappears.*

<img src="gallery/triptych_mnist_spectral.png" width="100%">

*Hero: loss along three paths between the same two MNIST networks, drawn as geological sections. The surface is the measured train loss, and each stratum is one digit class's share of it. The lower strip repeats each section at ×100 vertical exaggeration.*

## 1. The phenomenon

Train two copies of a network, A and B, from different random seeds. Then walk the straight line between them in weight space,

$$\theta(\lambda) = (1-\lambda)\,\theta_A + \lambda\,\theta_B,\qquad \lambda\in[0,1].$$

The loss rises steeply in the middle. The **barrier** is

$$\mathcal{B} = \max_\lambda\; L(\theta(\lambda)) - \big[(1-\lambda)L(\theta_A) + \lambda L(\theta_B)\big].$$

That rise makes it look as though A and B sit in isolated minima. Two findings argue otherwise:

1. **Curved paths.** Garipov et al. and Draxler et al. (2018) showed that low-loss curves connect such minima. We train a quadratic Bézier curve $\theta(t) = (1-t)^2\theta_A + 2t(1-t)\theta_C + t^2\theta_B$, keeping the endpoints fixed and learning the control point C.
2. **Permutations.** Hidden units are interchangeable. Permute the rows of $W_\ell$ and $b_\ell$ and the columns of $W_{\ell+1}$ by the same permutation $\pi_\ell$, and the network computes exactly the same function. Entezari et al. (2021) conjectured that most of the barrier comes from this symmetry. Git Re-Basin (Ainsworth et al., 2022) gave algorithms to remove it. **Weight matching** maximises $\sum_\ell \langle W_\ell^A,\, P_\ell W_\ell^B P_{\ell-1}^\top\rangle$ by coordinate descent, solving one linear assignment problem per layer per sweep. After matching, the straight line from A to $\pi(B)$ is nearly flat.

Two caveats matter and are measured below. Linear mode connectivity after matching **depends on width**: it fails for narrow networks. It is also **emergent during training**: it does not hold at initialisation in any interesting sense. REPAIR (Jordan et al., 2022) resets the per-unit activation statistics of the interpolated network, which removes a further part of the barrier caused by variance collapse.

## 2. Gallery

### 2.1 The triptych: three geological sections (MNIST, width 512)

<img src="gallery/triptych_mnist_survey.png" width="100%">

*Survey sheet. Measured: train loss along each path (201 points, all 60k images), stacked per-class contributions, test loss (dashed). Aesthetic: 19th-century map washes, hatched bedrock. ×100 strip = the same data rescaled.*

<table><tr>
<td><img src="gallery/triptych_mnist_night.png"><br><i>Night: strata in magma order.</i></td>
<td><img src="gallery/triptych_mnist_riso.png"><br><i>Riso: two inks alternate by class; the pink drum is offset by 0.25% of the path.</i></td>
</tr></table>

Panel I is the mountain. Panel II is the learned Bézier curve: seen at ×100 it runs *below* its endpoints. Panel III is the straight line to π(B): a 0.006-nat hill that only the ×100 strip shows. The strata show that every class loses confidence together on the mountain; no single class makes the barrier.

Fashion-MNIST, same pipeline (vertical exaggeration ×5, because the matched barrier is 0.05 nats):
<table><tr><td><img src="gallery/triptych_fmnist_spectral.png"></td><td><img src="gallery/triptych_fmnist_survey.png"></td></tr></table>

### 2.2 The plane through A, B and π(B)

<table><tr>
<td width="50%"><img src="gallery/plane_mnist_perm_spectral_basin.png"><br><i>Spectral split, the user favourite. The seam is the level set L = 0.011, the highest loss on the segment A–π(B). A and π(B) share one hourglass-shaped sublevel region; B sits in its own island. Declared aesthetic: rank-normalised per side.</i></td>
<td width="50%"><img src="gallery/plane_mnist_perm_topo.png"><br><i>Topographic survey: 44 log-spaced ink contours, with the same level set in red.</i></td>
</tr><tr>
<td><img src="gallery/plane_mnist_perm_aurora_basin.png"><br><i>palettes.py <code>aurora_ember</code> split, same seam.</i></td>
<td><img src="gallery/plane_mnist_perm_night.png"><br><i>Magma (reversed), linear in log-loss, with 30 faint contours.</i></td>
</tr></table>

### 2.2b The plane that contains the Bézier curve

A quadratic Bézier curve A → C → B lies exactly in the affine plane through its three control points, so this plane shows the whole trained curve with no projection error. The seam in both split plates is L = 0.006, the highest train loss along the curve.

<table><tr>
<td width="50%"><img src="gallery/plane_mnist_bezier_spectral_basin.png"><br><i>Plane {A, B, C}, Spectral split. The trained curve bends through a banana-shaped low-loss valley. The chord A–B (dashed) crosses the deep-red high-loss side. The control point C itself sits outside the basin. The maximum loss in the frame is 2005 nats.</i></td>
<td width="50%"><img src="gallery/plane_mnist_bezier_topo.png"><br><i>Same plane as a topographic survey: 44 log-spaced contours, with the seam level in red. Two shallow minima flank A and B inside one closed contour.</i></td>
</tr><tr>
<td><img src="gallery/plane_mnist_bezm_spectral_basin.png"><br><i>Plane {A, π(B), C′}: the Bézier curve trained <b>after</b> matching. The valley is wider and flatter, and the straight chord already runs inside it (matched barrier 0.0055).</i></td>
<td><img src="gallery/plane_mnist_bezier_night.png"><br><i>{A, B, C} in magma (reversed), linear in log-loss, with 30 faint contours.</i></td>
</tr></table>

Also: [bezier aurora split](gallery/plane_mnist_bezier_aurora_basin.png), [bezm topo](gallery/plane_mnist_bezm_topo.png). All three are MNIST train loss on a fixed 10k subset, 141² grid, bicubic-upsampled log-loss for display.

### 2.3 Width: the bookkeeping only works for wide networks

<table><tr>
<td width="50%"><img src="gallery/width_ridge_mnist_night.png"></td>
<td width="50%"><img src="gallery/width_ridge_mnist_paper.png"></td>
</tr></table>

*One row per width (32 to 2048), three independent pairs per width. Ghost band: the naive line (min–max over pairs). Solid: after weight matching. Orange/red line: matching + REPAIR. Every row uses the same vertical scale.*

<img src="gallery/width_barrier_mnist.png" width="100%">

WIDTH_PLANES_TBD

### 2.4 Emergence: the basin is grown, not found

<video src="gallery/emergence_mnist.mp4" autoplay loop muted playsinline width="100%"></video>

*[GIF](gallery/emergence_mnist.gif). The two runs are checkpointed at 28 epochs from 0 to 20. At each epoch k the checkpoints are weight-matched to each other (π_k) and interpolated.*

<table><tr><td><img src="gallery/emergence_mnist_strata_paper.png"></td><td><img src="gallery/emergence_mnist_strata_night.png"></td></tr></table>

*Each line is one checkpoint's profile, with log-loss on the same scale throughout. Left: raw. Right: matched.*

Fashion-MNIST shows the same story with a slower settling. The naive barrier grows to 0.96 by epoch 0.05 and ends at 1.43. The matched barrier reaches 0.70 at epoch 0.035, then falls to 0.35 by epoch 0.075 and stays around 0.04–0.05 from epoch 8 on. As on MNIST, applying the *final* permutation to early checkpoints gives a lower barrier than their own π_k does (0.13 vs 0.58 at epoch 0.05). So the permutation that connects the trained networks is already in place very early.

<table><tr><td><img src="gallery/emergence_fmnist_strata_paper.png"></td><td><img src="gallery/emergence_fmnist_strata_night.png"></td></tr></table>

FMNIST_FILMS_TBD

### 2.5 The permutation itself

<table><tr>
<td width="50%"><img src="gallery/perm_mnist_ink.png"><br><i>π₁ ∈ S₅₁₂, one dot per row. Dot size is aesthetic (2.4 cells).</i></td>
<td width="50%"><img src="gallery/perm_mnist_riso.png"><br><i>π₁, π₂, π₃ overprinted in three riso inks, with deliberate sub-cell misregistration.</i></td>
</tr></table>

<table><tr><td><img src="gallery/similarity_mnist_night.png"></td><td><img src="gallery/similarity_mnist_paper.png"></td></tr></table>

*The same 512×512 cosine-similarity matrix between A's and B's first-layer units, before and after reordering the columns by π₁. No values change; a diagonal appears.*

### 2.6 Units: quilts and twins

<img src="gallery/twins_mnist.png" width="100%">

*Twins: raw first-layer weights of 24 well-matched pairs (A left, π(B) right), Crameri vik.*

<table><tr>
<td><img src="gallery/quilt_mnist_vik.png"><br><i>All 512 units: A | π(B) | B. vik, per-tile scale.</i></td>
<td><img src="gallery/quilt_mnist_riso.png"><br><i>Two-ink riso variant.</i></td>
<td><img src="gallery/quilt_mnist_spectral.png"><br><i>Spectral sign split. A labelled variant: rank normalisation amplifies weight noise into confetti.</i></td>
</tr></table>

Fashion-MNIST: [quilt vik](gallery/quilt_fmnist_vik.png), [quilt riso](gallery/quilt_fmnist_riso.png), [quilt spectral](gallery/quilt_fmnist_spectral.png), [similarity night](gallery/similarity_fmnist_night.png), [similarity paper](gallery/similarity_fmnist_paper.png).

### 2.7 Films

<video src="gallery/walk_mnist.mp4" autoplay loop muted playsinline width="100%"></video>

*Walk ([GIF](gallery/walk_mnist.gif)): 900 test digits coloured by the class that the network at t predicts (colorcet glasbey_light, a nominal palette), with brightness showing confidence. On the naive line whole rows of digits collapse into one wrong colour near t = 0.5. On the other two paths nothing visibly changes.*

<video src="gallery/sorting_mnist.mp4" autoplay loop muted playsinline width="100%"></video>

*Sorting ([GIF](gallery/sorting_mnist.gif)): weight matching re-lists B's units sweep by sweep. The first sweep takes the test barrier from 1.33 to 0.007 nats. The remaining 12 sweeps polish it.*


## 3. What was computed

All numbers come from `cache/*.log` and `cache/*.npz`.

- **Model:** MLP 784 → W → W → W → 10, ReLU, PyTorch default (Kaiming-uniform) init. Hero width W = 512, the Git Re-Basin MLP architecture. Functional implementation in `common.py`.
- **Data:** MNIST and Fashion-MNIST, 60k train / 10k test, standardised with the global train mean and std. No augmentation.
- **Training:** Adam, lr 1e-3, batch 512, 20 epochs, float32 with TF32 off. Seed s sets both init and batch order. Hero pair: seeds 0 and 1. Width series: pairs (0,1), (2,3), (4,5).
- **Endpoints (MNIST):** train loss 0.0046 / 0.0070, test accuracy 98.39% / 98.11%. **(Fashion-MNIST):** train loss 0.107 / 0.118, test accuracy 89.5% / 88.9%.
- **Weight matching:** Git Re-Basin Algorithm 1, coordinate descent in random layer order, exact LAP per layer (`scipy.optimize.linear_sum_assignment`, float64), run until no layer improves. MNIST converged in 13 sweeps (under 1 s), Fashion-MNIST in 14. Check: π(B) and B give identical test loss to 1e-8.
- **Bézier:** quadratic, control point initialised at the midpoint, trained 20 epochs with Adam 1e-3 (cosine decay), one t ~ U(0,1) per minibatch.
- **REPAIR:** for every λ, each hidden unit's pre-activation mean and std are rescaled to the interpolation of the endpoints' statistics (measured on 10k train images), sequentially layer by layer, and folded into W and b.
- **Evaluation:** 1-D paths use 201 λ on the full train and test sets, with cross-entropy summed in float64 and batched with `baddbmm`. Width series: 25 λ. Barrier: max over λ of L(λ) − [(1−λ)L(0) + λL(1)]; the midpoint definition max L − (L₀+L₁)/2 is also logged.
- **Emergence:** checkpoints at 28 epochs (0 … 20) of both hero runs. At each one, weight-match the two checkpoints and evaluate 25-point lerps. Also logged: the final π applied to that checkpoint.
- **Planes:** θ = A + x·u + y·v with u, v orthonormal (Gram–Schmidt through the three points), in raw L2 weight units. Hero: 141² grid, 10k fixed train images (+ full test for the perm plane). Width planes: 97², 5k train images. Rendered as bicubic-upsampled log-loss.
- **Compute:** the GPU queue was full, and the Grace CPU turned out to be fast for these MLPs, so most runs used the CPU (`MC_DEV=cpu`). Hero MNIST: 3061 s CPU. Hero Fashion-MNIST: 5234 s CPU (training, paths, and 28-checkpoint emergence). Width 32–512: about 45 min CPU. Width 1024: 521 s GPU. Width 2048: 1821 s GPU. Hero planes: 1590 s GPU for the perm plane, 811 s GPU for the Bézier plane and 814 s GPU for the matched-Bézier plane (141², 10k train images each). Width planes: W811 s GPU for the Bézier plane and 814 s GPU for the matched-Bézier plane (141², 10k train images each). Rendering is CPU only.

Reproduce (from this directory; add `MC_DEV=cpu OMP_NUM_THREADS=4` to run on CPU):
```
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_hero.py mnist;  $PY compute_hero.py fmnist
$PY compute_width.py mnist            # widths 32..2048, 3 pairs (resumable)
$PY compute_planes.py hero mnist --res 141 --ntrain 10000 --planes perm,bezier,bezm --G 64
$PY compute_planes.py width mnist --res 97 --ntrain 5000
$PY analyze_units.py mnist; $PY analyze_units.py fmnist
$PY render_triptych.py mnist; $PY render_triptych.py fmnist
$PY render_planes.py mnist --plane perm; $PY render_planes.py mnist --plane bezier --styles spectral,topo
$PY render_width.py mnist; $PY render_emergence.py mnist
$PY render_units.py mnist --pieces perm,similarity,quilt,pairs; $PY render_units.py fmnist --pieces quilt,similarity
$PY render_walk.py mnist; $PY render_sorting.py mnist
```


## 4. Verification and honesty

**Did the phenomenon appear? Yes, in every form the docs describe, with the width caveat.**

| path (width 512) | MNIST barrier, train / test (nats) | Fashion-MNIST, train / test | worst test acc on path (MNIST) |
|---|---|---|---|
| naive A → B | **1.399** / 1.328 | 1.427 / 1.130 | 88.7% |
| B → π(B) (same function at both ends!) | 1.355 / 1.283 | 1.383 / 1.076 | 90.0% |
| naive + REPAIR | 0.106 / 0.082 | 0.407 / 0.226 | 95.2% |
| Bézier A → C → B | **0.000** / 0.000 (midpoint def. 0.0012) | 0.000 / 0.003 | 98.1% |
| matched A → π(B) | **0.0055** / 0.000 | 0.050 / 0.000 | 98.1% |
| matched + REPAIR | 0.0014 / 0.000 | 0.032 / 0.005 | 98.1% |

A test barrier of 0.000 means test loss never rose above the endpoint baseline; for Fashion-MNIST it actually dips in the middle. ‖A − B‖ = 53.2 and ‖A − π(B)‖ = 43.9: matching moves B closer, but not close.

**Width series (MNIST, mean over 3 pairs, train barrier):**

| width | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| naive | 2.66 | 1.66 | 0.93 | 0.93 | 1.26 | 1.75 | 2.01 |
| matched | 0.644 | 0.267 | 0.081 | 0.012 | 0.003 | 0.001 | 0.000 |
| naive + REPAIR | 2.62 | 1.72 | 0.57 | 0.22 | 0.082 | 0.060 | 0.048 |
| matched + REPAIR | 0.334 | 0.190 | 0.055 | 0.006 | 0.000 | 0.000 | 0.000 |

The matched barrier falls by about 3× per doubling of width. The test barrier after matching is exactly 0 from width 256 up. Width 32 is clearly *not* linearly connected after matching: 0.64 nats, and still 0.33 with REPAIR. The naive barrier is non-monotone and *grows* again above width 256. REPAIR on the unmatched line helps a lot at large width (0.05 at 2048) but slightly hurts at width 64 (1.72 vs 1.66 on average).

**Emergence (MNIST hero runs):** at initialisation the barrier is 0.001, trivially, because every point predicts roughly uniformly (loss 2.30). By epoch 0.05 the naive barrier is 1.16 and the matched one 0.92. After that the naive barrier plateaus at 1.2–1.45, while the matched barrier falls steadily: 0.28 (epoch 0.5), 0.15 (1), 0.036 (2.5), 0.006 (20). Unexpected: the *final* permutation applied to early checkpoints gives lower barriers than matching those checkpoints directly (for example 0.25 vs 0.92 at epoch 0.05). The alignment the runs end with is partly present long before the weights are distinctive enough for matching to find it. The permutation found at epoch k agrees with the final one on only 1–59% of units (layer 1), rising over training.

**Sorting:** the first weight-matching sweep takes the test barrier from 1.328 to 0.007 (objective 21.8 → 386.8). The remaining 12 sweeps raise the objective to 474.2 and the barrier reaches 0.000. Mean cosine similarity of matched first-layer units is 0.35 (unmatched diagonal: 0.07). 0 of 512 units are fixed points of π₁.

**Plane:** in the plane through A, B and π(B) (141², 10k train images), the sublevel set at L = 0.011 connects A and π(B) and excludes B, as the Spectral seam and the red contour show. This is a statement about one plane only (see Caveats).

**Fractal or self-similar claims:** none. The loss planes are smooth at the grid scale; there is nothing to box-count. The perm-matrix prints are uniformly random point sets on purpose.

**Negative results and what didn't work**
- The GPU queue was full for over 25 minutes, so the heavy MLP work moved to CPU. A 161² CPU plane was abandoned at 50 min under a load average of 40 and rerun at 141² on the GPU.
- Splitting the plane seam at chance level (L = ln 10) put almost the whole plane on one side, so it was dropped. The basin-level seam is used instead.
- The Spectral sign-split quilt amplifies weight noise and is weaker than the linear vik version; it is kept only as a labelled variant.
- REPAIR is not a uniform improvement for naive interpolation at small width (see above).
- The matched Bézier curve on Fashion-MNIST has a *test* barrier of 0.17 even though its train barrier is 0. The curve overfits the training set.
- CIFAR-10 / CNNs were not attempted (time and GPU). Git Re-Basin reports that those need much larger width multipliers, so the MLP result must not be generalised.

**Doc claims checked.** §5's story holds for these MLPs: naive barrier, near-flat Bézier curve, near-flat matched line, and emergence during training. "LMC after permutation" is width-dependent, as the task corrections say: it fails at width 32–64. The doc's "512 × 512 binary matrix" is exactly what the hero π₁ is.


## 5. Caveats

- **A plane is a slice.** The 2-D planes show exactly the loss on one affine plane through three trained points. Non-convexity in a slice implies non-convexity overall, but flatness or connectedness in a slice says little about the rest of the space. "One basin" here means that the segment A–π(B) and the sublevel set in *this plane* are connected. It does not mean the full loss surface has a single basin.
- **Euclidean distances in weight space are a choice.** The same function can sit at very different Euclidean distances (for example, permutations, or ReLU rescaling in the style of Dinh et al.). Weight matching itself relies on this: A and π(B) are closer in L2 than A and B, while B and π(B) are the *same function*.
- **"Barrier-free" is measured on a small MLP.** These are 3-hidden-layer MLPs on MNIST and Fashion-MNIST. Git Re-Basin reports that ResNets and VGGs on CIFAR need much more width, and that the barrier on ImageNet does not vanish. Nothing here tests those settings. See the negative results.
- **The Bézier curve's barrier is 0 by the linear-baseline definition** because the curve runs *below* the endpoints' loss for most of its length. The ×100 strip shows that shape. Its midpoint-definition barrier is 0.0012 nats.
- **Weight matching is a heuristic.** It finds a local optimum of a bilinear objective. It is not the permutation that minimises the barrier.
- **Colour.** Every Spectral or split image is rank-normalised per side. Colours encode order, not the size of loss differences, and are not comparable between images.

## 6. References

- Garipov, Izmailov, Podoprikhin, Vetrov & Wilson, *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs*, NeurIPS 2018. arXiv:1802.10026
- Draxler, Veschgini, Salmhofer & Hamprecht, *Essentially No Barriers in Neural Network Energy Landscape*, ICML 2018. arXiv:1803.00885
- Frankle, Dziugaite, Roy & Carbin, *Linear Mode Connectivity and the Lottery Ticket Hypothesis*, ICML 2020. arXiv:1912.05671
- Entezari, Sedghi, Saukh & Neyshabur, *The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks*, ICLR 2022. arXiv:2110.06296
- Ainsworth, Hayase & Srinivasa, *Git Re-Basin: Merging Models modulo Permutation Symmetries*, ICLR 2023. arXiv:2209.04836
- Jordan, Sedghi, Saukh, Entezari & Neyshabur, *REPAIR: REnormalizing Permuted Activations for Interpolation Repair*, ICLR 2023. arXiv:2211.08403
- Dinh, Pascanu, Bengio & Bengio, *Sharp Minima Can Generalize for Deep Nets*, ICML 2017. arXiv:1703.04933
- Sohl-Dickstein, *The boundary of neural network trainability is fractal*, 2024 (Spectral split colour convention). github.com/Sohl-Dickstein/fractal
