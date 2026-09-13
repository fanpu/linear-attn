# Weight Spectrum: a matrix learning to be less random

*Every eigenvalue of a weight matrix, printed while it trains: it starts as textbook random-matrix noise and grows a tail.*

<p align="center">
<img src="gallery/ridgeline_mlp_bs16_s1_FC1_sv_joy_logtime.png" width="46%">
<video src="gallery/ridge_film_mlp_bs16_s1_FC1_joy.mp4" autoplay loop muted playsinline width="46%"></video>
</p>

*Left: **Unknown Pleasures of SGD**. Each ridge is the measured singular-value density of the first layer (FC1, 1024×784) of an
MLP trained on FashionMNIST, from step 91 (top) to step 112,500 (bottom), with rows log-spaced in step. The bulk on the left is
the random-matrix part. The spikes that walk out to the right are eigenvalues training has pulled out of it.
Right: the same thing as a film. The bottom ridge is live, and a copy is printed and drifts up every 6 frames.*

## The phenomenon

Take a dense layer's weight matrix $W \in \mathbb{R}^{N\times M}$ (oriented so $N \ge M$, aspect ratio $Q = N/M$) and form the
correlation matrix $X = W^\top W / N$. Its eigenvalues $\lambda_i = s_i^2/N$ ($s_i$ are the singular values of $W$) make up the
**empirical spectral density** (ESD).

At initialization the entries are i.i.d. with variance $\sigma^2$. Random matrix theory then predicts the ESD exactly:
the **Marchenko–Pastur (MP) law**

$$\rho_{MP}(\lambda) = \frac{Q}{2\pi\sigma^2\lambda}\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)},\qquad \lambda_\pm = \sigma^2\left(1\pm \tfrac{1}{\sqrt Q}\right)^2 .$$

Nothing goes past $\lambda_+$ except finite-size fluctuations of order $M^{-2/3}$.

Martin & Mahoney's *heavy-tailed self-regularization* (HT-SR) program tracks how SGD pulls the ESD away from this law. First a few
eigenvalues bleed out past $\lambda_+$ ("bleeding out"), then they separate into spikes ("bulk + spikes"), and in well-trained
networks the tail looks like a power law $\rho(\lambda) \sim \lambda^{-\alpha}$ ("heavy-tailed"). They report that smaller batch
sizes push a network further along this sequence, and that smaller $\alpha$ tends to go with better test accuracy.

**In these runs (MLP 784-1024-1024-1024-10, FashionMNIST, SGD bs 16, 30 epochs):** at init every layer matches MP
(λmax / MP edge = 0.97–0.99, KS distance from the shuffled null ≤ 0.005). By the end, FC1 has **50 eigenvalues above the null bulk
edge**, λmax is **8.2× the MP edge** (FC2/FC3: 12×, 13×), and the tail fits a power law with **α = 1.94 ± 0.06** (FC2 2.42, FC3 3.05).
Over a batch-size series from 8 to 1024 (2 seeds each for bs 16–1024), the tail gets steadily lighter as batch size grows: FC1 α = 1.70,
1.94, 2.26, 2.83, 4.24, 7.8 at bs 8…256. Test accuracy does **not** follow: it sits flat at 0.895–0.898 from bs 16 to 256.

**Two controls are used throughout.**
1. **MP at the measured $\sigma^2$.** The theoretical curve uses the variance of the layer's current entries. Nothing is fitted.
2. **Shuffled-entries null.** At every checkpoint we also compute the ESD of the *same* matrix with its entries randomly permuted.
   This keeps the marginal distribution of the entries (including any heavy-tailed entries) and destroys all row/column
   correlations. Whatever survives the shuffle is not learned structure. "Eigenvalues above the null" counts
   $\lambda_i$ above the null's bulk edge, which is its *second* largest eigenvalue (see the rank-one gotcha below).

## Gallery

Unless noted, everything comes from the main art run `mlp_bs16_s1`, layer FC1. **Measured** means sorted eigenvalues of
$W^\top W/N$ (float64 SVD) at 197 checkpoints (150 linearly spaced plus 50 log-spaced, merged), the entry variance, and the shuffled null.
**Declared** means KDE bandwidths, height gammas, interpolation between checkpoints, colours, and misregistration.

### 1. Ridgeline stack (centrepiece)

<table>
<tr>
<td><img src="gallery/ridgeline_mlp_bs16_s1_FC1_sv_joy_logtime.png"></td>
<td><img src="gallery/ridgeline_mlp_bs16_s1_FC1_sv_joy.png"></td>
<td><img src="gallery/ridgeline_mlp_bs8_s0_FC1_sv_joy.png"></td>
</tr>
<tr>
<td>joy, rows log-spaced in step (58 checkpoints). Hero.</td>
<td>joy, rows linear in step (80). The tail edge becomes a straight diagonal: most outlier motion is spread evenly over linear time.</td>
<td>bs 8, seed 0, log rows. The heaviest tail in the series (α = 1.70), with sparse spikes at the far right.</td>
</tr>
<tr>
<td><img src="gallery/ridgeline_mlp_bs16_s1_FC1_sv_ink_logtime.png"></td>
<td><img src="gallery/ridgeline_mlp_bs16_s1_FC1_sv_gold_logtime.png"></td>
<td><img src="gallery/ridgeline_mlp_bs16_s1_FC1_sv_spectral_logtime.png"></td>
</tr>
<tr>
<td>ink: single-line plotter sheet on paper.</td>
<td>gold: palettes.py <code>klimt_gold</code>, shaded by row (declared).</td>
<td>Spectral (labelled variant): colour encodes time only. It is a sequential use of a diverging map.</td>
</tr>
<tr>
<td><img src="gallery/ridgeline_mlp_bs16_s1_FC1_sv_riso_logtime.png"></td>
<td><img src="gallery/ridgeline_mlp_bs16_s1_FC1_sv_riso.png"></td>
<td><img src="gallery/ridgeline_mlp_bs16_s1_FC1_sv_spectral.png"></td>
</tr>
<tr>
<td>riso, two drums: blue is the measured ESD, pink is the <i>same weights with entries shuffled</i>. The pink never grows a tail.</td>
<td>riso, linear rows.</td>
<td>Spectral, linear rows.</td>
</tr>
</table>

x is the singular value $s/\sqrt N = \sqrt\lambda$ on a linear axis. Height is (KDE density)^0.5 with bandwidth 0.012 × the init edge. Each ridge
occludes the ones behind it, as in the Joy Division cover. Also rendered: `ridgeline_..._sv_{ink,gold}.png` (linear rows).

### 2. Films

<table>
<tr>
<td><video src="gallery/ridge_film_mlp_bs16_s1_FC1_joy.mp4" autoplay loop muted playsinline width="100%"></video></td>
<td><video src="gallery/ridge_film_mlp_bs16_s1_FC1_gold.mp4" autoplay loop muted playsinline width="100%"></video></td>
<td><video src="gallery/esd_film_mlp_bs16_s1_FC1_night.mp4" autoplay loop muted playsinline width="100%"></video></td>
</tr>
<tr>
<td>Ridge film, joy (1080², 33 s; <a href="gallery/ridge_film_mlp_bs16_s1_FC1_joy.gif">GIF</a>). Linear training time. Between checkpoints the k-th largest eigenvalue is interpolated in log λ (declared).</td>
<td>Ridge film, gold (<a href="gallery/ridge_film_mlp_bs16_s1_FC1_gold.gif">GIF</a>).</td>
<td>Explanatory ESD film, night (<a href="gallery/esd_film_mlp_bs16_s1_FC1_night.gif">GIF</a>): histogram on log λ against the MP curve (measured σ²) and the shuffled null, plus a strip of every eigenvalue and the live fit numbers. Fit numbers always come from the last <i>measured</i> checkpoint.</td>
</tr>
</table>

### 3. Spectrograph plates: outliers peeling off the MP bulk

<table>
<tr>
<td><img src="gallery/plate_mlp_bs16_s1_FC1_magma.png"></td>
<td><img src="gallery/plate_mlp_bs16_s1_FC1_paper.png"></td>
<td><img src="gallery/plate_mlp_bs16_s1_FC1_riso.png"></td>
</tr>
<tr><td>magma: every eigenvalue exposed as a line. x is log₁₀ λ, rows run in linear step from init (top) to 112,500 (bottom).</td>
<td>paper: the same data as a photographic negative. The streamers are single eigenvalues.</td>
<td>riso: purple is measured, pink is the shuffled null. The null's edge also drifts right, because the weight norm grows, but it never streams.</td></tr>
<tr>
<td><img src="gallery/plate_mlp_bs16_s1_FC1_bio.png"></td>
<td><img src="gallery/plate_mlp_bs16_s1_FC1_spectral.png"></td>
<td><img src="gallery/plate_mlp_bs16_s1_FC2_paper.png"></td>
</tr>
<tr><td>bio: palettes.py <code>bioluminescence</code>.</td>
<td>Spectral (labelled variant on sequential exposure; it can band).</td>
<td>FC2 (1024×1024, Q = 1, so the bulk reaches down to λ≈0), paper. Also <code>plate_..._FC2_magma</code> and <code>..._FC3_magma</code>.</td></tr>
</table>

Rows between measured checkpoints track the k-th largest eigenvalue in rank order. That is interpolation, not eigenvector identity (declared).
Exposure is $1-e^{-\text{density}/D_0}$ (declared).

### 4. Random matrix theory against the measurement (riso)

<table><tr>
<td><img src="gallery/riso_mp_vs_esd_mlp_bs16_s1_fluo_pink_blue.png"></td>
<td><img src="gallery/riso_mp_vs_esd_mlp_bs16_s1_melon_indigo.png"></td>
<td><img src="gallery/riso_mp_vs_esd_mlp_bs16_s1_sunflower_federal_blue.png"></td>
</tr></table>

Solid mass is the MP law at each layer's measured σ², with nothing fitted. Bars and ticks are the measured eigenvalues. There are 3 layers × 5 checkpoints,
a sqrt vertical scale, and 2–4 px misregistration (declared).

### 5. Departure spectrograms, and the singular vectors

<table><tr>
<td><img src="gallery/departure_mlp_bs16_s1_FC1_aurora_ember_linear.png"></td>
<td><img src="gallery/departure_mlp_bs16_s1_FC1_sd_spectral.png"></td>
<td><img src="gallery/garments_mlp_bs16_s1_FC1_spectral.png"></td>
</tr><tr>
<td>log ρ_ESD − log ρ_null over (checkpoint, log λ), with the palettes.py <code>aurora_ember</code> split at 0 and a linear normalisation (the honest version). Warm means more eigenvalues than the null.</td>
<td>The same field in the Sohl-Dickstein Spectral split (each sign rank-normalised separately). Rank normalisation amplifies the frozen low-λ noise stripes, so treat it as decorative.</td>
<td>FC1's top-12 right singular vectors reshaped to 28×28, over training. Around step 1k they turn from noise into garment-like templates.</td>
</tr></table>

`gallery/ipr_mlp_bs16_s1.png`: inverse participation ratio of the singular vectors against rank. FC1's top two input-side (pixel) vectors reach IPR·d ≈ 7–8, against 3 for a Gaussian vector (about 2.5× more localised), and the bulk stays at ≈3.
The top output-side vector of FC2 and FC3 has IPR·d ≈ 1.4, *flatter* than random. That matches the near-uniform mean-shift direction (see the rank-one gotcha).

### 6. Verification plates

<table><tr>
<td><img src="gallery/verify_batch_series.png"></td>
<td><img src="gallery/verify_caveat_alpha_vs_generalization.png"></td>
</tr><tr>
<td><img src="gallery/verify_over_training_mlp_bs16_s1.png"></td>
<td><img src="gallery/verify_ccdf_mlp_bs16_s1.png"></td>
</tr></table>

## What was computed

- **Model:** MLP 784-1024-1024-1024-10, ReLU, Glorot-normal weights, zero biases. Layers FC1 (1024×784, Q=1.31), FC2 and FC3
  (1024×1024, Q=1). The 10×1024 head is not analysed.
- **Data:** FashionMNIST 60k train / 10k test, global mean/std normalisation, no augmentation.
- **Optimiser:** SGD with lr 0.01, momentum 0.9, no weight decay, 30 epochs, cross-entropy. The **same lr for every batch size**, deliberately,
  so that batch size is the only knob (see caveats).
- **Measurements:** at each checkpoint, float64 SVD of every layer, the eigenvalues of the element-shuffled matrix, the entry variance, a power-law fit
  (continuous Clauset MLE with xmin chosen by KS, the WeightWatcher estimator), top-32 singular vectors, and the IPR. Full W is saved at a few checkpoints.
- **Runs:** batch size ∈ {8, 16, 32, 64, 128, 256, 512, 1024}, seed 0; seed 1 for 16…1024; the main art run is bs 16 seed 1
  (112,500 steps, 197 checkpoints). All runs are on **CPU**, single-threaded, because the shared GPU was saturated: ≈20,000 s total
  (bs 16: 45 min, bs 8: 60 min). **GPU time: 0.**
- **Commands** (from this directory; the python is `/home/fzeng/ml/research/art/.venv/bin/python`):

```bash
bash run_series.sh                       # seed-0 series bs 16..1024 (5 in parallel)
bash run_seed1_queue.sh                  # seed-1 replicates bs 32..1024
python train.py --name mlp_bs16_s1 --data fmnist --widths 1024,1024,1024 --bs 16 --lr 0.01 --momentum 0.9 --epochs 30 \
  --n_ckpt 50 --n_lin 150 --n_full 12 --k_vec 32 --seed 1 --device cpu --threads 1          # main art run
python train.py --name mlp_bs8_s0 --data fmnist --widths 1024,1024,1024 --bs 8 --lr 0.01 --momentum 0.9 --epochs 30 \
  --n_ckpt 80 --n_full 12 --k_vec 32 --seed 0 --device cpu --threads 1
python render_verify.py mlp_bs16_s1      # prints the metrics table, writes cache/metrics_table.json + verify_*.png
python render_ridgeline.py mlp_bs16_s1 FC1 joy_log joy ink ink_log gold gold_log spectral spectral_log riso riso_log
python render_ridgeline.py mlp_bs8_s0 FC1 joy
python render_ridge_film.py mlp_bs16_s1 FC1 joy     # and gold; GIFs re-encoded at 480 px / 10 fps (see NOTES.md)
python render_esd_film.py mlp_bs16_s1 FC1 night
python render_plate.py mlp_bs16_s1 FC1 magma paper riso bio spectral
python render_riso_plate.py mlp_bs16_s1
python render_departure.py mlp_bs16_s1 FC1
python render_texture.py mlp_bs16_s1 garments ipr
```

## Verification and honesty

**Final metrics** (FC1 / FC2 / FC3; "above null" is the count of eigenvalues above the shuffled null's bulk edge; seeds listed as s0, s1):

| bs | steps | test acc (s0, s1) | train acc (s0) | α FC1 (s0, s1) | α FC2 | α FC3 | λmax/null edge FC1/2/3 (s0) | above null FC1/2/3 (s0) |
|---|---|---|---|---|---|---|---|---|
| 8 | 225,000 | .886 | .947 | 1.70 | 2.84 | 2.90 | 10.4 / 34 / 47 | 51 / 52 / 37 |
| 16 | 112,500 | .895, .896 | .976 | 1.94, 1.94 | 2.50, 2.42 | 3.12, 3.05 | 8.5 / 12.2 / 13.1 | 49 / 47 / 32 |
| 32 | 56,250 | .898, .895 | .992 | 2.26, 2.25 | 2.98, 3.00 | 3.57, 3.56 | 6.3 / 4.7 / 3.8 | 46 / 39 / 25 |
| 64 | 28,110 | .896, .898 | .989 | 2.84, 2.81 | 4.01, 3.94 | 4.05, 4.04 | 4.3 / 2.2 / 2.9 | 37 / 28 / 16 |
| 128 | 14,040 | .897, .897 | .991 | 4.33, 4.15 | 5.51, 5.55 | 4.40, 4.21 | 2.5 / 1.6 / 2.7 | 25 / 18 / 11 |
| 256 | 7,020 | .895, .895 | .985 | 7.42, 8.16 | 6.76, 7.09 | 5.08, 4.64 | 1.6 / 1.5 / 2.5 | 14 / 11 / 9 |
| 512 | 3,510 | .891, .886 | .978 | (13.5, 10.5) | 8.75, 8.72 | 5.97, 5.75 | 1.1 / 1.4 / 1.9 | 7 / 9 / 9 |
| 1024 | 1,740 | .860, .884 | .916 | (12.2, 7.6) | (12.4, 11.6) | 8.96, 9.16 | 1.0 / 1.2 / 1.4 | 4 / 7 / 9 |

α in parentheses: fewer than 5 eigenvalues sit above the null edge, so the "power law" is being fitted to the bulk and means nothing.

- **The phenomenon appeared, and it is learned structure.** At init, λmax / MP edge = 0.98 / 0.97 / 0.99 and the KS distance to the shuffled
  null is ≤ 0.005. In the bs-16 run, FC1 first has ≥ 5 eigenvalues above the null at step ~300, λmax passes 2× the MP edge at step ~1,240, and the
  final KS distance to the null is 0.26 (FC2 0.12, FC3 0.07). Shuffling the same entries destroys the tail at every checkpoint (pink in the riso pieces).
  Entry kurtosis of trained W stays ≤ 1.2, so the departure is **correlation structure, not heavy-tailed entries**.
- **α is meaningless in the random phase.** Clauset fits on exact MP matrices return α = 7–17 at init and 4–25 on shuffled nulls, and the fits jump
  around early in training. Films and plots only report α once ≥ 5 eigenvalues exceed the null edge.
- **Batch-size series: monotone and reproducible.** Seeds agree to within ±0.1 in α for bs ≤ 128. Smaller batches give heavier tails and more outliers,
  in the direction MM 2018 report.
- **Honest negative: accuracy is flat.** Test accuracy is 0.895–0.898 for every bs from 16 to 256, while FC1 α goes from 1.94 to 7.8 (a 4× change).
  Across the reliable runs, Spearman ρ(mean α, test acc) = −0.16 (n = 13), and **ρ = 0.00 for bs 16–256**. Where accuracy does move, it
  drops at *both* ends: bs 8 (the heaviest tail, α 1.70, test .886, train only .947, too noisy to fit) and bs 1024 (the lightest, only 1,740 steps,
  under-trained). The train−test *gap* does correlate with α (ρ = 0.45), but that is driven by how well each run fits the training set, which batch size
  and step count confound. At bs 1024, FC1's λmax sits at the MP edge and the network is still 86–88% accurate: this network does not need a heavy tail to
  generalise on FashionMNIST.
- **Gotcha found and fixed (rank-one spike in the null).** Shuffling entries keeps their global mean μ. FC2/FC3 weights drift negative during
  training (FC2 at bs 32: μ = −3.6e−3), and $\mu\,\mathbf{1}\mathbf{1}^\top$ adds one null eigenvalue near $\mu^2 M$, up to 2.7× the MP edge. The
  first version used the null's largest eigenvalue as its edge and undercounted outliers. The null bulk edge is now the *second* largest eigenvalue
  (`common.null_edge`). The same spike explains the falling stable rank of the null for FC2/FC3 in the over-training sheet, and part of the very large
  λmax/MP of FC2/FC3 at bs 8–16.
- **Weak pieces, not shown above:** textures of $\Delta W = W_T - W_0$ ("weave", "fields") are dominated by the SGD random walk and look like
  noise. The departure field in the rank-normalised Spectral split mostly shows frozen noise at low λ.
- **No fractal or self-similarity claim.** The ridges are densities; nothing here is box-counted.
- **Precision:** SVDs are float64 on float32 weights, and eigenvalues below ~1e−12 (FC2/FC3, Q=1) are numerical zeros and are ignored.

## Caveats

- **The generalisation claims are contested.** HT-SR's link between smaller α and better test accuracy comes from correlations across
  pretrained model families. Kothapalli et al. 2024 (arXiv:2406.04657) get heavy-tailed ESDs without SGD noise (a large lr is enough), and
  "Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias" (arXiv:2506.06280) shows the α estimate is biased by Q. Our flat accuracy is
  one more data point that α here tracks the *optimisation trajectory* (noise scale × steps), not test performance.
- **The batch series is confounded.** At a fixed lr and a fixed 30 epochs, a smaller batch means more SGD noise (lr/B) **and** more steps (bs 8 takes 129× as many
  as bs 1024). This series cannot separate the two.
- **α depends on the estimator:** xmin by KS, n_tail (147–286 at bs 16), Q, and the finite matrix size M ≤ 1024. Error bars are MLE standard errors
  plus a 2-seed range, not a population spread.
- One small architecture, one dataset, no weight decay, no lr schedule. Real networks (and MM's MiniAlexNet) may behave differently.
- The films and plates interpolate *rank-ordered* eigenvalues between checkpoints. A streamer is not the same eigenvector over time.
- Colour, height gamma, KDE bandwidth, exposure, misregistration and the Joy Division occlusion are aesthetic choices. The x positions of eigenvalues
  and the MP/null references are measured.
- Files over 20 MB (skipped by the commit script): none. The largest are the ridge-film MP4s (17–19 MB).

## References

- C. H. Martin, M. W. Mahoney, *Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning*, arXiv:1810.01075 (2018); JMLR 2021.
- C. H. Martin, M. W. Mahoney, *Heavy-Tailed Universality Predicts Trends in Test Accuracies for Very Large Pre-Trained Deep Neural Networks*, arXiv:1901.08278 (2019).
- V. A. Marchenko, L. A. Pastur, *Distribution of eigenvalues for some sets of random matrices*, Math. USSR-Sbornik 1 (1967).
- A. Clauset, C. R. Shalizi, M. E. J. Newman, *Power-law distributions in empirical data*, SIAM Review 51 (2009).
- V. Kothapalli et al., *Crafting Heavy-Tails in Weight Matrix Spectrum without Gradient Noise*, arXiv:2406.04657 (2024).
- *Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias*, arXiv:2506.06280 (2025).
- WeightWatcher, github.com/CalculatedContent/WeightWatcher. Peter Saville, *Unknown Pleasures* cover (1979), after the CP 1919 pulsar plot.

