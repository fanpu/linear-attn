# Ouroboros: generative models eating their own tail

*Fit a model to data, sample from it, fit the next model to those samples, and repeat. Watch a ring of eight Gaussians shrink to one needle, a fern fall apart into splinters, and a phase map of when that happens.*

<p align="center"><img src="gallery/spiral_ring_gmm_replace-accumulate_riso.png" width="720"></p>

## 1. The phenomenon

Self-consuming retraining ("model collapse") is a **random dynamical system on probability distributions**. It is not an iterated function system: nothing contracts toward a fixed attractor set. At generation $g$ a model $p_{\theta_g}$ is fit to a training set built from $n$ points:

$$\mathcal D_g=\underbrace{\{x_1,\dots,x_{\lfloor\lambda n\rfloor}\}}_{\text{real, fixed}}\ \cup\ \underbrace{\{\tilde x^{(g)}_1,\dots,\tilde x^{(g)}_{n-\lfloor\lambda n\rfloor}\}}_{\tilde x\sim p_{\theta_{g-1}}},\qquad \theta_g=\mathrm{fit}(\mathcal D_g).$$

- **replace** (Shumailov et al. 2024, Alemohammad et al. 2023): each generation's synthetic samples replace the previous ones. With $\lambda=0$ the chain has no anchor, so estimation noise is compounded. Each refit loses a little tail mass and variance, and the losses never come back. For a Gaussian the MLE variance shrinks in expectation by $(n-1)/n$ per generation, and the log-variance does a random walk with a drift toward $-\infty$.
- **anchored** replace, $\lambda>0$ (Bertrand et al. 2023): a fixed real fraction keeps the chain in a neighbourhood of the truth whose size shrinks as $\lambda$ and $n$ grow.
- **accumulate** (Gerstgrasser et al. 2024): $\mathcal D_g=\mathcal D_0\cup S_1\cup\dots\cup S_g$. The pool grows, so each new generation's error is diluted by $1/g$ and the total error stays bounded.

Targets live in $[-1.45,1.45]^2$: a **ring** of 8 Gaussians (radius 1, $\sigma=0.08$), an Archimedean **spiral**, and **Barnsley's fern** (chaos-game point cloud). Models: a **Gaussian mixture** fit by EM (well-specified for the ring with $K=8$; $K=64$, misspecified, for the fern) and a **Gaussian KDE** with leave-one-out-CV bandwidth.

Distance to the truth is measured by the **sliced 2-Wasserstein** distance (64 directions, 2048 model samples vs 2048 reference samples), $SW_2(p,q)^2=\mathbb E_{\theta}\,W_2^2(\theta_\#p,\theta_\#q)$. For the ring we also count **modes kept**: a mode counts if at least ¼ of its share of samples lies within $3\sigma$ of its centre.

**Common random numbers (CRN).** Sample $j$ of generation $g$ of seed $s$ uses the same uniform and normal draws in every chain, whatever its $(\lambda,n)$. Neighbouring cells of a phase map therefore differ only in their parameters, not in their luck.

## 2. Hero pieces

<table>
<tr><td width="50%"><img src="gallery/phase_ring_gmm_replace_split_sd_spectral.png"></td>
<td><img src="gallery/fern_sheet_sepia_ink.png"></td></tr>
<tr><td>Escape-time phase map, replace, 97 λ × 65 n chains (Sohl-Dickstein Spectral split).</td>
<td><i>Filix ouroborum</i>: Barnsley's fern re-learned 200 times, three regimes.</td></tr>
</table>

<video src="gallery/film_phase_ring_gmm_replace_sd_spectral.mp4" autoplay loop muted playsinline width="960"></video>

## 3. Gallery

### 3.1 Generations films (ring)
<video src="gallery/film_ring_gmm_dark.mp4" autoplay loop muted playsinline width="960"></video>

| file | measured | aesthetic |
|---|---|---|
| [film_ring_gmm_dark.mp4](gallery/film_ring_gmm_dark.mp4) / [.gif](gallery/film_ring_gmm_dark.gif) | GMM K=8, n=128, 200 generations, three chains side by side (replace λ=0, anchored λ=0.25, accumulate): 20k model samples per generation, sliced-W₂ trace | night ground, klimt_gold density LUT (log tone), sub-generation crossfade |
| [film_ring_gmm_paper.mp4](gallery/film_ring_gmm_paper.mp4) / [.gif](gallery/film_ring_gmm_paper.gif) | same | sepia ink on paper |
| [film_ring_kde_dark.mp4](gallery/film_ring_kde_dark.mp4) / [.gif](gallery/film_ring_kde_dark.gif) | KDE, n=128, 120 generations: the *opposite* failure. The ring **diffuses** instead of collapsing. | night ground, density LUT |
| [film_phase_ring_gmm_replace_sd_spectral.mp4](gallery/film_phase_ring_gmm_replace_sd_spectral.mp4) / [.gif](gallery/film_phase_ring_gmm_replace_sd_spectral.gif) | the (λ, n) phase map at each generation 0→80: red means $SW_2(g)-SW_2(0)>\tau=0.25$ now. Inset: share of the grid above τ. | Spectral split; each side rank-normalised over all frames; linear interpolation between generations |

### 3.2 Phase maps over (λ, n): where the snake eats its tail
Each cell is **one chain** (seed 0) of 80 refits of an 8-component GMM to the ring. n runs over 65 log-spaced values in [8, 512] and λ over 97 values in [0, 1]. Cells are drawn nearest-neighbour, with no smoothing.

<table>
<tr><td><img src="gallery/phase_ring_gmm_replace_sw2_crameri_batlow.png"></td><td><img src="gallery/phase_ring_gmm_replace_sw2_sepia.png"></td></tr>
<tr><td>batlow: log of $SW_2(80)/SW_2(0)$, the damage relative to the chain's own first fit (measured); colour map declared.</td><td>sepia paper variant of the same field (reversed so darker means worse).</td></tr>
<tr><td><img src="gallery/phase_ring_gmm_replace_split_aurora_ember.png"></td><td><img src="gallery/phase_ring_gmm_replace_split_hubble_sho.png"></td></tr>
<tr><td>aurora/ember split of the escape-time field. Red side: escaped past τ, paler means earlier. Purple side: survived, paler means further below τ.</td><td>Hubble-SHO palette, same field.</td></tr>
</table>

Also: [phase_ring_gmm_replace_sw2_crameri_lajolla.png](gallery/phase_ring_gmm_replace_sw2_crameri_lajolla.png).

**Replace vs accumulate diptych.** The same field and colour scale, with an accumulate map on the right (33 × 33 chains, nearest-upsampled onto the 97 × 65 layout).

<img src="gallery/phase_ring_gmm_split_sd_spectral.png" width="100%">

<table>
<tr><td><img src="gallery/phase_ring_gmm_sw2_crameri_batlow.png"></td><td><img src="gallery/phase_ring_gmm_sw2_sepia.png"></td></tr>
</table>

Also: [aurora/ember](gallery/phase_ring_gmm_split_aurora_ember.png), [Hubble SHO](gallery/phase_ring_gmm_split_hubble_sho.png), [lajolla](gallery/phase_ring_gmm_sw2_crameri_lajolla.png).
Under **accumulate**, 3.7% of cells escape, against 27.7% under replace. At λ=0 the escaped share is 6% of n columns for accumulate and 98% for replace. The only cells that escape are a few columns at n ≤ 13 plus isolated cells at n=26 and n=38, where the generation-0 fit is already poor. The median $SW_2(80)/SW_2(0)$ is 1.16 for accumulate (max 3.0) and 1.28 for replace (max 7.5). This reproduces Gerstgrasser et al.'s central claim on this toy.

**What to read in it.** Collapse is a small-$n$, small-$\lambda$ phenomenon. The escape boundary runs from λ≈0.75 at n=8 down to λ≈0.05 at n≈128. Above n≈128 only a thin band at λ≲0.05 still escapes within 80 generations. The **staircase** at small n is real, not a rendering artefact: only $\lfloor\lambda n\rfloor$ matters, so at n=8 the 97 rows contain just 9 distinct chains. The faint **vertical streaks** come from CRN: all λ in one column share one real dataset and one noise stream, so a lucky or unlucky column is lucky or unlucky all the way up.

### 3.3 Nested-generations spiral
Generation g is drawn at scale $0.94^{g/4}$ along a logarithmic spiral, so the chain's whole history coils inward toward generation 200 at the eye.

<table>
<tr><td><img src="gallery/spiral_ring_gmm_replace_dark.png"></td><td><img src="gallery/spiral_ring_gmm_replace_paper.png"></td></tr>
<tr><td>replace λ=0, night + klimt_gold</td><td>replace λ=0, sepia on paper</td></tr>
</table>

Measured: every tile is 20 000 samples of that generation's model (GMM K=8, n=128). Aesthetic: the spiral layout (one tile every 4 generations, ≈14 per turn), the tone map, blur proportional to tile size, and ink per sample ∝ 1/tile size, so the large outer tiles stay legible. The riso hero above overlays two point-reflected arms in two spot inks, with one drum deliberately misregistered. Fluorescent pink is replace, which collapses to 1 to 3 dots. Blue is accumulate, which keeps all 8 modes to the eye.

### 3.4 Tail-loss ridgelines
<table>
<tr><td><img src="gallery/ridgeline_ring_gmm_replace_night.png"></td><td><img src="gallery/ridgeline_ring_gmm_replace_paper.png"></td></tr>
<tr><td><img src="gallery/ridgeline_ring_gmm_anchored_night.png"></td><td><img src="gallery/ridgeline_ring_gmm_accumulate_night.png"></td></tr>
</table>

"The tails go first." One ridge per 2 generations, showing the density (per log unit) of each model sample's distance to the nearest centre of the *model's own* mixture components, in units of the true per-mode $\sigma$ (log axis). The dashed ridge on top is the true distribution. Right margin: share of samples beyond $2\sigma$. Under **replace** the ridge slides left from 1σ to ≈0.15σ by generation 150, and the share beyond 2σ falls from 10.8% (gen 0) to 0.8% (gen 10) to 0.0% (gen 50+). **Anchored** stays at 5–9% and **accumulate** at ≈7.5%. Line art and ground are aesthetic.

### 3.5 Filix ouroborum (fern)
<table>
<tr><td width="58%"><img src="gallery/fern_sheet_iron_gall.png"></td><td><img src="gallery/fern_hero_replace_sepia_ink.png"></td></tr>
<tr><td>iron-gall ink variant of the sheet</td><td>hero: generation 0 vs generation 200, replace</td></tr>
</table>

Measured: stipple dots are 12k (sheet) or 20k (hero) samples of the model at that generation. Engraved ellipses are the fitted components at 1σ and 2σ, with line darkness ∝ weight. The accumulate row has no ellipses because its parameters were not stored. Model: GMM K=64, n=4096, G=200. The first column is the true chaos-game fern. Aesthetic: paper, ink, dot size, and plate typography.

Sliced W₂ at gen 0/10/40/100/200: replace 0.021/0.066/0.086/0.114/0.122, anchored (λ=0.25) 0.021/0.043/0.022/0.026/0.021, accumulate 0.021/0.027/0.016/0.018/0.029. Under replace the components turn into needles and splinters (degenerate covariances hitting the 1e-6 floor) and 38 of the 64 components have lost essentially all their weight (π < 10⁻³) by generation 200, against 1 of 64 under anchoring. Anchoring keeps the silhouette but the "leaflets" become crossed needles.

## 4. What was computed

- **Code.** `common.py` holds targets, CRN pools, batched EM, LOO-CV KDE bandwidth, and metrics. `chains.py` has `run_batch`, the batched chains. Compute scripts (`compute_film.py`, `compute_phase.py`, `verify_compute.py`) write `cache/*.npz`. Render scripts (`render_*.py`, `verify.py`) read only the cache.
- **Precision/device.** float32 on CPU. The shared GPU was saturated, so everything ran on 2–4 CPU threads. Float32 is far below the metric noise: the sliced-W₂ noise floor is 0.046, and the covariance floor is 1e-6.
- **EM.** K components, deterministic farthest-point init, then 400 EM iterations at generation 0 and 10 warm-started iterations per generation after that. Covariance regulariser 1e-6.
- **KDE.** Isotropic Gaussian. Bandwidth from golden-section search on log₁₀h ∈ [-3, 0.3], maximising leave-one-out log-likelihood over 256 CRN query points; it matches a 400-point brute-force grid within 1%.
- **Seeds.** Seed 0 everywhere except the verification runs, which use seeds 0–4 (collapse table) and 0–1 (native window).

| piece | settings | wall (CPU) |
|---|---|---|
| ring films | GMM K=8 n=128 G=200; KDE n=128 G=120; 20k display samples/gen | 1.5 min, 40 s |
| fern | GMM K=64 n=4096 G=200 (replace+anchored with params, accumulate without) | ≈ 20 min |
| replace phase map | 97 λ ∈ [0,1] × 65 n ∈ [8,512] (log), G=80, 1 seed | 2 workers × ≈ 35 min |
| accumulate phase map | 33 λ × 33 n (the same range, nested sub-grid), G=80 | 2 workers × ACC_WALL |
| verification | seeds 0–4 at film settings; native window n=32…96 (all integers) × all distinct n_r, 2 seeds | ≈ 1 min; NATIVE_WALL |

```bash
P=/home/fzeng/ml/research/art/.venv/bin/python; export OUROBOROS_DT=float32 OMP_NUM_THREADS=2
$P compute_film.py ring gmm 128 200            # -> cache/film_ring_gmm_n128.npz
$P compute_film.py ring kde 128 120
$P compute_film.py fern gmm 4096 200 --K 64 --nd 20000 --regimes replace,anchored --tag _p
$P compute_film.py fern gmm 4096 200 --K 64 --nd 20000          # accumulate row (merge into _all, see NOTES.md)
$P compute_phase.py ring gmm replace --L 97 --N 65 --nmin 8 --nmax 512 --G 80 --bs 97   # (+ same with --rev in parallel; rerun to assemble)
$P compute_phase.py ring gmm accumulate --L 33 --N 33 --nmin 8 --nmax 512 --G 80 --bs 33
$P verify_compute.py all
# renders
$P render_film.py cache/film_ring_gmm_n128.npz dark;  $P render_film.py cache/film_ring_gmm_n128.npz paper
$P render_film.py cache/film_ring_kde_n128.npz dark
$P render_phase.py ring gmm --regimes replace;  $P render_phase.py ring gmm;  $P render_phase_film.py ring gmm
$P render_spiral.py cache/film_ring_gmm_n128.npz dark  --G 200 --stride 4 --r 0.94 --dtheta 0.45 --tile 0.18   # and paper
$P render_spiral.py cache/film_ring_gmm_n128.npz riso  --G 200 --stride 4 --r 0.94 --dtheta 0.45 --tile 0.09 --arms replace,accumulate
$P render_ridgeline.py cache/film_ring_gmm_n128.npz replace night --G 160 --step 2 --lo -1.6 --hi 0.7   # paper / anchored / accumulate
$P render_fern.py cache/film_fern_gmm_n4096_all.npz sheet [--ink iron_gall];  $P render_fern.py cache/film_fern_gmm_n4096_all.npz hero --gens 0,200
$P verify.py        # -> verify_results.txt, gallery/verify_boundary.png
```

## 5. Verification and honesty

Full output: [verify_results.txt](verify_results.txt). Figure: [gallery/verify_boundary.png](gallery/verify_boundary.png).

<!-- VERIFY -->

## 6. Caveats

- **Toy models, toy data.** Two-dimensional GMMs and KDEs are not LLMs. What transfers is the *mechanism*: finite-sample estimation error compounds when nothing re-anchors the chain. Collapse rates, and the λ needed to prevent collapse, do not transfer.
- **The model class decides the failure mode.** EM-GMM loses variance and modes (collapse), while LOO-CV KDE gains variance, because every generation adds $h^2$ of kernel variance and $h$ grows with the spread (explosion). "Model collapse" is not one direction.
- **One chain per phase-map cell.** Every pixel is a single random trajectory. The boundary's position varies from seed to seed by several $1/n$ steps (§5). Read the map as the location of a noisy transition, not as a sharp phase line.
- **λ has a resolution floor of 1/n.** Only $\lfloor\lambda n\rfloor$ enters, so the map is exactly piecewise constant in λ with steps of 1/n. Finer λ grids add no information.
- **τ and the "excess over generation 0" choice are declared.** At small n even the generation-0 fit is poor ($SW_2(0)$ ≈ 0.17 at n=8 vs 0.07 at n=512, noise floor 0.046), so the maps show *self-consumption damage* relative to each chain's own first fit, not absolute error. Some well-fit cells get *better* than generation 0 (ratios down to 0.44): EM warm-starting can escape a poor initial optimum.
- **Accumulate grid is coarser** (33×33, nested inside the 97×65 grid) and is shown nearest-neighbour upsampled.
- **Warm-started EM with 10 iterations per generation** is a declared protocol choice. Fully converged EM from scratch each generation would change the rates.
- **Accumulate at λ=0 in this code** starts from $\mathcal D_0$ (all n real points) and appends synthetic samples, which is Gerstgrasser et al.'s protocol. Its training set grows to 81n points by G=80.
- **Spectral split maps** rank-normalise each side separately. Colour distances within a side are ranks, not values, and pale hues near the ends do not mean "large" in absolute units.
- Not rendered, although caches exist: spiral-target films (`film_spiral_*`) and the fern KDE chain (`film_fern_kde_n2048.npz`).

## 7. References

- I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, Y. Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 2024.
- S. Alemohammad et al. *Self-Consuming Generative Models Go MAD.* arXiv:2307.01850, 2023 (ICLR 2024).
- Q. Bertrand, A. J. Bose, A. Duplessis, M. Jiralerspong, G. Gidel. *On the Stability of Iterative Retraining of Generative Models on their own Data.* arXiv:2310.00429, 2023 (ICLR 2024).
- M. Gerstgrasser, R. Schaeffer et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* arXiv:2404.01413, 2024.
- M. F. Barnsley. *Fractals Everywhere.* 1988 (fern maps).
- J. Sohl-Dickstein. *The boundary of neural network trainability is fractal.* arXiv:2402.06184, 2024 (Spectral split style).
