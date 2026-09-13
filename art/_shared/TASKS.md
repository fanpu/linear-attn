# Per-project task specs (compact handoff)

Each section holds the essentials a fresh agent needs: the goal, corrections to the proposal docs (these override the docs), state, and priorities. Read `BRIEF.md` first. Docs: `art/ml-art-directions.md` (main), `art/ml-art-fractals.md` (fractals).

Sister-project learnings apply everywhere:
- Render heroes at native resolution (never upsample a small grid).
- Choose zoom centres on the boundary, re-picked at each level.
- 2-parameter Lyapunov planes contain self-similar "shrimps" (Gallas 1993), which look spectacular in the Spectral split.
- READMEs written before their media exist end up with missing links, so always run the link check.
- Fill the canvas: no empty black thirds, no text running off edges.

---

## trainability-fractal (fractals §1)
- **State:** the batched float64 reimplementation of Sohl-Dickstein is done (`tfractal.py`, `styles.py`). Spectral style matches his colab. The Liu et al. "trivial non-convexity" diptych and hero are done. Box-count verification against the quadratic null exists. `PAUSED.md` has details.
- A `zoomB_run.sh` job was left running through `gpu_run.sh` (see `logs/zoomB.log`). Check its output before recomputing.
- `hero_overview_tanh_spectral_{print,labelled}.png` exist but are uncommitted; check they're native-resolution.
- **Remaining:** native ≥1024² Spectral overview hero; a deep zoom of ≥6 decades with boundary-centred windows, one plate per half-decade, plus an MP4/GIF; same-region tanh-vs-ReLU diptych; one semantic-axes sweep if cheap; box counting over ≥2 decades with a resolution check; the README body (currently placeholders).

## outcome-basins (fractals §5)
- **Piece A:** GD on ¼(xyz−1)² with 4 solutions is done and verified. At large η the basin boundary has D = 1.40±0.02 over a 1.3·10⁸× zoom; small-η null ≈ 1.0; uncertainty exponent 1.43; no riddling down to 10⁻¹³. Hero `f3_s0_e0.3_wide_newton.png` and film `f3_zoomfilm_newton.mp4` exist.
- **Piece B:** the XOR 2-2-1 net at η = 1.2. There are 16 raw solutions, and canonicalizing hidden units by permutation and sign leaves 2. That removes 58% of the between-solution boundary, but the fractal fan (solve vs diverge/plateau) remains, with D ≈ 1.78.
- **Remaining:** the η series animation (smooth → fractal as η grows); Piece B zoom and film; Spectral variants of the converge-speed maps; README; link check.

## signal-propagation (main §6 + fractals §3)
- **State:** phase plates are done. The analytic mean-field critical line is reproduced (σ_b² = 0.05 → σ_w² = 1.761). Empirical measurements match except noisy ξ_c on the chaotic side. The CRN finite-width setup follows D'Inverno et al. 2025 (arXiv:2508.03222, code github.com/jon-dong/fractal-deep-info-prop). float32 and float64 agree within 0.01%.
- **Findings:** the paper's D ≈ 1.85 is a max over thresholds, on one image, over < 2 decades, so it's likely inflated. Our finite-width frontier does **not** have a single fractal dimension; report the range and its dependence honestly.
- **Remaining:** Spectral frontier zoom plates (native resolution, boundary-centred) plus a video; the width-as-time animation (rough → smooth as width grows); README; link check.

## decode-map (fractals §6)
- **State:** a hand-written fp32 Qwen3-0.6B with static KV and no padding, verified against HF. An exact inverse-CDF sampler with shared per-position uniforms, 0 mismatches vs a float64 full-sort reference over 5,120 rows. A 64², L = 32 toy map in `cache/`. Styles built. `PAUSED.md` has resume commands.
- **Blocker:** ~0.9 s per token step on the shared GPU. Speed up first: bigger batches, torch.compile or CUDA graphs, fewer vocab ops such as top-k prefiltering that stays exact.
- **Corrections:**
  - `torch.multinomial` with a fixed seed isn't continuous in the probabilities, so keep shared uniforms.
  - Batched kernels are nondeterministic, so test batch placement and report the mismatch rate.
  - Finite L bounds the nesting depth, so make an animation with L as time and don't call it fractal beyond the refinement range.
- **Remaining:** (temperature, top-p) and (temperature, repetition-penalty) maps at 256²; the L-as-time animation; the batch-placement test; box counting over the refining range; styles (categorical mosaic, stained glass, boundary lines, Spectral for continuous metrics); README.

## diffusion-basins (fractals §4)
- **Correction:** the DDIM / probability-flow ODE is a finite composition of smooth, near-invertible maps with no folding. Expect smooth but stretched boundaries; test that, don't illustrate fractality. Slices through Gaussian noise must be norm-preserving (great-sphere coordinates). The iterated denoiser x ← D(x, t), or large-step overshoot, is a different, non-invertible map that can fold. Label it as such.
- **State:** code written. The 2D toy score nets for ring8 and grid25 are trained; scatter12 is not. Early looks: the exact-score ring8 ODE basins are straight rays by symmetry, a built-in smooth null. The iterated denoiser with overshoot gives Newton-like nested flowers just below stability (unverified). The grid mixture gives plain rectangles.
- **Remaining:** toy basin maps (ODE vs DDIM steps vs DDPM frozen noise vs iterated denoiser) with §11 verification; MNIST DDPM + classifier basins; the memorization version (32–64 images); thumbnail mosaic; step-count animation; README.

## loss-landscape (main §4)
- **Setup:** Li et al. 2018 filter-normalized directions (BN/bias directions zeroed), CIFAR-10 ResNet-20/56 with and without shortcuts. `common.py`, `train.py` and `landscape.py` (resumable) exist. Old 60-epoch runs were killed at epochs 8–18, and `train.py` has no resume, so retrain; you may shorten the schedule and state it.
- **Styles:** no rainbow 3D surfaces. Use contour-only survey sheets, hachures, hillshade with raking light, a hypsometric tint, and STL heightfield exports. Use log-loss height with a shared clip range across paired models.
- **Ideas:** the training trajectory as a trail on the contour map (PCA directions); the landscape at several checkpoints; a ridgeline stack of 1D slices; a 2×2 atlas; a zoom series testing whether no-skip roughness persists at finer spacing (report roughness scaling, don't overclaim).
- **Caveat (mandatory):** it's a 2D slice of ~10⁶ dimensions; non-convexity in a slice implies global non-convexity, smoothness implies little; Dinh et al. 2017 applies.

## mode-connectivity (main §5)
- **State:** `common.py` (weight matching, REPAIR, Bézier curves) and `compute_hero.py`. An earlier MNIST toy (width-512 MLP, 3 epochs) gave a naive barrier of 1.44, 0.038 after matching, 0.006 after matching + REPAIR, and ≈0 on the Bézier curve. Barrier vs epoch was ~0 at init, then grew, then the matched barrier fell. Those numbers were logged only, never saved, so rerun.
- **Corrections:** LMC after permutation depends strongly on width, so run a width series (32 → 2048). BN nets need statistics reset (REPAIR). LMC is emergent, not present at init.
- **Pieces:** the triptych (naive / Bézier / matched) as geological cross-sections; the width series; 2D loss planes through {A, B, π(B)}; the permutation-matrix print; a first-layer receptive-field quilt for A, B and π(B); weight matching "sorting" as an animation.

## edge-of-stability (main §1)
- **Setup:** Cohen et al. 2021 fc-tanh, full-batch GD on a CIFAR-10 subset, MSE, ≥4 learning rates. `eos_train.py` exists.
- **Corrections:**
  - Top Hessian eigenvectors rotate, so use the oscillation coordinate along the *current* top eigenvector, per-window eigenvectors, or windowed PCA, and declare which.
  - 2/η is the full-batch GD threshold only; Adam and momentum have their own thresholds.
- **Pieces:** the three-layer print (2/η hairline, λ_max stroke, braid ribbon); 4-LR small multiples; phase portraits and return maps; a plotter single-line braid; a braid-drawing animation; an η-sweep of oscillation patterns.

## neural-collapse (main §3)
- **Setup:** Papyan/Han/Donoho. Balanced 4-class and 10-class runs, trained far past zero error. `train_nc.py` exists; a toy got 3 epochs.
- **Corrections:**
  - For C = 4, the centred means span exactly 3D, so the tetrahedron is faithful; export STL.
  - For C = 10, project onto discrete-Fourier planes of the ideal ETF, where the ideal becomes regular {10/k} star polygons.
  - Show train vs test.
  - Balanced classes are required.
- **Pieces:** tetrahedron series I–V plus rotation video and meshes; nebula-condensing film via the star-polygon projections; NC1–NC4 curves; cosine-matrix grid over time.

## hessian-spectrum (main §7)
- **Setup:** own SLQ plus top-k Lanczos on HVPs, with the Gauss-Newton/Papyan decomposition. `common.py`, `train.py` and `analyze.py` exist; a toy spectrum is cached.
- **Corrections:**
  - Eigenvalues can be negative, so use a symlog axis.
  - SLQ smoothing hides mini-bulk structure; state the kernel width and iteration/probe counts.
  - Count outliers vs C in *your* data before captioning "count lines = count classes".
- **Pieces:** a class-count series (C = 2, 3, 4, 5, 7, 10); emission and absorption plates; a spectrograph-over-training animation; a barcode.

## weight-spectrum (main §8a)
- **Setup:** Martin & Mahoney HT-SR. ESD of WᵀW/N over training with a Marchenko–Pastur overlay, a power-law α fit, and a batch-size series. `train.py` exists.
- **Caveat:** the generalization claims are contested.
- **Pieces:** a tail-growing animation; a ridgeline stack over checkpoints; riso MP vs ESD; eigenvector localization / W texture.

## scaling-dimension (fractals §10)
- **Setup:** Sharma & Kaplan teacher/student with d = 2…12. Fit α and compare against 4/d using TwoNN and MLE ID of the data and student representations. `idlib.py` (tested) and `ts_train.py` exist.
- **Correction:** verify the doc's "GPT-2 d ≥ 90" claim against the paper.
- **Pieces:** diptych of α line vs d line; fan of slopes; agreement plate against y = x; a scale-invariance zoom through a manifold point cloud. Include a real-data check (MNIST/FashionMNIST/CIFAR ID).

## ouroboros (fractals §7)
- **Correction:** it's not an IFS; it's a random dynamical system on distributions. Use common random numbers across the grid.
- **Setup:** 2D targets (ring GMM, spiral, Barnsley fern point cloud); GMM-EM, KDE and optionally a tiny diffusion model; replace vs accumulate; λ real fraction. `common.py` and `chains.py` exist.
- **Pieces:** generations film; nested-generations spiral; phase map over (λ, n) with replace vs accumulate diptych; tail-loss ridgeline.
- **References:** Shumailov 2024, Alemohammad 2023, Bertrand 2023, Gerstgrasser 2024.
