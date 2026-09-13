# Neural Collapse: stars and tetrahedra from the terminal phase of training

*Train a ResNet far past zero training error and its last-layer class means drift toward a perfectly symmetric simplex. For 10 classes the Fourier planes of that simplex show {10/k} star polygons, and for 4 classes it is a regular tetrahedron you can 3-D print. This page shows how close our runs actually got, including where they didn't beat a random baseline.*

<p align="center">
<img src="gallery/stars_c10_night_k3_ep200.00.png" width="46%">
<img src="gallery/tetra_series_c4_brass.png" width="52%">
</p>

<p align="center">
<video src="gallery/film_stars_c10_night_720.mp4" autoplay loop muted playsinline width="46%"></video>
<video src="gallery/film_tetra_c4_brass.mp4" autoplay loop muted playsinline width="46%"></video>
</p>

## 1. The phenomenon

Papyan, Han and Donoho (2020) looked at the penultimate features h of a classifier trained past 100 % train accuracy (the *terminal phase*). Let μ_c be the class means, μ_G the global mean, Σ_W the within-class and Σ_B the between-class covariance, and W the last-layer weights. They found four properties:

- **NC1, variability collapse:** tr(Σ_W Σ_B⁺)/C → 0. Each class cloud shrinks onto its mean.
- **NC2, simplex ETF:** the centred means μ_c − μ_G become equal-norm and equiangular, with cos(μ_c−μ_G, μ_c'−μ_G) → −1/(C−1). This is the most spread-out way to place C points.
- **NC3, self-duality:** W/‖W‖ → M/‖M‖, so the classifier rows line up with the means.
- **NC4, nearest-class-mean:** the network's decision becomes argmin_c ‖h − μ_c‖.

**Why stars.** A C-point simplex ETF lives in the (C−1)-dimensional space 1⊥. The discrete Fourier vectors cos(2πkj/C) and sin(2πkj/C) for k = 1…⌊(C−1)/2⌋ form an orthonormal basis of that space. In plane k, vertex j sits at angle 2πkj/C on a circle. Joining j → j+1 therefore draws the regular star polygon {C/k}. For C = 10 that gives {10/1} a decagon, {10/2} two coincident pentagons, {10/3} a decagram and {10/4} two coincident pentagrams (k = 5 is a line). The measured means are mapped onto this frame by an **isometry**: an SVD basis of their span, orthogonal Procrustes onto the ideal vertices, and one global scale. Because there is no shear, a non-ETF configuration still looks non-ETF. The maths was checked numerically: an exact ETF gives residual 8e-16. **Misfit** = ‖aligned means − ideal‖_F / ‖ideal‖_F.

**Why a tetrahedron.** For C = 4 the centred means span exactly 3 dimensions, so the 3-D picture is faithful up to rotation and a single scale. Nothing is lost in projection. Only the sample clouds, which live in 256-d, are projected.

## 2. Gallery

### C = 10: star-polygon projections (CIFAR-10)
| | |
|---|---|
| <img src="gallery/stars_c10_night_k1234_ep200.00.png" width="100%"> | <img src="gallery/stars_c10_night_k3_sequence.png" width="100%"> |
| **Night, epoch 200.** Planes k=1..4. 10k train-subset and 10k test features, class hue (colorcet cyclic, declared), additive glow with per-plate auto exposure (declared). Bright line = measured means, faint line = ideal. | **{10/3} over training** (epochs 0, 0.1, 1, 5, 20, 60, 100, 200), with misfit and NC1 in each tile. |
| <img src="gallery/stars_c10_spectral_k1234_ep200.00.png" width="100%"> | <img src="gallery/stars_c10_riso_k1234_ep200.00.png" width="100%"> |
| **Spectral (Sohl-Dickstein split).** Per-sample logit margin (correct − best other). Red side = misclassified, purple/blue side = correct, rank-normalised per side, dark seam = decision boundary. The misclassified points (almost all test) fill the centre. | **Riso.** Fluo pink = train, blue = test, with a declared 4 px/1200 misregistration. Train collapses onto the vertices while test stays a haze inside. |
| <img src="gallery/plotter_stars_c10_ink.png" width="100%"> | <img src="gallery/plotter_stars_c10_spectral.png" width="100%"> |
| **Plotter sheet.** 31 distinct log-spaced checkpoints of the mean star, overdrawn (faint = early, heavy = late). Red dashes = ideal. The heavy late lines visibly miss the ideal, which matches the residual misfit of 0.11. | **Same sheet, Spectral by epoch.** A declared sequential use of Spectral, labelled variant only. |
| <img src="gallery/stars_c10_night_k1234_ep000.00.png" width="100%"> | <img src="gallery/stars_c10_night_k1234_ep010.00.png" width="100%"> |
| Initialisation (misfit 0.915): features share one dominant direction. | Epoch 10 (misfit 0.364). |

**Film.** [`film_stars_c10_night_720.mp4`](gallery/film_stars_c10_night_720.mp4) (26 s, 720², 10 MB) and a [GIF](gallery/film_stars_c10_night.gif) (360 px, 11 MB). It runs through all 93 checkpoints with 8 linearly tweened frames each, then a 2 s hold. Exposure is fixed for the whole film from the epoch-10 checkpoint, so condensation shows up as brightening. Each frame has its own global scale (feature norm grows 0.11 → 14.6). The checkpoint spacing is roughly logarithmic, so film time is not linear in epoch. A 1080² master `gallery/film_stars_c10_night_1080.mp4` (75 MB) is **not committed**; regenerate it with `film_stars.py`.

### C = 4: terminal-phase tetrahedra (airplane, automobile, bird, cat)
| | |
|---|---|
| <img src="gallery/tetra_series_c4_brass.png" width="100%"> | <img src="gallery/tetra_series_c4_plotter.png" width="100%"> |
| **Brass, I–V** = init, epoch 2, 16, 60, 250. Measured means as tubes, dotted ghost = ideal. The class-coloured dust is train + test samples projected into the 3-D mean span (lossy). Captions give the 6 measured angles. | **Plotter.** Line only, hidden edges dashed (declared depth cue). |
| <img src="gallery/tetra_series_c4_cyanotype.png" width="100%"> | <img src="gallery/tetra_series_c4_spectral.png" width="100%"> |
| **Cyanotype.** | **Spectral.** Edge colour = signed deviation of that pairwise angle from 109.47°, split at 0 (purple = more acute, red = more obtuse), rank-normalised across the series. |

**Rotation film.** [`film_tetra_c4_brass.mp4`](gallery/film_tetra_c4_brass.mp4) (25 s, 1080²) and a [GIF](gallery/film_tetra_c4_brass.gif). The shape morphs through all 101 checkpoints (4 tweened frames each) while the camera turns, then the final state makes a full 360° turn. Dust exposure is fixed from epoch 60.

**Meshes.** `gallery/stl/terminal_phase_{I..V}_ep*.stl` are rods, spheres and a hub, with the ideal circumradius 50 mm and measured vertex positions.

### Metrics as images
| | |
|---|---|
| <img src="gallery/misfit_c10_c4_paper.png" width="100%"> | <img src="gallery/curves_c10_c4_paper.png" width="100%"> |
| **Misfit vs epoch** for train means, test means and classifier rows, against **random-means null bands** (i.i.d. Gaussian means, d = 256 and d = 64, 5–95 %). | **NC1–NC4 and error**, train solid and test dashed, with log epoch and lr drops dotted. The grey band on the NC2 rows is the same random null. ([night version](gallery/curves_c10_c4_night.png), [night misfit](gallery/misfit_c10_c4_night.png)) |
| <img src="gallery/gram_c10_train_test_spectral.png" width="100%"> | <img src="gallery/gram_c10_means_W_pal_indigo_madder.png" width="100%"> |
| **Cosine-matrix grid, C = 10.** 48 log-spaced checkpoints. Upper triangle = train means, lower = test means. Colour = cos − (−1/9), split at the ideal and rank-normalised per side globally. Tiles darken toward the seam as the means approach the ETF. | **Means (upper) vs classifier rows W (lower)**, palettes.py `indigo_madder` split. Also [riso](gallery/gram_c10_means_W_riso.png). |
| <img src="gallery/gram_c4_train_test_spectral.png" width="100%"> | <img src="gallery/gram_c4_train_test_pal_aurora_ember.png" width="100%"> |
| C = 4, Spectral. | C = 4, palettes.py `aurora_ember`. |

## 3. What was computed

- **Model/data:** CIFAR-10 ResNet18 (3×3 stem, no maxpool) at **width 32** (penultimate d = 256), not the paper's width 64. No augmentation, per-channel standardisation, balanced classes (5000 per class).
- **Optimiser:** SGD, momentum 0.9, lr 0.05, wd 5e-4, batch 128, lr ÷10 at E/3 and 2E/3. **c10: 200 epochs** (all 10 classes), **c4: 250 epochs** (classes 0–3). Paper: 350 epochs and an lr sweep; we used one lr and one seed (0). bf16 autocast for training; features extracted in fp32; statistics in float64.
- **Checkpoints:** iterations 5–240 inside epoch 1, every epoch 0–30, every 2 to 100, every 5 after (93 for c10, 101 for c4). Per-checkpoint stats use a fixed class-balanced train subset (1000/class c10, 2500/class c4) and the full test set. Full-train NC1/accuracy is recorded every 10 epochs; subset and full NC1 agree (c10 final 0.0439 vs 0.0436).
- **Wall time:** c10 6664 s, c4 3774 s on a shared GB10 (≈2.9 GPU-h total). All rendering is CPU.
- **Reproduce:**
```bash
./run_train.sh                                         # both runs via gpu_run.sh -> cache/c10, cache/c4
python misfit.py compute c10 c4 && python misfit.py render c10 c4 --style paper   # also --style night
python render_curves.py c10 c4 --style paper           # night
python render_stars.py c10 --epoch 200 --style night   # riso | spectral ; --planes 3 --res 3000 for the hero
python plotter_stars.py c10 [--ink spectral]
python film_stars.py c10 --fpc 8 --hold 60             # 1080 master; 720 copy: ffmpeg -vf scale=720:720 -crf 30
python render_tetra.py c4 --epochs 0 2 16 60 250 --style brass --panel 1000   # plotter|cyanotype|spectral ; --stl
python film_tetra.py c4 --style brass --workers 3
python render_gram.py c10 --style spectral             # --pair means_W --style riso|pal_indigo_madder ; c4 pal_aurora_ember
```
(Python = `/home/fzeng/ml/research/art/.venv/bin/python`. The `stars_c10_night_k3_sequence.png` montage was composed with PIL from `render_stars.render` at 8 epochs.)

## 4. Verification and honesty

**Did collapse happen? Train: yes, strongly for NC1 and NC4. Test: only partly.**

| final | NC1 train | NC1 test | cos-std train | cos-std test | norm spread train | NC3 | NC4 test | test acc |
|---|---|---|---|---|---|---|---|---|
| c10 (ep 200) | **0.044** (from 50 at init) | 0.387 | 0.082 | 0.154 | 0.028 | 0.179 | 0.011 | 86.7 % |
| c4 (ep 250) | **0.0070** | 0.212 | 0.025 | 0.148 | 0.024 | 0.187 | 0.0023 | 90.2 % |

Train accuracy reaches 100 % in both runs and train NC4 is 0. Most of the NC1 drop happens at the first lr decay (see curves).

**Random-means null (key caveat).** In high dimension, random vectors are already nearly equiangular. For C i.i.d. N(0, I_d) means (4000 draws) the Procrustes misfit to the ETF is:

| | null d=256: 5 / 50 / 95 % | null d=64 median | measured train | test | W |
|---|---|---|---|---|---|
| C=10 | 0.081 / 0.097 / 0.114 | 0.196 | **0.112** | 0.231 | **0.037** |
| C=4 | 0.028 / 0.053 / 0.085 | 0.108 | **0.029** (min 0.016 @ ep 225) | 0.164 | **0.022** |

- **C = 10 train means never beat the random null at their own width.** Their final misfit of 0.112 sits inside the d = 256 band, and the NC2 cosine std (0.082) is worse than random (0.061). So the star shape of the 10 means alone is *not* evidence of an ETF in this run. What does show collapse is the **clouds**: NC1 falls 1000×, and random means would not make samples condense onto them. It also shows in the classifier rows, whose misfit of 0.037 is well below the null. The star picture's "starriness" is also partly geometric necessity: 10 generic points in 256-d already project near-regularly.
- **C = 4 train means do beat the null.** Misfit drops below the null median at epoch 76 (first lr decay) and ends at 0.029 ≈ the null's 5th percentile. The max angle error is 2.07° at epoch 250 (1.57° at epoch 125). Test means (0.164) do not beat even the d = 64 null.
- **Init is far from ETF, not near it.** Init misfit is 0.92 (c10) and 0.50 (c4), because init features share a dominant direction (cos-std 0.58). The trajectory really does go from far-from-ETF to near the random-level floor. The classifier W starts at the null level (≈0.10, random init), moves away, then comes back well below it.
- **NC3 plateaus at ~0.18** in both runs, so self-duality was not reached (it is still decreasing slowly on a log axis at the end). This is a partial negative result.
- Fractal/self-similar claims: none made. The star polygons are exact symmetry of the ideal, not fractal structure.

## 5. Caveats

- Width 32 and 200/250 epochs instead of width 64 and 350 epochs, with one lr and one seed. Stronger NC2 would plausibly need longer training or a sweep. Papyan et al. report their best lr per dataset.
- The class order in the Fourier frame is a **declared choice**. The ETF is symmetric under all permutations of the classes, so the particular star (which class sits at which vertex) is ours. A different order gives different but equally ideal star polygons, and Procrustes (reflection allowed) handles orientation.
- Projections of sample clouds are lossy: C=10 clouds are shown in 2 of 9 mean-span dims (and 256 total), and C=4 dust in 3 of 256. Only the **means** in the tetrahedron are exact.
- Per-checkpoint global scale hides the ~130× growth of feature norms. Exposure and glow are aesthetic. Tweened film frames are interpolations, not measurements.
- Stats use a class-balanced train subset (checked against full-train NC1 every 10 epochs).
- The misfit mixes norm inequality and angle error. The curves plate separates them (NC2 cos-std and norm spread).
- Git-skipped (>20 MB): `gallery/film_stars_c10_night_1080.mp4`.

## 6. References

- V. Papyan, X.Y. Han, D. Donoho, *Prevalence of neural collapse during the terminal phase of deep learning training*, PNAS 2020 (arXiv:2008.08186).
- X.Y. Han, V. Papyan, D. Donoho, *Neural collapse under MSE loss*, ICLR 2022.
- J. Sohl-Dickstein, *The boundary of neural network trainability is fractal* (Spectral split style), github.com/Sohl-Dickstein/fractal.
- Palettes: `/home/fzeng/ml/research/art/color-research/palettes.py` (indigo_madder, aurora_ember).
