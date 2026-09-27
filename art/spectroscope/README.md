# Spectroscope: the optimiser is a colour

*Every weight matrix of a small GPT, at every checkpoint, rendered as the colour of its singular-value spectrum treated as light. A matrix that looks like its own shuffled copy is white. AdamW turns the network blue. Muon keeps it close to white, with red threads. SGD lands in between. Moving the weights by AdamW's step sizes in random directions stays white.*

<img src="gallery/hero_dark_g1.png" width="720">

## The phenomenon

Each weight matrix W has an eigenvalue spectrum of WᵀW/N. Compare it with the spectrum of the same entries shuffled, which is the Marchenko–Pastur (MP) null: same entries, same Frobenius energy, no structure. Training can only **move energy between modes**, because shuffling preserves the total. The picture asks where each optimiser moves that energy, for every matrix and at every moment.

Result (6-layer GPT, TinyStories, 2,000 steps, one seed per optimiser):

| run | final val loss | mean shift, 36 hidden matrices | mean chroma (xy distance from white) | dominant λ of mean colour | rows red-shifted (> +2 nm) / blue-shifted (< −2 nm), of 180 |
|---|---:|---:|---:|---:|---:|
| **AdamW** (lr 3e-3) | 1.911 | **−68.2 nm** | 0.166 | 476 nm (blue) | 0 / 180 |
| **SGD + Nesterov 0.9** (lr 0.3) | 2.295 | **−22.7 nm** | 0.049 | c575 nm (violet-purple) | 0 / 150 |
| **Muon** (lr 0.02) | 1.776 | **−6.1 nm** | 0.012 | c582 nm (pale lavender) | **33** / 85 |
| **NULL** (AdamW-norm updates, random direction) | 9.843 | **−0.0 nm** | 0.0008 | none (white) | 0 / 0 |
| noise floor (shuffle A vs shuffle B) | | ±0.6 nm s.d. | ≈0.001 | | |

"Shift" is the power-weighted mean wavelength minus 550 nm. A flat spectrum gives exactly 0. Negative means energy moved into the top modes (blue), and positive means energy moved into the bottom modes (red, i.e. flatter than random). "cXXX" is a complementary wavelength, meaning the colour lies toward the purple line.

- **Different optimisers give different colours, and the ordering matches the hypothesis.** AdamW is strongly blue. SGD is intermediate. Muon stays near white, and it is the only optimiser that pushes any matrices red: 33 of 180 rows are flatter than their shuffle, up to +8.9 nm. That fits Muon's orthogonalised updates, whose singular values are all ≈1.
- **The colour is not just how far training got.** At the checkpoint where each run first reaches SGD's final train loss (2.30), AdamW (step 924) is already at −72.6 nm and Muon (step 621) at −5.7 nm.
- **The Muon run carries an internal control.** Muon trains only the hidden matrices. The embedding, position table and head are trained by AdamW in that same run, and they go blue: emb −56 nm and head −105 nm, against −84 / −105 nm under AdamW. In the Muon panel the bottom rows are the only saturated ones.
- **Trajectories differ too.** AdamW blue-shifts fast: −12 nm by step 50, −60 by step 200, and a minimum of −73.7 nm at step 663, after which it relaxes slightly toward cyan. Muon shows a faint early blue transient in its attention rows (steps ~50–300) that fades back to white. SGD's shift is concentrated in the **o and down projections** (−49 and −44 nm, against −8 to −13 nm for q, k, v and up).
  - **Confound, declared:** o and down are initialised 1/√12 smaller, as in GPT-2 practice. At a fixed step size they therefore move further relative to their initial scale. The SGD bars may partly be an init-scale effect and not an SGD effect.
- **The per-head split shows heads diverging in colour.** The within-(layer, type) standard deviation of the 6 heads' shift is 6.9 nm for AdamW (q 8.0, k 9.4, v 4.6, o 5.5), 3.6 nm for Muon and 3.4 nm for SGD. For the NULL it is 0.7 nm, which equals the shuffle-noise floor of 0.6–0.8 nm. So heads diverge at 5–10× the noise. Under AdamW, q and k heads diverge most. Under Muon, v and o heads diverge most.

## The null control

`null`: at every step AdamW computes its update from the model's own gradients. The update is then replaced by a Gaussian random matrix with the **same Frobenius norm**, for every parameter. The weights travel exactly as far as AdamW would send them, but in no particular direction. The model does not learn (val loss 9.84, worse than init). Every matrix stays white: mean shift −0.0 nm and chroma 0.0008, which is at the noise floor.

Two conclusions follow. The colour is caused by the *direction* of the updates, not their size. And the pipeline has no bias toward any hue. Rendered with the same treatment, the NULL is the fourth panel of the hero: a sheet of white light.

The noise check uses the two shuffles recorded at each checkpoint, rendered against each other. They give chroma ≈0.001 and a head-centroid spread of 0.6–0.8 nm.

The MP check replaces the shuffle null with exact MP quantiles at the measured variance. The final hidden-matrix shifts then agree to 0.1 nm: AdamW −68.3, Muon −6.2, SGD −22.8, NULL −0.1.

## What was measured

- **Model.** GPT: 6 layers, d_model 384, 6 heads, context 256, pre-LayerNorm, GELU MLP (4×), no biases, untied embedding and head, learned positions.
- **Data.** TinyStories, train shard 0 (119M tokens). GPT-2 BPE cut to the 8,191 most frequent tokens plus UNK (UNK rate 0.2%). Val loss uses 20 × 64 × 256 tokens from the official validation split.
- **Training.** 2,000 steps × 64 × 256 tokens, bf16 autocast, grad-clip 1.0, 100-step warmup, then cosine to 0.1×. No weight decay in any run.
- **Learning rates.** From 600-step probes: AdamW {1e-3: 2.52, **3e-3: 2.35**}, Muon {**0.02: 2.20**, 0.05: 2.27}, SGD {0.1: 3.34, **0.3: 3.12**, 1.0: 3.86}.
- **Muon.** Newton–Schulz quintic, 5 steps (coefficients 3.4445, −4.7750, 2.0315), Nesterov momentum 0.95, update scaled by √max(1, rows/cols). It is applied to q, k, v, o, up and down. Embedding, positions, head and LayerNorm gains use AdamW at 1e-3.
- **Spectra.** Recorded at **509 checkpoints**: every step to about step 150, then log-spaced to step 2,000. Each covers **183 matrices**:
  - per layer: whole q, k, v, o (384×384) and each of their 6 heads (64×384 row slices for q, k, v; 384×64 column slices for o), plus up (1536×384) and down (384×1536);
  - then emb (8192×384), pos (256×384) and head (8192×384).
- **Eigenvalues.** For each matrix: eigenvalues of the smaller Gram divided by max(rows, cols), for W and for **two fresh entry-shuffles per checkpoint**. They are computed on CPU workers (`eigworker.py`), because GPU `eigvalsh` was slow while the GPU was shared.

## Declared choices

- **Null.** The mean of the two fresh shuffles' sorted eigenvalues at each checkpoint. MP quantiles serve only as a check (they agree, see above).
- **Colour rule.** Settled in `probe_cached.py`; the module version is `spectral.py`, verified to reproduce the probe's output exactly.
  - Wavelength position is the null's cumulative energy fraction by rank. The top modes sit at 400 nm.
  - Power is dF_trained/dF_null.
  - The spectrum is integrated against the CIE 1931 2° CMFs (Wyman–Sloan–Shirley fit). Equal-energy white maps to white.
  - Out-of-gamut colours are desaturated toward white.
  - Every cell is shown at maximum displayable brightness, so colour carries the spectrum's *shape*, not its energy.
- **Known insensitivity.** A single spike in the very top modes lands near 400 nm, where the CMFs are weak. A rank-1 spike ×20 is only faintly violet at γ = 1 (see the key). Low-rank structure is therefore under-coloured, not over-coloured.
- **Gamma.** γ = 1 is faithful. γ = 2 raises P to the power 2 and renormalises. That keeps each hue's direction but roughly doubles its saturation, and it exists to make Muon's red and blue threads legible. Every γ = 2 file is labelled. AdamW at γ = 1 is already saturated, so **the faithful version is the primary one**. γ = 4 was tried and discarded because it turns AdamW into a single violet slab.
- **Layout.**
  - One native cell is one (matrix, checkpoint), enlarged by nearest-neighbour only (4 px wide × 5 px tall).
  - Row order, top to bottom: layers 0–5, each as [q, q.h0–h5, k, k.h0–h5, v, v.h0–h5, o, o.h0–h5, up, down], then emb, pos and head.
  - Columns are checkpoints in order.
- **Ground.** The hero uses a near-black ground (17, 16, 15): light seen in a dark room. The study sheets, key and inset use a cream ground.

## Images

- `gallery/hero_dark_g1.png` (**best**, faithful) and `hero_dark_g2.png`: the four runs stacked, each a Richter-*Strip* style panel of 183 rows × 509 columns.
  - Tonal check: 26% of pixels are below luminance 0.15, 56% above 0.85 and 13% in the mid-range. It is bimodal, lit panels on black, even though the mean (0.68) gives a "commitment" of only 0.18.
- `gallery/study_{adamw,muon,sgd,null}_g{1,2}.png`: labelled per-run panels.
- `gallery/strip_*`: the same panels, bare.
- `gallery/wall_g{1,2}.png`: the first iteration, all runs on the cream ground (critique below).
- `gallery/interleaved_adamw_muon_sgd_null_g{1,2}.png`: every matrix gets four adjacent 2-px rows (AdamW, Muon, SGD, NULL), so the comparison is made row by row. At print size it reads as ruled pinstripe, with one line in four blue.
- `gallery/key.png`: synthetic spectra against the 384×384 MP null. Rank-1 spike, top-10 boost and steeper power (null^1.5) give violet/blue. null^0.9, null^0.6, all-modes-equal (an orthogonal W) and lifted bottom half give salmon/orange/pink. Top modes removed gives near-white.
- `gallery/inset_spectra.png`: the measured spectra (eigenvalue vs rank, log-log) at the final checkpoint for L1.q, L3.up, L4.o.h2, emb and head in each run. Each panel's ground is the colour that spectrum makes.
- `gallery/contact_sheet.png`.
- `gallery/probe_*` come from the earlier CPU probe on cached MLPs.

### Critique and iteration

**Iteration 1: `wall_g1.png`, cream ground.** Key commitment was high (mean L 0.90), and AdamW's blue slab dominated as it should. But the thing the work is about includes the *white* of Muon and the NULL, and white panels on cream paper lose their edges. The NULL was nearly invisible, which buries the null result.

**Iteration 2: the hero, near-black ground.** White now reads as light, and the NULL becomes a luminous white panel rather than an absence. The subject (hue against white) is the most visible thing in the frame: AdamW's blue is the largest saturated mass, Muon a pale woven sheet, and SGD five blue bars.

The γ = 4 attempt was rejected: it saturated AdamW to one flat violet. At thumbnail size Muon and NULL are hard to tell apart at γ = 1. That is the honest reading (Muon's hidden matrices really are within 6 nm of white), and γ = 2 separates them.

## Prior art

- **Martin & Mahoney, *Traditional and Heavy-Tailed Self-Regularization* (ICML 2019) and WeightWatcher.**
  - They read training through the empirical spectral density of weight matrices against MP, with phases running from random-like through bleeding-out, bulk+spikes and heavy-tailed to rank-collapse.
  - Our blue shift is their "bleeding-out / bulk+spikes" seen as colour. They plot ESD histograms and fit power-law exponents. We found no use of ESDs as colour.
- **Muon vs AdamW spectra.** Recent work reports that Muon-trained weights keep higher stable rank and a more uniform spectrum than AdamW. Examples: "The Spectral Dynamics and Noise Geometry of Muon" (arXiv 2606.08388), and Jordan's Muon post on orthogonalised updates.
  - So the *direction* of our result is expected. The contribution here is the rendering (every matrix, head and checkpoint in one field) and the random-direction null.
- **Spectrum-to-colour.** Hyperspectral-to-RGB conversion and Michel-Lévy interference charts are the physical precedents (see `_shared/research-dense-colour.md` §4). U-CIE colours high-dimensional data through CIELAB. We found no precedent for rendering singular-value spectra as light.

## Compute

Summed pasar job time is about 134 min, all run `--mem`-shared beside another agent's jobs, which slowed every step.

| job(s) | what | time |
|---|---|---|
| 978 | smoke | ~3 min |
| 983 | eigvalsh timing | 1 min |
| 984–990 | 7 LR probes, 600 steps each, run concurrently | ~53 min |
| 996, 1005 | AdamW runs, cancelled while GPU eigvalsh was being replaced by CPU workers | ~14 min |
| 1004 | smoke | 5 min |
| **1006** | **adamw** | 16 min |
| **1007** | **muon** | 20 min |
| **1008** | **sgd** | 10.5 min |
| **1009** | **null** | 11 min |

Arrays are in `cache/runs/*.npz` (≈150 MB each, gitignored) and numbers in `metrics.json`.

## Code

- `prep_data.py`: tokenisation. Run it with `uv run --with pyarrow --with tiktoken`.
- `train_gpt.py`: training and spectrum recording.
- `eigworker.py`: CPU shuffles, Grams and eigenvalues.
- `spectral.py`: the colour pipeline, MP quantiles, centroid and dominant wavelength.
- `render.py`: strips, studies, wall, interleaved and metrics.
- `hero.py`, `key.py` (key and inset), `contact.py`.

## Not done, and next moves

- **Weight decay was not run** because the budget was spent. The cheapest informative next jobs:
  - AdamW with wd 0.1;
  - Muon with matched per-step shrink;
  - a shuffled-token AdamW run (does data structure, not just the optimiser, set the hue?).
- **One seed each**, so seed spread is unmeasured. The null and the loss-matched comparison make an artefact unlikely, but 2–3 seeds per optimiser would put error bars on the −68 / −23 / −6 nm ordering.
- **The init-scale confound** (o and down initialised smaller) should be removed with a uniform-init SGD run.
- **Next move: a longer AdamW vs Muon pair** (20k steps, with wd).
  - It would show whether AdamW's late relaxation toward cyan continues and whether Muon's red rows grow.
  - Its hero would be a single long diptych, AdamW blue against Muon white, at print width (≈8,600 px/m; 20k steps log-spaced to ~2,000 columns).
