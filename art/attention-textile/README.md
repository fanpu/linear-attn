# Attention as Textile: induction heads, woven

*When a transformer reads a sequence that repeats, some attention heads look back to the token that followed the last copy of the current token. Their attention matrices turn into cloth: parallel stripes at the repetition period, and, for self-similar sequences, patterns that repeat when you zoom in.*

<p align="center">
<img src="gallery/overlay_sierpinski-doubling_indigo.png" width="49%">
<img src="gallery/selfsim_sierpinski-doubling_indigo.png" width="49%">
</p>
<p align="center">
<video src="gallery/zoom_sierpinski-doubling.mp4" autoplay loop muted playsinline width="90%"></video><br>
<img src="gallery/proof_sierpinski-doubling_indigo.png" width="90%">
</p>

*Left: 255-letter Sierpinski-doubling word (each letter is 8 random tokens), read by Qwen3-0.6B. Indigo fill is the measured attention of the top-4 induction heads, shown as enrichment over uniform attention. Madder outlines mark the cells an ideal "attend to every previous occurrence" head would use. Right: the same matrix with a different declared normalisation (99.5th percentile, ^0.6) and dye bleed. Middle: continuous zoom into the top-left corner. Bottom: the proof plate, with measured matrices over ideal ones at 1, 1/2, 1/4 and 1/8 scale.*

---

## 1. The phenomenon

An **induction head** (Olsson et al., 2022) implements a two-step circuit.

1. A *previous-token head* in an earlier layer copies each token's identity into the next position's residual stream.
2. The induction head's query (the current token `A`) then matches keys that carry "my previous token was `A`". It attends to the token `B` that followed an earlier `A` and copies `B` into the logits, predicting `[A][B] ... [A] -> [B]`.

For a sequence of $T$ tokens $x_1..x_T$, the ideal induction attention from query $i$ is spread over

$$\mathcal{J}(i) = \{\, j \le i : x_{j-1} = x_i \,\},$$

and on a random segment of period $P$ this set is the off-diagonal stripes $j = i - kP + 1$. The quantities measured here:

| quantity | definition |
|---|---|
| induction (stripe) score | $\frac{1}{T-P}\sum_{i\ge P} a_{i,\,i-P+1}$ on repeated random tokens |
| prefix-matching score (Olsson) | $\mathbb{E}_i \sum_{j\in\mathcal{J}(i)} a_{ij}$ on Markov text with naturally recurring tokens (not repeated random) |
| previous-token score | $\mathbb{E}_i\, a_{i,i-1}$ |
| copying score (Olsson) | $\sum\lambda / \sum|\lambda|$ over eigenvalues of the direct OV circuit $W_U W_O^h W_V^h W_E$ |
| ablation | increase in loss on repeats when head $h$ is zero-ablated |
| in-context learning score | loss at late tokens (100-120) minus loss at early tokens (8-16); Olsson uses token 500 minus token 50 |

In small models these heads appear in a **phase change**: a sudden drop in loss on repeated material, while in-context learning improves at the same moment.

---

## 2. Gallery

### 2a. Self-similar words (the strongest pieces)

A word over a small alphabet (Thue-Morse, period-doubling, Fibonacci, Cantor, Sierpinski-doubling $X_{j+1}=X_jX_jz$, nested $x\,x\,x\,c$) becomes tokens by mapping each letter to a fixed block of random tokens. Cantor and Sierpinski filler letters `z` get *fresh* random tokens every time, so they never repeat. These sequences are up to 2048 tokens long. The **self-similarity is put into the input by construction**. What is measured is whether, and how faithfully, the model's attention reproduces it across scales.

| | |
|---|---|
| <img src="gallery/proof_cantor_indigo.png"> | <img src="gallery/proof_thue-morse_indigo.png"> |
| **Cantor proof plate.** Measured enrichment (indigo) vs ideal (madder) at 3x zoom steps. AUC 0.999 / 1.000 / 1.000 / 1.000 per scale ring. | **Thue-Morse proof plate.** AUC 0.989 / 0.995 / 0.996 / 0.991. |
| <img src="gallery/overlay_cantor_indigo.png"> | <video src="gallery/zoom_cantor.mp4" autoplay loop muted playsinline width="100%"></video> |
| **Cantor overlay print** (243 letters). | **Cantor zoom** (27x over 6 s). |
| <img src="gallery/selfsim_cantor_weave.png"> | <img src="gallery/selfsim_thue-morse_rownorm_weave.png"> |
| **Cantor, woven.** Warp colour, width and float encode attention; the weave is declared. | **Thue-Morse, woven, row-normalised** (declared) so the lower lattice reads. Raw version: `gallery/selfsim_thue-morse_weave.png`. |
| <img src="gallery/word_thue-morse_riso.png"> | <img src="gallery/word_fibonacci_riso.png"> |
| **Thue-Morse at token level**, 2048 tokens, head L20H14. Blue = ideal targets from the tokens; pink = measured. Mass on ideal targets 0.80. | **Fibonacci word**, same set-up. Mass on ideal targets 0.92. |

<img src="gallery/word_thue-morse_ideal_selfsimilar.png" width="100%">

*The ideal Thue-Morse "same letter" matrix is exactly self-similar under the substitution: $C(2n)=\begin{bmatrix}C(n)&1-C(n)\\1-C(n)&C(n)\end{bmatrix}$, checked numerically for n = 8..256. The last panel is the measured letter-level attention at 512 letters. Recency dominates there, and the lattice underneath is only faint.*

| | |
|---|---|
| <img src="gallery/selfsim_plate_indigo.png"> | <img src="gallery/selfsim_plate_riso.png"> |
| **All six words plus two controls**, four zoom levels each (indigo). | Same, two-ink riso: blue ideal, pink measured. |

### 2b. Pattern book: repetition rhythms (Qwen3-0.6B, head L20H14)

| | |
|---|---|
| <img src="gallery/book_rhythm_riso.png"> | <img src="gallery/book_rhythm_weave.png"> |
| **Riso, 2 inks.** Blue = where an ideal induction head looks (computed from tokens); pink = measured attention. The misregistration is deliberate. | **Weave.** |
| <img src="gallery/book_rhythm_indigo.png"> | <img src="gallery/book_rhythm_dark.png"> |
| **Indigo.** | **Dark / magma.** |
| <img src="gallery/book_series_riso.png"> | <img src="gallery/book_sampler_crossstitch.png"> |
| **Same thread gauge, different lengths**: period 20 repeated x2/3/6/10; period 16 at T = 64/128/256. Weave version: `gallery/book_series_weave.png`. | **Cross-stitch sampler**: a window of each swatch, floss by raw attention bins 0.03/0.10/0.25/0.50/0.80 (declared). |

Periods 5, 8, 13, 21, 34, 55 give stripe lattices of pitch P. The head attends to **every** earlier copy, not just the latest, so there are several parallel stripes. The Thue-Morse and Fibonacci swatches give broken, quasi-periodic lattices. The 15%-mutated swatch frays. The no-repetition control stays blank apart from the sink and a faint diagonal. Attention mass on ideal targets per swatch: 0.84 (P=5), 0.86 (P=8), 0.92 (P=13), 0.85 (P=21), 0.81 (P=34), 0.84 (P=55); repeats x2: 0.66, x3: 0.78, x6: 0.85, x10: 0.86; Thue-Morse 0.76, Fibonacci 0.86, nested 0.69, mutated 0.66.

### 2c. The whole model as a tapestry (Qwen3-0.6B, 28 layers x 16 heads)

| | |
|---|---|
| <img src="gallery/qwen_tapestry_quilt_P12R4.png"> | <img src="gallery/qwen_hero_dark_P50.png"> |
| **Patchwork quilt.** Period 12 x 4. Patch colour = head class from measured scores (red induction > 0.2, blue previous-token > 0.35, ochre first-token sink > 0.7, sage other; thresholds and palette declared). Pattern is per-head normalised. The red induction patches cluster in layers 16-24. | **Dark hero.** Period 50 x 4, 1 px = one attention weight, each head scaled to its own 99.5th percentile (declared), sink column in grey. |
| <img src="gallery/qwen_tapestry_weave_P12R4.png"> | <img src="gallery/qwen_tapestry_riso_P12R4.png"> |
| **Woven tapestry** (full size 4092x7114; the texture only reads at 1:1: `gallery/qwen_tapestry_weave_P12R4_detail_1to1.png`, `gallery/qwen_weave_closeup_L20H14.png`). | **Three-ink riso.** Pink = pattern, blue tint = induction score, yellow tint = previous-token score. |
| <img src="gallery/qwen_tapestry_indigo_P12R4.png"> | <img src="gallery/qwen_top8_weave_P50.png"> |
| **Indigo.** | **Eight strongest induction heads, woven** (period 50). |
| <img src="gallery/qwen_top8_crossstitch_P12.png"> | <img src="gallery/qwen_weave_closeup_L20H14.png"> |
| **Cross-stitch**, eight induction heads. | **Close-up**: one cell = one thread crossing. |

### 2d. The toy model: watching the stripe arrive

<p align="center"><video src="gallery/toy_induction_forming.mp4" autoplay loop muted playsinline width="95%"></video></p>

*34 s, 10 fps. Left: all 8 heads of the 2-layer attention-only model on a fixed 64-token probe (16 random tokens x 4), woven, one frame per checkpoint (every 5 steps through the transition, every 40 outside). Right: the loom record for the induction head and the previous-token head, woven down to the current step. Bottom: eval loss with a cursor. GIF: `gallery/toy_induction_forming.gif`.*

| | |
|---|---|
| <img src="gallery/toy_loom_indigo.png"> | <img src="gallery/toy_phase_change_curves.png"> |
| **Loom record.** Each weft row is a training moment (every 20 steps), each warp thread a look-back distance k = 0..127. Stripes at k = 31/63/95 appear in all four layer-1 heads in the same rows where layer-0 heads lock onto k = 1 (previous token) and k = 0 (current token). The rule is at the half-way point of the loss drop; the loss curve runs down the side. | **Phase-change curves** (seeds dashed; 1-layer control dotted). |
| <img src="gallery/toy_strip_riso.png"> | <img src="gallery/toy_strip_weave.png"> |
| **Film strip, riso**: pink L1H3 (induction), blue L0H3 (previous token), 12 checkpoints. | **Film strip, weave.** Cross-stitch version: `gallery/toy_strip_stitch.png`. |

<img src="gallery/toy_loss_vs_position.png" width="70%">

Weaker or duplicate styles are in `gallery/extra/` (dark Qwen tapestry at P12, dark and indigo series pages, dark and weave self-similar plates, dark word plates, indigo film strip, weave loom).

---

## 3. What was computed

**Environment.** `/home/fzeng/ml/research/art/.venv` (torch 2.14 + cu130, transformers 5.15) on a shared NVIDIA GB10. Computation and rendering are separate steps: raw arrays go to `cache/`, and every style renders from the cache.

### Part A: toy model (`toy_common.py`, `train_toy.py`, `analyze_toy.py`, `render_toy.py`)
- **Model.** 2 layers, 4 heads, d_model 128, attention-only (no MLP), pre-LayerNorm, learned absolute positions, untied unembedding. Vocab 256, context 128, about 215k parameters. Also a **1-layer control**.
- **Data** (generated on the fly). 75% "Markov text" from a fixed sparse random bigram chain (rows ~ Dirichlet(0.1)), with 2 earlier spans of 8-32 tokens copied verbatim later. 25% "repeated random": a uniform random segment with period drawn from 6..48 per batch, tiled.
- **Optimisation.** AdamW, lr 1e-3, betas (0.9, 0.98), no weight decay, 200 warm-up steps, batch 128, 6000 steps, float32, seeds 0 (main), 1, 2.
- **Logging.** Every 5 steps on fixed eval sets: 512 Markov sequences with one copied span, 256 repeated-random sequences with period 32. Logged: all scores in §1, per-position loss, the mean attention pattern on the repeated set, and a single probe pattern. Head ablation every 25 steps. Weights every 5 steps for the main run (1201 checkpoints, 520 MB float16). Post hoc from checkpoints: copying scores every 50 steps, 64-token probe patterns at every checkpoint, and the offset spectrum $S_t(k)=\mathbb{E}_{i\ge 32}\,a_{i,i-k}$.
- **Wall time.** Main run 39 min; seeds 1 and 2 and the 1-layer control about 25-40 min each. A discarded pilot ran 15 min. The shared machine was at load ~40, and the tiny model spent most of its time on data generation and evaluation, not the GPU.

```
python train_toy.py --steps 6000 --ckpt 5 --tag main
python train_toy.py --steps 6000 --ckpt 50 --seed 1 --tag seed1      # and --seed 2 --tag seed2
python train_toy.py --steps 6000 --ckpt 50 --layers 1 --tag onelayer
python analyze_toy.py --tag main
python render_toy.py curves strip loom movie
```

### Part B: Qwen3-0.6B (`compute_qwen.py`, `compute_qwen_tapestry.py`, `compute_qwen_selfsim.py`)
- Loaded from the local HF cache (`local_files_only=True`), float32, `attn_implementation='eager'`. Random tokens are drawn uniformly from the 151,643 non-special ids with no BOS prepended, so the first token acts as the attention sink.
- **Main probe.** Period 50 x 4 (200 tokens), scores averaged over 24 random draws. Null: 8 non-repeating sequences.
- **Tapestry probes.** P12x4, P8x6, P16x3, P6x8, 16 draws each.
- **Pattern book.** 18 sequences (periods 5..55, repeats 2..10, lengths 64..256, Thue-Morse/Fibonacci/nested with 4-token letters, 15% mutated, no-repeat control).
- **Self-similar words.** Letter blocks k = 2, 4, 8 tokens, sequences up to 2048 tokens, 4 independent token draws for k = 4 and 8. Attention is captured per layer with forward hooks, keeping only the top-8 induction heads at full resolution and all heads pooled to 256x256.
- **GPU wall time.** About 11 min in total (main 2.1 min, self-similar k = 8 and k = 4 about 4 min each).

```
python compute_qwen.py && python compute_qwen_tapestry.py
python compute_qwen_selfsim.py --k 2 --draws 1 --out cache/qwen_selfsim_k2.npz   # first pass, slightly different word set
python compute_qwen_selfsim.py --k 8 --out cache/qwen_selfsim_k8.npz
python compute_qwen_selfsim.py --k 4 --out cache/qwen_selfsim_k4.npz
python analyze_selfsim.py
python render_qwen.py; python render_book.py; python render_selfsim.py; python render_words.py
python render_proof.py movies; python replicates.py; python curate.py
```

(The k = 2 file came from an earlier version of `compute_qwen_selfsim.py` with Thue-Morse/period-doubling/Fibonacci at 1024 letters, Cantor depth 6 and nested depth 5; it is only used for the resolution check below.)

**Precision.** Everything is float32. The display is float16/uint8 after scaling. No deep-zoom precision issue arises: the zooms here are at most 27x over integer-indexed matrices.

---

## 4. Verification / honesty

### Did the phase change happen in my toy run? Yes.
- **Loss on repeats 2-4 of random tokens.** 5.56 nats (above ln 256 = 5.55) on a long plateau, then **0.15 nats**. The drop runs from 10% to 90% between **steps 1480 and 1850**, with the half-way point at step 1645. Seed 1: steps 1395-1865, half-way at 1505.
- **Copied spans in Markov text.** 3.46 nats before, 0.26 after. The **ICL score** goes from -0.14 to **-1.44** nats.
- **Loss bump.** Loss on *fresh* (non-copied) Markov tokens rises from 4.13 to about 4.42 at the transition, then slowly recovers to 3.94. This is the small bump the literature describes: capacity is re-allocated.
- **Heads.**
  - All four layer-1 heads become induction heads: stripe scores 0.64 / 0.61 / 0.51 / **0.74**, prefix-matching 0.40-0.44 against a uniform baseline of 0.018, copying scores 0.96-1.00.
  - L0H3 becomes a previous-token head (score 0.95). L0H2 attends to the current token (k = 0 stripe in the loom).
  - Ablation (+nats on repeats): L0H3 +7.4, L0H2 +4.8, each L1 head +0.06..0.38, reflecting redundancy across the four induction heads.
- **Co-emergence.** The first induction score > 0.2 comes at step 1665 and the first previous-token score > 0.2 at step 1650 (seed 1: 1535 and 1535). The two halves of the circuit appear together, within one logging interval. They do not appear sequentially.
- **Replicates** (`python replicates.py` writes `cache/toy_replicates_summary.json`).

  | run | 10%-90% of the loss drop (half-way) | loss on repeats: start -> end | max induction score | ICL score |
  |---|---|---|---|---|
  | seed 0 | steps 1480-1850 (1645) | 5.56 -> 0.15 | 0.74 | -1.44 |
  | seed 1 | steps 1395-1865 (1505) | 5.55 -> 0.15 | 0.67 | -1.43 |
  | seed 2 | steps 1390-1745 (1530) | 5.56 -> 0.16 | 0.72 | -1.44 |

  The **1-layer control** has no phase change and no induction head. Its loss on repeats drifts slowly from 5.62 to 4.81 over 6000 steps, its maximum induction score is 0.03 and its ICL score is -0.17. This matches Olsson et al.: induction needs two layers (key composition), and in-context learning barely improves without it.

### Qwen3-0.6B induction heads
- **Loss per repeat** of a 50-token random segment: 14.45 nats (first copy; random tokens are very unlikely), then 0.92, 0.41, 0.40 on copies 2-4.
- **Top induction (stripe) scores**: L20H14 0.68, L21H8 0.66, L6H11 0.61, L18H5 0.61, L3H10 0.54, L24H6 0.51, L16H14 0.51, L19H5 0.50. The same score on non-repeating sequences is at most **0.008** over all 448 heads.
- **Previous-token heads**: L15H3 0.83, L1H3 0.73, L2H12 0.72.
- **Head classes** on P12x4: 16 induction, 13 previous-token, 154 sink-dominated, 265 other.

### Self-similarity (`cache/selfsim_metrics.json`, `cache/proof_scale_aucs.json`)
- **Does the measured attention follow the ideal matrix at long range?** AUC of top-4-head letter attention for the ideal cells, at letter offsets 65+ (k = 8): Thue-Morse 0.99, Fibonacci 0.97, Cantor 0.998, nested 0.96, period-doubling 0.90, Sierpinski-doubling 0.83. With enrichment display, per zoom ring: Cantor 0.999-1.000, Thue-Morse 0.99-1.00, Sierpinski 0.81-0.87.
- **Where the real heads break self-similarity.**
  - *Sierpinski-doubling:* the ideal cells come in pairs (both earlier copies of $X_j$), but the heads mostly pick one. They prefer the copy whose longer preceding context also matches.
  - *Thue-Morse and period-doubling:* a strong recency bias makes the lattice fade with distance unless each row is normalised.
- **Decimation test** (offset-normalised $A[::q,::q]$ vs the top-left block, Pearson r):

  | word | measured | ideal | control |
  |---|---|---|---|
  | Thue-Morse | 0.87 | 0.98 | i.i.d. a/b word: 0.08 |
  | period-doubling (odd indices) | 0.82 | 0.92 | |
  | Cantor (q = 3) | 0.90 | 0.91 | shuffled Cantor: 0.12 |

- **Fractal claim: none.** Box counting was run on Cantor attention, binarised to the top-N cells with N = number of ideal cells, at eps = 1, 3, 9, 27, 81 letters (under two decades):
  - ideal matrix: slope 1.16 (asymptotic theory for Cantor x Cantor is log 4 / log 3 = 1.26; finite size);
  - measured raw attention: slope 1.09, but curved (0.95 fine vs 1.22 coarse);
  - measured enrichment: slope 0.99, precision 0.88;
  - shuffled-Cantor null: 0.99, strongly curved (0.50 vs 1.51);
  - **resolution check** (k = 2 blocks, 729 letters, 6 scales): measured precision only 0.13 and slope curved 0.60 -> 1.47, indistinguishable from the shuffled null.

  The extra structure the model adds (a previous-letter diagonal, recency) is one-dimensional and contaminates the count. **The self-similarity in these images is inherited from the constructed input and reproduced by the heads (AUC ~0.99); it is not an emergent fractal of the network.** There is less than two decades of scale, and the box-counting fit is not a straight line. A clean negative.

### Checks and fixes made along the way
- **Right-edge line (pattern book II).** The vertical pink line at the right edge of every triangle in the first pattern-book riso page was a **rendering bug**: `np.roll` misregistration wrapped the first-token sink column around to the right edge. It is fixed by a zero-filled shift.
- **Sierpinski bottom band.** The dense, blurry band at the bottom of the Sierpinski plate is **real**. The word ends in 8 consecutive fresh filler letters, the ends of all 8 construction levels. Filler queries have nothing to match, so their attention is diffuse (row max 0.02 vs median 0.047).
- **Pilot duplicate.** The pilot run was identical to the main run step for step (same seed), so it was stopped and discarded.

### What didn't work
- **Short letter blocks.** Thue-Morse with 2-token letters (only 4 distinct tokens) gives prefix matches everywhere. Recency then dominates and the letter lattice barely shows. 8-token letters were needed.
- **Raw scaling for Cantor.** Raw attention scaling makes Cantor look like tartan rather than dust, because attention per key is diluted as 1/i. Enrichment over uniform fixes the comparison. It is declared, and the raw version is still in the plates.
- **File size.** Full-grid weaves at 5 px per cell only just fit under 20 MB after replacing white-noise fibre texture with periodic twist highlights.

---

## 5. Caveats
- **Scope of the mechanism.** Induction heads explain *part* of in-context learning, not all of it. Olsson et al. argue causally only for small attention-only models; for large models the evidence is correlational. Qwen3-0.6B also has heads that do fuzzy or longer-prefix matching, which is visible in the Sierpinski deviations.
- **Constructed settings.** The toy setting is constructed. The data mixes in repeated random segments precisely so that induction pays off, and the phase-change timing depends on that mix, the learning rate and the batch size.
- **Self-similarity is imposed.** It comes from the word construction. The measured claim is only that the heads reproduce it.
- **Declared display choices.** Normalisation (per-head percentile, row max, enrichment over uniform), the power transform, dye bleed, riso grain and misregistration, weave float thresholds, cross-stitch bins and quilt class thresholds are all aesthetic choices, stated in each caption. Attention weights are never smoothed in any computed metric.
- **Qwen setup.** Attention was measured without a BOS token, so the first token of each probe absorbs a large share of attention (the sink). Qwen3 uses GQA (8 KV heads for 16 query heads), so the 16 query heads per layer are not independent.
- **Attention is not causation.** Attention patterns show where a head looks, not what it writes. Copying and ablation were measured only for the toy model.

---

## 6. Ideas explored / not pursued

Brainstormed ideas, with the two picked (plus the required pieces) marked:

1. **Self-similar words read by induction heads.** *Picked.* Most intricate, and the self-similarity can be verified against an exact ideal.
2. **Loom record / loom animation**: training time as weft, look-back distance as warp. *Picked.* The phase change becomes literally woven.
3. **Patchwork quilt** of the 28x16 grid, coloured by measured head class. Done.
4. **Pattern-book page** of rhythms and lengths. Done.
5. **Offset "ribbon" barcodes**: mean attention vs look-back distance for all 448 Qwen heads as a woven ribbon sampler. Not pursued (partly subsumed by the loom).
6. **Fraying series**: the same repeated text at mutation rates 0-50% as progressively worn cloth. Only one swatch (15%) was made.
7. **Scale series**: where induction heads sit in Qwen3 0.6B / 1.7B / 4B / 8B. Not pursued (budget).
8. **Physical output**: export the weave draft as a WIF file for a real Jacquard loom. Not pursued, but the weave renderer already computes an over/under map per cell.
9. **A fractal-dimension claim** from box counting. Tried; negative (see §4).
10. **Training the toy model on self-similar words** to see whether the transition changes. Not pursued.

---

## 7. References
- Olsson, Elhage, Nanda, et al., *In-context Learning and Induction Heads*, Transformer Circuits Thread, 2022. Definitions checked against the paper: ICL score = loss at the 500th token minus the 50th; induction heads = prefix matching plus copying measured on repeated random tokens; induction heads form in 2-layer but not 1-layer attention-only models.
- Elhage, Nanda, Olsson, et al., *A Mathematical Framework for Transformer Circuits*, 2021 (OV/QK circuits, copying eigenvalue score).
- Qwen Team, *Qwen3 Technical Report*, 2025 (Qwen3-0.6B).
- Allouche & Shallit, *Automatic Sequences*, 2003 (Thue-Morse, period-doubling, Fibonacci words).
- Main project doc: `../ml-art-directions.md` §8; verification protocol `../ml-art-fractals.md` §11.
