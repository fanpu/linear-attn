# Decode maps: where a language model's text changes as you turn the sampling knobs

*Fix the random numbers, sweep temperature and top-p over a 256 × 256 grid, and colour every pixel by the story Qwen3-0.6B writes there. The plane shatters into cells of identical text, and each new token cuts the cells again.*

<p align="center"><img src="gallery/hero/story_tp256_glass.png" width="720"></p>

<p align="center"><video src="gallery/film/refine_story256_glass.mp4" autoplay loop muted playsinline width="540"></video><br>
<i>L as time: the same 256² map after 1, 2, …, 64 tokens. Every frame is measured; the colours are declared and reshuffled per length so neighbours differ.</i></p>

## The phenomenon

A decoder with temperature *T* and nucleus threshold *p* samples token *t* from the truncated, tempered distribution

q<sub>T,p</sub>(v) ∝ softmax(z/T)<sub>v</sub> · 1[v ∈ nucleus<sub>p</sub>],

and with **inverse-CDF sampling** it picks the token where the sorted cumulative mass first passes a uniform u<sub>t</sub>. Here u<sub>1</sub>, …, u<sub>L</sub> are drawn **once** (seed 0) and shared by every pixel. Each pixel is then a deterministic function (T, p) ↦ text. The map is piecewise constant, and its boundaries are the (T, p) where some cumulative sum crosses u<sub>t</sub> exactly. After the first differing token the two texts condition on different prefixes, so every cell is cut again by the next token's boundaries. The partition can only refine with L, and finite L bounds how deep that nesting goes.

(`torch.multinomial` with a fixed seed is not continuous in the probabilities, which is why the uniforms are explicit.) The **Gumbel-max** variant picks argmax<sub>v∈nucleus</sub>(z<sub>v</sub>/T + g<sub>t,v</sub>) with shared Gumbel noise. It samples the same distribution but gives a completely different map (diptych below).
## Gallery

All plates show the story prompt “Write a short story about a lighthouse keeper.” on the (T ∈ [0, 1.5], top-p ∈ [0, 1]) plane at 256², L = 64, unless noted. Each grid sample is drawn as one square tile, with no interpolation. **Measured:** which pixels share identical text, the token index at which neighbours part, entropies, repetition. **Declared:** every palette, hash-to-colour assignment, lead width and misregistration.

### Cells of identical text
<table>
<tr><td><img src="gallery/hero/story_tp256_glass.png" width="400"><br><b>Stained glass.</b> Lead = tile edges whose two texts differ; glass colour is a hash-based proper colouring (declared).</td>
<td><img src="gallery/hero/story_tp256_ink.png" width="400"><br><b>Boundary lines, single ink.</b> Darker lines mark texts that part at an earlier token (opacity 1 − 0.75·t/L, declared).</td></tr>
<tr><td><img src="gallery/hero/story_tp256_mosaic.png" width="400"><br><b>Text-hash mosaic.</b> Colour = text cell; brightness = token at which the text leaves the greedy text.</td>
<td><img src="gallery/hero/story_tp256_age.png" width="400"><br><b>Boundary age.</b> Line colour = token index at which the neighbours part (batlow, 0 → 63).</td></tr>
<tr><td><img src="gallery/hero/story_tp256_riso.png" width="400"><br><b>Two-ink riso.</b> Blue = boundaries; pink = rank of mean model entropy; pink misregistered by 4 px (declared).</td>
<td><img src="gallery/hero/story_tp256_firstdiv_spectral.png" width="400"><br><b>First divergence, Spectral split</b> at token 16, the reproducibility horizon (labelled variant for a sequential quantity). Thin lines = boundaries born before token 16.</td></tr>
</table>

### Continuous metrics
<table>
<tr><td><img src="gallery/hero/story_tp256_coherence_spectral.png" width="400"><br><b>Coherent vs degenerate, Sohl-Dickstein Spectral split.</b> Signed = mean model entropy − 3.74 nats (valley of its histogram). Each side is rank-normalised, so the dark ends meet at the seam.</td>
<td><img src="gallery/hero/story_tp256_coherence_aurora.png" width="400"><br><b>Same field, palettes.py <code>aurora_ember</code></b> pairing.</td></tr>
<tr><td><img src="gallery/metrics/story_tp256_rep_spectral.png" width="400"><br><b>Repetition, Spectral split.</b> Red = the text repeats a 3-gram (rank of repetition rate); purple = no repeat (rank of 1 − distinct-2).</td>
<td><img src="gallery/metrics/story_tp256_entropy_magma.png" width="400"><br><b>Mean model entropy</b>, magma, 0.5–99.5 % clip.</td></tr>
</table>
<img src="gallery/metrics/story_tp256_metrics_sheet.png" width="820"><br>
<i>Metric sheet: repetition, distinct-2, length, model entropy, sampling entropy, first divergence.</i> Length/EOS is uninformative here: the story and list prompts never emit EOS within the horizon (0.02 % and 0 % of pixels), and the fact prompt ends at token 10 in over 95 % of pixels.

### Inverse CDF vs Gumbel-max (128², L = 48)
<img src="gallery/diptych/diptych_icdf_vs_gumbel_glass.png" width="820"><br>
<img src="gallery/diptych/diptych_icdf_vs_gumbel_ink.png" width="820"><br>
Both rules sample exactly the same distribution per step, but the maps differ. Inverse CDF gives **71 distinct first tokens** across the plane, **1945 distinct 48-token texts**, and 78 % singleton cells. Gumbel-max gives **2 first tokens, 269 texts** and 12 % singletons. With inverse CDF, the token under a fixed u moves whenever *any* probability mass above it shifts. With Gumbel-max, the argmax only changes when two perturbed logits swap order, and the perturbations are O(1) against logit gaps scaled by 1/T. (The Gumbel run used the eager forward pass and the inverse-CDF run used CUDA graphs. Per the horizon test below, that kernel difference alone changes about 18 % of texts by token 8. The first-token and cell-count contrast is far larger than that.)

### Other prompts
<table><tr>
<td><img src="gallery/hero/list_tp256_glass.png" width="265"><br>List prompt, glass (L = 48, 2348 texts)</td>
<td><img src="gallery/hero/list_tp256_ink.png" width="265"><br>List prompt, ink</td>
<td><img src="gallery/hero/fact_tp256_mosaic.png" width="265"><br>Fact prompt (“capital of Australia”): one answer covers ~97 % of the plane. Only the T > 1.3, p → 1 corner fractures (292 texts).</td>
</tr></table>

### Films
<video src="gallery/film/refine_story256_ink.mp4" autoplay loop muted playsinline width="540"></video>
`gallery/film/refine_story256_{glass,ink}.{mp4,gif}`: 1080², L = 1…64, each length held about 1 s with declared crossfades. The toy 64² version is `gallery/toy/refine_toy_glass.mp4`.

### Neighbouring cells read aloud
Walking along T at fixed top-p through the 256² map. Grey = prefix shared with the cell to the left; bold = where it departs. Transects whose first 48 tokens are identical to the previous cell are omitted.

<p><b>top-p = 0.498</b>, walking along T</p>
<table><tr><th>T range</th><th>tokens shared with the cell to the left</th><th>output (first 48 tokens; the part that differs from the cell to the left in bold)</th></tr>
<tr><td>0.000–0.082</td><td>–</td><td><span style='color:#888'></span><b>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a quiet coastal town, nestled between the cliffs and the sea, lived a lighthouse keeper named Elias. He had spent his life guarding the lighthouse, a structure that had stood</b></td></tr>
<tr><td>0.082–0.287</td><td>13</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a</span><b> foggy coastal town, nestled between the cliffs and the sea, lived a lighthouse keeper named Elias. He had spent his life guarding the lighthouse, a structure that had</b></td></tr>
<tr><td>0.568–0.604</td><td>21</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a foggy coastal town, nestled between the</span><b> crashing waves and the endless horizon, stood a lighthouse known as the *Whispering Beacon*. Its keeper, a man named Elias</b></td></tr>
<tr><td>0.604–0.680</td><td>36</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a foggy coastal town, nestled between the crashing waves and the endless horizon, stood a lighthouse known as the *</span><b>Starlight Beacon*. Its keeper, a man named Elias,</b></td></tr>
<tr><td>0.680–0.762</td><td>35</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a foggy coastal town, nestled between the crashing waves and the endless horizon, stood a lighthouse known as the</span><b> **Starlight Lighthouse**. Its tall tower, adorned with</b></td></tr>
<tr><td>0.791–0.873</td><td>18</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a foggy coastal town,</span><b> there stood a lighthouse known as *The Beacon*. Its tower stood tall and proud, casting long shadows over the sea. The keeper, a man</b></td></tr>
</table>
<p><b>top-p = 0.900</b>, walking along T</p>
<table><tr><th>T range</th><th>tokens shared with the cell to the left</th><th>output (first 48 tokens; the part that differs from the cell to the left in bold)</th></tr>
<tr><td>0.000–0.059</td><td>–</td><td><span style='color:#888'></span><b>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a quiet coastal town, nestled between the cliffs and the sea, lived a lighthouse keeper named Elias. He had spent his life guarding the lighthouse, a structure that had stood</b></td></tr>
<tr><td>0.059–0.076</td><td>13</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a</span><b> foggy coastal town, nestled between the cliffs and the sea, lived a man named Elias. He was known for his quiet demeanor and his deep love for the sea. Elias</b></td></tr>
<tr><td>0.076–0.205</td><td>35</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a foggy coastal town, nestled between the cliffs and the sea, lived a man named Elias. He was known</span><b> not only for his lighthouse but also for his quiet strength and</b></td></tr>
<tr><td>0.258–0.264</td><td>26</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a foggy coastal town, nestled between the cliffs and the sea,</span><b> stood a lighthouse known as the *Wharf Lighthouse*. Its keeper, a man named **Ethan</b></td></tr>
<tr><td>0.264–0.270</td><td>21</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a foggy coastal town, nestled between the</span><b> crashing waves and the distant horizon, stood a lighthouse known as the **Whispering Beacon**. Its tower stood tall and proud</b></td></tr>
<tr><td>0.270–0.281</td><td>42</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a foggy coastal town, nestled between the crashing waves and the distant horizon, stood a lighthouse known as the **Whispering Beacon**.</span><b> Built in the 19</b></td></tr>
<tr><td>0.281–0.287</td><td>18</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a foggy coastal town,</span><b> there stood a lighthouse known as the *Whispering Beacon*. Its keeper, a man named Elias, had spent his life guarding the lighthouse</b></td></tr>
<tr><td>0.287–0.316</td><td>13</td><td><span style='color:#888'>**The Lighthouse Keeper’s Tale** ⏎  ⏎ In the heart of a</span><b> stormy sea, where the waves crashed like angry spirits, there stood a lighthouse known as *The Beacon*. It was built in the 19th century, and</b></td></tr>
</table>
## What was computed

- **Model:** Qwen3-0.6B (HF weights, `local_files_only`). `qwen.py` is a hand-written reimplementation with a **bfloat16 body and fp32 LM head**, a static KV cache and no padding, verified against HF (`test_model.py`). Prompts use the chat template (`sample.py: chat_ids`).
- **Decoder engine** (`decode.py`): a prefix-trie that forwards each distinct prefix once, with CUDA-graph single-token steps (window Fb = 128, KV capacity Ncap = 384). The **certified sampler** computes per row an fp32 top-2048 plus a 0.04-logit histogram of all 151,936 logits, brackets the tail mass, and accepts a pixel only when the top-p cutoff and the inverse-CDF token agree at both brackets. Otherwise the pixel escalates K′ = 256 → 2048 → 32768 → a full float64 sort. Against a naive batch-1 float64 full-sort reference (`test_engine.py`), the certified and V-wide samplers give **identical maps (0 / 4096 pixels differ)** on the same forward kernels.
- **Noise:** seed 0, u<sub>t</sub> ~ U(0,1) float64, one per position, shared by every pixel. Gumbel: g<sub>t,v</sub> float64, per position and vocab entry.
- **Grids:** T ∈ [0, 1.5] and p ∈ [0, 1] at pixel centres. story_tp256 (L = 64, 1860 s, 246k node-steps), list_tp256 (L = 48, 292 s), fact_tp256 (L = 64, 131 s), story_tp128 (L = 48, 311 s), story_tp128_gumbel (44 s), plus placement runs (≈ 3400 s). Total GPU ≈ 2 h on the shared GB10, plus earlier toy runs.
- **Commands** (from `decode-map/`, `PYTHONPATH=. ../.venv/bin/python`, wrapped in `../_shared/gpu_run.sh`):
```
python decode.py --prompt story --res 256 --L 64 --x 0 1.5 --Fb 128 --Ncap 384 --out cache/story_tp256.npz
python decode.py --prompt fact  --res 256 --L 64 --x 0 1.5 --Fb 128 --Ncap 384 --out cache/fact_tp256.npz
python decode.py --prompt list  --res 256 --L 48 --x 0 1.5 --Fb 128 --Ncap 384 --out cache/list_tp256.npz
python decode.py --prompt story --res 128 --L 48 --x 0 1.5 --Fb 128 --Ncap 384 --out cache/story_tp128.npz
python decode.py --prompt story --res 128 --L 48 --x 0 1.5 --rule gumbel --nograph --out cache/story_tp128_gumbel.npz
python decode.py --prompt story --res 64 --L 32 --x 0 2 [--nograph | --slow --nograph | --fp32 | --shuffle --chunk_cols 2 --Ncap 32 --Fb 16] --out cache/place_story64_*.npz
python decode.py --prompt story --res 128 --L 48 --x 0 1.5 --shuffle --chunk_cols 8 --Ncap 96 --Fb 32 --out cache/place_story128_shuffle.npz
python verify.py placement cells horizon box        # -> cache/verify.json, gallery/verify_*.png
python heroes.py maps metrics diptych table         # CPU renders
python -c "import anim; anim.main('cache/story_tp256.npz','gallery/film/refine_story256_glass',s=4,mode='glass',hold=22,fade=8)"
```

## Verification and honesty

### 1. Placement and precision: how reproducible is a pixel's text?
Batched GPU kernels are not bitwise deterministic across batch shapes. Each row compares two full maps that differ in only one respect (`verify.py placement`, 64² at L = 32 unless noted):

| comparison | pixels differing (any of L tokens) | coherent region | median first differing token |
|---|---|---|---|
| brute force B = 256 vs trie Fb = 128 (same kernels) | **0.00 %** | 0.00 % | – |
| certified sampler vs V-wide sampler (same eager forward) | **0.00 %** | 0.00 % | – |
| trie, shuffled pixel order, Ncap 32 / Fb 16 vs Fb 128 | **1.95 %** | 0.25 % | 20 |
| eager forward vs CUDA-graph forward | 68.5 % | 67.0 % | 16 |
| bf16 body vs fp32 body | 57.9 % | 56.1 % | 13 |
| 128², L = 48: shuffled, Ncap 96 / Fb 32 vs Fb 128 | 78.5 % | 78.5 % | 13 |

<img src="gallery/verify_horizon.png" width="620"><br>
**Reproducibility horizon** (`verify.py horizon`). Solid lines: the fraction of pixels whose first *l* tokens agree. At *l* = 8 the figures are 100 / 99.6 / 82 / 81 / 95 % for the five pairs; at *l* = 16 they are 100 / 99.2 / 67 / 65 / 57 %. Dotted lines: Jaccard overlap of the boundary sets (0.81–0.86 at *l* = 8 for kernel/precision changes).

**Reading.** The sampler is exact: 0 mismatches wherever the logits are bit-identical. The bf16 body itself carries about 0.03-logit kernel noise (`scratch/graph_vs_eager.py`: median |Δlogit| ≈ 0.031, ≈ 1.5 % argmax flips on random inputs). Any change of kernel (eager vs graph, window size, fp32) re-rolls near-tied decisions, and the texts then diverge. Shuffling pixel order at a window size that keeps the same attention kernel costs only 2 %. Changing Fb from 128 to 32 at 128² changed the kernel and 78 % of texts. So the **coarse cell geometry (first ~8 tokens) is robust**, while **which exact text fills a small cell past ~16 tokens depends on precision and kernels**. It is a property of this bf16 model on this hardware, not of the model in the abstract.

### 2. Refinement and box counting
Distinct texts vs horizon *l* (story, 256²): 152 at *l* = 1, 955 at 4, 1604 at 8, 2056 at 16, 8376 at 64. At 128² the counts are 71 / 364 / 605 / 812 / 1945 (*l* = 48). The singleton fraction is **78 % at 128² and 79 % at 256²**: in the high-T, p → 1 corner, cells are smaller than the grid spacing at both resolutions.

<img src="gallery/verify_boxcount.png" width="760"><br>
Box counting of the boundary set (ε = 1/256 … 1/4 of the window). **Whole plane:** the local slopes are 1.3–1.8, drift with ε, and show no plateau. They agree at matched ε between 128² and 256² (e.g. *l* = 8: 1.49 / 1.56 / 1.44 / 1.32 / 1.36 vs 1.56 / 1.58 / 1.44 / 1.34 / 1.36). They sit between the nulls: a smooth curve gives 0.6–0.97, iid speckle 1.84–1.99. **Resolved half (T < 0.75)**, at *l* = 4, 8, 16 inside the reproducibility horizon: the slopes are 0.9–1.1 at both resolutions (256², *l* = 16: 0.92 / 1.02 / 1.00 / 1.09 / 1.26), with only 3, 4 and 9 cells.

**Verdict: not fractal in any measured sense.** Where the cells are resolved, the boundaries are piecewise-smooth curves (D ≈ 1), and refinement adds more curves as *l* grows. The higher whole-plane slope comes from the unresolved speckle corner, whose slope approaches the iid null. The nesting really is hierarchical (cells split at every token), but its depth is capped by L and, in practice, by the ~16-token precision horizon. Within that range the self-similarity is combinatorial (cells within cells), not metric. The film shows it best.

## Caveats
- A pixel is one sample, with one draw of uniforms (seed 0). Another seed gives a different, statistically similar map. Cell *areas* are not probabilities of texts.
- The map is also a map of the bf16 kernels: texts past ~16 tokens in small cells change with batch shape or precision (table above). The colour of a small late cell is not a stable property of the model.
- 256² is coarse for high T: 79 % of cells there are single pixels, so that corner reads as texture, not resolved geometry.
- Top-p = 0 is greedy by construction. The y axis is the nucleus mass, not a probability of the output.
- The inverse-CDF map depends on the (descending-probability) sort order of the CDF; a different order convention gives a different map. Gumbel-max has no such dependence.
- `story_tr192` ((T, repetition penalty) map) OOMed in the first attempt and was relaunched (see NOTES). It is not in this README yet.
- Files over 20 MB are not committed: none so far.

## References
- Holtzman et al. 2020, *The Curious Case of Neural Text Degeneration* (nucleus sampling).
- Qwen Team 2025, *Qwen3 Technical Report*.
- Sohl-Dickstein 2024, *The boundary of neural network trainability is fractal* (Spectral split style; github.com/Sohl-Dickstein/fractal).
- Gumbel-max trick: Maddison, Tarlow, Minka 2014, *A* Sampling*.
- Fractals companion doc §6 and §11 (`../ml-art-fractals.md`).
