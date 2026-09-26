# Overnight run — 2026-09-26 (GPU until 21:30 EDT)

New ML-phenomena art, at most two project agents at a time, all GPU work through pasar.
Shared brief: [_shared/OVERNIGHT-BRIEF.md](_shared/OVERNIGHT-BRIEF.md).

## Queue

| wave | project | idea | status |
|---|---|---|---|
| 1 | [ticket-shadow](ticket-shadow/) | lottery-ticket input mask as a perforated sheet that casts where the data lives | done |
| 1 | [one-road](one-road/) | function-space (InPCA) atlas: do all architectures take one road from ignorance to truth? | done |
| 2 | [drainage](drainage/) | greedy decoding's repetition loops as a river basin over the whole vocabulary | done |
| 2 | [palimpsest](palimpsest/) | does a network retrained on page B keep A's fine detail under B's broad strokes? | done |
| 3 | [to-scale](to-scale/) | massive activations at true scale; the one weight that breaks the model | done |
| 3 | [unsayable](unsayable/) | under-trained tokens as a type specimen, with the model's failed attempts to say them | done |

## Results

### ticket-shadow — real, and the optimiser decides the shadow

LeNet-300-100 lottery tickets on MNIST (20 IMP rounds, 5 seeds) are genuine: 98.06% test at 3.5% of
input weights vs 97.52% dense; re-initialised mask 96.3%, random pruning 92.1% at 1.4%. Most of the
shadow is the data (pixel-variance R² 0.75; first-run weight movement R² 0.83). The finding: with raw
pixels, **Adam's ticket draws the outline of every pixel MNIST ever lights** (a flat plateau: pixels lit
in 10–99 images keep as many connections as pixels lit in 20,000+), while **SGD's draws a soft core that
tracks variance** (r 0.88). Per-class digit ghosts are a null. Best: `ticket-shadow/gallery/diptych_adam_vs_sgd_r15.png`.
Likely mechanism (coordinator's reading, untested): Adam normalises each weight's step, so a weight that
receives *any* gradient moves about as far as one that receives a lot. Next: AdamW / large-ε Adam runs to
pin it, then cut the pair as two perforated sheets.

### one-road — one road with lanes; forks on held-out data

64 small runs (logistic, MLPs, CNN, ResNet-8, ViT, GRU) on MNIST, Fashion-MNIST, CIFAR-10 and a+b mod 97,
embedded per task with InPCA (Mao et al. 2024). On training examples every one of 309 cross-architecture
pairs is closer at matched progress than a within-class-shuffled null (median ratio 0.48–0.69), and all
untrained nets start at the uniform point. But architectures sit 1.6–2.5× further apart than seeds: lanes,
not one road. **On held-out CIFAR the road forks into three valleys** (MLPs; CNN/GRU; ResNet/ViT).
Shuffled-label runs leave ignorance in another direction; grokking runs on mod 97 walk away from truth on
held-out pairs and come back. Caveat: the flattest-null picture depends on the metric (Bhattacharyya vs
Hellinger); the numbers barely move. Best: `one-road/gallery/plate_roads_cifar_te.png`. Coordinator's
note: the framed-plot plates read as figures; the CIFAR fork is the image. Next: 5 seeds on held-out CIFAR
to test whether the three valleys are stable.

### drainage — an hourglass, not a basin

Qwen3-0.6B, greedy, no template, from all 151,643 ordinary tokens (256-token cap). 48.8% fall into a
loop; **31,128 distinct loops**, rank–size slope −0.93 (Zipf-like); the biggest are `0000…` (7,311 starts),
` \frac{1}{2} \left(` and ` 2^2 +`. The surprise: the whole vocabulary first funnels through **62 first
words** (`Question` 40%, ` Instructions` 25%) before fanning back out into the loops. 30% of pre-loop text is
shared token-for-token with other runs, so the rivers are real. Null (one token of memory): 4 terminal
states. bf16 vs fp32 changes ~a quarter of individual assignments but not the census. Qwen3-1.7B: a different
landscape, but `0000…` is the biggest sea there too. Best: `drainage/gallery/watershed.png`,
`drainage/gallery/census.png`. Coordinator's note: `Qwen/Qwen3-0.6B` is the post-trained model, so
"Question"/"Instructions" is very likely instruction-tuning showing through. Next: the same census on
Qwen3-0.6B-Base (needs download) and Qwen3-4B.

### palimpsest — no ghost at convergence; a real one on the way, and a hidden one in the weights

Coordinate networks trained on page A (Latin, italic), then on page B (bold, turned 90°); nulls C→B and
B-from-scratch. **At convergence nothing of A is left** on the pixel grid (<0.003 per band, same as controls).
On the way: **SIREN** does the hypothesised thing briefly — B's broad strokes land in 1–4 steps while A's
fine letters remain (overlap 0.34–0.49 vs 0.16–0.21 reversed). **Fourier features + Adam**: the page goes
blank for ~40 steps, then **A comes back** (amplitude 0.19–0.34 around step 100) before fading; C→B brings
back C, never A. After B, that network still responds to A's finest detail 4–8× more than B's, and relearns A
to 26.3 dB in 150 steps vs 13 dB for controls — the under-text survives in the weights, not the image. Adam
erases A 100–300× faster than it writes B; SGD erases at the rate it writes. Best:
`palimpsest/gallery/resurface_ff32_w256.png`, `palimpsest/gallery/palimpsest_slit_siren_n512.png`,
`palimpsest/gallery/uv_relearn_ff32_w256.png`. Open question: is A's return an Adam-restart artefact
(zeroed moment estimates)? Next: carry optimiser state over / warm-up / AdamW.

### to-scale — a 39-metre bar, no softmax needed, and a committee instead of a super weight

Qwen3-0.6B layer 2: the first token's dim 35 is **43,691× the median** activation (1.7B: 54,117×; 4B:
38,346×; same model with random weights: 15×). At 1 mm per median, **that bar is 39.1 m** — a 16.6 cm × 4.71 m
floor-to-ceiling strip at house print density. **Massive activations are not tied to softmax**: Gated
DeltaNet (4,855×) and GLA (3,495×) have them with no softmax anywhere; DeltaNet (733×) and RWKV-7 (208×)
don't; OLMo-2-1B, a softmax model, has neither massive activations (158×) nor an attention sink. **Qwen3-1.7B
has a textbook super weight** ([1793, 1821] writes 99.8% of the activation; zeroing it: perplexity 16.6 →
109.6). **Qwen3-0.6B has none — a committee of six** weights in one row; any one removed barely matters, all
six → perplexity 1,562 (all 64 subsets measured). Random-weight nulls < 0.01. First-order fragility
underestimates the 1.7B super weight 41×, so the maps show estimate as discs and measured ablation as rings.
Best: `to-scale/gallery/superweight_detail_1.7b.png`, `to-scale/gallery/committee.png`,
`to-scale/gallery/banner_film.mp4`, `to-scale/gallery/typology.png`.

### unsayable — the words a model has but cannot say, and what each size says instead

Land & Bartolo's indicator (top 2% of tokens) plus three repeat prompts; unsayable = p < 0.01 in all three.
Qwen3 0.6B/1.7B/4B/8B: 743/703/860/1,105 unsayable (72/69/84/38% of candidates); OLMo-2-1B 349 (18%).
**Null** (random ordinary tokens, same test): 0.10–0.17% fail for Qwen3, 0.86% for OLMo-2. 696 tokens are
unsayable at all four Qwen3 sizes (Jaccard 0.61–0.74), mostly rare Hangul, then Han/Arabic/Hebrew; OLMo-2's
are code identifiers and spam. **Each size fails in its own way**: 0.6B answers 526 of them with a lone `'`
(0 of 1,029 random tokens); 1.7B with "The string '" (365; 0 random); 4B with its own `<tool_call>` token
(207; 4 random); 8B with `</think>` loops (693; 13 of 2,906 random). Best:
`unsayable/gallery/instead_qwen3-0.6b.png` (a single madder apostrophe over 526 tokens — one of the night's
strongest images), `concordance_qwen3.png`, `register_qwen3.png`. Caveats: chat checkpoints, not Base; the
plates set real spam tokens verbatim (`_sexkontakte`, `_swingerclub`) — worth a curatorial decision before hanging.
