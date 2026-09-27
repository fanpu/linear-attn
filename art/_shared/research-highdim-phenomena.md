# High-dimensional phenomena: where each pixel carries a vector

*Scouting report, 2026-09-26. Brief: the collection so far is mostly one scalar per mark
(loss, λ, converge/diverge, a rank), coloured through a colormap. Find ML phenomena whose
natural output is a dense, many-channel field, so that colour can come from the data. Prefer
small networks. Check prior art candidly.*

---

## 0. What "colour from the data" can honestly mean

A colormap turns one number into a hue, and that hue carries nothing the number doesn't. In
every candidate below, a pixel carries a vector. Four mappings keep the colour faithful, and
each proposal says which one it uses.

1. **Three-way simplex → exact colour.** If the pixel's vector is a probability over exactly
   three things (three classes, three memories, three reference networks), the barycentric map
   to an OKLab triangle (or RGB primaries) is a bijection. Nothing is lost and nothing is added.
   It is worth designing experiments so that K = 3. This is the strongest argument for colour
   anywhere in the collection.
2. **K-way distribution → linear mixing.** Give each outcome a hue at equal OKLab lightness,
   and colour the pixel by Σ pₖ cₖ. Because the map is linear, a mixture of outcomes is a
   mixture of colours. Uncertainty loses chroma, since opposite hues cancel to grey, so it
   needs no separate channel. The only declared choice is where each hue goes on the circle.
3. **Cyclic quantity → hue.** Use hue only when the quantity really is on a circle: a residue
   mod 10, a Fourier phase, or the mantissa of log₁₀.
4. **Hidden vector → linear projection.** Take PCA or a fixed readout (such as the
   unembedding) to 3 dimensions and map that to OKLab. State the variance explained, fit the
   projection once on the whole image and never per tile, and never use UMAP or t-SNE, which
   do not preserve distances. Colours from nonlinear embeddings are the Refik Anadol trap:
   pretty, and meaningless as distances.

A lesson carried over from [CRITIQUE.md](../CRITIQUE.md): linear mixing lands in mid-tones,
and mid-tones are where value structure dies. Every proposal below therefore gives
**lightness to one declared scalar**, usually confidence (max p, 1 − entropy/log K, or a
resultant length). Hue and chroma come from the vector, and the image can commit to a dark
key where the network is unsure.

---

## 1. The candidates

Nineteen ideas, including the twelve seeds. **Novelty** is my verdict after searching the web
(September 2026). "Done" means I found the same image. "Adjacent" means the measurement exists
but I found no dense, vector-coloured rendering of it. Compute is for the one GB10.

| # | Idea | Channels per pixel | Density (pixels of real data) | Novelty verdict / prior art | Compute (GB10) | Risk |
|---|---|---|---|---|---|---|
| 1 | **Three Minds**: a weight-space plane through 3 independently trained small nets, each pixel coloured by *what the net computes* there (agreement with each vertex net, or PCA of logits on a probe set) | 3 exact (agreement with A, B, C); or 10⁴ logits → PCA3 | 1024² weight points | **Adjacent.** Loss on 3-solution planes: [Garipov 2018](https://arxiv.org/abs/1802.10026), [Li 2018](https://arxiv.org/pdf/1712.09913), [Ideami's Loss Landscape art](https://losslandscape.com/gallery/). Function similarity *along paths*, and t-SNE of predictions: [Fort, Hu & Lakshminarayanan 2019](https://arxiv.org/abs/1912.02757). No per-pixel function-coloured plane found. | MLP or small CNN, 1024² × 1k probes ≈ 10¹⁵ FLOP; the first layer is linear in (α, β), so it is precomputed. **~30–60 min** | Low. Worst case the centre is flat and the colour lives only at the edges. |
| 2 | **Fold stack**: a plane through 3 training images, with *every layer's* representation coloured, plus activation-region hairlines weighted by Hamming distance | PCA3 of 64–512-d features per layer; final layer is a 3-class simplex | 1024² × ~10 layers | **Largely done for the final layer.** Decision regions on a 3-image plane: [Somepalli 2022](https://arxiv.org/abs/2203.08124). Exact regions: [SplineCam](https://arxiv.org/abs/2302.12828). Region migration in grokking: [Humayun 2024](https://arxiv.org/abs/2402.15555). Region counts: Hanin & Rolnick 2019. The per-layer colour stack and Hamming-weighted edges appear new. | ResNet-18/CIFAR over 1M images ≈ 5·10¹⁴ FLOP, with PCA on the fly. **~15 min** per checkpoint | Medium. The most-published idea on the list. |
| 3 | **Warp and weft**: a residual-stream tapestry of every token × every layer of a long text, coloured by the logit-lens expected unembedding | 1024–5120-d → shared W_U PCA3; lightness = lens p(next token) | 4096 tokens × 28–40 layers per model | **Adjacent.** Logit-lens grids of about 10 tokens: [nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), [tuned lens](https://arxiv.org/abs/2303.08112), [LogitLens4LLMs](https://arxiv.org/pdf/2503.11667). PCA scatter per layer: [ICLR 2026 blogpost](https://iclr-blogposts.github.io/2026/blog/2026/vis-llm-latent-geometry/). No dense, vocabulary-anchored colour weave found. | Seconds (0.6B) to minutes (14B) | Medium. Tokens are high-frequency, so it may read as static without seriation. |
| 4 | Attention of every head on one document, woven | 1 per cell, ×448 heads | — | **Done in this repo** ([attention-textile](../attention-textile/README.md), full 28×16 tapestry). Also BertViz. | — | Reject |
| 5 | **The Loom**: every training example × every checkpoint, coloured by the full predicted class distribution | 10-d softmax → linear OKLab mix; lightness = max p | 10k–60k rows × 500 checkpoints (5–30 M cells) | **Adjacent.** Per-example scalars: [Toneva 2019](https://arxiv.org/abs/1812.05159), [Dataset Cartography](https://arxiv.org/abs/2009.10795), [first-learned](../first-learned/README.md) (argmax only). No full-distribution weave found. | MNIST/CIFAR small nets, **~5 GPU-min per run**, about 600 MB of fp16 per run | Low |
| 6 | Toy-model superposition phase diagrams | a small Gram-matrix glyph per pixel | 64² glyphs | **Done**: [Elhage 2022](https://transformer-circuits.pub/2022/toy_model/index.html) phase diagrams and small multiples | minutes | Low novelty |
| 7 | **Illuminated manuscript**: SAE features per token, the colour a mixture of active features | k active features → decoder-direction PCA3 | about 10⁵ tokens of text | **Adjacent/done.** Single-feature token highlighting everywhere ([Neuronpedia](https://www.neuronpedia.org), [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)). No Qwen3 SAE is cached, so one would have to be trained. | Train an SAE on Qwen3-0.6B, **~4–8 GPU-h** | High: mixing many features gives mud |
| 8 | NCA / Lenia / CPPN weight interpolation | RGB native | — | **Done**: [Mordvintsev 2020](https://distill.pub/2020/growing-ca/), Chan 2019 (Lenia), Ha 2016 (CPPN); [palimpsest](../palimpsest/README.md) here | — | Reject |
| 9 | Diffusion x₀-predictions across timesteps and seeds | an image per cell | — | **Done**: DDPM Fig. 6 ([Ho 2020](https://arxiv.org/abs/2006.11239)); [diffusion-basins](../diffusion-basins/README.md) here | — | Reject |
| 10 | Token-embedding or unembedding matrices as images | 1 | 151k × 1024 | **Done** in several forms, including [block-quant](../../hardware/block-quant/README.md) here | — | Low value |
| 11 | NTK Gram matrix over a dataset through training | 1 per cell (the 10×10 output block is impractical) | 2048² | **Done**: [Fort 2020](https://arxiv.org/abs/2010.15110), [Seleznova 2023](https://arxiv.org/abs/2305.16427) (class blocks) | ~1 GPU-h | Scalar per cell, so it misses the brief |
| 12 | **Cayley tables in colour**: the grokking MLP hidden state over the whole (a, b) table, for Z_p and S₅ | 512-d → PCA3; in Z_p, PCs pair into cos/sin(ω(a+b)), so hue is a real phase | 113² … 120² per frame, a film over training | **Adjacent.** Single-neuron heatmaps: [Nanda 2023](https://arxiv.org/abs/2301.05217) Fig. 12. Irreps: [Chughtai 2023](https://arxiv.org/abs/2302.03025). No whole-state colour table found. | minutes | Medium: small frames, and it overlaps [grokking](../grokking/README.md) |
| 13 | **Leading Digit**: Qwen's multiplication table, the next-digit belief at "=" for every a×b with a, b < 1000 | 10-d digit distribution → hue on the log₁₀-mantissa circle; chroma = certainty | 1000² = 10⁶ prompts per model | **Not found as an image.** Measured in aggregate: first digit easy, last digit hard ([Gambardella 2024](https://arxiv.org/abs/2406.02356)). Per-neuron operand grids: [Nikankin 2025](https://arxiv.org/abs/2410.21272). Fourier features: [Zhou 2024](https://arxiv.org/abs/2406.03445), [Kantamneni & Tegmark 2025](https://arxiv.org/abs/2502.00873). | 8 M tokens of prefill: **~5 min (0.6B), ~1 h (8B), ~2–3 h (14B)** | Low: the ground truth is exact |
| 14 | **In-context borders**: Qwen classifies a 2-D grid of points in context, with 3 labels | 3 exact (P(A), P(B), P(C)) | 256²–512² queries | **Extension of published work**: [Zhao, Nguyen & Grover 2024](https://arxiv.org/abs/2406.11233) (binary, irregular boundaries) | Shared prefix cached, **minutes** | Medium |
| 15 | Dense associative memory: basins with 3 stored images, coloured by retrieval weights | 3 exact | 1024² | **Done in monochrome**: basin plots with white metastable states ([Hopfield–Fenchel–Young](https://arxiv.org/abs/2411.08590)); [Ramsauer 2020](https://arxiv.org/abs/2008.02217) | seconds | Low novelty; close to diffusion-basins |
| 16 | **Which Function**: the trainability fractal coloured by the *learned function* instead of by converged or not | 256 probe outputs → PCA3; diverged pixels go black | 1024²–4096² training runs | **Adjacent.** [Sohl-Dickstein 2024](https://arxiv.org/abs/2402.06184) colours by convergence and speed (scalar); [transformer follow-up](https://arxiv.org/abs/2501.04286) likewise; [outcome-basins](../outcome-basins/README.md) here is discrete toy solutions | Tiny nets × 500 steps, **~10–40 min** per 1024² | Medium: the interior could be flat |
| 17 | **Emission spectra**: each token's next-token distribution as a spectrum over a seriated vocabulary | 4096 vocabulary bins; hue = seriation coordinate of W_U | 4096 × N tokens | **Not found** | seconds–minutes | Medium–high: 1-D vocabulary order is lossy, and it echoes hessian-spectrum's idiom |
| 18 | MLP triplets: the (gate, up, downᵀ) weights of each neuron–dimension pair as one RGB pixel | 3 exact | 3072 × 1024 × 28 layers | **Not found as RGB triplets**; single weight heatmaps are common ([Distill, Visualizing Weights](https://distill.pub/2020/circuits/visualizing-weights/)) | seconds | High: likely reads as static apart from massive-activation stripes |
| 19 | All heads' attention patterns across training, for a small transformer we train ourselves | behaviour vector (prev, induction, sink, self, local) → mix | 32 heads × 2000 checkpoints | **Done as heatmaps**: [Olsson 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), [Tigges 2024](https://arxiv.org/abs/2407.10827); partly in attention-textile | ~1 GPU-h | Low novelty |

Not tabled because they are weak for this brief: RL policy maps (action distributions over
state slices are standard in RL papers), the pareidolia field (a CIFAR classifier slid over a
large photograph: pretty, but no finding), and per-example memorisation in LMs (a scalar).

---

## 2. Ranked top six

Ranked by (finding × density × faithfulness of colour × novelty) ÷ risk, with small networks
favoured when the other factors tie.

### 1. *Leading Digit* — the multiplication table as Qwen believes it (#13)

**Quantity.** For every a, b ∈ [0, 1000), the prompt `a*b=` is given to Qwen3. Qwen splits
numbers into single digits (checked: `487*923=` is 8 tokens), so the next-token distribution
over the tokens 0–9 is the model's belief about the **leading digit** of the product. Place
digit d at hue angle 2π·log₁₀(d). The hue circle is then the circle of the log-mantissa, which
is a real cyclic quantity and Benford's circle. The pixel's colour is the circular mean of the
belief. Chroma is the resultant length (a confident pixel is saturated, a split one is grey),
and lightness is max p. A second plate forces the correct prefix and reads the belief about
the **last digit**. Its hue goes on the 10-circle of residues, since the last digit is
(a mod 10)(b mod 10) mod 10.

**Why it is dense and why colour is earned.** The ground truth is a pair of beautiful exact
fields. The leading digit follows the nested hyperbolas a·b = d·10ᵏ: a smooth rainbow of
log-spaced hyperbolic bands. The last digit is a 10×10 plaid repeated ten thousand times. So
magnitude is hyperbolic and residue is periodic, and the model is known, in aggregate, to get
the first right and the second wrong ([Gambardella 2024](https://arxiv.org/abs/2406.02356)).
The mechanism literature splits the same way: MLPs carry magnitude on low-frequency features and
attention carries residues on high-frequency ones ([Zhou 2024](https://arxiv.org/abs/2406.03445)).
The diptych therefore draws that division of labour.
The image shows *where* each succeeds: blur and grey fringes along the hyperbolas where carries
decide the digit, and a plaid that holds for small operands and melts into noise for large
ones. The scale ladder 0.6B → 1.7B → 4B → 8B → 14B shows the table coming into focus.
Tokenisation is a free second axis: OLMo-2 chunks numbers into three digits (`487`, `*`, `923`),
so its first answer token is a 1000-way belief, and the hue can carry the mantissa almost
continuously.

**Null.** The exact table, rendered with the same mapping, hung beside it. Also a
"Benford-only" baseline that ignores the operands, and a shuffled-operand prompt.

**The image.** *At 10 m:* a square of concentric rainbow hyperbolas sweeping from the lower
left, like moiré or interference fringes, sharp in the bottom-left corner and softening toward
the top right, where the products grow long. *At 30 cm:* each fringe edge is a ragged, greyed
seam where the model hesitates between two digits. The seams are ragged in steps of 10 and
100, following the operands' own digits, and a few islands have the wrong hue: confident
mistakes. The last-digit companion is a fine tartan that is crisp in one corner and dissolves
into coloured noise.

**Compute.** 8 M tokens of prefill per model: minutes for 0.6B, about 1 h for 8B, 2–3 h for 14B.
It can be run at 512² first.

### 2. *Three Minds* — the function landscape of a weight plane (#1)

**Quantity.** Train three small nets from different seeds, for example an MNIST MLP and a
CIFAR-10 small CNN. Take the plane through their three weight vectors (Garipov's construction).
At each of 1024² points, evaluate the net on a probe set chosen as the examples on which A, B
and C disagree. The pixel is **(agreement with A, with B, with C)**: three numbers that land
exactly on the RGB primaries at the vertices. Lightness is the accuracy at that point.
Variant: PCA3 of the full logit vector over 1k probes (report the variance explained).

**Why it matters.** Every loss-landscape picture, including our own
[loss-landscape](../loss-landscape/README.md) and [mode-connectivity](../mode-connectivity/README.md),
shows *how bad* a point is. None shows *what* the net computes there. On a function-coloured
plane a high-loss ridge could be a region computing garbage (dark), a region that computes a
fourth, different function (a new hue off the triangle), or a region that collapses to always
predicting one class (a flat plateau). These are three different claims, and the loss surface
cannot tell them apart.

**Nulls.** (a) The same plane through three *permuted copies* of one net. The vertices compute
identical functions, so they should all read white, and the interior shows what permutation
alone costs. (b) The same after weight matching ([Ainsworth 2023](https://arxiv.org/abs/2209.04836)),
which should turn into one country. (c) A random-direction plane around one net.

**The image.** *At 10 m:* three saturated islands (red, green, blue) at the corners of a
triangle on a dark ground, with luminous bridges where their functions blend and dark
barriers where they don't. *At 30 cm:* the rims of the barriers, where hue shifts before
lightness falls, so the net changes its mind before it gets worse. Far from the triangle, flat
single-colour plateaus appear where the net predicts one class everywhere.

**Compute.** The first layer is linear in (α, β), so it is three precomputed matmuls. The rest
is batched forward passes, about 30–60 min at 1024² for a 270k-parameter MLP.

### 3. *Warp and Weft* — the residual stream read through the unembedding (#3)

**Quantity.** Take one long, varied document (4096 tokens) and every layer ℓ and position t.
Compute the logit-lens distribution p_ℓ,t (after the final norm) and its expected output
embedding ē = E_p[W_U row]. Project ē onto the top three principal components of W_U itself.
Those axes are fixed by the model's vocabulary, not by the document, which is what makes the
colour faithful: two cells share a colour iff they are "about to say" similar words. Lightness
is p_ℓ,t(the actual next token). The whole Qwen3 family shares a tokenizer, so one reference
W_U basis colours 0.6B through 14B comparably. Layers are the warp and tokens are the weft.

**Why.** Logit-lens figures show about ten tokens as text. At full density a known structure
should become visible as colour: bottom layers carry the *current* token's colour, top layers
the *next* token's, and the switch-over layer moves with model size. A few tokens (proper
nouns, closing brackets, the second half of a repeated phrase) resolve early, as bright
columns that light up deep in the stack.

**Nulls.** A token-shuffled document, where the late-layer colour should become noise while
the early layers are unchanged. A randomly initialised Qwen3-0.6B.

**Mind the sink.** The first token and massive-activation dimensions dominate raw hidden
states (see [to-scale](../to-scale/README.md)). The lens colour avoids this because it goes
through the final norm and the vocabulary, which is a second reason to prefer it over raw PCA.

**The image.** *At 10 m:* a long horizontal cloth, 4096 threads wide and 28 bands tall, with
a lower half of busy, many-coloured weave (token identity) that settles upward into a
calmer, lighter upper half with bright vertical streaks. *At 30 cm:* individual threads, and
you can find the word "the" by its recurring colour in the bottom band.

**Compute.** Seconds to minutes.

### 4. *The Loom* — every example's belief, every step (#5)

**Quantity.** Train a small CNN on CIFAR-10 (and an MLP on MNIST as a quieter twin). Log the
full softmax on all 10k test (or 50k train) examples at 500 log-spaced checkpoints. Place the
ten class hues at equal OKLab lightness around the circle. The pixel is Σ pₖ cₖ, with lightness
max p, so uniform ignorance is dark grey and certainty is bright and saturated. Rows are
ordered by the learning time from [first-learned](../first-learned/README.md) and columns are
steps. The class-hue placement puts the most-confused pairs (cat/dog, 4/9) about 90° apart,
declared, so that their confusion reads as an intermediate hue rather than cancelling to grey.

**Why.** First-learned kept only the argmax. The distribution shows *what an example was
mistaken for* on its way. Mislabelled digits that are pulled toward their visual class before
memorisation drags them to the label should show up as a two-colour stripe. The
forgetting events of Toneva 2019 appear as flicker between two specific hues.

**Nulls.** A random-label run (the whole cloth should flicker, then turn to memorised hues
late). A second seed (same order, different flicker?).

**The image.** *At 10 m:* a vast darkness at the left edge that turns, along a diagonal front
from top to bottom, into ten interleaved bands of colour. Most of the right half is calm
saturated colour, and the bottom rows stay grey and flickering to the end. *At 30 cm:* single
rows are one digit's biography: violet, a flash of teal, violet again, then settling on
teal.

**Compute.** About 5 GPU-minutes per run, and roughly 600 MB of fp16.

### 5. *Which Function* — the trainability fractal, coloured by what was learned (#16)

**Quantity.** Sohl-Dickstein's setup: a one-hidden-layer tanh net, full-batch GD on 16 points,
with a grid over the two layers' learning rates (or init scale × lr). For converged pixels,
colour by PCA3 of the trained net's outputs on 256 probe inputs *off* the training set. For
diverged pixels use black, with lightness from divergence time. The projection is fitted on
all converged pixels.

**Why.** The published fractal answers "did it train". This asks "*what* did it learn", and
whether the edge of stability changes the answer. The implicit bias of large-step GD is
predicted to select flatter interpolants, so the function should drift in colour as one
approaches the boundary. If the fractal boundary also has fractal *colour* texture on the
convergent side, that is a new measurement.

**Null.** The same grid with a tiny learning-rate range, where gradient flow gives one function
and should give one flat colour. Also a shuffled-probe control for the PCA.

**The image.** *At 10 m:* a black diverging field with a coastline, and on the convergent side
colour grading smoothly toward the coast. *At 30 cm:* the coast's filigree, each tendril of
convergence tinted by the solution it reached.

**Risk.** If the interior is flat, the image falls back to the existing piece. A 128² pilot
settles this in a minute.

### 6. *Fold Stack* — one input plane, every layer (#2)

**Quantity.** Train a CIFAR ResNet-18 (and a 4-layer MLP) on a 3-class CIFAR subset
(airplane, cat, frog), so the final layer's softmax is an exact colour. Take the plane through
three training images, one per class, at 1024². For each layer, PCA3 of that layer's features
is fitted on the whole plane and mapped to OKLab. The final plate is the barycentric softmax.
Across every pixel edge, draw a hairline whose darkness is the **Hamming distance** between the
two activation patterns: how many ReLUs flip there. That is a faithful weight, unlike the
arbitrary hash colours of region plots.

**Why.** Somepalli shows the last plate and SplineCam shows the partition. The stack shows the
*transformation*: a nearly linear colour gradient at layer 1 (the input plane is almost
affine), hardening through the middle layers into flat stained-glass class cells whose
hairlines crowd toward the decision boundary. That crowding is the region migration of
[Humayun 2024](https://arxiv.org/abs/2402.15555), which a film over training can show.

**Null.** A randomly initialised net of the same architecture: all plates smooth, and region
density uniform.

**The image.** *At 10 m:* a row of ten square plates going from a soft tricolour gradient to
bold stained glass. *At 30 cm:* the hairline net, fine and dense near the boundaries and sparse
inside the cells.

**Compute.** About 15 min per checkpoint. It is candidly the least novel of the six. It stays
in because it is the purest small-net piece and the easiest to verify.

**Next in line.** *In-context borders* (#14) is the three-label, colour-exact extension of
Zhao 2024. Expect boundaries that snap to decimal digit changes, because Qwen reads coordinates
one digit at a time. *Cayley tables* (#12) are clean, but their frames are small and the idea
overlaps the grokking piece.

---

## 3. Cross-cutting notes for whoever builds these

- **Choose K = 3 where you can.** Three reference nets, three classes, three labels. Exact
  colour is the collection's strongest defence against the "it's just a colormap" critique.
- **Lightness from one declared scalar.** This is what lets these images commit to a key.
- **One projection per image.** Fit PCA or the reference basis once. Per-tile or per-frame
  refits make colour drift that looks like data.
- **Report what the colour discards.** Give the variance explained for every PCA3, and the
  chroma lost for every K > 3 mix, in the label.
- **Hang the null beside the piece.** Items 13 and 1 have exact nulls (the true table; the
  permuted copies), which make the strongest diptychs in the list.
