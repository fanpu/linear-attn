# Art from Real ML Phenomena — Project Directions

*Beautiful first, educational second. Every piece here is driven by data you compute yourself from a real training run or a real theoretical object — nothing simulated for looks.*

---

## 0. The governing constraint

The failure mode for "ML art" is almost always the same: the *phenomenon* is real, but the *image* is a matplotlib default with a prettier colormap. That's a diagram, not art. The second failure mode is the reverse: something visually arresting that has been so heavily post-processed that the data is decorative, and a viewer who looks closer learns nothing true.

So a working rule for this project:

> **Every visual decision must be either (a) a faithful rendering of a real quantity, or (b) an explicit, declared aesthetic choice.** Nothing in between. If a curve is smoothed, you know by how much and you say so.

This is the same discipline as documentary photography: you can choose the framing, the light, and the print, but you don't move the subject.

A second rule that makes things beautiful rather than merely correct: **choose phenomena whose mathematical structure is already a form.** A circle. A simplex. A phase boundary. A braid. Loss curves are not forms — they're line graphs. The projects below are ranked partly by how much intrinsic geometry they carry.

---

## 1. *Progressive Sharpening* — Edge of Stability as a braid

### The phenomenon
Full-batch gradient descent on a neural net does something classical optimization theory forbids. The sharpness — the largest eigenvalue $\lambda_{\max}$ of the loss Hessian — rises steadily during training (*progressive sharpening*), until it reaches $2/\eta$ where $\eta$ is the learning rate. Classical theory says that past this point GD on a quadratic diverges. Instead the sharpness hovers right at (and slightly above) that threshold for the rest of training, the loss oscillates up and down, and yet the loss still decreases over long timescales. This is the **Edge of Stability** (Cohen et al., ICLR 2021).

The reason it doesn't diverge is a self-stabilization feedback loop: when the iterates start to blow up along the sharpest direction, third-order terms in the loss push the trajectory back toward flatter regions (Damian et al., 2022). Cohen et al. (ICLR 2025) later showed the *time-averaged* trajectory follows a clean differential equation they call the **central flow** — gradient flow constrained to the region where sharpness stays below $2/\eta$.

### Why it's beautiful
There are two curves living on top of each other: a fast, violent oscillation along the top Hessian eigenvector, and a slow, smooth drift underneath it. That's a braid around a spine. Physically it's a wire humming at exactly its resonant frequency without snapping — and the reason it doesn't snap is that the humming itself keeps detuning the wire.

### The piece
A large-format print, dark ground. The horizontal axis is training step. Three layers:
1. A single hairline at height $2/\eta$ — dead straight, the only straight line in the image. This is the law.
2. The measured sharpness $\lambda_{\max}(t)$: rising smoothly, then crashing into the line and oscillating around it forever. Rendered as a thin, nervous stroke.
3. Underneath, the loss trajectory projected into the top-2 Hessian eigendirections, plotted as an actual 2D path — the braid. The oscillation amplitude encoded as stroke width or as a ribbon.

Variant: run the same net at four learning rates. Four panels. The straight line sits at a different height in each, and the trajectory finds it every time. That's the whole argument of the paper, told without a word.

Variant: a **pen plotter** version. The oscillation is genuinely at pen-stroke frequency, and a plotter drawing 40,000 steps of EoS oscillation is a performance piece as well as an object.

### How to compute it
Small net (2-layer MLP or a small CNN), small dataset (a 1,000-example CIFAR-10 subset), **full-batch** GD, MSE or CE loss, constant $\eta$. Sharpness via power iteration on Hessian-vector products (`torch.autograd.grad` twice — no explicit Hessian). Log $\lambda_{\max}$ every step or every few steps. Project $\theta_t$ onto the top-2 eigenvectors measured at some reference step. This is minutes of compute on your GB10; it's the logging, not the training, that costs.

### Pitfalls
- Use **full-batch** GD for the canonical picture. Mini-batch SGD lives in a related but distinct regime — recent work (Andreyev & Beneventano, 2024) argues that what stabilizes at $2/\eta$ under SGD is *batch sharpness* (expected directional curvature of mini-batch Hessians along their own gradients), with $\lambda_{\max}$ sitting lower.
- Adam/momentum have their own thresholds, not $2/\eta$ — Cohen et al. (2022) show each optimizer has its own edge of stability with a hyperparameter-dependent threshold on a *preconditioned* sharpness. Don't label an Adam run "$2/\eta$."

---

## 2. *Circuit Formation* — grokking as a crystallization

### The phenomenon
Train a one-layer transformer on modular addition $a + b \bmod 113$ with a restricted training fraction and weight decay. It memorizes fast, then sits at chance test accuracy for a long time, then abruptly generalizes. Nanda et al. (ICLR 2023) reverse-engineered what happens: the model maps each input integer onto a **rotation** — the embedding matrix represents each token as sines and cosines at a sparse set of key frequencies — and composes rotations via trigonometric identities to do the addition. Training splits into three phases: memorization, circuit formation, cleanup.

### Why it's beautiful
Because the learned structure is a *circle*. A cloud of 113 meaningless points, over training, arranges itself into a ring in the right 2D subspace, in the right order. There are very few cases in ML where a network's internal state converges on a shape a Greek geometer would recognize, and this is one.

### The piece
Two options, both strong:

**(a) The crystallization film.** Project the token embeddings onto the top-2 principal components of the relevant Fourier subspace, one frame per checkpoint. Points drift, then snap onto the ring. Each point labeled with its integer, so you can see the ordering emerge. Silent, slow, ~90 seconds.

**(b) The spectral print.** Take the DFT of the embedding matrix along the input dimension and take the $\ell_2$ norm along the other axis — the same measurement Nanda et al. use. You get a near-empty spectrum with a handful of spikes: the "key frequencies." Print this as a tall, sparse, almost-silent image — a spectrogram of a network's private music. Pair it with the pre-grokking version, which is dense noise. Diptych: *Before / After*.

### Educational payload
This is the cleanest available refutation of "neural nets are inscrutable." The caption can literally state the algorithm.

### Pitfalls
Grokking on modular addition is a specific, carefully constructed setting (restricted data + weight decay). Don't let the piece imply that every network groks, or that grokking is a general law. It's a phenomenon with known preconditions.

---

## 3. *Terminal Phase* — neural collapse as a simplex

### The phenomenon
Train a classifier past zero training error and keep going (the **terminal phase of training**). Papyan, Han & Donoho (PNAS 2020) found four coupled things happen: (NC1) within-class variability of last-layer features collapses to zero — every dog image maps to the same point; (NC2) the class means converge to the vertices of a **simplex equiangular tight frame** — a maximally spread-out, perfectly symmetric arrangement where every pair of class-mean directions has the same angle; (NC3) the classifier weights align with those same directions; (NC4) the classifier reduces to nearest-class-mean.

*Simplex ETF, plainly:* take $C$ vectors of equal length, centered at the origin, arranged so the angle between any two is identical and as large as possible. For $C=4$ in 3D, that's a regular tetrahedron.

### Why it's beautiful
A regular polytope, arrived at by accident. Nobody asked for a tetrahedron; SGD produced one. It's also physically buildable — the piece can leave the screen.

### The piece
**Sculpture.** Train on a 4-class subset so the ETF lives in 3D and is a tetrahedron. Then: a physical object — brass rods or fine wire — where the vertex positions are the *measured* class means at a sequence of checkpoints. Early checkpoints give a lopsided, wobbly frame; late checkpoints give something nearly regular. Cast five of them as a series: *Terminal Phase I–V*. The last one is almost, but not exactly, a Platonic solid. That "almost" is the honest part and the emotional part.

**Or:** point-cloud prints. Render the last-layer features of 10,000 test images at each of ~8 checkpoints, projected to 2D. A nebula condensing into ten hard points.

### Pitfalls
Neural collapse as described requires **balanced classes**; with imbalanced data you get *minority collapse* instead (Fang et al., PNAS 2021) — interesting, but a different piece. Also, it's a terminal-phase phenomenon: you have to train well past zero error.

---

## 4. *Filter-Normalized* — the landscape, done honestly

### The phenomenon
The famous images: a deep net without skip connections has a chaotic, crumpled loss surface; add skip connections and it becomes a smooth bowl. Li et al. (NeurIPS 2018) produced these using **filter normalization**: you pick two random directions in weight space, but rescale each filter's direction to match the norm of the corresponding filter in the trained weights. Without this, ReLU nets' scale invariance (you can double one layer and halve the next) makes apparent sharpness meaningless and cross-model comparison impossible.

### Why it's beautiful
Terrain. Humans have thousands of years of practice reading topographic form.

### The piece
Don't do another rainbow 3D surface plot — that's the most reproduced image in deep learning and adding one more contributes nothing. Instead, borrow a cartographic idiom:

- **Contour lines only**, in the style of a USGS or Ordnance Survey sheet. Hairlines, one ink, hand-set labels giving loss values instead of elevations. Titled like a map sheet: *ResNet-56 (no skip), sheet 1 of 4*.
- **CNC-milled or 3D-printed relief**, single material, raking light. Two objects side by side: with and without skips. This is the strongest version — it's an object people want to touch, and the argument lands in the hand.
- **Hachures** (the 19th-century technique of indicating slope with short strokes perpendicular to contours) — beautifully suited to a plotter.

### Pitfalls — and this is the important one
A 2D slice through a $10^7$-dimensional surface is a *slice*. Li et al. themselves note that 1D interpolation plots can be misleading because they ignore normalization and symmetry invariances. The useful asymmetry: if the slice is non-convex, the full landscape definitely is; if the slice looks smooth, the full landscape is *probably* locally smooth-ish — but that direction of the inference is much weaker. If you make this piece, the caption should say so. An artwork that teaches a false confidence about high-dimensional geometry is worse than no artwork.

---

## 5. *One Basin* — mode connectivity and permutation symmetry

### The phenomenon
Two networks trained from different random seeds end up at different points in weight space. Linearly interpolate between them and the loss spikes in the middle — a **barrier**. This looks like evidence of isolated minima. But Garipov et al. and Draxler et al. (both 2018) showed you can find simple *curved* paths between those minima along which loss stays essentially flat. And Entezari et al. (2021) conjectured, with Ainsworth et al.'s Git Re-Basin (2022) providing algorithms, that the barrier is largely an artifact of **permutation symmetry**: hidden units are interchangeable, so the two solutions are in "the same" basin, just labeled differently. Permute one to align with the other and the linear path often becomes nearly barrier-free.

Note the honest caveat from that same paper: linear mode connectivity is an *emergent property of training* — it doesn't hold at initialization.

### Why it's beautiful
It's a story about identity. Two things that look different are the same thing wearing a different arrangement of its own parts. That's a genuinely moving idea and it has an exact mathematical statement.

### The piece
A triptych, all three panels showing loss along a path from model A to model B:
1. **Naive linear path** — a mountain in the middle.
2. **Learned curve** (Garipov-style Bézier path) — a low, snaking valley.
3. **After permutation alignment** — the mountain has vanished from the straight line.

Rendered as cross-sections through terrain, like geological survey diagrams. The visual joke is that panel 3 is a straight, boring line, and that's the point: the drama in panel 1 was bookkeeping.

Companion piece: the permutation matrix itself. A $512 \times 512$ binary matrix as a hard black-and-white print — one dot per row. Pure noise to look at, and it's the entire difference between two "different" neural networks.

---

## 6. *Order and Chaos* — the initialization phase diagram

### The phenomenon
Take a randomly initialized deep MLP with weight variance $\sigma_w^2$ and bias variance $\sigma_b^2$. Mean-field analysis (Poole et al. 2016; Schoenholz et al. 2017) shows the $(\sigma_w^2, \sigma_b^2)$ plane splits into two phases. In the **ordered** phase, two different inputs become asymptotically identical as they propagate through layers — signal dies, gradients vanish. In the **chaotic** phase, nearby inputs are driven apart and folded — gradients explode. The critical line between them is the **edge of chaos**, and Schoenholz et al. showed the closer you initialize to it, the deeper a network you can actually train.

### Why it's beautiful
It's a real phase diagram, in the physics sense — the same kind of object as a water/ice/steam diagram. And the underlying quantity (correlation between two inputs' representations as a function of depth) is a flow you can draw.

### The piece
A large print of the $(\sigma_w^2, \sigma_b^2)$ plane. Each pixel is a genuine measurement: initialize a deep MLP with those hyperparameters, push two nearby inputs through, and color by the depth at which their correlation reaches its fixed point (the *depth scale*). The critical line appears as a bright ridge where that depth diverges. It emerges from the data — you never draw it.

**A caution worth knowing:** recent work (2025) reports that when you account for finite width, the boundary between these propagation regimes has *fractal* structure rather than being a clean curve. That's a gift for an artwork — zoom sequence, four panels, increasing magnification — but it's recent and worth reading before you commit a print run to it.

---

## 7. *Bulk and Outliers* — the Hessian spectrum

### The phenomenon
Histogram the eigenvalues of a trained network's Hessian and you don't get a formless mess. You get a dense **bulk** near zero, plus a small number of isolated **outliers** — and the number of outliers is often equal to the number of classes $C$ (Sagun et al., 2016/2017). Papyan (JMLR 2020) traced this to structure in the gradient class means, and found a further level: an additional "mini-bulk" of roughly $C(C-1)$ eigenvalues, giving a three-level hierarchy — bulk, mini-bulk, outliers.

### Why it's beautiful
It's a spectrum in the literal sense — an emission spectrum. The visual language of spectroscopy plates (dark ground, bright lines at specific positions) is already gorgeous and already understood by viewers as "a fingerprint of a substance."

### The piece
Trained-network Hessian spectra rendered as astronomical spectral plates, one per architecture/dataset. Log-scaled horizontal axis, bright lines for outliers, a diffuse band for the bulk. A series of ten, framed like plates from an observatory archive: *ResNet-18 / CIFAR-10*, *VGG-11 / MNIST*, etc. You can count the classes by counting the lines. That's the educational payload and it takes the viewer about thirty seconds to discover unaided, which is exactly the right amount of work.

Compute: stochastic Lanczos quadrature via Hessian-vector products (`pyhessian` or similar) — cheap enough for real nets on your hardware.

---

## 8. Two smaller ideas worth a sketch

- **The weight-matrix spectrum over training.** Martin & Mahoney's "heavy-tailed self-regularization" line of work tracks how the eigenvalue distribution of weight matrices departs from the random-matrix (Marchenko–Pastur) prediction as a network learns. An animation of a spectrum *growing a tail* is lovely, and it's a rare case where you can see learning as a departure from randomness. (Worth reading critically — the generalization claims attached to it are contested.)

- **Attention as textile.** Attention matrices are literally woven-looking. A jacquard weaving or risograph of induction-head attention patterns, at the training step where induction heads form, would be a genuinely novel object — and the phase transition in the loss curve at that moment gives you a narrative.

---

## Media, and why it matters

Your pieces will be stronger the further they get from a screen. Ranked roughly by how much the medium adds:

| Medium | Fits | Notes |
|---|---|---|
| Pen plotter (AxiDraw) | EoS trajectories, contour maps, hachures | Vector-native; the drawing time is proportional to the training time, which is conceptually nice |
| CNC relief / 3D print | Loss landscapes, ETF sculpture | Highest impact per piece; people touch it |
| Risograph | Spectral plates, permutation matrices, phase diagrams | Limited spot colors force real aesthetic decisions; misregistration is beautiful |
| Large archival inkjet | Phase diagrams, point clouds | Your photography printing knowledge transfers directly |
| Silent video loop | Grokking crystallization, spectrum evolution | Only medium that shows *time*, which is half of these phenomena |

On color: use perceptually uniform maps (viridis, magma, cividis) or restrict to one or two inks. The classic rainbow/jet colormap creates false visual boundaries where the data has none — it will make your phase diagram look like it has structure it doesn't. This is a correctness issue, not a taste issue.

---

## Suggested sequencing

**Start with #1 (Edge of Stability).** Reasons: the compute is trivial, the phenomenon is currently live research so the work feels contemporary, the visual has an intrinsic form (the braid against the straight line), and the story compresses to one sentence — *"it runs right along the edge of falling apart, and that's why it works."*

Then #3 (neural collapse sculpture) as the physical piece, and #2 (grokking film) as the time-based piece. Those three cover three media and three completely different phenomena — that's a coherent first show.

A realistic first pass:
- Week 1: instrumented training loop + logging (sharpness via power iteration, checkpoint dumps, eigenvector projections). One codebase serves projects 1, 3, 4, 7.
- Week 2: make the ugly version of all three. Confirm the data is real and the phenomenon actually shows up in *your* runs.
- Week 3: throw away every plotting default and design one piece properly.

The third week is the entire project. The first two are engineering you already know how to do.

---

## Blind spots worth checking before you commit

These are places where the popular ML-art version of a claim is subtly wrong, and where you'd be embarrassed to have a print on a wall asserting it:

1. **"Flat minima generalize better" is not settled.** Dinh et al. (2017) showed that for ReLU nets you can take a "flat" minimum and, using the scale invariance $(\alpha W_1, \alpha^{-1}W_2)$, produce an arbitrarily "sharp" minimum computing *the exact same function*. So sharpness as usually defined isn't reparameterization-invariant, and any piece titled something like "flatness is generalization" is asserting something contested. Filter normalization is a partial patch for exactly this, not a solution to it.

2. **Low-dimensional pictures of high-dimensional objects are lossy in a specific direction.** Non-convexity in a slice implies non-convexity overall; smoothness in a slice implies very little. Most loss-landscape art quietly asserts the wrong direction of that implication.

3. **"The" edge of stability is optimizer-specific.** $2/\eta$ is the full-batch GD statement. SGD, momentum, and Adam each have different thresholds on different (sometimes preconditioned) curvature quantities.

4. **Grokking is a setting, not a law.** It needs particular data fractions and regularization. Also there are competing explanations for *why* it happens (circuit formation and cleanup; efficiency of the generalizing circuit under weight decay) — the mechanism is not one agreed thing.

5. **Loss landscape ≠ optimization difficulty.** A lot of landscape imagery implicitly says "training is hard because the terrain is rough." But EoS says something almost opposite: the dynamics of the discrete optimizer, not the static terrain, determine where you end up. The optimizer's own instability is what selects the solution. These two framings sit in tension and most ML art uncritically adopts the first.

6. **Adjacent literatures you may not have hit yet**, each of which is a source of further pieces: random matrix theory in deep learning (Marchenko–Pastur baselines, Pennington & Worah); information geometry and the Fisher information metric (loss landscapes have a natural non-Euclidean geometry, and the "distances" in every Euclidean landscape plot are arguably the wrong distances); the NTK/feature-learning distinction (what changes during training vs. what's fixed at init); statistical-mechanics treatments of learning (Bahri et al.'s review is the entry point).

7. **On the art side:** look at Vera Molnár, Manfred Mohr, and the plotter-art lineage from the 1960s–70s before making plotter work — they solved a lot of the compositional problems already, and the field has a real history that ML-art often reinvents badly. On the scientific-illustration side, the 19th-century tradition of geological cross-sections and astronomical spectral plates is a richer visual vocabulary than anything in contemporary data viz.

---

## Core references

- Cohen et al., *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*, ICLR 2021 — arXiv:2103.00065
- Cohen et al., *Understanding Optimization in Deep Learning with Central Flows*, ICLR 2025 — arXiv:2410.24206
- Damian, Nichani & Lee, *Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability*, 2022
- Andreyev & Beneventano, *Edge of Stochastic Stability*, arXiv:2412.20553
- Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability*, ICLR 2023 — arXiv:2301.05217 (code + checkpoints at neelnanda.io/grokking-paper)
- Papyan, Han & Donoho, *Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training*, PNAS 2020 — arXiv:2008.08186
- Li et al., *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018 — arXiv:1712.09913 (code: github.com/tomgoldstein/loss-landscape)
- Garipov et al., *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs*, 2018 — arXiv:1802.10026
- Ainsworth, Hayase & Srinivasa, *Git Re-Basin*, ICLR 2023 — arXiv:2209.04836
- Poole et al., *Exponential Expressivity in Deep Neural Networks Through Transient Chaos*, NeurIPS 2016; Schoenholz et al., *Deep Information Propagation*, ICLR 2017
- Papyan, *Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra*, JMLR 2020 — arXiv:2008.11865
- Dinh et al., *Sharp Minima Can Generalize for Deep Nets*, ICML 2017 — arXiv:1703.04933
