# Fractals from Real ML Systems — Project Directions

*Companion to the main ML-art directions doc. Same rule applies: every visual decision is either a faithful rendering of a measured quantity or a declared aesthetic choice, nothing in between.*

---

## 0. The recipe

Every classical fractal image is built from the same three ingredients:

1. **An iterated map.** Something you apply over and over: $z \mapsto z^2 + c$, Newton's method, a gradient descent step.
2. **A discrete outcome.** Bounded or divergent. Which root you converged to. Which class you produced.
3. **A 2D slice of the space of things you can vary.** In Mandelbrot's case, the parameter $c$. The image *is* that slice, colored by the outcome.

The structure appears at the boundary between outcomes, because near that boundary the iteration is maximally sensitive — arbitrarily small changes flip the result, at every scale.

Machine learning has iterated maps everywhere. What's been explored is a narrow corner of the recipe: **vary hyperparameters, ask converge-or-diverge.** That's Sohl-Dickstein (2024) and its follow-ups. Every other choice of ingredient 2 and ingredient 3 is open ground.

> **The design move for this whole project: keep the iterated map, change the question you ask about it, or change which space you sweep.**

Two things worth stating up front, because they govern everything below:

- **Cost per pixel is a full run of something.** A $1024^2$ image is a million training runs, or a million sampler calls. Every project here lives or dies on making the per-pixel unit tiny. Design for a toy version first, at $128^2$, and only scale what shows structure.
- **"It looks intricate" is not "it is fractal."** There is a verification protocol in §8 and it is not optional. Zoom sequences, JPEG artifacts, and aliasing all manufacture convincing fake self-similarity.

---

# Part I — Established, with images already made

These are real. The catch is that the pretty pictures partly exist already, so your contribution has to be the *object*, not the discovery.

## 1. *Trainability* — the Mandelbrot set of your own training loop

### The phenomenon
Sohl-Dickstein (2024) points out the structural analogy directly: Mandelbrot and Julia sets come from iterating a function and asking whether the series diverges or stays bounded; neural network training iterates an update function and can likewise converge or diverge with extreme sensitivity to hyperparameters. Sweeping a 2D grid of hyperparameters and coloring by convergence, he finds the boundary is fractal over more than ten decades of scale, across every configuration tested — tanh and ReLU, full-batch and minibatch. A 2025 follow-up extends the finding to decoder-only transformers trained with Adam, sweeping learning rates for the fully-connected and attention paths.

### Honest positioning
He published the images, the paper, and a colab. If you make a screen-resolution zoom video you are making a worse version of something that already has 16,000 likes on Twitter. The contribution has to be elsewhere:

- **Medium.** A 1-metre riso or archival print of a region nobody has looked at, where the *print* is the artifact. A deep-zoom sequence as a bound book, one plate per decade of scale, captioned with the actual $(\eta_1, \eta_2)$ coordinates and the zoom factor. Ten decades is ten plates — a natural edition.
- **Axis choice.** Sweep axes with semantic weight rather than two learning rates: (weight decay, data fraction) for a grokking setup; (learning rate, batch size); ($\sigma_w^2$, learning rate) to link the initialization phase diagram to the training one.
- **Same-region diptychs.** The identical hyperparameter window under two architectures. If the fractal differs, that difference *is* the architecture's fingerprint.

### How to compute it
One hidden layer, ~16 units, MSE loss, dataset size equal to the parameter count. Converged if the mean of the last several losses falls below a fixed threshold relative to the start. Per-pixel cost is a few hundred GD steps on a tiny net, which vectorizes across the whole grid — this is one big batched `vmap` on your GB10, not a million serial runs.

---

## 2. *Cascade* — the bifurcation diagram of gradient descent

### The phenomenon
This is the fractal hiding inside project #1 of the main doc. Past the edge of stability, raising $\eta$ further doesn't produce chaos immediately — it produces a cascade. For deep linear networks, loss oscillations beyond EoS follow a *period-doubling route to chaos*, visible in both the singular values and the Hessian eigenvalues. Even for a product of four scalars, the boundary separating converging from diverging initializations shows fractal structure, and trajectories starting near it oscillate chaotically before settling onto the two-step orbit.

### Why it's beautiful
The bifurcation diagram — $\eta$ horizontal, the attractor's visited values vertical — is the Feigenbaum object. Fig trees, period-doubling, chaotic bands with clean periodic windows inside them. It is one of the most recognizable images in 20th-century science, and here it is being produced by a neural network rather than a population model.

### The piece
A single very wide print. $\eta$ from stable through the first bifurcation, through the cascade, into the chaotic bands. Plot every visited loss value after a burn-in, one dot per iterate, at enough resolution that the periodic windows inside the chaos are visible. Hairline vertical rule at $2/\lambda_{\max}$ — the edge of stability is the *first* bifurcation, and labeling it that way is the entire educational payload.

**The measurement that makes it art rather than illustration:** compute the ratio of successive bifurcation intervals, $\delta_n = (\eta_n - \eta_{n-1})/(\eta_{n+1} - \eta_n)$. If it approaches $\delta \approx 4.669$, your network is in the same universality class as the logistic map, and the caption can say so. **Measure it, don't assume it** — Feigenbaum universality is a theorem about one-dimensional unimodal maps, and whether the effective map for your network reduces to that class is an empirical question you are answering, not a fact you are illustrating. A measured $\delta$ that *isn't* 4.669 is an equally good piece with a more interesting caption.

### How to compute it
Deep linear network on a matrix factorization loss is the cleanest setting — the cascade is documented there. Or the minimalist product-of-scalars objective. Per-$\eta$ cost is thousands of cheap iterations. Sweep $10^4$ values of $\eta$.

---

## 3. *Finite Width* — the fractal frontier of information propagation

### The phenomenon
Project #6 in the main doc, zoomed in. Mean-field theory says the $(\sigma_w^2, \sigma_b^2)$ plane has a clean critical line separating ordered from chaotic signal propagation. Recent work accounting for finite width reports that the frontier between these propagation regimes is instead fractal, with a fractal dimension estimated by box counting across several zoom levels, and the same holds for the backpropagation version.

### Why it's beautiful
It is a phase diagram that dissolves when you look closely. The clean physics picture is the infinite-width idealization; the real, finite object has structure all the way down. That's a genuine and rather moving statement about the relationship between theory and instance.

### The piece
Four panels, increasing magnification, each labeled with its zoom factor and its measured box-counting dimension. The first panel is the textbook phase diagram. The fourth is unrecognizable. Print large enough that a viewer walking closer replicates the zoom.

### Pitfall
This is a 2025 result. Reproduce it yourself at your own widths before committing to a print run, and state the width in the caption — the whole point is that the answer depends on it.

---

# Part II — Unexplored. Nobody has made these pictures.

Ranked by (novelty × plausibility × cheapness).

## 4. *Which Dog* — the latent basin map of a diffusion model

**The strongest idea in this document.**

### The setup
Deterministic sampling (DDIM, or the probability-flow ODE) makes a diffusion model an *iterated map* from a noise vector to an image. Fix everything except the starting noise. Take a 2D affine slice through latent space — pick $z_0$, two directions $u, v$, and evaluate the map on the grid $z_0 + \alpha u + \beta v$. Color each pixel by a **discrete** property of the output.

That is structurally a **Newton fractal**: Newton's method converges to one of several roots, and coloring the plane by *which root* produces the famous interlocking basins with fractal boundaries. A diffusion sampler contracts toward a small number of modes. Same shape of question.

### The discrete outcome, in order of cost
1. **Toy first.** Train a tiny diffusion model on a 2D Gaussian mixture with $k$ modes. The outcome is "which mode did this noise land in." Free to evaluate, runs in seconds, and it is the *exact* Newton-fractal analogue. If the boundaries here are smooth, you learn that immediately and cheaply.
2. **Class label.** Small class-conditional or unconditional model on CIFAR-scale; color by a classifier's verdict on the output.
3. **Memorization.** For a model trained on a small dataset enough to memorize, color by nearest training image. The boundary between "this noise reproduces image #412" and "#1893" is a picture of memorization geometry.

### Why it might be fractal, and why it might not
*For:* contraction toward few attractors with sensitive dependence is the canonical fractal-basin setup, and diffusion samplers are famously sensitive to the seed. *Against:* the ODE is a smooth flow, so the map is smooth; the basin boundaries could be perfectly tame codimension-1 surfaces, and the apparent sensitivity may be low-dimensional rather than infinitely nested. **This is an open experimental question and that is what makes it worth doing.** A clean negative — smooth boundaries — is itself a true and publishable image.

### Cost control
Few sampling steps, low resolution, batched across the whole grid. Start at $128^2$ with 10 DDIM steps at $32\times32$.

---

## 5. *Not Whether, But Which* — basin maps of outcome identity

### The idea
Everything published colors initialization or hyperparameter space by a *binary*: converged or diverged. Replace that with the identity of the solution reached:

- final test accuracy (continuous, but banded — color by band)
- which feature a given neuron learned
- the sign pattern of the first-layer weights
- which permutation class the solution falls into, computed by weight-matching against a reference model (the Git Re-Basin machinery from #5 of the main doc becomes a *coloring function*)

### Why it's a different picture
Converge/diverge is a Mandelbrot question. "Which of several attractors" is a Julia/Newton question, and those images look completely different — interlocking, many-colored, with the delicate filigree where basins meet. Same compute as work already done; a strictly richer image; and it answers something better: not "does training work here" but "how finely interleaved are the outcomes."

### Pairs beautifully with
The permutation-matrix print from the main doc. Two panels: the basin map colored by raw outcome (chaos), and the same map colored *modulo permutation* (large calm regions, if the single-basin story holds). That contrast is the whole Git Re-Basin argument rendered as two images.

---

## 6. *Decode* — the hyperparameter map of a language model

### The idea
Fix the prompt and the random seed. Sweep (temperature, top-$p$), or (temperature, repetition penalty), on a grid. Color each pixel by a scalar of the generated text: repetition rate, distinct-$n$ ratio, output length, entropy, or the binary "did it terminate."

Autoregressive generation is an iterated map, and it has *genuine* discontinuities — infinitesimal changes in temperature flip which token gets sampled, and that flip propagates through everything after it. All three ingredients of §0 are present.

### The honest uncertainty
With a fixed seed, the map is piecewise-constant: flat cells separated by discontinuity curves. You might get a clean mosaic rather than something self-similar. **Whether the cells nest under zoom is the experiment.** My guess is that near-ties in the token distribution create a cascade of finer and finer cells, because a flip early in the sequence changes every subsequent distribution — that's a plausible mechanism for scale-free structure, but it is a guess, not a result.

### Why do it anyway
Nobody has made this map. It's cheap on your hardware with a small model. And it has an unusual property for an artwork: practitioners tune these two knobs by feel constantly, so a picture of what that space actually looks like has direct practical bite.

---

## 7. *Ouroboros* — self-consumption as an iterated function system

### The idea
Training a generative model on its own outputs, repeatedly, is an **iterated function system** on distribution space — the same class of object that generates the Sierpiński triangle and Barnsley fern. Sweep (fraction of real data retained, generation number) and color by a collapse metric: variance of the output distribution, coverage of the original modes, distance to the original distribution.

### Status
Model collapse under self-consumption is a documented phenomenon. Whether the *boundary* between collapse and stability has fractal structure is, as far as I know, untouched.

### Cost
The most expensive project here — each pixel is a chain of training runs. Do it on a 2D toy distribution or a tiny VAE/diffusion model where a full generation takes seconds, and accept a $128^2$ grid. The image is the map; the sculpture is the nested set of generations.

---

## 8. *Depth as the Dial* — roughness that grows with layers

### The phenomenon
Di Lillo et al. (2025) study the boundary volumes of excursion sets of random networks — the level sets $\{x : f(x) = c\}$ — as depth increases. For non-regular activations such as the Heaviside step function, these boundaries show fractal behavior, with Hausdorff dimension increasing monotonically with depth. For regular activations (ReLU, tanh, logistic), the expected boundary volume instead converges to zero, stays constant, or diverges exponentially, controlled by a single easily-computed spectral parameter.

*Hausdorff dimension, plainly:* a number measuring how much of the space a set fills. A smooth curve in the plane has dimension 1; a curve so crinkled that it starts behaving like an area has dimension between 1 and 2. Growing with depth means the network's level sets get progressively crinklier.

### The piece
A grid of tiles. Columns are activation function; rows are depth. Each tile is the level set of a randomly initialized network over a 2D input slice, rendered as a single-ink line drawing. Reading down the Heaviside column, the lines get progressively wilder; the ReLU column stays orderly. The measured dimension is printed under each tile.

### Why it's a good object
It's the only project here where the fractal lives in **input space** rather than parameter space — so the image is a picture of the *function*, not of the training process. Also the cheapest thing in this document: it requires no training at all, only random initialization and forward passes.

---

## 9. *Two Players* — chaos in learning against an opponent

### The phenomenon
Learning dynamics in simple games are chaotic — this is well established in the game-theory and evolutionary-dynamics literature, with the canonical case being replicator dynamics on rock-paper-scissors-like games. (*Replicator dynamics:* the continuous-time model where a strategy's share grows in proportion to how much better it does than average.)

Two agents doing gradient ascent against each other on a small game, with initialization space colored by limit behavior — converges to equilibrium, cycles, or wanders chaotically.

### Status
Chaos here is known; the fractal-basin rendering, and specifically framing it as a picture of *self-play*, is not something I've seen done. Trivially cheap — this is a handful of coupled ODEs, not a neural network.

### Caveat before committing
Verify the chaos result in your own simulation before writing a caption that asserts it. The literature is older and the exact conditions matter.

---

# Part III — Self-similarity without a fractal image

## 10. *Four Over d* — the slope as a dimension

### The idea
Fractals are defined by scale-invariance, and there is a *deep* scale-invariance in ML that produces no fractal picture at all: the neural scaling law.

Sharma & Kaplan (JMLR 2022) argue that if a network is effectively doing regression on a data manifold of intrinsic dimension $d$, then the loss scales as $L \propto N^{-\alpha}$ with $\alpha \approx 4/d$. They test this by independently measuring the intrinsic dimension and the exponent in a teacher/student setup where $d$ can be dialed.

So, under that theory: **the slope of a scaling law is a measurement of the dimension of the data.** And intrinsic dimension, measured by nearest-neighbor methods, is generally *not an integer*.

### The piece
Two measurements, side by side, for several datasets. Left: the log-log scaling law, a straight line, slope $\alpha$. Right: the intrinsic-dimension estimate, itself a log-log line whose slope is $d$. Caption gives $4/\alpha$ and $d$ as two numbers that should match.

No fractal imagery at all — and that restraint is the point. The piece is about a power law being the signature of a scale-free object, which is the actual mathematical content of "fractal."

### The honest caveat, which belongs *in* the piece
The fit is good in controlled teacher/student settings and looser for real models: for GPT-2, measured intrinsic dimensions across layers came out at $d \geq 90$, consistent with the *inequality* the theory requires but not achieving equality. Print both numbers and let them disagree. A diptych where the two panels almost-but-don't-quite agree is more honest and more interesting than one where you cherry-picked the dataset that matched.

---

# 11. Verification protocol — do not skip this

Intricate ≠ fractal. Before any of these gets printed:

1. **Box counting across decades.** Cover the boundary with boxes of side $\epsilon$, count occupied boxes $N(\epsilon)$, and fit $\log N$ against $\log(1/\epsilon)$. A straight line over several decades with non-integer slope is the claim. One decade is not evidence. Report the range of $\epsilon$ you fitted over.
2. **Resolution independence.** Recompute a region at 2× and 4× resolution. Real structure refines; aliasing artifacts change. If new detail stops appearing, you've hit your numerical precision floor, not the end of the fractal — say which.
3. **Precision floor.** Deep zooms in float32 die early. Know the zoom depth at which your step sizes stop being representable, and stop before it. Note it in the caption the way an astronomer notes seeing conditions.
4. **Never JPEG.** Compression invents self-similar-looking texture at block boundaries. PNG or raw throughout, lossless into the print pipeline.
5. **Continuous coloring.** Binary escape-time coloring creates hard bands that read as structure. The standard fix is a smoothed iteration count. Whichever you pick, it is a *declared aesthetic choice* and belongs in the caption.
6. **A null model.** Run the same pipeline on a boundary you know to be smooth — a quadratic's stability threshold, say. If your rendering makes *that* look fractal, your rendering is the fractal.

---

# 12. Media

Fractals are the one family in this whole project where **the screen might genuinely be the right medium**, because zoom is the content and only a moving image carries it. Ranked:

| Medium | Fits | Notes |
|---|---|---|
| Deep-zoom video loop | Trainability, finite-width frontier | The only medium that carries scale-invariance directly. Silent, slow, no captions until the end |
| Bound plate book | Any zoom sequence | One plate per decade, coordinates and zoom factor printed. Turns the zoom into a physical act |
| Large archival print | Basin maps, latent maps, bifurcation diagram | The bifurcation diagram in particular wants to be very wide and not very tall |
| Riso | Depth/activation tile grid | Limited inks suit line-only level-set drawings |
| Pen plotter | Poorly suited | Fractal boundaries are not vector-native; a plotter will either lie or run for a week |

On color: escape-time images are where the rainbow colormap does the most damage, because false banding reads as genuine period structure. Perceptually uniform maps or two spot inks. The classic plotter-era artists — Molnár, Mohr — got extraordinary range out of one ink and recursion, and are the right reference here rather than contemporary fractal-art conventions.

---

# 13. Sequencing

**Start with #8 (depth as the dial).** No training at all, only random init and forward passes; the result is already published so you're reproducing rather than gambling; and it produces a grid of line drawings, which is a complete object in one week.

**Then #4 (diffusion latent basins), toy version.** 2D Gaussian mixture, few modes, color by which mode. This is a day of work and it tells you whether the flagship version is worth pursuing. If those boundaries are smooth, you've saved yourself a month.

**Then one of #2 or #5** depending on which way the first two go — #2 if you want the guaranteed-beautiful published object (the cascade will be there), #5 if you want the unexplored one.

Rough budget: one week building a batched sweep harness that takes an arbitrary per-pixel function and a 2D parameter grid and returns an array. That single harness serves #1, #2, #4, #5, #6, #7. Everything after that is choosing the per-pixel function.

---

# 14. Blind spots

1. **Fractality may not be a deep-learning fact.** A 2024 paper argues that complex fractal trainability boundaries can arise from trivially non-convex objectives — i.e. the phenomenon may be generic to iterating anything near an instability, not something neural networks specially possess. If so, the honest caption is *"this is what iteration near an instability always looks like, and training is iteration near an instability"* — which is still a good caption, and a considerably more sophisticated one than "neural networks are secretly fractal." Read this before writing wall text.

2. **The 2D slice problem, again.** Every image here is a 2D slice of a high-dimensional space, and the same asymmetry from the loss-landscape doc applies: fractal structure in a slice is strong evidence of fractal structure overall; smoothness in a slice is weak evidence of anything. Don't let a smooth latent slice convince you a diffusion model's basins are tame globally.

3. **"Fractal" has a definition and it isn't "pretty and recursive."** Non-integer Hausdorff or box-counting dimension, or exact/statistical self-similarity across scale. Heavy-tailed weight spectra are *statistically* scale-free (power laws have no characteristic scale) but are not fractal sets. Use the word precisely or it costs you credibility with exactly the audience you want.

4. **Adjacent theory you may not have hit:** the dynamical-systems toolkit these projects implicitly use — Lyapunov exponents, riddled basins (where basins are so intertwined that every neighborhood of a point in one basin contains points of another), Feigenbaum universality, and the theory of when 1D reductions of high-dimensional maps are valid. A weekend with a dynamical-systems text would let you say much more precise things about your own images. Riddled basins in particular would be a spectacular thing to find in a training landscape and I don't believe anyone has looked.

5. **On the art side:** fractal art has a large, aesthetically conservative amateur tradition (Mandelbrot zooms, Apophysis flames) and it is easy to accidentally produce work that reads as belonging to it. The escape route is the same as everywhere else in this project — the *subject* is a real system, so caption it, site it, and choose a visual idiom (scientific plate, survey sheet, single-ink line drawing) that signals measurement rather than decoration.

---

# Core references

- Sohl-Dickstein, *The Boundary of Neural Network Trainability is Fractal*, 2024 — arXiv:2402.06184 (code and colab at github.com/Sohl-Dickstein/fractal)
- *Mapping the Edge of Chaos: Fractal-Like Boundaries in the Trainability of Decoder-Only Transformer Models*, 2025 — arXiv:2501.04286
- *Complex Fractal Trainability Boundary Can Arise from Trivial Non-Convexity*, 2024 — the deflationary counterpoint; find and read before writing captions
- *Revisiting Deep Information Propagation: Fractal Frontier and Finite-Size Effects*, 2025 — arXiv:2508.03222
- *Learning Dynamics of Deep Linear Networks Beyond the Edge of Stability*, 2025 — arXiv:2502.20531 (period-doubling route to chaos)
- Zhu et al., *Understanding Edge-of-Stability Training Dynamics with a Minimalist Example*, 2023 — arXiv:2210.03294 (fractal convergent/divergent initialization boundary)
- Liang & Montúfar, *Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region*, 2025 — arXiv:2509.25351
- Di Lillo, Marinucci, Salvi & Vigogna, *Fractal and Regular Geometry of Deep Neural Networks*, 2025 — arXiv:2504.06250
- Sharma & Kaplan, *A Neural Scaling Law from the Dimension of the Data Manifold*, JMLR 2022 — arXiv:2004.10802
- Ainsworth, Hayase & Srinivasa, *Git Re-Basin*, ICLR 2023 — arXiv:2209.04836 (for the permutation-modulo coloring in #5)
- Peitgen, Jürgens & Saupe, *Chaos and Fractals: New Frontiers of Science* — the standard reference for the dynamical-systems and box-counting machinery
