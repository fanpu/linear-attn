You are the agent for **theory project #4: "Lazy vs. rich training, and μP hyperparameter transfer"**. Your directory is `/home/fzeng/ml/research/theory/04-lazy-rich-mup/` (create it).

{{HEADER}}

## Project-specific guidance (suggestions, not a script — use your judgment)

**Framing for the reader:** a wide enough network trained normally barely moves its features and behaves like a fixed kernel machine (the NTK). Whether a network actually *learns features* depends on how you scale things with width. μP is the scaling that keeps feature learning alive, and as a bonus, the best learning rate stops depending on width.

**Reproductions** (read the papers for exact parameterizations):
- **NTK width scaling** (Jacot et al. 2018; Lee et al. 2019): 2-layer MLP on a small binary subset (e.g. 1k downsampled CIFAR or MNIST-PCA images), widths 64…16384.
  - Measure ‖θ_t−θ₀‖/‖θ₀‖ and ‖Θ_t−Θ₀‖_F/‖Θ₀‖_F against width; both should go like ~1/√m.
  - Compare the network's predictions with the linearized model / NTK kernel regression.
  - Compute the empirical NTK via `torch.func` JVP/VJP products; don't materialize full Jacobians.
- **Chizat, Oyallon & Bach 2019 α-scaling** (loss rescaled by 1/α²): the lazy → rich transition.

**Visual ideas** (pick, improve, or replace):
- **Hero animation:** a 2D-input toy (e.g. a spiral or two-circle task) with a two-layer ReLU net. Each hidden neuron is a particle (e.g. its weight direction × output magnitude, Chizat–Bach style). Rich regime: particles swarm into structure. Lazy regime: they barely jiggle. Show them side by side with the decision boundaries forming. This can be really beautiful.
- **Kernel movement:** a log-log plot vs width with theoretical slope guides.
- **Kernel matrix animation:** a Θ_t heatmap over training for narrow vs wide nets. Wide stays frozen; narrow reorganizes into block structure aligned with the labels (kernel alignment).
- **The canonical μP picture:** loss vs learning rate curves across widths, with SP optima drifting and μP optima stacked. Consider an animated build-up as widths are added.
- **Widget:** α / width slider scrubbing through precomputed particle-field animations and boundaries.
- **Widget:** μP explorer. Choose the parameterization (SP/μP), then hover widths to see LR landscapes and the location of the optimum.

**Build on — μP transfer (section 4):**
- **Setup:** small GPT-style models (e.g. 4 layers, widths 128–1024) on a small text corpus.
  - Check `~/.cache/huggingface` for datasets. A TinyStories-scale subset downloaded into `cache/` is fine (network works).
  - A char- or byte-level corpus is acceptable if tokenizers are a hassle.
- **Implement μP yourself.** Follow Tensor Programs V's table: init variances, per-layer Adam LRs, 1/d attention scaling, output multiplier. Verify with a "coordinate check" plot, itself a nice figure: activation scale vs width stays flat under μP and blows up under SP.
- **Reproduce first:** SP optimum drifting vs μP stable.
- **Then extend with at least one of:**
  - transfer across depth with depth-μP-style residual scaling (1/√L branch multipliers; find the papers)
  - transfer across token budget
  - per-layer feature movement ‖ΔW‖_op/‖W‖_op vs width: does "rich" stay rich?

**Budget:** the GPU is shared; keep the whole sweep to a few GPU-hours through the slot limiter. Use short runs, few LRs (e.g. 7 on a log grid), 2 parameterizations, 4 widths, and add seeds only where noise matters.

Commit at checkpoints: NTK scaling figures; particle hero; coordinate check; μP sweep; draft; widgets; polish.

Finish with the final report described in the brief.
