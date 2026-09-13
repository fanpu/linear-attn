You are the agent for **theory project #5: "Sample complexity of feature learning: parities and single-index models"**. Your directory is `/home/fzeng/ml/research/theory/05-parities-single-index/` (create it).

{{HEADER}}

## Project-specific guidance (suggestions, not a script — use your judgment)

**Framing for the reader.** Some functions are "hard to find, easy to represent." A 2-layer net can easily represent the parity of k hidden bits out of n. Finding them with SGD takes ~n^k steps. Loss is flat for ages while real progress happens invisibly underneath, then it suddenly drops. Single-index models make this quantitative. How long SGD is stuck depends on a single number, the information exponent, i.e. the first nonzero Hermite coefficient of the link function.

**Reproductions** (read the papers for exact setups and claims):
- **Barak, Edelman, Goel, Kakade, Malach & Zhang 2022, "Hidden progress in deep learning: SGD learns parities near the computational limit".**
  - (n,k)-sparse parity, n ∈ {20,30,40,50}, k ∈ {2,3,4}, 2-layer MLP, online SGD, ≥16 seeds vmapped.
  - Fit the exponent of steps-to-solve vs n.
  - Show their hidden-progress measures (e.g. weight mass / Fourier gap on the relevant coordinates) growing while loss is flat.
- **Ben Arous, Gheissari & Jagannath 2021.**
  - Single-index targets with link He_k, k* = 1…4, d = 32…512.
  - Fit the sample-complexity exponent in d and compare with k* − 1 (log factors at k* = 2; verify).
  - Use spherical online SGD on a single neuron and/or a two-layer net.

**Visual ideas** (pick, improve, or replace):
- **Hero animation:** the first-layer weight matrix (neurons × n input coordinates) of the parity net as a heatmap over training.
  - It starts as noise; the k relevant columns slowly brighten (the hidden progress), then crystallize.
  - Synchronized with a loss curve that sits flat and then falls off a cliff.
- **Sphere animation:** the overlap m = ⟨w, w*⟩ for single-index learning.
  - Many seeds of w on a sphere or on a 1D m-axis, jittering near the equator (m ~ 1/√d) for a long "search" phase, then escaping to the pole.
  - Very striking with many seeds as a rain of trajectories, each escaping at a different time.
- **Scaling plot:** log-log steps-to-solve vs n or d with theoretical slopes as guides; beautifully typeset.
- **Widget:** sliders for d and k*. Simulate the overlap ODE/SDE (dm/dt ∝ m^{k*−1} + noise) live in JS with many seeds, showing the escape-time distribution.
- **Widget:** parity playground. Scrub training time on precomputed runs and toggle between the "loss view" and the "hidden progress view".

**Build on (section 5):**
- **Question:** does reusing data beat the online information-exponent barrier? See Dandi et al. 2024, "The benefits of reusing batches for gradient descent in two-layer networks: breaking the curse of information and leap exponents"; verify title and claims.
- **Measurement:** empirical sample-complexity exponent in d for online SGD vs 2-pass vs many-pass at matched sample count.
- **Optional:** staircase functions (Abbe, Boix-Adsera & Misiakiewicz), where leap complexity predicts learning order. That gives a lovely staged-learning figure.

**Budget:** tiny models, lots of seeds. The main cost is long runs at large n, k, and d. Cap steps and extrapolate honestly. A few GPU-hours through the slot limiter.

Commit at checkpoints (parity scaling; single-index scaling; hero; draft; widgets; build-on; polish).

Finish with the final report described in the brief.
