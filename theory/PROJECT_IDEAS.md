# Theory: Project Ideas

This directory holds experiments that reproduce classical ML-theory results and then push past them. Every project should be small enough to run comfortably on a single GB10.

## Ground rules

- **Put a number or curve from the theory on every plot.** A reproduction counts only when a measured quantity lands on a predicted one: an exponent, a closed-form curve, or a limiting solution. "Looks qualitatively similar" doesn't count.
- **Keep a "reproduce" stage and a "build on" stage.** The reproduction calibrates the setup. The extension should ask a question whose answer we don't already know.
- **Write down the discrepancy.** Finite width, finite n, finite steps, and float32 all move results away from asymptotic predictions. Measuring *how far* and *in which direction* is often the most interesting result.
- **Projects are self-contained,** like `dayN/`: each subdirectory has its own scripts, results, and a short `README.md` with the prediction, the measurement, and the gap between them.

## Sizing for the GB10

- There is ~120 GiB of unified memory (shared CPU/GPU), but memory bandwidth is modest. The sweet spot is **many tiny models in parallel** (`torch.func.vmap` over seeds or hyperparameters), not one big model.
- Many of these projects need **float64** to compare cleanly against closed forms, which is cheap at these sizes.
- The machine is aarch64, so JAX and `neural-tangents` may be painful to install. Use `torch.func` (`jvp`, `vjp`, `vmap`) for empirical NTKs and per-sample quantities.
- Several projects (2, 3, and parts of 1) run fine on the 20 Grace CPU cores.

## Already covered elsewhere in the repo

`art/` has compute code for edge of stability, grokking, neural collapse, loss landscapes, mode connectivity, signal propagation, Hessian spectra, and weight spectra (see `art/PROGRESS.md`). This doc leaves those out on purpose. Their code is worth reusing, especially the HVP/Lanczos code in `art/hessian-spectrum/common.py` and the mean-field quadrature in `art/signal-propagation/sp_core.py`.

---

## 1. Double descent where the answer is exact

**Classical result:** For ridgeless least squares, test risk blows up at the interpolation threshold (p ≈ n) and comes back down in the overparameterized regime. For isotropic Gaussian features, the risk has a closed form (Hastie, Montanari, Rosset & Tibshirani 2019). Mei & Montanari (2019) give the precise asymptotics for random-features regression.

**Reproduce:**
- **Linear model:** n = 400, sweep γ = p/n from 0.1 to 10, 50 seeds per point. Overlay the Hastie et al. formula: risk = σ²γ/(1−γ) for γ < 1, and r²(1−1/γ) + σ²/(γ−1) for γ > 1. Error bars should shrink onto the curve as n grows.
- **Random features:** a ReLU RF model on a Gaussian single-index target. Overlay the Mei–Montanari asymptotic curve, or the simpler Gaussian-equivalent linear model (Hu & Lu 2022).
- **Optimal ridge removes the peak** (Nakkiran, Venkat, Kakade & Ma 2021). Verify this on both models.

**Build on:**
- **Trained networks:** reproduce model-wise and epoch-wise *deep* double descent (Nakkiran et al. 2019) at small scale, using a 4-layer CNN with width sweep k = 1…64 on a 10k CIFAR-10 subset with 15–20% label noise. Test whether their "effective model complexity" threshold predicts the peak location *quantitatively*, not just roughly.
- **Anisotropic covariance:** with power-law feature covariance or a misspecified target, where does the peak move, and does the RF formula still track trained two-layer nets once the first layer is allowed to learn?

**Scale:** The linear and RF parts take minutes. The CNN sweep is a few hours for ~8 widths × 2 noise levels.

---

## 2. Implicit bias: which minimum does the optimizer pick?

**Classical results:**
- **Separable logistic regression:** gradient descent converges *in direction* to the L2 max-margin SVM, but only at rate O(1/log t) (Soudry, Hoffer, Nacson, Gunasekar & Srebro 2018).
- **Diagonal linear networks** w = u⊙u − v⊙v: the initialization scale α interpolates between the min-L2 solution (α → ∞, kernel regime) and the min-L1 solution (α → 0, rich regime) (Woodworth et al. 2020).
- **Deep matrix factorization:** biased toward low rank, and more so with depth (Arora, Cohen, Hu & Luo 2019). The bias is *not* captured by any norm (Razin & Cohen 2020).
- **Steepest descent** with respect to a norm converges to the max-margin solution *under that norm*, e.g. sign GD → L∞ margin (Gunasekar, Lee, Soudry & Srebro 2018).

**Reproduce:**
- **Logistic GD** on separable data (d = 50, n = 40): plot the angle to the SVM direction against log t out to 10⁶+ steps in float64. Confirm the 1/log t crawl, and that normalized GD is much faster.
- **Sparse regression** with a diagonal linear net (d = 100, n = 40, 5-sparse truth): sweep α over six decades. Compare the solution to min-L2, to basis pursuit (an LP), and to Woodworth et al.'s closed-form potential Q_α.
- **Matrix completion** at depths 1/2/3: track effective rank, and reproduce the Razin–Cohen example where nuclear norm diverges while rank goes down.

**Build on:** Implicit bias of *modern* optimizers.
- **Adam:** does it reach the L∞ max-margin solution on separable data as recent theory predicts, and on what timescale, with realistic ε, β₂, and a finite horizon?
- **Muon / spectral-norm descent:** for linear multiclass classifiers, is the limit the spectral-norm max-margin solution?
- **Two-layer homogeneous ReLU nets:** normalized margin should increase monotonically toward a KKT point (Lyu & Li 2020). Check this, and check which optimizer's margin geometry predicts test accuracy on a small real task.

**Scale:** CPU-friendly. Everything is tiny, but runs are long, so vectorize across seeds.

---

## 3. Stagewise learning and saddle-to-saddle dynamics

**Classical result:** Saxe, McClelland & Ganguli (2014) solved gradient-flow dynamics for deep *linear* networks from small initialization. Each singular mode of the input–output correlation matrix is learned along a sigmoid, with a transition time ∝ 1/sᵢ. Stronger modes are learned first, producing plateaus and sudden drops. Later work shows the network jumps between saddles of increasing rank, both in deep linear nets (Jacot et al. 2021) and in diagonal nets (Pesme & Flammarion 2023).

**Reproduce:**
- **Two-layer linear net, whitened inputs:** choose target singular values (e.g. 5, 2, 1, 0.5). Overlay the exact analytic trajectory of each mode strength on the measured one, in float64, from small balanced initialization.
- **Break the assumptions** one at a time: non-whitened inputs, unbalanced init, larger init, depth 3–4. Quantify how quickly the analytic solution stops predicting plateau lengths.

**Build on:** Does the same mechanism explain plateaus in *nonlinear* and *attention* models?
- Transformers reportedly learn through gradual rank increase of ΔW (Boix-Adsera et al. 2023). Train a small attention-only model on a low-rank linear-regression task and a toy copying task. Track the singular values of ΔW_QK and ΔW_OV over time.
- **The test:** do plateau durations scale as 1/(signal strength), as the linear theory predicts? Vary the target's singular values and fit the exponent.

**Scale:** Minutes on CPU for the linear parts. The attention toy takes ≤1 GPU-hour for a full sweep.

---

## 4. Lazy vs. rich training, and μP hyperparameter transfer

**Classical results:**
- **NTK regime:** as width → ∞ under NTK parameterization, the tangent kernel freezes and training is equivalent to kernel regression (Jacot, Gabriel & Hongler 2018; Lee et al. 2019).
- **Laziness from output scale:** scaling the output by α makes *any* model lazy as α → ∞ (Chizat, Oyallon & Bach 2019).
- **μP:** a different parameterization keeps feature learning alive at infinite width, so optimal hyperparameters transfer across width (Yang & Hu 2021; Yang et al. 2022, Tensor Programs V).

**Reproduce:**
- **Width sweep:** 2-layer MLP on a 1k-example binary CIFAR subset (downsampled so the NTK is computable), widths 64…16384.
  - Measure ‖θ_t − θ₀‖/‖θ₀‖ and ‖Θ_t − Θ₀‖_F/‖Θ₀‖_F against width. Both should scale as ~1/√m.
  - Compare network test predictions with those of the linearized model / kernel regression.
  - Build the empirical NTK from `torch.func` Jacobian-vector products rather than materializing Jacobians.
- **Chizat α sweep:** with the loss rescaled by 1/α², show the lazy → rich transition in kernel movement and in test error.

**Build on: μP transfer.**
- **Setup:** small GPT-style models (4 layers, widths 128–1024) on a TinyStories-scale corpus, sweeping learning rate under standard parameterization and under μP.
- **Reproduce first:** the Tensor Programs V picture, where the SP optimum drifts with width and the μP optimum stays put.
- **Then extend:**
  - Does transfer also hold across *depth* with depth-μP-style residual scaling?
  - Does it hold across *token budget*?
  - Per layer, measure how much features actually move (‖ΔW‖_op/‖W‖_op) as width grows. Does "rich" stay rich?

**Scale:** The NTK part takes an hour or two. The μP sweep (~64 short runs) is one overnight job.

---

## 5. Sample complexity of feature learning: parities and single-index models

**Classical results:**
- **Sparse parities:** SGD on neural nets learns (n, k)-sparse parities in roughly n^O(k) steps, close to the statistical-query lower bound. Progress is *hidden*: loss stays flat while the Fourier weight on the correct subset grows steadily (Barak, Edelman, Goel, Kakade, Malach & Zhang 2022).
- **Single-index models** y = σ(⟨w*, x⟩): online SGD needs ~d^(k*−1) samples when the information exponent k* ≥ 3, ~d log d when k* = 2, and ~d when k* = 1 (Ben Arous, Gheissari & Jagannath 2021).

**Reproduce:**
- **Sparse parity:** n ∈ {20, 30, 40, 50}, k ∈ {2, 3, 4}, a 2-layer MLP trained by online SGD on fresh batches, ≥16 seeds per cell (vmapped).
  - Fit the exponent of steps-to-solve against n.
  - Plot the hidden-progress measure against the flat loss curve.
- **Single index:** Hermite targets He_k with k = 1…4 and d = 32…512. Fit the sample-complexity exponent in d, and compare it with k* − 1.

**Build on:** The online-SGD bound is not the whole story.
- **Batch reuse:** reusing data (multi-pass SGD) reportedly beats the information-exponent barrier (Dandi et al. 2024).
- **The measurement:** the empirical exponent in d for online SGD vs. 2-pass vs. many-pass SGD, at matched sample count.
- **Staircase functions:** compare with staircase targets (Abbe, Boix-Adsera & Misiakiewicz), where leap complexity predicts which pieces are learned in what order.

**Scale:** Tiny models, lots of seeds. The main cost is the long runs at large d and k*. Budget a few GPU-hours and cap step counts.

---

## 6. A solvable scaling law: power-law random features

**Classical result:** Solvable models explain neural scaling laws (Bahri et al. 2021; Maloney, Roberts & Sully 2022; Bordelon, Atanasov & Pehlevan 2024). Paquette, Paquette, Xiao & Pennington (2024, "4+3 phases") solve one-pass SGD on a *power-law random features* model:
- data covariance eigenvalues ∝ j^(−2α),
- target coefficients ∝ j^(−β),
- a random projection to d features.

They derive the loss as a function of model size d and steps t, a phase diagram in (α, β), and the compute-optimal exponent d*(compute) in each phase.

**Reproduce:**
- Implement PLRF with a large ambient dimension (v ≈ 10⁴–10⁵; the d × v projection easily fits in memory).
- Choose one (α, β) point in each of 3–4 phases, sweep d and t, and build the IsoFLOP frontier.
- Check the fitted compute-optimal exponents against the paper's predictions. Pay attention to near-phase-boundary behavior, where finite-size effects should be worst.

**Build on:** Can PLRF predict a *real* small model's scaling exponent?
- Estimate α and β from a real dataset: the eigenspectrum of input features or frozen embeddings, and the target's projection onto those eigenvectors.
- Train small MLPs or transformers across sizes on that data.
- Compare the measured compute-optimal exponent with the PLRF prediction for the estimated (α, β). When it fails, identify which assumption broke (feature learning, multiple epochs, or non-Gaussian data).

**Scale:** PLRF sweeps take 1–3 GPU-hours. The real-model comparison needs a modest overnight sweep.

---

## 7. Non-vacuous generalization bounds and the "rethinking generalization" puzzle

**Classical results:**
- **Random labels:** deep nets can fit random labels, so uniform-convergence bounds based on capacity alone can't explain generalization (Zhang, Bengio, Hardt, Recht & Vinyals 2017).
- **A non-vacuous bound:** Dziugaite & Roy (2017) optimized a PAC-Bayes bound over a Gaussian posterior around SGD's solution. On binary MNIST with a 1-hidden-layer MLP, they got a *non-vacuous* bound (≈0.16–0.2 vs. ~2% test error), and the bound turns vacuous with random labels.

**Reproduce:**
- **Random labels:** a small CNN on a CIFAR-10 subset with label-corruption fraction swept 0 → 1. Plot time-to-memorize and test error against the corruption fraction.
- **PAC-Bayes bound:** binary MNIST (<5 vs ≥5), 1-hidden-layer MLP with 600 units. Center the prior at initialization, optimize the PAC-Bayes-kl bound over N(w, diag σ²), and report the bound with true and random labels.

**Build on:** Do generalization measures *predict* anything at small scale?
- **Sweep:** ~100 cheap configurations (width, depth, LR, batch size, weight decay, dropout, data size).
- **Measures:** ~10 per run, including spectral-norm products, path norm, margin-normalized norms, PAC-Bayes sharpness, flatness, and the optimized bound itself.
- **Evaluate as Jiang et al. (2020)** did (Kendall τ, conditional-independence tests) and **as Dziugaite et al. (2020)** did (worst-case across environments). Which measures survive the robust criterion?
- **Optional:** how much tighter do data-dependent priors make the bound (Pérez-Ortiz et al. 2021)?

**Scale:** Each run takes minutes. The full sweep with bound optimization is one overnight job.

---

## 8. In-context regression with linear attention (ties into the linear-attention sprint)

**Classical results:**
- **In-context learning:** transformers trained on sequences (x₁, y₁, …, x_N, y_N, x_q) with y = ⟨w, x⟩ and fresh w per sequence learn to regress in context (Garg, Tsipras, Liang & Valiant 2022).
- **Linear self-attention as GD:** one layer of linear self-attention can implement one step of preconditioned GD, and trained layers converge to it (von Oswald et al. 2023; Ahn et al. 2023). Zhang, Frei & Bartlett (2024) prove gradient flow converges to a closed-form global minimum.
- **Task diversity:** with M pretraining tasks, the model switches from a Bayes-optimal "memorize the task set" solution to ridge regression once M crosses a threshold (Raventós, Paul, Chen & Ganguli 2023).

**Reproduce:**
- **One-layer linear self-attention** with d = 20 and N = 40 in-context examples. Check that the learned key/query/value blocks match the Zhang–Frei–Bartlett closed form, and that in-context risk matches one-step preconditioned GD.
- **Multi-layer linear attention:** compare the risk at depth L with L-step preconditioned GD.
- **Task-diversity sweep:** sweep M over 2⁰…2¹⁶. Overlay the discrete-prior Bayes predictor and ridge, and locate the transition.

**Build on:** Which in-context *algorithm* do different recurrences learn?
- **Architectures:** softmax attention, linear attention, DeltaNet, and Gated DeltaNet at matched depth. They all use the kernels already installed via `flash-linear-attention`.
- **Why DeltaNet is interesting:** its state update S_t = S_{t−1} − β_t(S_{t−1}k_t − v_t)k_tᵀ is literally an online SGD step on in-context regression. Does it learn a *better* algorithm than GD, such as something closer to recursive least squares?
- **Baselines:** compare against ridge, GD-k, and RLS, and track how the gap changes with depth, noise level, and covariance shift at test time.

**Scale:** Small models (≤10M params). The full sweep takes a few GPU-hours.

---

## Suggested order

1. **Warm-ups with exact answers: #3 (Saxe dynamics) and #2 (implicit bias).** Each takes under a day, runs on CPU, and yields measured curves that sit directly on analytic ones. They also set up the float64 + vmap tooling the other projects reuse.
2. **#8 (in-context regression):** the most direct link to the linear-attention sprint, with a genuinely open "build on" question.
3. **#4 (μP transfer):** the most practically useful. It also produces a reusable small-transformer sweep harness for #6.
4. **#1, #5, #6, #7** as interest dictates. #6 and #7 have the most open-ended extensions.
