# Topic landscape: linear attention and recurrent sequence models (as of 2026-09-15)

Every entry below was confirmed via web search or the arXiv abstract page (title + ID) on 2026-09-15. One-line claims paraphrase the abstract. Non-arXiv sources (blogs, GitHub) are marked as such.

## 1. Architecture families

### 1.1 Linear attention, RetNet, GLA (additive or decayed outer-product state)
- **Transformers are RNNs** — Katharopoulos, 2020, 2006.16236. Kernelized attention gives an O(N) recurrent form.
- **Linear Transformers Are Secretly Fast Weight Programmers** — Schlag, 2021, 2102.11174. Linear attention = fast-weight memory. Points out its capacity limit and proposes a delta-rule update.
- **RetNet** — 2023, 2307.08621. Fixed (data-independent) decay on the state.
- **Gated Linear Attention (GLA)** — 2023, 2312.06635. Data-dependent diagonal decay plus a hardware-efficient chunked kernel.
- **Based** ("Simple linear attention LMs balance the recall-throughput tradeoff") — Arora, 2024, 2402.18668. Recall vs state-size tradeoff.
- **Gated Slot Attention** — 2024, 2409.07146. **HGRN2** — 2024, 2404.07904 (state expansion).
- **Forgetting Transformer (FoX)** — 2025, 2503.02130. Adds a forget gate to softmax attention.
- **Wall Attention** (Tilde blog + fla, 2026; no arXiv ID found). Diagonal forget gates in softmax attention, aimed at length generalization.

### 1.2 Delta-rule family
- **DeltaNet** ("Parallelizing Linear Transformers with the Delta Rule over Sequence Length") — Yang, 2024, 2406.06484. WY/UT chunkwise-parallel training of the delta rule.
- **Gated DeltaNet** — Yang, 2024, 2412.06464. Scalar decay α plus delta rule; beats Mamba-2 and DeltaNet at 1.3B.
- **DeltaProduct** — Siems, 2025, 2502.10297. n_h Householder steps per token give a diagonal-plus-rank-n_h transition. Better state tracking and length extrapolation. Includes finite-precision theory.
- **RWKV-7 "Goose"** — Peng, 2025, 2503.14456. Generalized delta rule with vector gates. Claims it recognizes all regular languages.
- **Comba** — 2025, 2506.02475. Closed-loop-control variant of bilinear RNNs.
- **PaTH Attention** — 2025, 2505.16381. Accumulated Householder products used as position encoding.
- **Kimi Linear / KDA** — Kimi Team, 2025, 2510.26692. Channel-wise-gated delta rule with a DPLR chunk kernel. Claims a 3:1 KDA:MLA hybrid beats full MLA under a fair comparison.
- **Gated DeltaNet-2** — Hatamizadeh, 2026, 2605.22791. Separate channel-wise erase and write gates. Claims best at 1.3B vs Mamba-2 / GDN / KDA / Mamba-3.
- **Kaczmarz Linear Attention** — Zou, 2026, 2605.08587. Step size β = η/(‖k‖²+ε) derived from the Kaczmarz projection. Reports gains over GDN at 0.4B.
- **Preconditioned DeltaNet** — Tumma, 2026, 2604.21100. Diagonal curvature preconditioning of GDN and KDA.
- **OSDN** — 2026, 2605.13473. Online diagonal preconditioner learned by hypergradient, with a convergence bound.
- **MDN** (momentum delta rule) — 2026, 2605.05838. **Erase-then-Delta Attention** — 2026, 2606.26560.
- **Sparse Delta Memory** — Cabannes, 2026, 2607.07386. Sparse addressing for a much larger delta-rule state at iso-FLOP.

### 1.3 Mamba and other SSMs
- **Mamba** — Gu & Dao, 2023, 2312.00752. Selective (input-dependent) SSM.
- **Mamba-2** ("Transformers are SSMs") — Dao & Gu, 2024, 2405.21060. Structured state-space duality; scalar-decay matrix state.
- **Mamba-3** — Lahoti, 2026, 2603.15569 (ICLR 2026). Exponential-trapezoidal discretization, complex-valued transitions, MIMO. +0.6 pt over GDN at 1.5B (SISO).
- **Samba** — 2024, 2406.07522. Mamba + sliding-window attention hybrid.

### 1.4 Test-time-training / online-learning views
- **TTT** ("Learning to (Learn at Test Time)") — Sun, 2024, 2407.04620. The hidden state is a model updated by self-supervised SGD.
- **Longhorn** — Liu, 2024, 2407.14207. The SSM update is derived as an implicit online-regression step.
- **Titans** — Behrouz, 2024/25, 2501.00663. Deep neural memory updated at test time, with momentum and weight decay.
- **Test-time regression** — Wang, 2025, 2501.12352. Unifies linear attention, SSMs, fast-weight programmers, and softmax attention via three choices: regression weights, function class, optimizer.
- **Miras** ("It's All Connected") — Behrouz, 2025, 2504.13173. Framework of memory, attentional bias, retention, and optimizer.
- **Atlas** — Behrouz, 2025, 2505.23735. Optimizes memory over a window of past tokens (not just the last one); gives capacity bounds for deep memory.
- **MesaNet** — von Oswald, 2025, 2506.05233. Solves the in-context least squares to optimality per token with chunked conjugate gradient.
- **Gated KalmaNet** — Peng, 2025, 2511.21016 (CVPR 2026). Exact Kalman gain / online ridge. Shows GDN-type layers are KF approximations with identity error covariance.
- **Kalman Linear Attention** — Shaj, 2026, 2602.10743. Information-form Kalman filter as a parallel scan (Möbius precision recursion).
- **Palimpsa** ("Learning to Remember, Learn, and Forget in Attention-Based Models") — Bonnet, 2026, 2602.09075. Bayesian metaplasticity; Mamba-2 is a special case.
- **Kalman Delta Networks** — Bui, 2026-09-07, 2609.07816. Tracks memory uncertainty; the delta rule is the isotropic-covariance special case. Gains at 750M and 1.3B.

### 1.5 Log-linear attention and other state expansion
- **Log-Linear Attention** — Guo, 2025, 2506.04761 (ICLR 2026). Fenwick-tree set of O(log T) states on top of Mamba-2 / GDN.
- **Adaptive Memory Decay for Log-Linear Attention** — 2026, 2605.06946.
- **Memory Caching: RNNs with Growing Memory** — Behrouz, 2026, 2602.24281. Caches hidden-state checkpoints so memory grows with length.
- **Hybrid Associative Memories** — Lufkin, 2026, 2603.22325. An RNN plus an attention cache that stores only tokens the RNN predicts poorly.
- **A Hippocampus for Linear Attention (HOLA)** — 2026, 2607.02303. Delta-rule state plus a bounded exact KV cache.
- **Dynamic Linear Attention** — Wang, 2026, 2606.10650. Adaptive merging of multiple states.

### 1.6 Hybrids and the ratios they use
| Model | Linear layer | Linear : full attention | Source |
|---|---|---|---|
| Jamba | Mamba | 7 : 1 (a:m = 1:7) | 2403.19887 |
| MiniMax-01 | Lightning (linear) attention | 7 : 1 | 2501.08313 |
| Qwen3-Next (80B-A3B) | Gated DeltaNet | 3 : 1 (with gated attention) | Qwen blog; fla README |
| Qwen3.5 | Gated DeltaNet | 3 : 1 | HF blog (mlabonne, 2026-02-17) |
| Kimi Linear (48B-A3B) | KDA | 3 : 1 (with MLA) | 2510.26692 |
| Olmo Hybrid (7B) | Gated DeltaNet replacing SWA layers | see paper | 2604.03444 |
| Nemotron 3 Super (120B-A12B) | Mamba | hybrid Mamba-attention MoE | 2604.12374 |
| MiniMax M2 / M2.5 | none | full attention (reverted) | MiniMax blog "Why did M2 end up as a full attention model?" |

- **A Systematic Analysis of Hybrid Linear Attention** — Wang, 2025, 2507.06457. 72 models; recommends GDN or HGRN-2 at 3:1–6:1. The best standalone linear layer is not necessarily the best in a hybrid.
- **Rethinking the Role of Efficient Attention in Hybrid Architectures** — Qiao, 2026, 2606.15378. The efficient layer mainly changes the *pace* at which long-context skill develops. Hybrids converge with enough training.
- **Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing** — Cerruti, 2026, 2607.07953. Controlled comparison; KDA+Muon has the lowest loss, GDN the highest throughput.

## 2. Theory

### 2.1 Associative-recall capacity and interference
- **Zoology** — Arora, 2023, 2312.04927. 82% of the perplexity gap to attention is explained by associative recall (MQAR).
- **Repeat After Me** — Jelassi, 2024, 2402.01032. Fixed-state models provably cannot copy long strings; two-layer transformers can.
- **Sharp Capacity Thresholds in Linear Associative Memory** — Barnfield, 2026, 2605.05189. Top-1 threshold at d²/(n log n) = 2 for isotropic Gaussian embeddings (optimized linear memory).
- **Factual recall in linear associative memories: sharp asymptotics** — Giorlandino, 2026, 2605.10795. Capacity p log p / d² = 1/2; Hebbian vs optimal storage.
- **State Rank Dynamics in Linear Attention LLMs** — Sun, 2026, 2602.02195. In Qwen3-Next GDN heads, state rank splits into near-zero vs saturating heads. Low-rank heads are needed for NIAH.
- **The Key to State Reduction in Linear Attention: A Rank-based Perspective** — Nazari, 2026, 2602.04852. Trained states are low-rank; low rank amplifies query noise; 50% of q/k channels can be pruned.
- **Quantifying Memory Utilization with Effective State-Size** — Parnichkun, 2025, 2504.19561.
- **Kernelized Linear Attention (KATA)** — Ghriss, 2026, 2607.17419. Welch-bound interference floor for feature maps.
- **Variational Linear Attention** — Pandey, 2026, 2605.11196. A whitened (RLS-like) write avoids interference from correlated keys; qualitative analysis of DeltaNet only.

### 2.2 In-context learning as online optimization (LMS / RLS / Kalman)
- **Transformers learn in-context by gradient descent** — von Oswald, 2022, 2212.07677. One linear self-attention layer = one GD step on regression.
- **Trained Transformers Learn Linear Models In-Context** — Zhang, Frei, Bartlett, 2023, 2306.09927. Gradient flow on one LSA layer converges to a global minimum.
- **Test-time regression** (2501.12352), **Longhorn** (2407.14207), **MesaNet** (2506.05233): the delta rule is first-order online least squares; Mesa is the exact solution.
- **Gated KalmaNet** (2511.21016), **KLA** (2602.10743), **KDN** (2609.07816): Kalman-filter generalizations of the delta rule.
- **Learning to Adapt: In-Context Learning Beyond Stationarity** — Qin, 2026, 2604.10946 (ICLR 2026). GLA with a global λ on AR(1)-drifting regression. Closed-form optimum; error is minimized at λ<1; empirically beats LMS/RLS.
- **In-Context Learning Under Regime Change** — Dudley, 2026, 2604.16988. Transformers trained on piecewise-linear regression match change-point-optimal baselines.
- **Understanding Generalization and Forgetting in In-Context Continual Learning** — Li, 2026, 2605.28705 (ICML 2026). Bias / variance / interference decomposition for linear attention on multi-task prompts.
- **Transformers as Implicit State Estimators** — 2024, 2410.16546. Transformers approximate Kalman filtering in-context on linear dynamical systems.

### 2.3 Expressivity and state tracking
- **The Illusion of State in State-Space Models** — Merrill, 2024, 2404.08819. Linear and Mamba SSMs are in TC⁰; they cannot compose permutations.
- **Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues** — Grazzi, 2024, 2411.12537. With finite precision and only positive eigenvalues, parity is impossible; eigenvalues in [−1,1] fix this.
- **DeltaProduct** (2502.10297), **RWKV-7** (2503.14456): Householder products and generalized delta rules give regular-language recognition.
- **Learning State-Tracking from Code Using Linear RNNs** — Siems, 2026, 2602.14814. Linear RNNs learn state tracking from code traces; limits under partial observability (norm decay).
- **Olmo Hybrid: From Theory to Practice and Back** — Merrill, 2026, 2604.03444. Hybrids express tasks beyond both parents and scale better at 7B.
- **Provably Shorter Scratchpads in Hybrid DeltaNet-Attention Decoders** — Steifer, 2026, 2605.16640.

### 2.4 Length generalization of recurrent models
- **Understanding and Improving Length Generalization in Recurrent Models** — Goomba Lab (Gu group), 2025, 2507.02782 (ICML 2025). "Unexplored states" hypothesis; state-noise / state-passing post-training takes 2k to 128k.
- **Stuffed Mamba** — Chen, 2024, 2410.07145. The minimum training length to learn forgetting scales linearly with state size.
- **DeciMamba** — Ben-Kish, 2024, 2406.14528. Effective receptive field is limited by the training length.
- **LongMamba** — 2025, 2504.16053 (ICLR 2025). Global vs local channels by receptive field.
- **Mamba Modulation** — Lu, 2025, 2509.19633 (NeurIPS 2025). Failure is tied to the spectrum of A; spectrum scaling fixes it.
- **Optimal Decay Spectra for Linear Recurrences** — Cao, 2026, 2604.07658. Geometric decay spacing is minimax-optimal; random init collapses the spectral gap.
- **LongSSM** — 2024, 2406.02080.

### 2.5 Numerical precision of chunked / WY kernels and recurrent states
- **Mamba SSMs Are Lyapunov-Stable Learners** — Halloran, 2024, 2406.00209. Mamba's recurrence is provably stable to mixed-precision perturbations.
- **DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization** — Zhang, 2026-08, 2608.27513. Error-propagation model for GDN/KDA states; INT8/FP8 states already hurt reasoning.
- **Why Gated DeltaNet Survives 4-Bit Quantization** — Kozyrev, 2026-09, 2609.04098. State error plateaus (≈12–13% relative from 256 to 32k tokens); delta writes delete old errors key by key.
- **Diagnosing Training Inference Mismatch in LLM RL** — 2026, 2605.14220. General training–inference mismatch (not specific to linear attention).
- Non-arXiv: "Reliable RL Scaling Requires Accounting for Prefill-Decode Kernel Mismatch" (Yifan Zhang, blog, 2026-08-09) — argues chunk schedule changes finite-precision operator, no measurements. vLLM IsoExec blog (2026-08-21). MiniMax M2 blog: linear attention "far more sensitive to numerical precision". fla issue #389: GDN with gate fixed at 1 is numerically unstable.

## 3. Known / Open / Contested

### Known (strong evidence)
- Fixed-state recurrent models lag attention mainly on in-context recall and copying (Zoology 2312.04927; Jelassi 2402.01032; Based 2402.18668).
- Linear attention, SSMs, DeltaNet, TTT, and Titans are all online regressors on key→value pairs, differing in objective, forgetting, and optimizer. The delta rule is one SGD step (2102.11174; 2501.12352; 2407.14207; 2504.13173).
- With finite precision, positive-eigenvalue diagonal transitions cannot do parity. Negative eigenvalues and Householder products unlock state tracking (2411.12537; 2502.10297; 2503.14456; 2404.08819).
- A minority of full-attention layers recovers recall. Production hybrids use 3:1 (Qwen3-Next/3.5, Kimi Linear) to 7:1 (Jamba, MiniMax-01) (2507.06457; 2510.26692; 2501.08313; 2403.19887).
- Trained linear-attention states are low-rank, and heads stratify into low- and high-rank groups (2602.02195; 2602.04852).
- Recurrent models trained short fail long, and cheap state-distribution interventions largely fix it (2507.02782; 2410.07145).
- A gated forget factor provably helps on drifting in-context regression; optimum λ<1 depends on drift (2604.10946).
- Kalman / second-order generalizations of the delta rule improve LM and recall at ≤1.3B (2511.21016; 2602.10743; 2609.07816; 2506.05233; 2604.21100).

### Open
- Do the gates of trained gated delta-rule layers match the optimal tracking filter under controlled drift? No paper measures learned gates against a derived optimum (2604.10946 uses one global λ on GLA; 2609.07816 has no synthetic drift study).
- Closed-form recall error of the *single-pass* delta rule as a function of load n/d, key correlation, and β (capacity results exist for Hebbian/optimized memories with isotropic keys: 2605.05189; 2605.10795).
- Why pretraining produces the rank stratification in 2602.02195. Which data or gate statistics cause it?
- A quantitative predictor of the length-extrapolation failure point for delta-rule models. The existing explanations (2507.02782; 2509.19633; 2406.14528) are qualitative or Mamba-specific.
- Precision behavior of chunked WY kernels for non-contractive transitions (β→2, DeltaProduct, no decay). The existing plateau result (2609.04098) is for gated GDN with quantization noise.
- Is compute better spent on better optimizers (Mesa, Kalman, preconditioning) or on bigger/sparser states (2607.07386; 2602.24281; 2506.04761)?

### Contested
- **Linear hybrids vs full attention in production.** Kimi Linear claims hybrids beat full attention (2510.26692); Qwen3.5 ships GDN; MiniMax went back to full attention for M2, citing precision and infrastructure (MiniMax blog).
- **Best linear mixer.** Mamba-3 (2603.15569), Gated DeltaNet-2 (2605.22791), and KDA (2607.07953) each come out best in their own 1–1.5B comparisons.
- **Cause of length-generalization failure.** Candidates: unexplored states / state norm (2507.02782); spectrum of A (2509.19633); limited receptive field (2406.14528; 2504.16053); failure to learn forgetting (2410.07145).
- **Does recurrent-state numerical error accumulate?** The MiniMax blog says linear attention is precision-fragile, and DAMP (2608.27513) finds FP8 states hurt. 2609.04098 finds GDN state error plateaus even at 4-bit.
- **Does the choice of linear layer matter in hybrids?** Yes per 2507.06457; mostly it changes the pace per 2606.15378.
- **Is state capacity wasted?** Low-rank findings (2602.02195; 2602.04852) vs gains from larger or sparser states (2607.07386; 2602.24281).

## 4. Newest (Jan–Sep 2026) worth reading first
- 2603.15569 Mamba-3 (ICLR 2026)
- 2605.22791 Gated DeltaNet-2
- 2609.07816 Kalman Delta Networks (Sep 7)
- 2602.10743 Kalman Linear Attention
- 2604.10946 Learning to Adapt: ICL Beyond Stationarity (GLA + drift theory)
- 2602.02195 State Rank Dynamics in Linear Attention LLMs
- 2602.04852 The Key to State Reduction in Linear Attention
- 2609.04098 Why Gated DeltaNet Survives 4-Bit Quantization
- 2608.27513 DAMP recurrent-state quantization
- 2604.03444 Olmo Hybrid
- 2604.21100 Preconditioned DeltaNet; 2605.13473 OSDN; 2605.08587 Kaczmarz Linear Attention
- 2605.05189 Sharp Capacity Thresholds in Linear Associative Memory; 2605.10795 Factual recall in linear associative memories
- 2604.07658 Optimal Decay Spectra for Linear Recurrences
- 2607.07386 Sparse Delta Memory; 2602.24281 Memory Caching; 2603.22325 Hybrid Associative Memories; 2607.02303 HOLA
- 2606.15378 Rethinking Efficient Attention in Hybrids; 2607.07953 Linear Attention Architectures (controlled comparison)
- 2602.14814 Learning State-Tracking from Code; 2605.16640 Provably Shorter Scratchpads in Hybrid DeltaNet-Attention
- 2604.16988 ICL Under Regime Change; 2605.28705 In-Context Continual Learning theory
