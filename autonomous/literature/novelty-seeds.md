# Novelty check for seeds S1–S5 (2026-09-15)

Method: web search (general + arXiv), then the arXiv abstract or HTML page for every candidate close paper. Exact query strings are listed under each seed. The "closest prior work" descriptions come from the fetched abstracts or HTML, not from memory.

---

## S1. Gated delta rules as optimal filters for drifting in-context tasks

**Searches run**
- `gated deltanet kalman filter in-context learning non-stationary`
- `test-time regression unifying framework associative memory sequence models arXiv`
- `linear attention optimal forgetting drift in-context regression theory 2026`
- `Longhorn state space models are amortized online learners arXiv`
- `MesaNet sequence modeling locally optimal test-time training arXiv`
- `"Learning to Adapt: In-Context Learning Beyond Stationarity" arXiv`
- `delta rule linear attention recursive least squares forgetting factor in-context learning`
- `transformers in-context Kalman filter drifting linear regression learned forgetting`
- `gated delta rule in-context learning theory non-stationary tasks tracking step size forget gate learned`
- `DeltaNet normalized LMS in-context linear regression trained one-layer theory`
- `"delta rule" "Kalman" linear attention forget gate process noise random walk`
- `in-context learning piecewise stationary regression change points linear recurrent models gating theory`
- `learned decay gates match optimal forgetting drift rate synthetic in-context regression Mamba Gated DeltaNet`
- `in-context learning random walk regression weights transformer or linear RNN learns optimal tracking step size LMS Kalman comparison experiments`
- `"non-stationary" in-context regression "Gated DeltaNet" OR "Mamba" learned forget gate analysis synthetic`

**Closest prior work**
1. **Learning to Adapt: In-Context Learning Beyond Stationarity** — Qin, Jiang, Zhu, 2604.10946 (ICLR 2026). **Very close.**
   - Model: one-layer GLA with a single *global* forget factor λ (S_i = λS_{i-1} + v_i k_iᵀ).
   - Data: regression weights follow AR(1) drift w_i = γw_{i-1} + e_i.
   - Theory: gradient-flow convergence and closed-form train/test error as a function of (λ, γ). An optimal λ<1 exists.
   - Experiments: loss minimum at λ<1; GLA beats LMS/RLS baselines.
   - Not done: no delta rule, no β (write strength), no data-dependent gates, no comparison of *learned* gates to the optimum, no Kalman bound.
2. **Gated KalmaNet** — Peng et al., 2511.21016 (CVPR 2026).
   - Shows GDN-type layers are Kalman-filter approximations with identity error covariance.
   - Builds an exact-gain (online ridge) layer.
   - No drift experiments and no analysis of the learned gates.
3. **Kalman Linear Attention** — Shaj et al., 2602.10743.
   - Kalman filter as a parallel scan with time-invariant OU dynamics; the decay is read as a forget factor.
   - Evaluated on LM and state tracking, not on drifting regression.
4. **Kalman Delta Networks** — Bui, Huang, Ying, 2609.07816 (2026-09-07).
   - Memory is a latent drifting key→value map; the Kalman update keeps the delta-write form; GDN-style update = isotropic-covariance special case.
   - Only an ablation of learned vs fixed process/observation noise.
   - No synthetic drift sweep and no closed-form optimal α, β.
5. **Test-time regression** (2501.12352), **Longhorn** (2407.14207), **MesaNet** (2506.05233): the online-regression view of the delta rule. All assume a stationary target; none derives drift-optimal gates.
   - Also related: **ICL Under Regime Change** (2604.16988; transformers, change points) and **Transformers as Implicit State Estimators** (2410.16546; transformers ≈ Kalman on LDS).

**Verdict: partially covered.**
- Covered:
  - (a) the Kalman / optimal-filter *framing* of the delta rule (2511.21016, 2602.10743, 2609.07816);
  - (b) the fact that some λ<1 is optimal under drift, with closed forms, for GLA (2604.10946).
  - Classical adaptive-filter theory (NLMS/RLS misadjustment vs tracking lag) also gives optimal step sizes and forgetting factors under random-walk drift. Those textbook results were not re-checked here.
- Still open:
  1. Closed-form α*(q,σ), β*(q,σ) for the *gated delta rule* (NLMS with leakage), not GLA.
  2. Whether the *data-dependent* gates of a trained GDN converge to those values as q and σ are swept.
  3. The gap between the best gated-delta-rule risk and the Kalman risk, as a function of key anisotropy. This gap is what KDN / GKA / KLA are meant to close, and no paper has quantified it.

**Nearest still-open variant.** "When does a Kalman memory beat the gated delta rule?"
- Derive the risk of the optimal-gate GDN and of the Kalman filter on drifting in-context regression, as a function of (q, σ, key covariance spectrum).
- Show that trained GDN gates reach the GDN optimum.
- Predict the regime (anisotropic keys, low drift) where covariance tracking helps.
- This yields a falsifiable prediction for 2609.07816 / 2511.21016 and stays cheap (1-layer models, synthetic data).
- Must cite 2604.10946 as the GLA precursor.

---

## S2. Effective rank of linear-attention states

**Searches run**
- `effective rank of linear attention recurrent state trained language models spectrum analysis`
- `linear attention state low-rank memory utilization DeltaNet state matrix singular values`
- `recurrent state saturation predicts recall failure linear attention long context memory capacity analysis 2026`

**Closest prior work**
1. **State Rank Dynamics in Linear Attention LLMs** — Sun et al., 2602.02195.
   - Per-layer effective rank of GDN states in Qwen3-Next (48 layers).
   - Finds "state rank stratification": about half the heads sit near zero rank, the rest saturate. Head identity is fixed by pretraining.
   - Pruning low-rank heads drops NIAH from 93.8% to 46.9%; pruning high-rank heads costs little. Gives a rank-saturation / norm-accumulation theory.
   - Uses these findings to prune the KV/state (38.9% reduction). **Covers the measurement, the layer dependence, and the link to recall.**
2. **The Key to State Reduction in Linear Attention: A Rank-based Perspective** — Nazari & Rusch, 2602.04852.
   - Trained states are low-rank ("at most half of capacity" used).
   - Theory: low effective rank amplifies query noise in retrieval.
   - Removes 50% of q/k channels with a marginal perplexity cost.
   - **Covers the "wasted memory / free lunch" claim.**
3. **Quantifying Memory Utilization with Effective State-Size** — Parnichkun et al., 2504.19561. A general effective-state-size metric for input-varying linear operators.

**Verdict: covered (kill).** The prediction ("rank ≪ d_k, varies by layer, predicts recall failure") and the practical payoff (state reduction) are both published.

**Nearest still-open variant (weak).**
- A causal account of *why* stratification emerges, e.g. a 1-layer synthetic model in which key anisotropy plus decay produce low-rank states.
- Best folded into S4 or S1 rather than pursued alone.

---

## S3. A formula for when linear attention stops extrapolating in length

**Searches run**
- `length generalization recurrent models unexplored states hypothesis Mamba linear attention arXiv 2025`
- `predict length extrapolation failure from learned decay spectrum state space model training length`
- `linear attention length extrapolation failure theory Gated DeltaNet beyond training context 2026 arXiv`
- `DeciMamba exploring length extrapolation potential of Mamba effective receptive field arXiv`
- `LongMamba training-free long context global channels local channels receptive field arXiv`
- `"length generalization" linear RNN "product of transition" eigenvalue cumulative decay out-of-distribution state norm theory`
- `why linear recurrent models fail beyond training length state norm effective memory horizon predict extrapolation 2026 arXiv`
- `"training length" "forget gate" horizon length extrapolation linear attention empirical law decay statistics`

**Closest prior work**
1. **Understanding and Improving Length Generalization in Recurrent Models** — 2507.02782 (ICML 2025).
   - "Unexplored states" hypothesis: failure happens when training covers only part of the attainable state distribution. Tied to state-norm growth.
   - Fixes: state-noise or state-passing initialization. Covers Mamba and linear attention.
2. **Mamba Modulation** — Lu et al., 2509.19633 (NeurIPS 2025). Links failure to the *spectrum of A*; spectrum scaling restores long-context performance. Mamba only.
3. **DeciMamba** — 2406.14528 and **LongMamba** — 2504.16053.
   - The effective receptive field, set by cumulative decay (Δ_t), is limited by the training length.
   - Channels split into local and global by receptive field.
4. **Stuffed Mamba** — Chen et al., 2410.07145. The minimum training length needed to learn forgetting scales linearly with state size; passkey-retrieval length scales exponentially with it.
5. **Optimal Decay Spectra for Linear Recurrences** — Cao, 2604.07658. Theory linking decay-rate spacing to memory horizon (minimax rates).

**Verdict: partially covered.**
- Covered: the idea that decay statistics or the transition spectrum measured at training length explain the failure (2509.19633, 2406.14528, 2504.16053), and a strong competing explanation via state norms / unexplored states (2507.02782). The kill criterion ("dominated by position-dependent norms") is partly supported by 2507.02782.
- Still open: a *quantitative* predictor of the failure length, fit across several training lengths, for *delta-rule* models (GDN, DeltaProduct), where transitions are not diagonal and the delta write also erases. Also a head-to-head test of the competing hypotheses.

**Nearest still-open variant.** A discriminating experiment.
- Train GDN at L = 256/512/1024.
- Compute (i) the cumulative-decay horizon, (ii) the state-norm trajectory, and (iii) state-distribution coverage.
- Test which one predicts the failure length and which intervention (spectrum rescaling vs state-noise init) fixes which failure.
- Risk: it may turn out to be a replication of 2507.02782 for GDN.

---

## S4. Recall capacity of the delta rule with correlated keys

**Searches run**
- `delta rule associative memory capacity correlated keys closed form linear attention random matrix theory`
- `associative recall capacity DeltaNet versus linear attention theory interference number of key-value pairs dimension`
- `Kaczmarz method random sequential projections error correlated rows convergence rate closed form online delta rule memory`
- `statistical mechanics in-context associative recall linear attention delta rule capacity phase transition sequential writes`
- `single-pass online delta rule versus Hebbian storage capacity correlated patterns analytical`
- `delta rule retrieval error key correlation step size beta theorem linear transformer associative memory interference analysis`
- `"delta rule" "Hebbian" in-context recall linear attention capacity high-dimensional asymptotics 2025 2026`
- `online least squares memory sequential projection key overlap forgetting curve closed form DeltaNet random keys`
- `linear attention associative recall correlated keys anisotropic key covariance recall accuracy theory delta rule whitening`
- `How catastrophic can catastrophic forgetting be in linear regression Kaczmarz projections Evron arXiv`
- `memory capacity of DeltaNet theoretical analysis number of associations retrievable fixed-size state proof`
- `Kohonen novelty filter orthogonal projection associative memory correlated patterns recall error analysis`

**Closest prior work**
1. **Sharp Capacity Thresholds in Linear Associative Memory** — Barnfield et al., 2605.05189.
   - Exact thresholds for data-dependent linear memories: top-1 at d²/(n log n) = 2, plus a closed-form critical load for top-k.
   - Assumes isotropic Gaussian embeddings and ERM-trained memory, not a single-pass delta rule or correlated keys.
2. **Factual recall in linear associative memories: sharp asymptotics** — Giorlandino, Goldt, Maillard, 2605.10795. Capacity p log p / d² = 1/2; Hebbian vs optimal storage. No sequential delta rule.
3. **How catastrophic can catastrophic forgetting be in linear regression?** — Evron et al., 2205.09588.
   - Exact forgetting expressions for sequential projections (Kaczmarz / alternating projections), with task-order effects.
   - This is mathematically the β=1 block delta rule, but framed as continual learning with task-level subspaces, not key-value recall with a correlation parameter.
4. **Linear Transformers Are Secretly Fast Weight Programmers** — Schlag et al., 2102.11174. Points out the d_k capacity limit of additive memory and motivates the delta rule; no closed form under correlation.
5. **Variational Linear Attention** (2605.11196) and **Don't Read Everything: A Curvature-Conditioned Query** (2606.01294). Discuss correlated/anisotropic keys and interference qualitatively. The fetched HTML confirms neither gives a closed-form delta-rule recall error.

**Verdict: open (narrow), with caveats.**
- No arXiv paper found gives recall error of the single-pass (gated) delta rule as a function of (n/d_k, key correlation, β).
- Caveats:
  - (i) The ingredients are classical: Kaczmarz, Kohonen's orthogonalizing / novelty filter, pseudo-inverse Hopfield rules with correlated patterns. The pre-arXiv literature (1980s–90s neural-network journals) was **not** searched thoroughly and must be checked before any claim of novelty.
  - (ii) The seed's "sharp capacity threshold" is doubtful for mean-squared recall error, which should degrade smoothly. The sharp thresholds in 2605.05189 / 2605.10795 come from top-1 decoding with extreme-value (log n) effects.
  - (iii) The impact is moderate unless it explains something in trained models, e.g. why GDN-2 / KDA channel gates help.

**Suggested sharpening.** Derive MSE recall and top-1 capacity for the β-delta rule with a spiked or AR key covariance, and compare against the Hebbian rule and the whitened / RLS write (VLA, GKA). The one-figure claim would be: "delta rule capacity vs correlation, with Hebbian and RLS as bounds."

---

## S5. Numerical drift of chunked parallel forms at long context

**Searches run**
- `numerical precision chunkwise parallel DeltaNet WY representation bfloat16 error accumulation long sequence`
- `flash-linear-attention chunk vs fused_recurrent numerical mismatch gated deltanet github issue`
- `training-inference mismatch linear attention kernels RL prefill decode logprob discrepancy Gated DeltaNet Qwen3-Next`
- `"prefill-decode kernel mismatch" linear attention arXiv`
- `floating point error propagation linear RNN state chunked parallel scan vs sequential recurrence sequence length growth analysis`
- `"Diagnosing Training Inference Mismatch in LLM Reinforcement Learning" linear attention recurrent state`
- `Mamba selective scan numerical instability mixed precision long sequence state divergence bf16 fp32 analysis paper`
- `hybrid linear attention model RL training rollout mismatch recurrent state precision fp32 MiniMax Kimi Linear chunk kernel`
- `linear attention recurrent state quantization low precision inference error accumulation 2026 arXiv Gated DeltaNet`
- `Mamba State-Space Models Are Lyapunov-Stable Learners mixed precision perturbation theorem arXiv 2406.00209`

**Closest prior work**
1. **Why Gated DeltaNet Survives 4-Bit Quantization** — Kozyrev & Maiboroda, 2609.04098 (Sep 2026).
   - Lockstep FP32 simulation of injected quantization error in the GDN state of a 27B hybrid.
   - Relative state error is flat (12.96% at token 256 vs 12.31% at 32,768).
   - A 1% impulse decays to 1/e within 80–1,382 steps, much faster than the decay-gate horizon, because delta writes overwrite errors key by key.
   - The per-position quantization cost *shrinks* with position.
   - Does not compare chunked vs recurrent kernels (they check consistency only to 1e-4).
2. **DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization** — Zhang et al., 2608.27513.
   - Error-propagation decomposition through the transitions for GDN/KDA states; decay-based persistence estimate.
   - INT8/FP8 states degrade reasoning. No explicit length bound, no chunk-vs-recurrent study.
3. **Mamba SSMs Are Lyapunov-Stable Learners** — Halloran et al., 2406.00209. Proves Mamba's recurrence is stable to mixed-precision perturbations; empirical divergence is lower than in transformers.
4. Practice reports (non-arXiv):
   - Yifan Zhang blog "Reliable RL Scaling Requires Accounting for Prefill-Decode Kernel Mismatch" (2026-08-09): the chunk schedule changes the finite-precision operator; no measurements.
   - vLLM IsoExec blog (2026-08-21): bitwise-aligned kernels for Qwen3.5.
   - MiniMax M2 blog: linear attention is "far more sensitive to numerical precision".
   - fla issue #389: GDN with gate fixed at 1 is numerically unstable.
   - General training–inference mismatch: 2605.14220.

**Verdict: partially covered, and likely a kill for gated models.**
- Covered, in effect: the kill criterion. For gated delta rules, injected state errors do not grow with length (2609.04098), and contractive recurrences are provably stable to perturbations (2406.00209).
- Nobody has published a systematic chunk-vs-recurrent gap vs length × dtype × transition conditioning.
- By the same mechanism, the gap is expected to sit at the rounding floor for GDN/KDA.

**Nearest still-open variant.** Non-contractive transitions.
- Cases: DeltaNet without decay, β∈(1,2] (negative eigenvalues, near-orthogonal Householder transitions), DeltaProduct, and Mamba-3 complex/rotational transitions. Here errors are not forgotten.
- Also the *backward* pass (gradient mismatch in chunked WY backward) and RL logprob mismatch in hybrids.
- This is a side note at best. Best run as a quick check inside `hardware/`, not as the main project.

---

## Summary

| Seed | Verdict |
|---|---|
| S1 | partially covered — open: delta-rule optimal gates vs learned GDN gates, and the GDN-vs-Kalman risk gap vs key anisotropy |
| S2 | covered (kill) — 2602.02195, 2602.04852 |
| S3 | partially covered — open: quantitative predictor for delta-rule models, discriminating among competing explanations |
| S4 | open (narrow) — check pre-arXiv associative-memory literature; drop the "sharp threshold" framing for MSE |
| S5 | partially covered / likely kill for gated models — open only for non-contractive transitions |
