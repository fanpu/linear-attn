# Idea backlog

Each idea has a **prediction** (what I expect to see), a **cheap probe** (a small fraction of the token budget), and a **kill criterion** (what would make me drop it).

**Scoring.** Each idea is scored 1–5 on:
- **N**ovelty (after checking the literature),
- **I**mpact (would it change what people believe or build?),
- **F**easibility on a shared GB10 in two weeks,
- **C**larity (can it be explained with one figure?).

**Status:** `seed` (unchecked) → `novelty-checked` → `probing` → `promoted` / `killed` (with a reason and a link to evidence).

The seeds below were written on 2026-09-15 from memory, before any literature search. **Their novelty is unverified.** Several may already be published; checking that is the first job.

---

## Seeds (2026-09-15)

### S1. Gated delta rules as optimal filters for drifting in-context tasks
- **Intuition.** A delta-rule state is an online linear regressor, updated one token at a time. If the task being regressed *drifts* along a sequence, the best online estimator is a Kalman filter. Its optimal "forgetting" depends on the drift speed and the noise level. The forget gate α and the write strength β in Gated DeltaNet look like exactly those two knobs.
- **Builds on** theory/08's finding that a trained 1-layer DeltaNet ≈ normalized LMS (a least-mean-squares filter with normalized step size), extending it from stationary to non-stationary tasks.
- **Prediction.** On in-context regression with random-walk task drift (drift rate q, noise σ), the gates a trained model learns track a closed-form optimum α*(q, σ), β*(q, σ). Its risk approaches the Kalman / optimal-forgetting-LMS bound.
- **Probe.** 1-layer Gated DeltaNet on synthetic drifting regression. Sweep q and compare the learned α, β to the formula (CPU/GPU minutes).
- **Kill if** the learned gates don't vary systematically with q, or the "online learning" view of these models already covers this with measurements.
  - Check Longhorn, "test-time regression", MesaNet, and Titans first.
- **Status:** novelty-checked 2026-09-15, **partially covered** → refined into **S1′** below.

### S2. Why is the state never full? Effective rank of linear-attention states
- **Intuition.** A state of size d_k × d_v could store d_k independent key→value associations. If trained models use only a low-rank slice, most of that memory is wasted, and there's a free lunch: smaller states, or a better allocation.
- **Prediction.** State effective rank ≪ d_k on real text, it differs systematically across layers, and it predicts where recall fails.
- **Probe.** Train small LMs (or use released `fla` checkpoints, if downloadable) and measure the spectra of S_t over long documents.
- **Kill if** the rank is near-full, or this spectrum analysis already exists.
- **Status:** killed 2026-09-15, covered by 2602.02195 (State Rank Dynamics) and 2602.04852 (rank-based state reduction).

### S3. A formula for when linear attention stops extrapolating in length
- **Intuition.** A recurrent state is a sum of old writes, each multiplied by a product of transition matrices. Whether old information fades, persists, or blows up depends on the eigenvalues of those products. Training at length L only ever tests products up to length L.
- **Prediction.** The length at which extrapolation fails can be predicted from statistics of the learned transitions (e.g. the distribution of per-token decay), measured at training length only.
- **Probe.** Train small models at L = 256 / 512 / 1024 and evaluate up to 16×. Fit failure length against a spectral statistic.
- **Kill if** there's no clean relation, or the failure is dominated by something else (e.g. position-dependent norms).
- **Status:** novelty-checked, **partially covered** (2507.02782 unexplored states; 2509.19633 Mamba spectrum; DeciMamba). Still open: a quantitative, discriminating test for delta-rule models. **Backup** direction; risks becoming a replication of 2507.02782.

### S4. Recall capacity of the delta rule with correlated keys: a closed form
- **Intuition.** Additive (Hebbian) memory suffers interference when keys overlap. The delta rule subtracts the old prediction before writing, which partially undoes the interference. How much capacity does that buy as a function of key correlation?
- **Prediction.** A random-matrix calculation gives recall error vs (number of stored pairs / d_k, key correlation, β), with a sharp capacity threshold. Trained models on multi-query associative recall sit on the curve.
- **Probe.** A pencil derivation plus a float64 simulation of the recurrence (CPU), then one small trained-model check.
- **Kill if** a closed form already exists in the associative-memory literature, or the simulation shows no clean threshold.
- **Status:** novelty-checked, open but narrow (pre-arXiv Kaczmarz / Kohonen literature not searched; a sharp MSE threshold is unlikely). **Merged into S1′** as its key-anisotropy axis.

### S5. Numerical drift of chunked parallel forms at long context
- **Intuition.** Training kernels compute the delta rule in chunks, using a compact "WY" representation of products of Householder-like matrices. Rounding errors in bf16 may accumulate differently than in the step-by-step recurrence. This connects to the precision measurements in `hardware/`.
- **Prediction.** The gap between chunked and recurrent outputs grows with sequence length at a rate set by the conditioning of the transitions, and matters for downstream loss past some length.
- **Probe.** Compare `fla` chunk vs recurrent kernels on random and trained weights in float32/bf16/float64 across lengths.
- **Kill if** the gap stays at the rounding floor (~1e-3 relative in bf16) with no length dependence.
- **Status:** killed 2026-09-15. See R-001: the error is flat in T (Fan Pu's day-1 data, T = 256–4096).

## Shortlist after the literature check (2026-09-15)

### S1′. When does a smarter memory write help? Gated delta rule vs Kalman filter under drift ★ PROMOTED at Gate A (2026-09-15)
- **Intuition.**
  - A gated delta rule (GDN) makes *one* correction step per token, along the current key. A Kalman filter (KF) keeps a covariance, so it knows which directions it has already pinned down.
  - Noiseless, isotropic keys, drift q per token, d dimensions:
    - GDN at β = 1 is a random projection (Kaczmarz). Its steady-state error is ≈ qd: the drift accumulated in the ~d tokens it takes to refresh every direction.
    - KF never re-measures a known direction. A back-of-envelope estimate puts its error at ≈ qd/2, **a factor of only ~2**.
  - With anisotropic keys (condition number κ), the delta rule slows down in weak directions while KF is unaffected. The gap should grow with κ.
  - A *learned static key map* can whiten keys (cf. theory/08: linear attention learns the Γ⁻¹ preconditioner), which should remove most of that gap.
- **Candidate claim.** Kalman-style memories (Gated KalmaNet 2511.21016, Kalman Linear Attention 2602.10743, Kalman Delta Networks 2609.07816) can beat a well-trained gated delta rule by more than a small constant only through (i) key statistics that change within a sequence, (ii) the early-sequence transient, or (iii) anisotropy the key projection fails to whiten. The theory gives the size of each gain, which is a falsifiable account of where their LM gains come from.
- **Prediction.**
  - (a) A closed-form optimum α*, β* for the leaky NLMS rule under AR(1) drift, matching simulation.
  - (b) An isotropic GDN*/KF risk ratio of ≈ 2 as qd → 0 (noiseless), approaching 1 as qd grows.
  - (c) A ratio that grows with κ unless keys are whitened.
  - (d) Trained 1-layer GDNs learn α ≈ α*, β ≈ β*, and a whitening key map.
- **Probes.** Experiment 001: theory plus float64 simulation, CPU only. Experiment 002: trained 1-layer GDN on drifting regression (GPU, minutes).
- **Kill if** the isotropic ratio is ≫ 2 across the board (then "gains come from anisotropy" is wrong), or ≈ 1 everywhere, including under anisotropy (then this setting explains nothing).
- **Must cite and position against:**
  - Classical adaptive filtering: optimal LMS step and RLS forgetting under random walk (Eweda 1994 IEEE TSP; Haykin). Those results are low-dimensional and small-step; ours are proportional, d ~ memory length.
  - Qin et al. 2604.10946 (GLA, one global λ, no delta rule, no learned-vs-optimal comparison).
  - Fan Pu's theory/08 (1-layer DeltaNet ≈ NLMS, stationary).

### S3′. Discriminating test of length-extrapolation explanations for delta-rule models (backup)
See S3 status. Only if S1′ dies at Gate A.

---

## Killed
- **S2** (2026-09-15): covered by 2602.02195 and 2602.04852.
- **S5** (2026-09-15): chunk-kernel float32 error is a constant ~1e-4 floor, not length-dependent. R-001.

## Promoted
- **S1′** (Gate A, 2026-09-15): R-002 (isotropic gap ≤ 1.47×) and R-003 (square-root law). The next claim to test is that trained GDNs learn gates α ≈ √(1−q) and key metric AᵀA ∝ Σ^(−½) under drift (vs Σ⁻¹ stationary).
