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
- **Status:** seed.

### S2. Why is the state never full? Effective rank of linear-attention states
- **Intuition.** A state of size d_k × d_v could store d_k independent key→value associations. If trained models use only a low-rank slice, most of that memory is wasted, and there's a free lunch: smaller states, or a better allocation.
- **Prediction.** State effective rank ≪ d_k on real text, it differs systematically across layers, and it predicts where recall fails.
- **Probe.** Train small LMs (or use released `fla` checkpoints, if downloadable) and measure the spectra of S_t over long documents.
- **Kill if** the rank is near-full, or this spectrum analysis already exists.
- **Status:** seed.

### S3. A formula for when linear attention stops extrapolating in length
- **Intuition.** A recurrent state is a sum of old writes, each multiplied by a product of transition matrices. Whether old information fades, persists, or blows up depends on the eigenvalues of those products. Training at length L only ever tests products up to length L.
- **Prediction.** The length at which extrapolation fails can be predicted from statistics of the learned transitions (e.g. the distribution of per-token decay), measured at training length only.
- **Probe.** Train small models at L = 256 / 512 / 1024 and evaluate up to 16×. Fit failure length against a spectral statistic.
- **Kill if** there's no clean relation, or the failure is dominated by something else (e.g. position-dependent norms).
- **Status:** seed.

### S4. Recall capacity of the delta rule with correlated keys: a closed form
- **Intuition.** Additive (Hebbian) memory suffers interference when keys overlap. The delta rule subtracts the old prediction before writing, which partially undoes the interference. How much capacity does that buy as a function of key correlation?
- **Prediction.** A random-matrix calculation gives recall error vs (number of stored pairs / d_k, key correlation, β), with a sharp capacity threshold. Trained models on multi-query associative recall sit on the curve.
- **Probe.** A pencil derivation plus a float64 simulation of the recurrence (CPU), then one small trained-model check.
- **Kill if** a closed form already exists in the associative-memory literature, or the simulation shows no clean threshold.
- **Status:** seed.

### S5. Numerical drift of chunked parallel forms at long context
- **Intuition.** Training kernels compute the delta rule in chunks, using a compact "WY" representation of products of Householder-like matrices. Rounding errors in bf16 may accumulate differently than in the step-by-step recurrence. This connects to the precision measurements in `hardware/`.
- **Prediction.** The gap between chunked and recurrent outputs grows with sequence length at a rate set by the conditioning of the transitions, and matters for downstream loss past some length.
- **Probe.** Compare `fla` chunk vs recurrent kernels on random and trained weights in float32/bf16/float64 across lengths.
- **Kill if** the gap stays at the rounding floor (~1e-3 relative in bf16) with no length dependence.
- **Status:** seed. Smaller-scope; may be a side note rather than the paper.

---

## Killed
(None yet.)

## Promoted
(None yet.)
