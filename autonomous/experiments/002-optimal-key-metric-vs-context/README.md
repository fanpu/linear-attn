# 002: Which key metric should a trained delta-rule memory learn: Σ⁻¹ or Σ^(−½)?

*Pre-registered 2026-09-15, before any code was run. Sections 1–4 are frozen.*

## 1. Question

Experiment 001 found that, *in steady state* under drift, the best static key metric for a delta-rule memory is Σ^(−½) (the square-root law, R-003).

Linear attention trained on *stationary* in-context regression instead learns the full inverse covariance Γ⁻¹ ≈ Σ⁻¹ (Zhang–Frei–Bartlett 2023; reproduced in theory/08).

A trained model minimizes the loss **averaged over its whole context**, which mixes the start-up phase with steady state. So: **what is the best exponent s in M = Σ^(−s), as a function of context length T and drift q?** The answer is the curve that trained models (experiment 003) should land on.

## 2. Intuition and hypothesis

There are two phases in every sequence:
- **Learning from scratch (the first ~d tokens).** The memory knows nothing, and every direction needs to be learned once. Full whitening (s = 1) makes all directions equally easy to learn, so the delta rule converges as fast as it would on isotropic inputs. This phase favours s = 1.
- **Tracking (after the memory has converged).** Directions go stale at a rate proportional to their importance λ. The budget argument from 001 says to refresh them in proportion to √λ, which favours s = ½.

The context-averaged loss weights the two phases by how long each lasts. Hypothesis: **s\* slides from 1 to ½ as the tracking phase comes to dominate the context,** i.e. as T grows past the relaxation time and when q > 0.

## 3. Predictions and kill criteria

- **P1:** with no drift (q = 0), s\* ≈ 1 (within ±0.15) for every T ≥ d, with or without noise.
- **P2:** with drift (qd ∈ {0.1, 1}) and long contexts (T = 64d), s\* ≈ ½ (within ±0.15).
- **P3:** at fixed drift, s\* decreases monotonically with T, and the crossover sits near T ≈ a few relaxation times.
- **Kill if:** s\* doesn't depend on T or q (then the trained-model prediction is uninteresting), or s\* at long T with drift is far from ½ (> 0.3 away; then the 001 law doesn't carry over to context-averaged loss).

## 4. Method

- **Code:** reuse `../001-gdn-vs-kalman-theory/sim.py`, which has the same data model and delta rule with static preconditioner M = Σ^(−s).
- **Grid:**
  - d = 32 and κ = 100 (log-uniform spectrum);
  - qd ∈ {0, 0.01, 0.1, 1}, σ² ∈ {0, 0.1};
  - s ∈ {0, 0.125, …, 1.25} (11 values);
  - α ∈ {a, 1−2(1−a), 1−(1−a)/2} (α = 1 when q = 0), and 16 log-spaced β values in [0.02, 1.9].
- **One run per (q, σ², s)** with T_max = 64d = 2048 steps and 512 sequences. For each T ∈ {d/2, d, 2d, 4d, 16d, 64d}, the context-averaged risk of every (α, β) is the prefix mean of its risk curve. Take the min over (α, β), then the argmin over s, refined by fitting a parabola through the three s values around the grid minimum.
- **Reference:** the Kalman filter's context-averaged risk, run once per (q, σ²).
- **Error bars:** 8 batch chunks. Re-take the argmin per chunk for the spread of s\*.
- **Compute:** GPU, a few minutes. Output: `cache/metric.json`. Figure: s\* vs T/d, one line per qd.

---

## 5. What actually happened

Run as planned (one run per configuration, 8 chunks, `sweep_metric.py`; about 1 minute on GPU). Best exponent s\* (± spread across chunks):

| qd, σ² | T = d/2 | d | 2d | 4d | 16d | 64d |
|---|---|---|---|---|---|---|
| 0, 0 | 0.18 | 0.28 | 0.36 | 0.43 | 0.48 | 0.48 (±0.01) |
| 0, 0.1 | 0.06 | 0.20 | 0.33 | 0.43 | 0.48 | 0.50 |
| 0.1, 0 | 0.18 | 0.27 | 0.35 | 0.40 | 0.44 | 0.45 |
| 0.1, 0.1 | 0.07 | 0.20 | 0.32 | 0.39 | 0.42 | 0.44 |
| 1, 0 | 0.17 | 0.23 | 0.27 | 0.29 | 0.31 | 0.32 |
| 1, 0.1 | 0.08 | 0.16 | 0.21 | 0.24 | 0.27 | 0.28 |

(qd = 0.01 is indistinguishable from qd = 0.)

- **P1 (s\* ≈ 1 with no drift): refuted.** s\* never exceeds 0.5. With no drift it *rises* from ≈ 0.2 to ≈ 0.5 as the context grows.
- **P2 (s\* ≈ ½ at T = 64d with drift): holds at qd = 0.1 (0.44–0.45); misses at qd = 1 (0.28–0.32, more than 0.15 from ½).** Not killed: the kill line was 0.3 away.
- **P3 (s\* decreases with T): refuted in direction.** s\* increases monotonically with T in every configuration.

**Unplanned: a finite-context budget model** (`budget_theory.py`, derived after seeing the results).
- *The model.* Direction i's risk decays as λ_i e^(−r_i t), with refresh rate r_i ∝ λ_i^(1−s). The context-averaged risk is Σ λ_i (1 − e^(−r_i T))/(r_i T).
- *Long contexts:* this becomes the same Σ λ_i/r_i problem as tracking, so s\* → ½. It predicts 0.497 and 0.500 at T = 16d and 64d; the simulation gives 0.48.
- *Short contexts:* the model predicts greedy allocation, with s\* = −0.5 (the grid edge), −0.18, and 0.15 at T = d/2, d, 2d. The simulation gives 0.18, 0.28, 0.36. **The direction is right, but the numbers are wrong.** The rate picture fails when one direction would receive a large share of refreshes per step: a projection can't remove more than all the error along a direction, and errors along different directions are coupled.

## 6. Interpretation

- **My hypothesis confused two different algorithms.** "Whitening makes learning from scratch as fast as isotropic" is true, but the context-averaged loss is a *time integral*, and the integral of an exponential decay is 1/rate. So learning from scratch and tracking both reduce to the same Σ λ_i/r_i allocation problem, and the √ law governs both.
- **Linear attention is different.** It is a *one-pass Hebbian* estimator, ŵ ∝ Σ y_i x_i ≈ Σ w. It needs Σ⁻¹ to be unbiased, which is why Zhang–Frei–Bartlett find Γ⁻¹. The delta rule corrects itself iteratively, so no bias correction is needed, and the budget logic takes over.
- **Sharper prediction for trained models (experiment 003).** On the same long-context in-context regression task:
  - trained **linear attention** should learn a key metric with exponent ≈ 1;
  - trained **DeltaNet** should learn ≈ ½;
  - DeltaNet with short contexts (T ≲ 2d) or fast drift (qd ~ 1) should learn ≈ 0.2–0.35.
- **Side observation.** With a context-averaged objective and no drift, the Kalman filter's advantage grows with T (1.12× at T = d/2, 2.0× at T = 4d, noiseless), because it becomes exact after d steps while the delta rule only decays exponentially. **The early-sequence transient is where Kalman memories gain most**, consistent with 001's interpretation, point 4.

## 7. Takeaway
A delta-rule memory should use key metric ≈ Σ^(−½) for long contexts, with or without drift, and a smaller exponent for short contexts. That contrasts with the Σ⁻¹ learned by linear attention. (R-004.)
