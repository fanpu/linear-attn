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
*(to be written)*

## 6. Interpretation
*(to be written)*

## 7. Takeaway
*(to be written)*
