# 003: Do trained DeltaNet and linear attention learn different key metrics on the same task?

*Pre-registered 2026-09-15, before any code was run. Sections 1–4 are frozen.*

## 1. Question

Experiment 002 predicts, for a delta-rule memory, a best key metric M ∝ Σ^(−s\*) with s\* rising to ≈ ½ at long contexts, while one-pass linear attention needs Σ⁻¹ (Zhang–Frei–Bartlett). **Does gradient training actually find these?**

If it does, "DeltaNet learns the square root of the whitening that linear attention learns" becomes a statement about trained networks, not just ideal algorithms.

## 2. Intuition and hypothesis

Both models have a learnable key/query projection A, which sets the geometry in which the memory stores and retrieves:
- **DeltaNet.** With key = query = Ax, a normalized key, and value y/‖Ax‖, the delta rule is exactly the preconditioned update w ← w + β r · AᵀA x / (xᵀAᵀA x). So AᵀA *is* the key metric M.
- **Linear attention.** ŷ_t = γ/(t−1) · Σ_{i<t} y_i x_iᵀ AᵀA x_t. Here M = AᵀA must undo the input covariance to make the sum unbiased.

**Hypothesis.** Training drives DeltaNet's M toward Σ^(−s\*(T,q)) (002's table) and linear attention's M toward ≈ Σ⁻¹.

## 3. Predictions and kill criteria

**Setup:** d = 32, log-uniform Σ with κ = 100 (diagonal), σ² = 0.1. The learned exponent ŝ is the slope of −log diag(M) against log λ; off-diagonal mass is reported separately.

- **P1 (DeltaNet, no drift).** ŝ ≈ 0.20, 0.43, 0.48 at T = d, 4d, 16d (002's s\*), each within ±0.1.
- **P2 (DeltaNet, drift qd = 1, learnable α).** ŝ ≈ 0.16, 0.24, 0.27 at T = d, 4d, 16d, within ±0.1.
- **P3 (linear attention, no drift).** ŝ ≥ 0.85 at T = 4d and 16d.
- **P4 (training finds the optimum).** The trained DeltaNet's context-averaged excess risk is within 10% of the best NLMS-with-Σ^(−s) risk from 002 at the same (T, q).
- **P5 (initialization doesn't matter).** Starting from A = I (s = 0) or A = Σ^(−½) (s = 1) gives the same ŝ within ±0.1.

**Kill if:** at T = 16d with no drift, DeltaNet's ŝ is outside [0.3, 0.7], *or* linear attention's ŝ is < 0.7 (then there's no clean contrast). If P4 fails badly (> 30%), the trained model is not the idealized algorithm, and ŝ comparisons are suspect.

## 4. Method

- **Models** (1 layer, 1 head, value dim 1, trained end to end):
  - **DeltaNet-WB.** Learnable A ∈ ℝ^{d×d}, β = 2·sigmoid(b), and α = sigmoid(a) (α fixed at 1 when there's no drift).
    - Sequence layout: interleaved tokens [query x_t, write (x_t, y_t)], so ŷ_t reads the state before y_t is written.
    - Kernel: fla `chunk_gated_delta_rule`, float32 with IEEE matmul. At query tokens β = 0 and the key is 0; at write tokens there is a normalized key and value y/‖Ax‖. The query is A x_t (unnormalized), so ŷ = S A x_t.
    - A `test_models.py` check will compare against the float64 NLMS recurrence.
    - Ordering note: the fla gate decays at the write token, so the prediction uses the state *after* the previous decay. This differs from 001's decay-then-predict by one factor of α, which the learned α absorbs.
  - **LinAttn-WB.** Learnable A and a scalar γ; ŷ_t = γ/(t−1) Σ_{i<t} y_i (Ax_i)ᵀ(Ax_t), computed with prefix sums. No drift only.
- **Training.** Fresh tasks every batch (256 sequences). Adam with lr 3e-3 and cosine decay over 4000 steps. Loss = mean squared error over all positions t ≥ 2. Initial A = I or Σ^(−½) (P5).
- **Runs:**
  - DeltaNet: T ∈ {d, 4d, 16d} × q ∈ {0, qd = 1} × 2 inits = 12 runs;
  - LinAttn: T ∈ {4d, 16d} × 2 inits = 4 runs;
  - 2 extra seeds at the headline configuration (T = 16d, no drift) for both models = 4 runs.
  - 20 runs in total.
- **Evaluation.** 4096 fresh sequences. Report ŝ (fit), the fit's R², the off-diagonal fraction ‖offdiag M‖_F/‖M‖_F, learned α and β, and the context-averaged excess risk.
- **Compute:** GPU, through `gpu_run.sh`, with pause checkpoints every 500 steps. Estimated at 1–2 h.

---

## 5. What actually happened
*(to be written)*

## 6. Interpretation
*(to be written)*

## 7. Takeaway
*(to be written)*
