# 001: How much better is a Kalman memory than the best gated delta rule, under drift?

*Pre-registered 2026-09-15, before any code was run. Sections 1–4 are frozen.*

## 1. Question

A gated delta rule (as in Gated DeltaNet) and a Kalman filter both keep a running estimate of a linear map and correct it with each new token. The Kalman filter additionally tracks *how uncertain* it is in every direction.

Several 2025–26 architectures add this uncertainty tracking to linear attention:
- Gated KalmaNet (2511.21016),
- Kalman Linear Attention (2602.10743),
- Kalman Delta Networks (2609.07816).

They report perplexity gains, but nobody has measured **how much the extra machinery can help in principle, and in which regime.** This experiment measures it on the cleanest possible task, using exact formulas and float64 simulation. No training is involved.

## 2. Intuition and hypothesis

**The task.** In-context linear regression where the true weights drift.
- The data: $x_t \sim \mathcal N(0, \Sigma)$ and $y_t = w_t^\top x_t + \sigma\varepsilon_t$.
- The drift: $w_{t+1} = a\,w_t + \sqrt{q/d}\;\xi_t$, with $a = \sqrt{1-q}$. This keeps $\mathbb E\|w_t\|^2 = 1$ at every step.
- $q$ is "how much of the task is replaced per token", and $d$ is the dimension.
- Before seeing $y_t$, the memory predicts $\hat y_t = u_t^\top x_t$. The risk is $\mathbb E(y_t - \hat y_t)^2$.

**The two memories.** Both first *decay* the old estimate, then *correct* it along the new input.

| | decay | correction |
|---|---|---|
| gated delta rule (NLMS form) | $u_t = \alpha\,\hat w_{t-1}$ | $\hat w_t = u_t + \beta\,(y_t - u_t^\top x_t)\,x_t/\|x_t\|^2$ |
| Kalman filter | $u_t = a\,\hat w_{t-1}$, $P^-_t = a^2 P_{t-1} + \tfrac{q}{d}I$ | $\hat w_t = u_t + K_t(y_t - u_t^\top x_t)$, $K_t = P^-_t x_t / (x_t^\top P^-_t x_t + \sigma^2)$ |

The gated delta rule is exactly the Kalman filter with the covariance $P$ replaced by a constant multiple of the identity.

**A worked example of why the gap might be small.** Take isotropic inputs and no noise.
- *Delta rule.* With $\beta = 1$ it becomes a projection: after one step, the estimate is exactly right along $x_t$, and unchanged in the other $d-1$ directions. A random direction gets refreshed about once every $d$ tokens, during which it drifts by about $q$ in total. So the error should settle near $qd$ (for small $qd$).
- *Kalman filter.* It never wastes a measurement on a direction it already knows. The average "age" of its knowledge is about half as long, so its error should be ≈ $qd/2$.

This predicts a gap of only **about 2×**.

**Why the gap might be large.** With anisotropic inputs (condition number $\kappa$), the delta rule mostly refreshes the high-variance directions and neglects the low-variance ones. The Kalman filter's covariance corrects for exactly that.

**Hypothesis.** The Kalman advantage over the best-tuned gated delta rule is a small constant (≲ 2×) for isotropic inputs, and grows with input anisotropy $\kappa$.

## 3. Predictions and kill criteria

**P1: Exact formula (a correctness check).** For isotropic Gaussian inputs, the delta-rule error $m = \mathbb E\|u_t - w_t\|^2$ obeys an exact linear recursion in two moments, $m_t$ and $c_t = \mathbb E[w_t^\top(u_t - w_t)]$. The fixed point is

$$c = \frac{a(\alpha-a) - q}{1 - a\alpha(1-\beta/d)},\qquad
m = \frac{(\alpha-a)^2 + 2\alpha(\alpha-a)(1-\beta/d)\,c + \alpha^2\beta^2\sigma^2/(d-2) + q}{1 - \alpha^2\big(1 - (2\beta-\beta^2)/d\big)},\qquad R = m + \sigma^2 .$$

Simulation must match within Monte-Carlo error (target: ≤ 2% relative at d ≥ 16). If it doesn't, the derivation is wrong: fix it before anything else.

**P2: Isotropic gap.** Let $R^*_{\text{GDN}}$ be the risk at the best $(\alpha, \beta)$ and $R_{\text{KF}}$ the Kalman risk, both measured as excess risk (the part above $\sigma^2$).
- *Prediction:* as $qd \to 0$ with $\sigma = 0$, $R^*_{\text{GDN}}/R_{\text{KF}} \to$ about 2. The ratio approaches 1 as $qd$ grows.
- *Kill if:* the isotropic ratio is ≫ 2 (say > 4) over most of the $(qd, \sigma)$ grid. Then the "small constant" story is wrong.

**P3: Anisotropy.** With $\Sigma$ having condition number $\kappa$ (normalized so that $\operatorname{tr}\Sigma = d$):
- *Prediction:* the ratio grows with $\kappa$.
- *Kill (of the anisotropy half) if:* the ratio stays within 1.5× of its isotropic value up to $\kappa = 100$.

**P4 (exploratory, no prediction): static preconditioning.** Does feeding the delta rule pre-whitened keys $\Sigma^{-s/2}x$, for some $s \in [0,1]$, recover the isotropic ratio?
- Note: whitening the inputs makes the *drift* anisotropic in the new coordinates, and the Kalman filter's risk is unchanged by any fixed linear reparametrization. So this is genuinely unclear. A trained key projection can only implement this kind of *static* preconditioning, so the answer tells us how much a Kalman memory can add beyond what training already provides.

**P5 (exploratory): the transient.** Starting from $\hat w_0 = 0$, how does the ratio behave over the first few multiples of $d$ tokens, before steady state? The early-sequence gap could be much larger than the steady-state gap.

## 4. Method

- **Code.** `sim.py` runs batched float64 simulations of (i) the NLMS-form gated delta rule over a grid of $(\alpha, \beta)$ and (ii) the Kalman filter with the true $a$, $q$, $\sigma$, starting from $\hat w_0 = 0$, $P_0 = I/d$. `theory.py` holds the closed form (P1) and optimizes it over $(\alpha, \beta)$.
- **Check against known answers** (`test_sim.py`):
  - the P1 closed form vs simulation;
  - with $q = 0$ and $\sigma = 0$, the Kalman filter recovers $w$ exactly after $d$ steps;
  - the Kalman filter is never worse than the best delta rule (it is the optimal linear estimator).
- **Grid.**
  - $d \in \{32, 128\}$;
  - $qd \in \{0.01, 0.03, 0.1, 0.3, 1, 3, 10\}$;
  - $\sigma^2 \in \{0, 0.1, 1\}$;
  - $\kappa \in \{1, 10, 100\}$, with log-uniformly spaced eigenvalues.
- **Measurement.**
  - Steady-state risk averages the last half of $T = \max(2000,\ 20/(q + 1/d))$ steps over a batch of ≥ 256 sequences.
  - Report the ratio with a bootstrap 95% CI.
  - For the anisotropic delta rule, grid-search $(\alpha, \beta)$ in simulation, since the closed form doesn't apply.
- **Compute.** CPU only, minutes. `OMP_NUM_THREADS=4`.
- **Outputs.** `cache/*.json`. Figures: ratio vs $qd$ (one line per $\sigma^2$), and ratio vs $\kappa$.

---

## 5. What actually happened

*Complete (2026-09-15).*

**Deviations from the plan in §4.**
- The first isotropic run used $T = \max(2000, 20/(q+1/d))$. That is too short when the optimal $\beta$ is small (noisy, low-drift settings): at $d=128$, $qd=0.01$, $\sigma^2=1$ the relaxation time is ≈ 700 steps, and the simulation read 0.168 against a theory value of 0.100. I discarded that run and reran everything with $T = \max(2000, 20\tau)$, where $\tau = 1/(1-\alpha^2(1-(2\beta-\beta^2)/d))$ is the delta rule's relaxation time at its optimum.
- I added a $d$-scaling run (`sweep_dscale.py`, $d$ = 16…1024) to test a limit I conjectured after seeing the first numbers. It was not pre-registered.

**P1 (exact formula): confirmed.** Across all 42 isotropic configurations ($d \in \{32, 128\}$, 7 drift rates, 3 noise levels), simulated steady-state risk matches the closed form within 0.5%. The unit tests also match the full transient recursion within 3% on 50-step blocks. At the *joint* optimum over $(\alpha, \beta)$, $\alpha^* = a = \sqrt{1-q}$ (to ~2e-8 after grid refinement): **the best forget gate equals the true persistence of the task.** Only $\beta^*$ depends on the noise. At a fixed, non-optimal $\beta$, $\alpha^*$ moves away from $a$. Qin et al. (2604.10946) report that the analogous identity does *not* hold for GLA.

**P2 (isotropic gap): the prediction of ≈ 2 was wrong. The gap is smaller.** Ratio of excess risks, best delta rule / Kalman, at $d = 128$:

| $qd$ | $\sigma^2 = 0$ | $\sigma^2 = 0.1$ | $\sigma^2 = 1$ |
|---|---|---|---|
| 0.01 | 1.458 | 1.044 | 1.017 |
| 0.1 | 1.399 | 1.105 | 1.027 |
| 1 | 1.138 | 1.079 | 1.015 |
| 10 | 1.004 | 1.003 | 1.001 |

- **Noiseless $d$-scaling.** At $qd = 0.01$ the ratio is 1.389, 1.423, 1.444, 1.453, 1.459, 1.466, 1.465 (±0.004) for $d$ = 16…1024. It levels off at **≈ 1.465**.
- **Why.** The Kalman error settles at ≈ 0.675–0.68 $qd$, while the delta rule's is $qd/(1+qd)$.
- **Neither conjectured constant fits.** An "age" model, in which each Kalman measurement fully resets the most uncertain direction, predicts $2/\pi \approx 0.637$ and a ratio of $\pi/2$. $1/\ln 2 = 1.443$ is exceeded from $d = 64$ on.
- **Status of the constant.** The exact high-dimensional constant remains open. It needs the steady-state spectrum of $P$ under random rank-one Riccati updates.


**P3 (anisotropy): the pre-registered criterion kills it. The gap grows, but slowly.**
- *Setup:* d = 64, log-uniform input variances with condition number κ, and the best (α, β) found by grid search per configuration. Steady state is the last quarter of T ≤ 100k steps; the change from the third quarter is ≤ 1.2% everywhere.
- *Result:* without preconditioning, the noiseless ratio at qd = 0.01 is 1.44, 1.59, 1.95, 2.35 for κ = 1, 10, 100, 1000. At κ = 100 that is 1.35× the isotropic value, below the 1.5× kill line. With σ² = 0.1 it is 1.05, 1.11, 1.23, 1.31.
- *Caveat:* the β grid is coarse near 1 (0.80, 0.958, 1.19), which biases the noiseless ratios up by a few percent.

**P4 (static preconditioning): the exponent s = ½ recovers the isotropic gap exactly; s = 1 (full whitening) is as bad as none.**
- *Setup:* the key metric is M = Σ^(−s).
- *Result:* with s = ½, the ratio equals the isotropic value at every κ, drift, and noise level (within about 1% for qd ≤ 0.1; +2–7% at qd = 1). With s = 1, the ratio is 1.60, 2.04, 2.66 at κ = 10, 100, 1000 (noiseless, qd = 0.01).

**Unplanned: a square-root law explains P3 and P4 quantitatively.** (Derived after seeing the P4 numbers, then checked against every configuration.)
- *The argument.* Think of each token as one unit of "refresh" budget spread over directions. Direction i, with input variance λ_i, goes stale at a rate proportional to λ_i in risk units. Refreshed at rate r_i, it contributes ≈ λ_i / r_i to the risk. Minimizing Σ λ_i / r_i subject to Σ r_i fixed gives **r_i ∝ √λ_i**.
- *Consequences.* A key metric Σ^(−s) refreshes at r_i ∝ λ_i^(1−s), so its risk relative to the optimum is (E λ^s)(E λ^(1−s)) / (E λ^½)². By Cauchy–Schwarz this is ≥ 1, with equality at s = ½, and it is symmetric under s ↔ 1−s. The Kalman filter, being optimal, should also sit at the square-root allocation, so its risk should scale by (E λ^½)² relative to isotropic.
- *Checks* (noiseless, qd ≤ 0.1):
  - Kalman risk / isotropic: measured 0.902, 0.710, 0.544 against predicted 0.900, 0.705, 0.537 (κ = 10, 100, 1000).
  - s = 1 penalty: measured 1.112, 1.417, 1.845 against predicted 1.111, 1.419, 1.862.
  - s = ½: measured 0.993–1.004 against predicted 1.
- *Where the law fails:* with no preconditioning (s = 0) it overpredicts the penalty at large κ (1.63 measured vs 1.86 predicted at κ = 1000). There the weak directions are refreshed so rarely that their error saturates at the prior variance, which caps the damage. The Kalman scaling also breaks with noise (0.71 vs 0.54 at κ = 1000, σ² = 0.1).
- *Novelty* (checked afterwards; see `literature/novelty-sqrt-law.md`):
  - **Classical, in the noisy small-step regime:** optimal metric ∝ R^(−½) for Q ∝ I, the s ↔ 1−s symmetry (Eweda 1994), the penalty √[(E λ^s)(E λ^(1−s))]/E λ^½, and Kalman ∝ E λ^½ (Sayed 2008 formulas; P R P = σ²Q).
  - **Different in the noiseless projection regime (our result):** every factor is **squared**. The noisy σ² = 0.1 runs sit close to the classical law (s = 0 at κ = 100: 1.17 vs classical 1.19 vs noiseless 1.42), so noise sets the exponent.
  - **Not found:** exact preservation of the isotropic gap under s = ½, the Kalman constant, and the key-projection interpretation.

## 6. Interpretation

1. **A Kalman memory's advantage over a *well-configured* gated delta rule is a small constant on stationary input statistics:** ≤ 1.47× noiseless and ≤ 1.13× at σ² = 0.1, for any input anisotropy, *provided* the delta rule's key metric is Σ^(−½). The advantage does not grow with anisotropy.
2. **The right metric is Σ^(−½), not the Σ⁻¹ that "whitening" or "natural gradient" intuition suggests.** Under drift, a memory should spend refreshes in proportion to √(importance). This is known for noisy LMS; we find it also holds, with a larger (squared) penalty, in the noiseless projection regime relevant to exact recall.
3. **This is learnable.** With tied query/key projections A, a GDN's update is w ← w + β r AᵀA x / xᵀAᵀA x, so its key metric is M = AᵀA. A trained GDN can in principle sit exactly at the s = ½ allocation.
   - Prediction for trained models: under drift, learned AᵀA ∝ Σ^(−½).
   - Contrast with stationary one-shot in-context regression, where linear attention learns Γ⁻¹ (Zhang–Frei–Bartlett; reproduced in theory/08), i.e. s = 1.
   - So the learned exponent should move from 1 toward ½ as the task changes from "learn from scratch within the context" to "track a drifting task".
4. **Implication for Kalman-style linear attention** (Gated KalmaNet, Kalman Linear Attention, Kalman Delta Networks): on this model, their gains over a well-trained GDN must come from things a static key metric can't express. Candidates are key statistics that change within a sequence, the early-sequence transient, or failure of GDN training to find the right metric. All three are testable.

**Not yet checked:** the transient (P5; curves are saved in cache/iso.json but not analyzed); finer s grids; non-Gaussian keys; drift that isn't isotropic in w-space.

## 7. Takeaway
On stationary key statistics, a Kalman memory beats the best gated delta rule by at most ≈ 1.5×, at any anisotropy, as long as the delta rule's key metric follows a square-root law (Σ^(−½)), which a key projection can learn. (R-002, R-003.)
