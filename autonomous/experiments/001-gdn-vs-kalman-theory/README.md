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
*(to be written)*

## 6. Interpretation
*(to be written)*

## 7. Takeaway
*(to be written)*
