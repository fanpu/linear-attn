# Novelty check: the square-root law and related claims from exp 001 (2026-09-15)

Scope: claims C1–C5 from `experiments/001-gdn-vs-kalman-theory/README.md` §5–6.

Method:
- General web search, plus the OpenAlex API for verbatim abstracts. IEEE Xplore, ScienceDirect, Springer and Wiley pages returned 403 or a login redirect.
- arXiv HTML pages were fetched directly.
- The Semantic Scholar, arXiv and Google Books APIs were rate-limited (429), so book full texts (Sayed 2008, Macchi 1995, Benveniste et al. 1990, Haykin) could **not** be searched.
- "Confirmed" below means the title, authors and venue were seen, and the quote comes from a fetched abstract or page. A statement not quoted is marked *(not accessed)*.

---

## C1. Optimal static metric under isotropic drift is Σ^(−1/2); penalty (Eλ^s)(Eλ^(1−s))/(Eλ^½)²; KF risk ∝ (Eλ^½)²

**Queries:** Macchi 1986 "Optimization of adaptive identification for time-varying filters"; `optimal step-size matrix LMS tracking random walk square root input correlation matrix`; Ljung Gunnarsson survey optimal gain matrix; Guo Ljung 1995 performance analysis; Eweda 1994 RLS LMS sign tracking; Benveniste 1987 design tracking; `"optimal gain" tracking LMS "R^{-1/2}"`; `Sayed tracking "Tr(R)Tr(Q)"`; Kalman random-walk steady-state `"P R P" = Q` slow variation; Gustafsson adaptive filtering LMS optimal step matrix; Aitchison Bayesian filtering root-mean-square; OpenAlex: `lag misadjustment LMS nonstationary`, `statistical efficiency LMS nonstationary`, `optimal gain matrix tracking slowly time-varying`.

**Closest prior work (confirmed)**
1. **E. Eweda, "Comparison of RLS, LMS, and sign algorithms for tracking randomly time-varying channels," IEEE TSP 42(11):2937–2944, 1994.** Regime: small step, noisy, random walk.
   - Abstract: "ξ does not depend on the spread of eigenvalues of the input covariance matrix, R, in the cases of the LMS algorithm and the SA, while it does in the case of the RLS algorithm … the minimum values of ξ and η attained by the RLS algorithm are equal to the ones attained by the LMS algorithm … (2) if the fluctuations of the individual elements of the optimal vector are mutually uncorrelated and have the same mean-square value."
   - This is exactly the s=0 ↔ s=1 symmetry at Q ∝ I, in the noisy regime.
2. **A. H. Sayed, *Adaptive Filters*, Wiley 2008** (tracking chapter), as restated in Table II of **Arenas-García et al., "Combinations of Adaptive Filters," arXiv 2112.12245**:
   - LMS: ζ(μ) = ½(μσ_v² Tr R + μ⁻¹ Tr Q), with ζ_o = √(σ_v² Tr R · Tr Q).
   - RLS: ζ(β) = ½(βσ_v² M + β⁻¹ Tr(QR)), with ζ_o = √(σ_v² M Tr(QR)).
   - Their text: "LMS will outperform RLS if Q is proportional to … R, and the opposite will occur when Q ∝ R⁻¹."
   - Regime: small step, noisy, energy-conservation approximations.
   - **Consequence (one line of algebra from these formulas, not stated there):** for a step matrix μM with M = R^(−s), ζ_o = σ_v √(Tr(R^(1−s)) · Tr(Q R^s)). With Q = (q/d)I this is minimised at s = ½ (Cauchy–Schwarz), with penalty **√[(Eλ^s)(Eλ^(1−s))]/Eλ^½**.
3. **Kalman steady state in the same regime** (standard; derived here, no verbatim source accessed). The averaged Riccati equation in the slow-drift limit gives P R P = σ²Q, so P ∝ R^(−1/2) when Q ∝ I. The gain is therefore the LMS gain with M ∝ R^(−1/2), and the KF excess risk is σ·tr((R^½QR^½)^½) ∝ **Eλ^½** (not squared).
   - Expected homes: **Ljung & Gunnarsson, "Adaptation and tracking in system identification—a survey," Automatica 26:7–22, 1990**, and **L. Guo & L. Ljung, "Performance analysis of general tracking algorithms," IEEE TAC 40(8):1388–1402, 1995**. Both exist (confirmed), but the optimal-gain formula inside them was *(not accessed)*.
   - Guo–Ljung abstract: "Approximate, and easy-to-use, expressions for the covariance matrix of the parameter tracking error are developed."
4. **O. Macchi, "Optimization of adaptive identification for time-varying filters," IEEE TAC 31(3):283–287, 1986.** Scalar gain μ only. Abstract: tracking MSE "results from the tradeoff between the gradient part which is μ-increasing and the lag contribution which is μ-decreasing … In two important cases the optimum is exact. One of these cases is 'slow-variations.'" No matrix gain in the abstract.
5. **B. Widrow & E. Walach, "On the statistical efficiency of the LMS algorithm with nonstationary inputs," IEEE TIT 30(2), 1984.** Compares LMS, "orthogonalized LMS" (R⁻¹) and exact LS by misadjustment; scalar step. Abstract does not mention R^(−1/2).
6. **A. Benveniste, "Design of adaptive algorithms for the tracking of time-varying systems," Int. J. Adapt. Control Signal Process. 1:3–29, 1987.** Tracking criterion plus design methodology for slow variation. Gain-matrix content *(not accessed)*.
7. **L. Aitchison, "Bayesian filtering unifies adaptive and non-adaptive neural network optimization methods," NeurIPS 2020 (arXiv 1807.07540).** Diagonal, noisy, ML-optimizer setting.
   - Abstract: natural-gradient methods "were unable to recover … a root-mean-square gradient normalizer, instead getting a mean-square normalizer. To recover the root-mean-square normalizer, we find it necessary to account for the temporal dynamics of all the other parameters."
   - This is the same "drift ⇒ square root, stationary ⇒ inverse" dichotomy, stated for Adam.
8. Also confirmed, less close: **Lindbom, Sternad & Ahlén, "Tracking of time-varying mobile radio channels, Part I: The Wiener LMS algorithm," IEEE TCOM 2001**, and **Sternad, Lindbom & Ahlén, "Wiener design of adaptation algorithms with time-invariant gains," IEEE TSP 2002**. Both optimize constant-gain adaptation laws for a steady-state error covariance.

**Cross-check against exp 001 data** (this supports the classical noisy law being the *square root* of ours):
- **Unpreconditioned delta rule at σ²=0.1** (noisy s=0 ratios normalised by the isotropic 1.05), κ = 10, 100, 1000:
  - measured: 1.06, 1.17, 1.25
  - classical noisy (√ of our penalty): 1.05, 1.19, 1.37
  - our noiseless law: 1.11, 1.42, 1.86
- **Kalman at κ=1000, σ²=0.1:** measured 0.71; classical Eλ^½ = 0.733; our noiseless (Eλ^½)² = 0.537.

**Verdict: known in a different regime.**
- *Known (noisy, small step):* optimal metric ∝ R^(−1/2) for Q ∝ I; the s ↔ 1−s symmetry (Eweda); the penalty √[(Eλ^s)(Eλ^(1−s))]/Eλ^½; KF ∝ Eλ^½.
- *What remains new:* the **noiseless / projection (β≈1) regime**, where the penalty is the **square** (Eλ^s)(Eλ^(1−s))/(Eλ^½)² and KF risk ∝ (Eλ^½)². This is plausibly because no noise term exists to trade against the step size.
- *Also new:* the exact preservation of the finite isotropic GDN/KF gap under s = ½, and the crossover between the two laws as σ² varies.
- The README's "novelty" note should say this explicitly. Do not claim the √ law itself.

---

## C2. Optimal leak α* equals the AR coefficient a, independent of noise

**Queries:** `leaky LMS OR leaky NLMS tracking "first-order Markov" OR "AR(1)" optimal leakage factor equals Markov parameter`; OpenAlex: `leaky LMS tracking first-order Markov time-varying system`, `LMS algorithm tracking first order Markov model parameter a analysis`.

**Closest prior work**
- **Z. Qin, J. Jiang, Z. Zhu, "Learning to Adapt: In-Context Learning Beyond Stationarity," arXiv 2604.10946 (Apr 2026).** GLA with a global forget factor λ, AR(1) weights w_i = γw_{i−1} + e_i. It explicitly does **not** get λ* = γ: "it does not necessarily imply that choosing λ=γ minimizes the recovery error."
- For the Kalman filter, using a in the prediction step is textbook. No LMS/NLMS paper stating α* = a was found.

**Own check** (closed form in `theory.py`, grid-refined to ~1e-10):
- At the **joint** optimum (α*, β*): |α* − a| ≤ 2e-8 for d ∈ {32, 128}, qd ∈ {0.1, 1, 10}, σ² ∈ {0, 0.1, 1}. So this is a sharp, noise-independent identity, not a small-q artefact.
- **Caveat:** at a *fixed, non-optimal* β (e.g. β = 0.1 or 1.7 at d=32, qd=10), α* moves ≥ 0.05 away from a. The identity is a property of the joint optimum (and of β=1, σ=0), not of α alone. The README wording should be tightened.

**Verdict: not found** for the leaky NLMS (a joint-optimum identity). Low standalone significance. It is worth proving, since the obvious GLA analogue is explicitly reported as false.

---

## C3. Exact non-small-step steady-state MSD of leaky NLMS, Gaussian regressors, drift (two-moment recursion, E‖x‖⁻² = 1/(d−2))

**Queries:** Slock 1993 NLMS convergence; Tarrab Feuer NLMS uncorrelated Gaussian; Bershad NLMS Gaussian inputs; Rupp 1993 spherically invariant; `normalized LMS tracking analysis random walk nonstationary exact Gaussian`; `Mean-square Analysis of the NLMS Algorithm`.

**Closest prior work (confirmed)**
- **M. Tarrab & A. Feuer, "Convergence and performance analysis of the normalized LMS algorithm with uncorrelated Gaussian data," IEEE TIT 34(4), 1988.** "a comprehensive study of the first and second-order behavior in the NLMS algorithm." Stationary, white Gaussian, exact.
- **D. T. M. Slock, "On the convergence behavior of the LMS and the normalized LMS algorithms," IEEE TSP 41(9):2811–2825, 1993.** Proposes a simple input model (radial × direction) that makes NLMS analysis tractable, including the optimal step sequence for white input. Stationary.
- **M. Rupp, "The behavior of LMS and NLMS algorithms in the presence of spherically invariant processes," IEEE TSP 41(3), 1993.** First- and second-order moments of NLMS with SIRP inputs. Stationary.
- **N. J. Bershad, "Analysis of the normalized LMS algorithm with Gaussian inputs," IEEE TASSP 34(4):793–806, 1986** (confirmed via citation in Chan & Zhou, J. Signal Process. Syst. 2009).
- **T. Y. Al-Naffouri, M. Moinuddin, A. Ali, "Mean-square Analysis of the NLMS Algorithm," arXiv 2108.03721.** Closed forms for colored Gaussian inputs covering "transient, steady-state, and tracking mean-square behavior"; the drift model was not visible in the abstract.
- Sayed 2008: NLMS tracking EMSE via energy conservation *(not accessed; approximate)*.

**Verdict: known in a different regime.**
- Exact spherical-symmetry NLMS analysis is classical for the stationary case, and tracking analyses exist but are approximate or use a pure random walk.
- The leak + AR(1) cross-moment c = E[wᵀe] is a routine extension. It is not a novelty claim; cite Tarrab–Feuer / Slock.

---

## C4. High-d constant for noiseless KF with random isotropic rank-one observations under drift (≈ 0.68·qd; ratio ≈ 1.465)

**Queries:** `Kalman filter random rank-one measurements high dimension random Riccati equation steady-state error covariance spectrum free probability`; `Vakili Hassibi Stieltjes transform random Riccati recursion`; `randomized Kaczmarz time-varying linear system tracking random walk`; OpenAlex: `exponentially weighted RLS large dimensional random matrix steady state tracking`, `Kalman filter Riccati random regressors large system limit eigenvalue distribution`.

**Closest prior work (confirmed)**
- **A. Vakili & B. Hassibi, "A Stieltjes transform approach for studying the steady-state behavior of random Lyapunov and Riccati recursions," IEEE CDC 2008.** "we do obtain explicit formula for the asymptotic eigendistribution of certain classes of Lyapunov and Riccati recursions … we have not yet developed a full theory."
- **Vakili & Hassibi, "On the steady-state performance of Kalman filtering with intermittent observations for stable systems," IEEE CDC 2009.** "For systems with a stable system matrix and i.i.d. time-varying measurement matrices, we obtain explicit equations that allow one to compute the asymptotic eigendistribution of the error covariance matrix."
  - Our system A = aI is stable, so this is the right toolkit.
  - Their measurement dimension per step is not rank-one-with-σ=0 in the abstracts, so the constant is not given there.
- **Vakili & Hassibi, "A Stieltjes transform approach for analyzing the RLS adaptive filter," Allerton 2008.** Eigendistribution of the RLS error covariance for white regressors (exponentially weighted, no drift).
- **Kar, Sinopoli & Moura, "Kalman filtering with intermittent observations: weak convergence to a stationary distribution," IEEE TAC 2011.** Random Riccati equation, Bernoulli arrivals. Qualitative only.
- Randomized Kaczmarz: only static or noisy-system results were found (e.g. arXiv 2403.19874). No drift-tracking constants.

**Own observation:** the deterministic-equivalent (mean-field) Riccati P ← a²(P − P²/trP) + (q/d)I has the isotropic fixed point tr P = qd/(1+qd). That is exactly the NLMS value. The entire ≈1.465× KF advantage therefore comes from the **fluctuation-induced spread of P's spectrum**, which averaged (Guo–Ljung-type) analyses discard.

**Verdict: not found.** Vakili–Hassibi is the method to adapt: a Stieltjes fixed point for the stationary spectral law of P under rank-one updates in the d→∞, qd-fixed scaling.

---

## C5. ML statement that learned key metric under drift should be Σ^(−1/2) (vs Σ⁻¹ stationary)

**Queries:** `linear attention DeltaNet key projection learned preconditioner drifting in-context regression inverse square root covariance`; `in-context learning non-stationary drifting linear regression transformer theory optimal preconditioner`; `gated delta rule Kalman filter linear attention comparison tracking random walk theory`; `online least squares drifting target optimal preconditioner inverse square root`.

**Closest prior work (confirmed; each fetched and checked for Σ^(−1/2))**
- **Ahn, Cheng, Daneshmand, Sra, arXiv 2306.00297**, and **Zhang, Frei, Bartlett, "Trained Transformers Learn Linear Models In-Context," JMLR 25** (2306.09927): stationary, learned preconditioner ∝ inverse covariance.
- **Qin, Jiang, Zhu, arXiv 2604.10946:** AR(1) drift, GLA. Learned W_KQ(∞) ∝ Λ̃⁻¹ (inverse of an *effective* covariance), not Λ^(−1/2).
- **Preconditioned DeltaNet, arXiv 2604.21100:** diagonal Gram-inverse preconditioner, fixed least-squares objective, no drift analysis.
- **OSDN (Zhou, Li, Liu, Lin), arXiv 2605.13473:** comparator D⋆ = Σ_k†, plus a learned "Adaptive Preconditioner Forgetting". No square-root analysis.
- **Kalman Delta Networks, arXiv 2609.07816:** delta rules are "fixed-gain approximations" of the KF. No Σ^(−1/2) and no constant-gap result.
- **Gated KalmaNet, arXiv 2511.21016** (Peng et al.): exists (confirmed); not analysed for this.
- **"Adapt or Forget: Provable Tradeoffs Between Adam and SGD in Nonstationary Optimization," arXiv 2605.04269:** random-walk target, Adam vs SGD bounds. It does not derive an optimal preconditioner.
- **Aitchison 2020** (C1 item 7) is the nearest *ML* statement of "drift ⇒ root-mean-square scaling": diagonal, optimizer, not attention.

**Verdict: not found** for linear attention / delta-rule key metrics. The idea has a clear ancestor in Aitchison (optimizers) and in classical LMS tracking (C1).

---

## Summary: what remains novel
1. The Σ^(−1/2) metric and the s ↔ 1−s symmetry are **classical in the noisy, small-step regime** (Eweda 1994; Sayed 2008 formulas; KF Riccati P R P = σ²Q). There the penalty is the square root of ours and KF ∝ Eλ^½; our σ²=0.1 data match this.
2. New: the **noiseless/projection regime law**, where the penalty (Eλ^s)(Eλ^(1−s))/(Eλ^½)² and the KF (Eλ^½)² scaling are squared, together with the noise-driven crossover between the two laws.
3. New: under s = ½ the finite isotropic GDN/KF gap (≈1.465 noiseless) is preserved at all κ. So "Kalman ≤ ~1.5× a well-preconditioned GDN" holds on stationary key statistics.
4. Open, and apparently unknown: the high-d noiseless KF constant (≈0.68·qd). It arises purely from spectral fluctuations of P; Vakili–Hassibi Stieltjes methods are the route.
5. Not found in ML: key projections as a learned Σ^(−1/2) metric under drift (vs Σ⁻¹ stationary), and α* = a at the joint optimum for leaky NLMS. Both should cite the classical and Aitchison ancestry.
