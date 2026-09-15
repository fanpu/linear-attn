# Results ledger

One entry per concluded experiment, **positive or negative**, newest first. This is the single place to answer "what do we actually know?"

Each entry follows this format:

```markdown
### R-NNN · <claim, stated as a finding> · <confidence>
- **Date / experiment:** YYYY-MM-DD · experiments/NNN-slug/
- **Question:** what was asked, in one sentence.
- **Result:** the numbers (mean ± spread, n seeds, how many runs were tried in total).
- **Figure:** experiments/NNN-slug/figures/xxx.png
- **Caveats:** what could make this wrong; what wasn't tested.
- **Takeaway:** one plain sentence a non-specialist could repeat.
```

**Confidence levels** (defined in `CHARTER.md` §7):
- **confirmed**: ≥3 seeds, controls done, red-teamed.
- **likely**: consistent evidence, not yet stress-tested.
- **preliminary**: one run, or one setting.
- **refuted**: tested and false.

---

### R-004 · For a delta-rule memory, the best key metric exponent is ≈ ½ at long contexts even without drift, and smaller for short contexts. Pre-registered "s\* ≈ 1 when stationary" was refuted · likely
- **Date / experiment:** 2026-09-15 · `experiments/002-optimal-key-metric-vs-context/`
- **Question:** With the loss averaged over a context of length T (what in-context training optimizes), which exponent s in M = Σ^(−s) is best, as a function of T and drift?
- **Result:**
  - d = 32, κ = 100, best (α, β) per s.
  - No drift: s\* = 0.18, 0.28, 0.36, 0.43, 0.48, 0.48 for T/d = ½, 1, 2, 4, 16, 64 (±0.01–0.06).
  - With noise, similar but lower at short T.
  - qd = 0.1: s\* → 0.45 at long T. qd = 1: s\* → 0.3.
  - A budget model predicts the long-T limit ½ but fails quantitatively for T ≤ 2d.
- **Figure:** not yet.
- **Caveats:** one κ, one d, a single seed with 8 chunks. An idealized NLMS with a static metric, not a trained network. The short-T theory is not yet understood.
- **Takeaway:** a memory that corrects itself (delta rule) should tilt its attention toward important directions only by their square root, while a memory that just adds things up (linear attention) needs full whitening. So the two architectures should learn different key projections on the same task.

### R-003 · Square-root law: under drift, the best key metric for a delta-rule memory is Σ^(−½), and with it a Kalman memory's advantage stays at the isotropic value at any anisotropy · likely
- **Date / experiment:** 2026-09-15 · `experiments/001-gdn-vs-kalman-theory/` (sweep_aniso)
- **Question:** With anisotropic inputs (condition number κ up to 1000), does a Kalman memory's advantage over the best gated delta rule grow, and can a static key metric Σ^(−s) remove the growth?
- **Result:**
  - No preconditioning (s = 0): the noiseless ratio grows slowly, 1.44 → 2.35 over κ = 1 → 1000. That fails the pre-registered 1.5×-at-κ = 100 criterion.
  - s = ½: the ratio equals the isotropic value to about 1% at every κ, drift (qd ≤ 0.1), and noise level.
  - s = 1 (full whitening): the ratio reaches 2.66 at κ = 1000.
  - A budget argument (minimize Σ λ_i/r_i with Σ r_i fixed ⇒ r_i ∝ √λ_i) predicts a penalty of (E λ^s)(E λ^(1−s))/(E λ^½)² and a Kalman risk scaling of (E λ^½)². In the noiseless regime these match within about 1–2% (s = ½, s = 1, and Kalman); s = 0 is lower than predicted at large κ, because weak directions saturate. 24 configurations × 3 values of s; d = 64; 256 sequences; 1 seed.
- **Figure:** not yet.
- **Caveats:**
  - A coarse β grid near 1 (a few % bias).
  - Only a log-uniform spectrum, Gaussian inputs, and isotropic drift in weight space.
  - The √ form likely has classical roots (steady-state Riccati for random-walk tracking). The novelty check isn't done.
  - The law was derived after seeing the data, then checked on all configurations; it was not pre-registered.
- **Takeaway:** a memory that tracks a changing task should refresh each direction in proportion to the square root of its importance. A linear-attention key projection can learn that, and once it does, uncertainty tracking buys only a small constant factor.

### R-002 · Under isotropic drift, a Kalman memory beats the best gated delta rule by at most ≈ 1.47×, and by ≤ 1.13× once there is noise · likely
- **Date / experiment:** 2026-09-15 · `experiments/001-gdn-vs-kalman-theory/` (sweep_iso, sweep_dscale)
- **Question:** On in-context regression whose weights drift (q per token, dimension d, noise σ²), how much lower is the Kalman filter's steady-state excess risk than that of the best-tuned gated delta rule (NLMS form)?
- **Result:**
  - The delta rule's steady-state risk has an exact closed form (two-moment recursion). It matches float64 simulation within 0.5% on 42 configurations.
  - The optimal forget gate is α* = √(1−q), the true task persistence, to 4–5 digits.
  - Ratio (delta rule / Kalman) at d = 128:
    - noiseless: 1.46 (qd = 0.01) → 1.14 (qd = 1) → 1.00 (qd = 10);
    - σ² = 0.1: at most 1.13;
    - σ² = 1: at most 1.04.
  - The noiseless small-drift ratio grows with d and levels off at 1.465 ± 0.004 (d = 512–1024). Batch of 16–512 sequences, 8 chunks for error bars, 1 seed per config. 63 configurations in total; one earlier run was discarded because its runs were too short (see the experiment README).
- **Figure:** not yet.
- **Caveats:**
  - Isotropic Gaussian inputs, a known drift model, and a Kalman filter given the *true* q and σ², so this is an upper bound on what a Kalman memory can gain here.
  - The NLMS form assumes a value/key scaling that a trained GDN must learn.
  - The limiting constant (Kalman error ≈ 0.68·qd) is not derived. My "age model" guess (2/π) was wrong.
- **Takeaway:** when keys look like random directions, tracking the memory's uncertainty buys at most ~1.5× and usually only a few percent. Any big win for Kalman-style linear attention has to come from something else: structured keys, noise-free recall, or the start of a sequence.

### R-001 · The fla chunk kernel's float32 error does not grow with sequence length (kills S5) · likely
- **Date / experiment:** 2026-09-15 · `common/test_common.py`, plus Fan Pu's `day1/results.txt`
- **Question:** Does rounding error in the chunked (WY / UT-transform) delta-rule kernel accumulate along the sequence, as seed S5 predicted?
- **Result:**
  - Fan Pu's day-1 table (Gated DeltaNet, float32 with IEEE matmul): the max relative forward error is 0.7–1.3×10⁻⁴ at T = 256, 1024, and 4096 for every head size. It's flat in T.
  - My check (T = 64–512, D = 16–128) agrees. Chunk kernel: 5×10⁻⁵ (gated) to 6×10⁻⁴ (ungated) relative error. Recurrent kernel: ~10⁻⁶. Chunk with Triton's default TF32 matmul: ~2×10⁻³.
  - fla forces IEEE matmul only on pre-Ampere cards, so on the GB10 you must set `TRITON_F32_DEFAULT=ieee` yourself. Fan Pu's day 1 had already found this too.
- **Figure:** none. The numbers are in `day1/results.txt` and the journal.
- **Caveats:** only random inputs, not trained weights, where transitions could be closer to singular. Only T ≤ 4096. bf16 is not covered by my check.
- **Takeaway:** the training kernel is a little less precise than the step-by-step one, but the error is a fixed floor rather than something that snowballs with context length. There's no paper in it.
