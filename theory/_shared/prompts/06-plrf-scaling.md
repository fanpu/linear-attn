You are the agent for **theory project #6: "A solvable scaling law: power-law random features"**. Your directory is `/home/fzeng/ml/research/theory/06-plrf-scaling/` (create it).

{{HEADER}}

## Project-specific guidance (suggestions, not a script — use your judgment)

**Framing for the reader.** Why does loss fall as a clean power law in model size, data, and compute? And why does the compute-optimal model size itself scale as a power of compute (the "Chinchilla" question)? A surprisingly simple solvable model reproduces all of this. Linear regression on random features of data with power-law spectra is enough, and the exponents follow from two numbers describing the data.

**Reproduction:**
- **Main paper:** Paquette, Paquette, Xiao & Pennington 2024, "4+3 phases of compute-optimal neural scaling laws". Read it carefully; the setup, parameter names (α, β, v, d), phase boundaries, and predicted exponents must match the paper, not the ideas doc.
- **Implementation:** PLRF with large ambient dimension v (10⁴–10⁵), one-pass SGD, sweeps over d and steps.
- **Analysis:** build IsoFLOP curves and the compute-optimal frontier. Check the fitted exponents against the predictions at one (α, β) point in each of 3–4 phases, paying attention to finite-size effects near phase boundaries.
- **Theory curves:** if the paper gives a deterministic-equivalent (e.g. Volterra-equation) loss curve that is implementable, overlay full theoretical loss curves, not just exponents. If it's too hard, compare exponents and say so.
- **Context:** Bahri et al. 2021, Maloney, Roberts & Sully 2022, and Bordelon, Atanasov & Pehlevan 2024. Explain briefly how they relate.

**Visual ideas** (pick, improve, or replace):
- **Hero animation:** loss-vs-compute curves for many model sizes d drawn one after another on log-log axes. Their lower envelope, the compute-optimal frontier, emerges as a glowing line. The frontier's slope is the exponent.
- **Phase-diagram plate:** the (α, β) plane with phases colored and the predicted compute-optimal exponent as a continuous field. Mark measured points and their fitted exponents. Make it a beautiful scientific plate.
- **Spectra illustration:** "what α and β mean" — data covariance eigenvalues and target coefficients as power-law bar spectra, with the random-feature projection dropping the tail.
- **Loss-curve anatomy:** decompose loss into its components (e.g. approximation limit, SGD noise, bias). Show which dominates in each phase, perhaps as a stacked-area animation over training.
- **Widget:** click a point in the (α, β) phase diagram to see the predicted loss curves, the compute-optimal frontier, and (where you ran it) the measured curves. Closed-form exponents can be computed live in JS.
- **Widget:** compute-budget slider. Pick a FLOP budget and see the IsoFLOP curve over d with the optimal d marked.

**Build on (section 6): can PLRF predict a *real* small model's scaling exponent?**
- **Estimate α and β from real data.** For example:
  - eigenspectrum of CIFAR-10 pixel/patch covariance or of frozen random-feature/embedding covariance
  - the target's projection onto the eigenvectors, e.g. one-vs-rest labels
- **Measure:** train small MLPs or other simple models across sizes on that data and measure the compute-optimal exponent.
- **Compare:** set it against the PLRF prediction. When it fails, identify which assumption broke: feature learning, multiple epochs, or non-Gaussianity.

**Budget:** PLRF sweeps are matrix-heavy but small. Keep the total to a few GPU-hours through the slot limiter; float32 is likely fine for the sweeps, but check against float64 on a small case. Datasets are at `/home/fzeng/ml/research/art/data`.

Commit at checkpoints (PLRF implementation + tests; IsoFLOP reproduction; phase plate; draft; widgets; build-on; polish).

Finish with the final report described in the brief.
