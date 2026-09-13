You are the agent for **theory project #1: "Double descent where the answer is exact"**. Your directory is `/home/fzeng/ml/research/theory/01-double-descent/` (create it).

{{HEADER}}

## Project-specific guidance (suggestions, not a script — use your judgment)

**Framing for the reader.** Classical statistics says more parameters mean more overfitting: a U-shaped curve. Modern ML happily uses more parameters than data. Both are right, because the test-error curve has a *second* descent. At the interpolation threshold the model is forced to fit noise with a barely-possible solution, and that is exactly where things blow up. In the simplest settings this is *exactly* computable.

**Reproductions** (read the papers; verify the formulas before plotting):
- **Hastie, Montanari, Rosset & Tibshirani, "Surprises in high-dimensional ridgeless least squares interpolation".**
  - Isotropic linear model: closed-form ridgeless risk, and the ridge version via Marchenko–Pastur.
  - Measured points, many seeds, error bars shrinking onto the curve as n grows.
- **Mei & Montanari 2019, random-features regression.**
  - The asymptotic curve, or the Gaussian-equivalent linear model (Hu & Lu 2022 or similar; verify).
- **Nakkiran, Venkat, Kakade & Ma 2021.** Optimally tuned ridge removes the peak.
- **Bias/variance decomposition.** Show that the peak is a variance explosion; its source is the smallest singular value of the feature matrix going to 0 at p = n. Directly visualizing the spectrum of the feature matrix near p = n (Marchenko–Pastur edge hitting zero) is a great teaching figure.

**Visual ideas** (pick, improve, or replace):
- **Hero animation:** 1D regression with random ReLU / Fourier features fit to ~20 noisy points.
  - As the number of features p sweeps up through n, the min-norm interpolant goes from smooth to wildly oscillating at p ≈ n, then calms down again as p → ∞.
  - Synchronized: a dot tracing the test-risk curve below.
  - Intuitive and striking.
- **Theory curve overlays:** closed-form curves overlaid on measured risk across SNRs, drawn as an elegant family of curves.
- **Marchenko–Pastur histogram animation:** as γ → 1 the lower edge touches zero, and 1/λ_min explodes.
- **Deep double descent heatmaps:** width × epochs test error (Nakkiran et al. 2019 style), for small CNNs on a CIFAR-10 subset with label noise.
- **Widget:** sliders for γ, SNR, and ridge λ, with the live closed-form risk curve in JS (Marchenko–Pastur Stieltjes transform is implementable in JS) and bias/variance split.
- **Widget:** live 1D random-features fitting in JS. The user drags a feature-count slider (and maybe ridge) and watches the interpolant. Small linear algebra, e.g. p ≤ 200, is fine in the browser.

**Build on (section 1):**
- **(a)** Model-wise and epoch-wise deep double descent at small scale:
  - 4-layer CNN, width k = 1…64, 10k CIFAR-10 subset, 15–20% label noise.
  - Test whether the "effective model complexity" threshold predicts the peak location *quantitatively*.
- **(b) and/or:** does the peak move under anisotropic (power-law) covariance or misspecification, and does the RF formula still track trained two-layer nets once the first layer learns?

**Budget:** the linear/RF parts take minutes. Cap the CNN sweep at a few GPU-hours through the slot limiter: fewer widths/epochs, one seed, run in parallel within a slot. Datasets are at `/home/fzeng/ml/research/art/data`.

Commit at checkpoints (closed-form reproductions; hero; CNN sweep; draft; widgets; polish).

Finish with the final report described in the brief.
