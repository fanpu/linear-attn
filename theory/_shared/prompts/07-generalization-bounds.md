You are the agent for **theory project #7: "Non-vacuous generalization bounds and the 'rethinking generalization' puzzle"**. Your directory is `/home/fzeng/ml/research/theory/07-generalization-bounds/` (create it).

{{HEADER}}

## Project-specific guidance (suggestions, not a script — use your judgment)

**Framing for the reader.** A network that can perfectly memorize random labels somehow generalizes on real labels. So any explanation of generalization based only on "how big is the model class" is doomed. PAC-Bayes gives a different kind of answer: a *certificate*. If the solution survives a lot of weight noise and stays close to initialization, then the test error provably can't be much worse than training error, and the bound is computable with real numbers.

**Reproductions** (read the papers; use their exact setups and compare numbers):
- **Zhang, Bengio, Hardt, Recht & Vinyals 2017.** Small CNN on a CIFAR-10 subset. Sweep label corruption 0→1. Plot time-to-memorize and test error vs corruption. Optionally include shuffled pixels / Gaussian inputs.
- **Dziugaite & Roy 2017.** Binary MNIST (verify the exact label split and architectures, e.g. 1 hidden layer of 600 units).
  - SGD-train the net.
  - Optimize the PAC-Bayes-kl bound over a diagonal Gaussian posterior with the prior centered at init. Take the prior variance over a discrete grid with a union bound, as they do.
  - Report the bound for true and random labels; compare with their reported numbers.
  - Implement the kl-inverse numerically (bisection) and test it.

**Visual ideas** (pick, improve, or replace):
- **Hero animation:** two identical networks trained side by side, real labels vs random labels.
  - Train accuracy climbs for both; random-label learning is slower.
  - Test accuracy: one soars, one flatlines at chance.
  - Show sample images with their (random) labels flickering.
  - End on the PAC-Bayes certificate: non-vacuous for one, vacuous for the other.
- **"Noise-robustness" picture:** a 2D-input toy classifier. Sample many weight perturbations from the optimized posterior and draw their decision boundaries as a translucent cloud. Flat, wide solutions give a tight cloud; memorizing ones give a shattered cloud. Beautiful, and it gets at the core intuition.
- **Bound anatomy:** a stacked bar or waterfall from empirical error → + KL term → kl-inverse → final bound, compared with actual test error.
- **Posterior-optimization animation:** σ per weight growing (weights that can tolerate noise), while the bound value descends.
- **Widget:** PAC-Bayes-kl calculator in JS. Sliders for n, KL, empirical error, and δ; live kl-inverse bound, compared with the looser square-root (McAllester) form.
- **Widget:** generalization-measure explorer for the build-on sweep. Dropdown to choose a measure; scatter of measure vs actual gap; Kendall τ shown; points colored by the hyperparameter varied.

**Build on (section 7): do generalization measures predict anything at small scale?**
- **Sweep:** ~100 cheap configurations (width, depth, LR, batch size, weight decay, dropout, data size) on a small dataset.
- **Measures:** ~10, e.g. spectral-norm products, path norm, Frobenius distance to init, margin-normalized norms, PAC-Bayes sharpness/flatness, and the optimized bound itself.
- **Evaluation:** Jiang et al. 2020 "Fantastic generalization measures and where to find them" (Kendall τ, granulated τ, conditional-independence tests) and Dziugaite et al. 2020 "In search of robust measures of generalization" (worst case over environments).
- **Report:** which measures survive.
- **Optional:** tighter bounds with data-dependent priors (Pérez-Ortiz et al. 2021). Verify the reference.

**Budget:** each run takes minutes; the whole sweep plus bound optimization should be a few GPU-hours through the slot limiter. Pack many small runs per slot. Datasets are at `/home/fzeng/ml/research/art/data`.

Commit at checkpoints (random labels; PAC-Bayes bound matched; hero; sweep; draft; widgets; polish).

Finish with the final report described in the brief.
