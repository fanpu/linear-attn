# 01 — Double descent where the answer is exact

Blog post: `post.md` (rendered: `post.html`, via `_shared/render_post.py`). It reproduces the closed-form double-descent curves
for ridgeless and ridge regression (Hastie et al. 2022) and random ReLU features (Mei & Montanari 2022), checks that optimal ridge
removes the peak (Nakkiran et al. 2021), and builds on them in two ways: deep double descent in a small CNN family with a quantitative
test of the effective-model-complexity rule (Nakkiran et al. 2020), and frozen vs. trained two-layer networks.

**Prediction vs. measurement.**
- Isotropic ridgeless: measured risk (n = 400, 50 draws) within 0.3–0.5% (median) of the formula for |γ−1| > 0.25.
- Ridge: within the same accuracy, and λ* = σ²γ/r² is confirmed numerically.
- Random features: within 0.9% at d = 200; the peak height near N = n is underestimated at small d.
- CNN and two-layer results: see §6 and §7 of the post.

Layout
- `dd_core.py` holds the closed forms: MP Stieltjes, ridge/ridgeless risk, the general-Σ deterministic equivalent, and the Mei–Montanari equations. It also has the simulators; `test_core.py` tests them.
- CPU sweeps: `compute_linear.py`, `compute_rf.py`, `compute_aniso.py`, `compute_hero.py`, `compute_twolayer.py` write to `cache/`.
- GPU: `train_cnn.py`, launched by `run_cnn.sh` (4 processes in one `_shared/gpu_run.sh` slot, checkpointed and resumable); `cnn_data.py` loads the results.
- Rendering: `render_*.py` read `cache/` and write `figures/`; `style.py` is the shared palette.
- Widgets: `export_widgets.py` writes `widgets/data_*.js`; the widgets themselves are `widgets/{common,rf,risk}.js`. Open `post.html#selftest` to run their checks against Python.

Reproduce: see "Reproduce it" at the end of `post.md`. `cache/`, `_preview/`, `refs/`, `scratch/` are gitignored.
