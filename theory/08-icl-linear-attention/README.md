# 08 — In-context regression with linear attention

The deliverable is the blog post `post.md`, rendered to `post.html`. It reproduces Zhang–Frei–Bartlett (one-layer linear self-attention converges to preconditioned GD with preconditioner Γ⁻¹), von Oswald / Ahn (depth ↔ GD steps), and Raventós et al. (the task-diversity transition). It then compares softmax attention, linear attention, DeltaNet, and Gated DeltaNet on in-context regression.

**Prediction vs measurement (short version, details in the post):**
- The exact population gradient flow from ZFB's initialization reaches W* to about 1e-11. Adam on minibatches gets close for isotropic inputs; for correlated inputs it is slow in low-variance directions.
- The trained layer's risk vs context length and under covariate scaling matches the closed form. On random-covariance training the error plateaus at the closed-form value.
- Deep linear attention beats k-step GD. The noiseless proportional-limit GD-k optimum is γ^k(1−γ)/(1−γ^{k+1}).

**Layout**
- `icl_core.py`, `seqmodels.py`: math, baselines, and token mixers. Tests are `test_core.py` and `test_seqmodels.py`.
- Compute (writes to `cache/`, gitignored): `lsa_gradflow.py`, `train_lsa1.py`, `eval_lsa1.py`, `train_lsa_deep.py`, `taskdiv.py`, `train_seq.py` (+ `run_seq_grid.sh`), `analysis_seq.py`.
- Render (writes to `figures/`, `widgets/data_*.js`): `render_hero.py`, `render_think.py`, `render_mechanism.py`, `render_repro.py`, `render_taskdiv.py`, `render_seq.py`, `build_widget_data.py`. Shared styling is in `style.py`.
- Widgets: `widgets/explorer*.js` (live Marchenko–Pastur closed forms plus trained points) and `widgets/delta_widget.js` (live delta-rule vs Hebbian memory). Open `post.html#selftest` to run their self-tests in the console.

**Reproduce:** see the "Reproduce it" section at the end of `post.md`. Render the post with `../.venv/bin/python ../_shared/render_post.py post.md --shot`.
