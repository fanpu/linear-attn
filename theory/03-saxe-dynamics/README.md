# 03 — Stagewise learning and saddle-to-saddle dynamics

Blog post: `post.md` (rendered: `post.html`). Reproduces Saxe, McClelland & Ganguli (2014) for deep linear networks,
breaks its assumptions one at a time, and builds on it with a softmax attention head.

**Prediction.** From small init each singular mode of Σyx follows u(t) = s e^{2st}/(e^{2st} − 1 + s/u0), t½ = ln(s/u0 − 1)/2s.
**Measurement.** Decoupled init, float64 GD: max |u − theory| = 3.5e-4·s (∝ learning rate). Random small init: strong modes
within 0.02%, weak modes 2–5% late (mode competition). Break sweeps in `figures/breaks.png`, `figures/depth.png`.
**Build-on.** One softmax attention head on y = M x₁: OV phase ∝ s^-0.86 (Saxe with log), attention phase ∝ s^-1.96;
a 4-vector reduced model started from the same init predicts the attention transition within 3.5%.

Layout
- `saxe_core.py` closed forms, datasets, batched float64 GD; `test_core.py` tests
- `compute_reproduce.py`, `compute_breaks.py` (CPU) and `compute_attention.py` (GPU, via `_shared/gpu_run.sh`), `attn_reduced.py`
- `render_*.py` read `cache/*.npz` and write `figures/`; `style.py` holds the shared palette
- `widgets/` hand-written canvas widgets (`#selftest` in the URL runs their self-checks)

Reproduce: see the "Reproduce it" section at the end of `post.md`. `cache/`, `_preview/`, `refs/` are gitignored.
