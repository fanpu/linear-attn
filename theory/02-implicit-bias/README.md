# 02 · Implicit bias: which minimum does the optimizer pick?

A blog post (`post.md` → `post.html`) that reproduces four classic implicit-bias results and then measures the bias of modern optimizers.

| | prediction | measurement |
|---|---|---|
| Soudry et al. 2018 | $w(t)=\hat w\log t+\tilde w$, angle $=O(1/\log t)$ | 2D: residual matches the SVM-dual $\tilde w$ to 1e-8 at $t=10^{100}$; d=50: angle·ln t flattens |
| Gunasekar et al. 2018 | steepest descent → max-margin in its own norm | sign GD → L∞, coordinate descent → L1, NGD → L2 |
| Woodworth et al. 2020 | GF on $u^2-v^2$ → $\arg\min Q_\alpha$ | agreement to 1e-8 over 7 decades of α; discrete GD deviates by O(η) |
| Arora et al. 2019 / Razin & Cohen 2020 | σ-ODE; norms diverge while rank → 1 | ODE ratio 1.0000 (depth 2); float64 eventually breaks det>0 |
| **new**: Adam | L∞ (ε=0) vs L2 (ε>0) | L∞ phase lasts t× ≈ ln(1/ε)/min(γ∞·lr,(1−β₂)/2) steps |
| **new**: Muon / spectral descent | spectral-norm max margin | exact spectral descent & exact-polar Muon reach 0.9985 of it; Newton–Schulz Muon stalls at 0.92–0.93 (NS maps singular values into 0.68–1.13, not 1) |

**Reproduce** (CPU only, `OMP_NUM_THREADS=1`, from this directory, python = `../.venv/bin/python`):
compute scripts write `cache/*.npz` (gitignored); render scripts write `figures/`.

```
python compute_soudry.py hero; python compute_soudry.py gd 1e8; for m in ngd ngd_sqrt sign; do python compute_soudry.py $m 1e7; done
python compute_diag.py; python compute_matrix.py razin; (see post "Reproduce it" for the full list)
python compute_adam.py ...; python compute_spectral.py 1e6; python compute_ns.py 1e6
python render_hero.py; python render_soudry.py; python render_diag.py; python render_matrix.py; python render_adam.py; python render_spectral.py
python export_widgets.py; python test_core.py
../.venv/bin/python ../_shared/render_post.py post.md --shot
```

Layout: `core.py` (math), `compute_*.py` (measurements), `render_*.py` (figures), `widgets/` (JS + data), `style.py` (the shared visual system).
