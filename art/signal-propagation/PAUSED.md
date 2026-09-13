# PAUSED: signal-propagation (Order and Chaos + Finite Width)

Paused at a clean checkpoint on request (rate limits). No background jobs are running. `cache/` holds only toy/probe arrays, which are gitignored.

## Done

- **Paper verified.** arXiv:2508.03222 (D'Inverno, Hu, Davy, Unser, Rozza, Dong; v2 Jan 2026) exists. The code is at github.com/jon-dong/fractal-deep-info-prop; a copy was cloned to /tmp/fdip, which may be gone. Its actual setup:
  - erf MLP, z = σ_w W h/√N + σ_b b, with W~N(0,1), b~N(0,1) redrawn per layer.
  - Axes are (σ_w, σ_b) ∈ [0,4]², not squared. D=1000; the text says N=1000.
  - **The same random draw is reused for all pixels (CRN).**
  - The measured quantity is L = ‖x₁ᴰ−x₂ᴰ‖² for two independent unit-norm inputs, binarized at τ.
  - Box counting uses box sizes of 1–49 px on a single 1000² image (under 2 decades). The reported dimension is the **maximum over a τ grid** (logspace 1e-5..1), which biases it upward.
  - Reported dimensions: MLP ≈1.85, CNN ≈1.8, backprop ≈1.6. Their compute_frontier.py defaults also include residual connections and "struct" mode (not the paper MLP).
- **`sp_core.py`**: CRN per-layer seeded draws (identical for f32/f64, all zooms and resolutions); a compiled batched forward returning L_D, L_avg (last 20 layers), and t_hit; the closed-form mean-field erf L^(D) as the null model; boundary/box-count/zoom-centre helpers.
- **`compute_meanfield_tanh.py`**: q*, χ₁, c*, χ_c, ξ_q, ξ_c by quadrature. It subtracts the quadrature bias at c=1, which brought the fallback pixel count to 0. The critical point σ_b²=0.05 → σ_w²=1.7610 matches Schoenholz et al. The toy run looks good; the full run was not done.
- **`compute_empirical_tanh.py`**: toy works.
  - Ordered-phase ξ_c, and χ₁, c*, q* everywhere, match theory to within ~2%.
  - Chaotic-side ξ_c is biased low or high depending on the fit window, because of the finite-N noise floor (~0.03 at N=1000, K=1). An offline test found a better window: AB pair only, δ ≤ 0.3·δ₀, floor 2σ, giving a median ratio of 0.95 but 16% NaN. **Next step: put that into the script.** Use K=8 and consider dropping the AC pair.
- **Toy findings (128², N=100, D=1000):**
  - In the chaotic phase, float32 and float64 disagree by O(1) in L itself, because chaos amplifies roundoff.
  - The binarized frontier agrees to 0.01% for τ ∈ [1e-8, 1e-3].
  - The frontier shifts strongly at N=20 and approaches mean-field for N ≥ 300.
  - The L histogram is bimodal with a sparse band in between.
- **Zoom probes (f32, 256², auto-centred ×4 per level)** are cached for N=100 (9 levels) and N=50. The N=1000 run was killed. The probes have not been viewed yet.

## Next

1. View the probe chains to find the scale where the frontier stops refining (finite D makes L analytic in σ, so it should eventually look smooth). Decide the main width and depth for the zoom pieces.
2. Full runs, all via gpu_run in background:
   - `compute_meanfield_tanh.py --W 2400 --H 1200` (~20–40 min).
   - Empirical tanh at 480×240, N=1000, D=400, K=4–8, after fixing the fit window.
   - Width/seed maps with erf at [0,4]², 512², N ∈ {20,50,100,300,1000} × seeds {0,1,2} in f32. Validate one against f64.
   - A f64 zoom chain at 512–1024² native resolution, for box counting stitched across levels.
   - Resolution check: rerun the same windows at 2× and 4× via `--centers windows.json`.
   - Null model: `L_mf` is already saved per level.
   - Precision mask: f64 run with `--perturb 1e-13`.
3. Trainability check: tanh MNIST, σ_b²=0.05, a σ_w² × depth grid, overlaying 6ξ_c.
4. Renders (render_*.py, Spectral split primary via `color-research/palettes.py` `render_split`):
   - Phase plate.
   - Four-panel zoom plates showing zoom factor and dimension.
   - Deep-zoom MP4/GIF from native keyframes.
   - Correlation-flow animation.
   - Width-as-time animation.
   - At least 3 styles each.
   - Brainstorm at least 5 ideas and build 2.
   - Then the README, a link check, and the final report.

## Resume commands

```
cd /home/fzeng/ml/research/art/signal-propagation
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_meanfield_tanh.py --W 160 --H 80 --out cache/meanfield_tanh_toy.npz   # toy check (<1 min)
../_shared/gpu_run.sh $PY compute_meanfield_tanh.py --W 2400 --H 1200 > logs/meanfield_tanh.log 2>&1 &
../_shared/gpu_run.sh ./logs/probe.sh > logs/probe.log 2>&1 &    # N=100,50,1000 f32 zoom probes
```
