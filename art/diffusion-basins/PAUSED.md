# PAUSED: Which Dog (diffusion basins)

Stopped on coordinator request (token budget). All background jobs were killed. The GPU slots were full for most of the session, so almost nothing heavy has run yet.

## Done
- `common.py`: VP schedule, DDIM (Euler in sigma), RK4 probability-flow ODE, DDPM with frozen noise, great-sphere slice (exp-map coordinates, norm sqrt(d)), box counting, video writer.
- `toy.py`: GMM layouts (ring8/ringK, grid25, scatter12, tri3, pair2), exact Tweedie denoiser and its Jacobian (checked against autograd, error 4e-15), and the eps-MLP trainer. Trained so far: `cache/toy_ring8.pt` and `cache/toy_grid25.pt`. scatter12 still needs training.
- `toy_compute.py`: tasks `maps analytic|learned`, `steps`, `iter hero|gamma|learned`, `zoom <tag>`, `verify` (2048² box counting, resolution check, uncertainty exponent, circle and straight-ray nulls, Newton z^3-1 positive control). None of these tasks has run to completion.
- `mnist.py`: PatchUNet DDPM (1.28M params, about 4 it/s under contention), classifier, and a memorization model (40 images). Smoke-tested only; not trained.
- `mnist_compute.py`: great-sphere slice maps, step sweep, zoom, uncertainty exponent, and a float64 check. Not run.
- `render_common.py`, `render_toy.py`: palettes (validated), night/ink/riso styles, a Spectral split of sign(det J), animations, and verification plots. Not run; there is no data for them yet.
- Prototype findings (scratch, 256-320²): the analytic ring8 ODE and DDIM give exactly straight rays (symmetry null). The over-relaxed denoiser x <- x + g(D(x)-x) at sigma=0.4 is invertible for g <= ~1.06 and folds above that. Ring8 at g=2.10-2.12 (mode stability limit 2.125) gives Newton-like nested flowers. This is **not yet verified as fractal**. grid25 separates into rectangles, and tri3 is tame.

## Next (in order)
1. `gpu_run.sh python toy.py train scatter12`, then `mnist.py memo --steps 12000 --bs 64` and `mnist.py clf+ddpm --steps 15000 --bs 128`.
2. `toy_compute.py iter hero && verify && iter gamma`; `maps analytic && steps analytic && zoom ode_scatter12`; `zoom iter_ring8 iter_ring6`; then `maps learned`, `steps learned`, `iter learned`.
3. `render_toy.py atlas hero iter diptych steps gamma verify zoom:ode_scatter12 zoom:iter_ring8`; look at every PNG.
4. MNIST: `mnist_compute.py map ddpm hero 192 50`, `map memo memo_hero 192 50`, `map memo_exact memo_exact 192 50`, `stepsweep 128`, `zoom 128 4`, `uncert`, `f64check`; then write render_mnist.py (basin map, thumbnail mosaic, memorization cells).
5. README.md (brief format), link check, commit.

## Resume notes
- Cost under contention: an analytic GMM eval at 1024² takes about 0.24 s. Width-256 MLP float64 is about 15x slower than float32. The PatchUNet forward on a 2048 batch takes about 1.8 s.
- Brainstormed ideas: (1) smooth-flow atlas with stretch/FTLE; (2) over-relaxed denoiser gamma-sweep fractal; (3) MNIST digit atlas plus mosaic; (4) memorization cells vs the exact empirical-score map; (5) Spectral split of orientation sign(det J) (fold curves as seams); (6) whole-sphere globe projection. Build (2)+(5) and (3)+(4) first.
