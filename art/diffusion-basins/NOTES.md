# NOTES: Which Dog (diffusion basins)

## Session 3 (final successor) -- state
- DONE: MNIST renders (gallery/mnist/*: basin confidence/riso/paper/margin-spectral at 2304^2 with bilinear prob interpolation + digit legend; boundary mosaic 24x24 window (72,160) classes 0/3/5/6/8/9; full mosaic; memo_cells (smoothed one-hot); memo mosaic; mnist_steps mp4/gif; zoom plate with caption). All viewed.
- DONE: gamma film (gallery/animations/gamma_sweep_ring8_night.*, viewed frame: good); steps_scatter12 film + gallery/verify/toy_dimension.png viewed (good).
- DONE: zoom_iter_ring8 re-encoded: mp4 CRF21 16.6 MB, gif 360px/8fps/128 colours bayer 13.8 MB.
- DONE: README.md full (verdict table, invertibility explanation, commands); link check passes (inline python regex over src/href/]( ) ).
- common.write_video now pads odd frame sizes to even; caption_strip word-wraps.
- MNIST numbers: uncert M=2048 alpha=0.825 D=1.17 (weak, 4 flips at 1e-3); f64 mismatch 0/2304; zoom 5->2 classes, straight boundary at width 0.012 rad; stepsweep agreement w/ N=45: 2%(1) 55%(2) 81%(4) 92%(8) 98%(23); boundary edges saturate ~1170 by N=8; memo net vs exact same image on 44.5% of pixels, 17 vs 16 images reached, net median d1/d2 0.04, exact median d1 0; clf test acc 99.3%.
- RUNNING at handoff (logged, OK to leave):
  * GPU `mnist_compute.py uncert 16384` (eps to 1e-4) -> cache/mnist_uncert_M16384.json, logs/mnist_uncert16k.log. When done: put alpha/D into README section 4 table (MNIST row) and remove "added below if it finished" sentence.
  * GPU `toy_compute.py maps learned; zoom ddim50_learned_scatter12` -> logs/toy_learned.log. When done: `python render_toy.py atlas verify` (atlas renders learned too: gallery/toy/atlas_learned_*), view, add a learned-net row to README gallery 2b and verdict table (zoom_dimensions.json key ddim50_learned_scatter12).
  * CPU `render_toy.py gamma zoom:ode_scatter12 zoom:iter_ring6` -> logs/render_toy_rest.log; gamma done, zoom films pending. When done: check sizes (<20 MB mp4, <15 MB gif; re-encode as for ring8 if not), view plates, add to README 2a/2b.


## Session 2 (CPU-first; GPU saturated until ~21:00)
- `common.gpu_setup` honours `DB_DEVICE=cpu DB_THREADS=2` (CPU mode).
- `toy_compute.iterate_map` now uses active-set compaction; sign/logdet are of F^nu (frozen at convergence). 256^2 ring8 2000 it: 10.7 s tracked / 5 s untracked on 2 CPU threads.
- render: added measured confidence shading (`margin_confidence` = 1-d1/d2 of sample for samplers; rank log nu for iterated map) -> `atlas_*_confidence`, `hero_*_confidence`, `iter_*_confidence`.
- Running (CPU, 2 threads each, logs/): toyA = iter hero -> verify -> iter gamma; toyB = maps analytic -> steps analytic -> zoom ode_scatter12; toyC = zoom iter_ring8 iter_ring6.
- Queued on gpu_run.sh (logs/mnist_memo.log, mnist_ddpm.log, toy_train_scatter12.log): mnist.py memo --steps 12000 --bs 64; clf+ddpm --steps 15000 --bs 128; toy.py train scatter12.

### Session 2 results so far
- box counting 2048^2 (fit 2-256 px): ODE scatter12 D=1.094+-0.015, DDIM50 1.093, DDIM10 1.087; nulls: circle 1.061, ring8 exact rays 1.070, iter gamma=1 1.070; iter ring8 g=2.12 D=1.571+-0.021 (1.47 @1-16px, 1.73 @32-512px), iter ring6 1.408; Newton z^3-1 control 1.470 (flat across windows).
- resolution check (boundary px at 256/512/1024/2048): ODE 1157/2349/4747/9578 (x2.0 = smooth), iter_ring8 window (2.2,1.1,hw .3) 851/1728/3494/7192 (x2.03, locally smooth there!), Newton 3708/10241/27891/75877 (x2.7).
- uncertainty exponent: DDIM50 scatter12 alpha=1.017 (D=0.98). Others: see cache/verify_toy.json when done.
- ODE RK4 convergence: label mismatch 100 vs 400 steps 7.6e-6.
- stretch: log10 sigma_max of ODE sampler Jacobian ranges -0.97..2.0; det<0 fraction 6e-5 (finite-diff noise) -> invertible, smooth but stretched.
- DDPM frozen noise: z-plane map is a SINGLE label for n=10,30,100,1000 steps (x_T coefficient ~0.015 -> start noise irrelevant). New task `toy_compute.py ddpm` maps a great-sphere slice through the injected-noise sequence instead (cache/ddpm_scatter12.npz, log logs/toyD_ddpm.log). Needs a render function (not written yet).
- iterated heroes: ring8 = nested flowers (best image); ring6 flowers only at 6-fold junctions; scatter12 nearly Voronoi + a few tongues.
- zooms: iter_ring8 centre (2.83511,2.83511), level 32 (hw 9.3e-10) still 19.5k boundary px at 768^2; iter_ring6 centre found at the origin (6-fold junction).
- Rendered so far: gallery/iterated/*, gallery/toy/atlas_analytic_*, hero_*, stretch_*, gallery/diptych/*. Viewed: iter ring8 confidence+spectral (strong), atlas confidence (DDPM column trivial -> replace by noise-slice), stretch fire (good), diptych spectral (good).

### Session 2 verdicts (for README section 6)
- uncertainty exponent (M=2^18, eps 1e-1..1e-5, D=2-alpha): DDIM50 scatter12 D=0.983, PF-ODE scatter12 D=0.995 (SMOOTH); iter ring8 g=2.12 D=1.270; iter ring6 D=1.177; Newton z^3-1 control D=1.441.
- ODE zoom (ode_scatter12, centre (0.40653,-0.41867)): by level 16 (hw 4.6e-5) boundary = 1536 px = one straight line across 768^2, 2 labels -> smooth.
- iter_ring8 zoom (36 levels x2, hw 4 -> 5.8e-11, float64): windowed box-count D (2-128 px) repeats with PERIOD 9 LEVELS (scale 512): 1.56 1.59 1.35 1.34 1.49 1.60 1.50 1.25 1.42 | same...; period mean D=1.46; boundary px per 768^2 window 16k-81k, never decays to 2R=1536 -> genuine discrete self-similarity (nested flowers), not resolution artefact. Global 2048^2 resolution window (2.2,1.1) locally smooth: fractal set is concentrated on the flower accumulation, not everywhere.
- DDPM noise-slice (512^2): 30 steps D[2,128]=1.22, 1000 steps 1.19 at 512 px = finite-res smooth curves; images viewed, smooth organic lobes. gallery/toy/ddpm_noise_slices.png.

### Running at handoff (all logged; OK to leave)
- CPU: toyA -> `iter gamma` (cache/iter_ring8_gamma.npz; then render `render_toy.py gamma`); toyB -> `zoom ode_scatter12` (then `render_toy.py zoom:ode_scatter12`); toyC -> zoom iter_ring6 (then `zoom:iter_ring6`); render_zoom8.log (zoom:iter_ring8 video+plate, not yet viewed); render_steps_verify.log (steps anim + gallery/verify/toy_dimension.png, not yet viewed).
- GPU: mnist memo + clf+ddpm training (logs/mnist_memo.log, mnist_ddpm.log), then run_mnist_chain.sh auto-runs all mnist_compute tasks (logs/mnist_compute.log). scatter12 net trained? check logs/toy_train_scatter12.log -> then `DB_DEVICE=cpu python toy_compute.py maps learned` (or on GPU), `steps learned`, `iter learned`, render `atlas`.
- render_mnist.py written (basin, mosaic, memo, steps, zoom) but UNTESTED.

### Next
0. zoom_iter_ring8 plate VIEWED: excellent nested 8-petal flowers repeating from width 8 to 2.3e-10 (hero candidate). But gallery/zooms/zoom_iter_ring8.gif = 54 MB and .mp4 = 35 MB: both exceed commit.sh 20 MB cap and GIF target 15 MB -> re-encode (GIF 360 px / 8 fps / fewer frames_per_level; MP4 higher CRF) before commit.
1. View: zoom_iter_ring8 plate/gif, steps anim, toy_dimension.png; fix weaknesses (captions clip if long: caption_strip does not wrap).
2. gamma anim, ode zoom, ring6 zoom renders; learned-score maps.
3. MNIST renders when cache/mnist_hero.npz etc. exist; thumbnail mosaic.
4. README.md (brief format) with the numbers above; commit.

## Session 1 (previous handoff)

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
