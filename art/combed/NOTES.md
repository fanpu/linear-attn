# NOTES: Combed (art/ml-art-3d.md §8)

## State
- M1 DONE (2026-09-15, CPU): all 10 fields sampled, order checks, t->1 limit, null, previews in `cache/preview/`. Report: `docs/superpowers/plans/reports/combed-M1.md`.
- M2 DONE 2026-09-15: dense trajectories + basin volumes (GPU via gpu1.sh), SVGs + box counting + all heroes (CPU, 4 threads,
  controller-approved; chain `logs/cpu_m2_heroes.sh`, 689 s total). Report: docs/superpowers/plans/reports/combed-M2.md.
- M3 (film, README) not started.

## M2 files and commands (from art/combed/)
- `compute_m2.py dense|basin --device cuda` (via gpu1.sh, `logs/gpu_m2_compute.sh`): `cache/dense_{kind}_N{N}.npz` (states (269,20000,3) float32:
  257 RK4 states on [0,1-1e-3] + every 8th of 96 geometric tail steps to 1-1e-6; end_nn/d1/d2 at 1-1e-6),
  `cache/basin_N16_R{128,256}.npz` (labels, ratio, null_voronoi_x0). N=16 dense files were made on CPU (device stored in npz).
- `compute_m2.py boxcount` -> `cache/basin_boxcount.json`; `plot_verify.py` -> `gallery/verify/basin_boxcount.png`.
- `render_hair.py diptych|stereo`, `render_tubes.py tubes|svg`, `render_basin.py --res R`: read cache only; r3d.
- Rulings from controller (M1 gate): all renders use the t = 1-1e-6 run; every memorised fraction is printed with the fresh-knot null; width/budget fixed and declared.

## Files
- `combed_common.py`: trefoil data, closed-form field v*, `VelocityMLP`, RK4, Gu et al. criterion.
- `compute.py`: `train` -> `cache/mlp_N{N}.pt`; `sample closed|mlp` -> `cache/flow_{kind}_N{N}.npz`;
  `converge closed|mlp` -> `cache/converge_{kind}_N{N}.json`; `summary` -> `cache/summary.json` (copy in `logs/summary.json`).
- `preview.py`: matplotlib previews -> `cache/preview/` (not gallery renders).
- `test_combed.py`: pytest (continuity equation FD check, eq. 6 direct sum, nested sets, RK4 exact on a straight line, criterion).

## npz contents (`flow_{kind}_N{N}.npz`)
`data` (N,3); `x0` (20000,3) noise, seed 1; `traj` (64,20000,3) float32 from the 256-step run at grid indices `keep`
(times `t_keep`, uniform in step index); `end256`, `end512` raw endpoints at t = 1-1e-3 (float64);
`xhat256`, `xhat512` = end + (1-t) v(end, t) (one Euler step to t = 1); `{end256,xhat256}_{nn,d1,d2}`;
closed only: `wmax256` largest softmax weight at the stop.

## Resume commands (CPU, from art/combed/)
```
../.venv/bin/python -m pytest -q test_combed.py
OMP_NUM_THREADS=2 ../.venv/bin/python compute.py sample closed --threads 2 > logs/sample_closed.log 2>&1
OMP_NUM_THREADS=2 ../.venv/bin/python compute.py train --threads 2 > logs/train.log 2>&1
OMP_NUM_THREADS=4 ../.venv/bin/python compute.py sample mlp --threads 4 > logs/sample_mlp.log 2>&1
OMP_NUM_THREADS=4 ../.venv/bin/python compute.py converge closed --threads 4; ... converge mlp
OMP_NUM_THREADS=4 ../.venv/bin/python compute.py limit closed --threads 4; ... limit mlp
../.venv/bin/python compute.py summary && ../.venv/bin/python preview.py
```
Every stage skips outputs that already exist.

## Decisions
- Decision: closed-form field is Bertrand et al. (arXiv:2506.03719) Prop. 1 / eq. 6 verbatim: û*(x,t) = Σᵢ λᵢ (x⁽ⁱ⁾ − x)/(1 − t), λ = softmax_j(−‖x − t x⁽ʲ⁾‖²/(2(1 − t)²)); t = 0 noise N(0, I), t = 1 data. It matches the plan's formula exactly — checked against the paper.
- Decision: memorisation criterion is Gu et al. (arXiv:2310.02664), after Yoon et al. 2023: a sample is memorised if its ℓ2 distance to the nearest training point is < 1/3 of the distance to the second nearest. Memorised fraction = share of the 20k samples meeting it.
- Decision: CPU, not GPU — the models are tiny (whole M1 ≈ 1 h on ≤ 4 threads), the GPU queue was held by autonomous/ at launch, and CPU float64 sampling is deterministic. Hardware caption: GB10 CPU (Grace), torch 2.14.0+cu130 CPU path.
- Decision: training points s ~ U[0, 2π) from `np.random.default_rng(0)`, 4096 draws, smaller N are prefixes. Consequence: points 7 and 14 are only 1.5e-3 apart (a near-duplicate pair present in every N).
- Decision: noise seeds `np.random.default_rng(1).standard_normal((20000, 3))`, shared by all 10 fields.
- Decision: MLP = [x, sin/cos(f t) for 16 f geometric in [1, 1000]] -> 4 hidden layers × 256, SiLU -> R³ (207,363 params, width fixed across N per the spec's capacity pitfall). Budget: 20,000 Adam steps, batch 1024, lr 1e-3 cosine to 0, t ~ U[0,1], float32, torch seed 0, identical for every N. Also logged: excess loss E‖v_θ − v*‖² on a fixed 4096-point probe set.
- Decision: both fields are integrated in float64 (the MLP is cast to double for sampling) so solver roundoff is not a factor.
- Decision: headline memorised fraction uses x̂ = x + (1 − t) v(x, t) at t = 1 − 10⁻³ (one Euler step to t = 1; for v* it is the posterior mean E[x₁ | x_t]), not the raw endpoint. Why: the raw endpoint of the closed-form flow sits (1 − t)·|x₀,eff − x₁| ≈ 2e-3 from its training point, which is comparable to the median neighbour spacing at N ≥ 1024 (2.9e-3 at 1024, 8.3e-4 at 4096); the criterion d1 < d2/3 would then call exact copies "not memorised" because of the declared stop, not because of the field. The raw-endpoint fraction is reported alongside.
- Decision: trajectories cached at 64 time points uniform in step index of the 256-step run (t_k = k(1 − 10⁻³)/256, k = round(linspace(0, 256, 64))).
- Decision: step-doubling check reports max, 99.9th percentile and median of ‖end256 − end512‖ over all 20k seeds, plus an order check (`converge`) on the 256 worst and 256 random seeds at 1024 and 2048 steps.
- Decision: added a `limit` stage — continue the 256-step endpoints from t = 1-1e-3 to 1-1e-6 with 96 RK4 steps geometric in (1-t) (checked against 192) — because at N >= 1024 the declared stop's residual width 1e-3 exceeds the training-point spacing, so the closed-form field's x_hat blends neighbours; the limit column shows the field itself memorises (~100%).
- Decision: added a null through the identical criterion — 20k fresh points on the continuous trefoil (s ~ U[0,2pi), seed 7) — because on a 1D curve a perfect generaliser is still called "memorised" ~26-37% of the time. Also report median distance of x_hat to the continuous knot (400k-point KD-tree).
- Decision: `limit`/`converge` are diagnostics; the headline table remains x_hat at t = 1-1e-3 (the declared stop), with limit, raw and null columns beside it.

## M1 results (cache/summary.json)
| N | closed x_hat | closed raw | closed 1-1e-6 | MLP x_hat | MLP raw | MLP 1-1e-6 | null (fresh knot) | MLP median dist to knot |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.954 | 0.881 | 1.000 | 0.704 | 0.700 | 0.704 | 0.263 | 3.9e-3 |
| 64 | 0.988 | 0.962 | 1.000 | 0.434 | 0.430 | 0.434 | 0.367 | 5.3e-3 |
| 256 | 0.909 | 0.762 | 1.000 | 0.183 | 0.174 | 0.183 | 0.313 | 5.4e-3 |
| 1024 | 0.719 | 0.379 | 0.999 | 0.049 | 0.044 | 0.049 | 0.333 | 5.6e-3 |
| 4096 | 0.471 | 0.061 | 1.000 | 0.005 | 0.005 | 0.005 | 0.345 | 5.5e-3 |
- Closed x_hat median distance to knot ~6e-6 at every N. Non-memorised closed samples at N=16/64 all sit between the near-duplicate pair (points 7, 14; 1.5e-3 apart).
- Step doubling (256 vs 512, raw endpoint, 20k seeds): closed max 3.1e-3 / 2.9e-3 / 9.0e-3 / 1.4e-2 / 4.2e-3; MLP max 0.277 (one seed flips basin at N=16) / 1.2e-2 / 1.3e-2 / 8.6e-3 / 4.6e-3; p99.9 <= 3.9e-3 everywhere.
- Order check: MLP gaps shrink ~300x then ~50x per doubling (converged at 512); closed-form worst seeds shrink only ~2-10x per doubling (late basin commitment near t -> 1 is stiff), random seeds ~5-10x.
- Trained MLP samples lie ~5e-3 off the knot at every N (training excess loss 0.005-0.026); at N >= 256 that blur exceeds the point spacing, so the MLP's low memorised fraction is below the null: it means "blurred along/off the knot", not "new points exactly on the knot".
- Wall clock (CPU, 2 threads each, contended): training 255-355 s per N; closed sampling 4 s (N=16) .. 2106 s (N=4096); MLP sampling 300-613 s per N; limit closed 369 s at N=4096.

## M2 decisions
- Decision: render from dense RK4 states (257 + 12 tail) instead of the 64 cached points — linear interpolation of the 64 points misses true states by up to 1.6e-2 (p99.9 7-10e-3, ~5 px at 2048 px); dense vertices are <= 1/256 in t apart.
- Decision: hair glow maps hue = density-weighted mean t (colorcet bmy) and brightness = (log1p(W/1)/log1p(150))^0.9, W = hair length density per 2048-px-equivalent pixel; raw additive 1-exp tonemap either saturated the core or hid the outer hairs. Endpoints: separate Gaussian glow (sigma 1.6 px at 2048), tonemap exposure 0.05, screen blend.
- Decision: MLP renders also use the 1-1e-6 run (the ruling mandates it for closed-form; same stop for both keeps the diptych matched). MLP endpoints move < 1e-3 between stops.
- Decision: plotter SVG draws t >= 0.5 only (declared window; full-length hairs are a uniform starburst that hides the knot), hidden lines by a tube depth buffer r = max(0.004, 2.5 px) with eps = 10 r (curves heading along the view axis otherwise self-occlude into dashes), runs < 3 px dropped.
- Decision: tubes have no AO or shadow (r3d has no occluder for splatted tubes); plaster = matte off-white, one raking Lambert light at 0.62 ambient.
- Decision: basin palette colorcet glasbey_category10 in training-set index order; null = Voronoi cell of the start point x0 (piecewise planar, D = 2 exactly) through the identical voxel render and box counting.
- Decision: box-counting fit over eps = 1..R/16 voxels; larger boxes saturate (local slope -> 3 for both basins and the planar null).
- Decision: views = orthographic az -60 el 40; hair/tube ortho height 3.4, SVG 2.6, basin 9.2.
- Decision: hero N for tubes/SVG = 64 (tufts) and 1024 (knot), both fields; hair diptychs at all five N (feeds the M3 film); stereo pairs closed N=64 and MLP N=1024.

## M2 numbers
- Basin N=16 (t = 1-1e-6): 16 labels; agreement with Voronoi-of-x0 null 76.6%. Box counting eps 1-16 vox at 256^3: D = 2.13 (null 2.10); at 128^3 eps 1-8: 2.16 (null 2.13). Boundary voxels 256^3/128^3 = 4.11.
- Dense stop 1-1e-6 memorised fractions (raw endpoint): closed 1.000/1.000/1.000/0.998/0.998, MLP 0.704/0.434/0.183/0.049/0.005.
- Decision: heroes (>= 2000 px) rendered on CPU with 4 threads, not through gpu1.sh — the queued GPU render job waited 05:02-06:35 behind ~9 art jobs (solid-edge m2 alone ~3 h left); controller approved; own queued job (gpu1 pid 662025 + its flock) killed before it took the lock. CPU timings: tubes 8 s per (N, kind) at 2400 px (4 images each), basin 9 s (128^3) / 17 s (256^3) at 2400 px, 20k-hair diptych 68-107 s at 2048 px/panel, whole chain 689 s.
- Decision: stereo uses rotation stereo (azimuth -60 -/+ 2.5 deg): r3d.stereo_pair shifts an orthographic camera sideways, which gives zero parallax.
- Decision: plaster ground darkened to grey 0.36 (tubes at 0.95 x (0.62..1) were indistinguishable from a 0.8 ground).
