# Invariant Tori (§5), milestone M1 report

**Status: DONE.** All tests pass, and the ε = 0 preview shows nested closed tori. Commit `e363a6d` (`art/invariant-tori: M1 chart, orbits, density, membrane, tests`). CPU only, no GPU used, and no packages installed.

## What was done

- **`chart.py`** implements the chart from logits to u ∈ ℝ⁴, then to S³, then to stereographic ℝ³, plus its inverse.
  - **Zero-mean coordinates:** the basis is e_a = (−1,1,0)/√2 and e_b = (−1,−1,2)/√6 for each player.
  - **Energy:** H = lse(Eᵀw_x) + lse(Eᵀw_y), with minimum 2 log 3 at u = 0.
  - **Pole basis:** an orthonormal basis of p⊥ comes from a QR decomposition.
  - **Inverse:** X → q → t solving H(tq) = h → strategies. The solve uses 64 bisection steps on the proven bracket [0, √6·h] (from H(tq) ≥ t/√6), then a Newton polish.
  - It also provides the section function g, dg/dt under the flow, and the stereographic scale.
- **`replicator_c.py`** is copied from game-chaos. The only change is that the trajectory buffer stores float64 logits instead of probabilities.
- **`section.py`** holds game-chaos's `section_seed`, with an added `root=` argument, and a first-return helper.
  - The section plane (x_R, y_P) is 2-to-1: each (p, q) can carry two section points.
- **`compute_orbits.py`** computes all orbits in float64 with RK4, h = 0.01, on H = 2.8.
  - **ε = 0:** 12 tori, T = 2·10⁴, sampled every 0.1.
  - **ε = 0.5:** one chaotic orbit to T = 2·10⁵ (sampled every 0.05) and 8 regular orbits to T = 2·10⁴.
  - **ε-sweep:** 26 values in [0, 0.5] × the same 12 starts, T = 10⁴, sampled every 0.2. It checkpoints one file per ε.
  - Section crossings and λ are stored for every orbit.
- **`compute_chart.py`** produces the chart products:
  - the pole search;
  - stereographic coordinates for every orbit and every section dot;
  - the 256³ density;
  - the 128³ membrane.
- **`test_chart.py`** has 5 pytest tests: the energy formula, basis/stereographic exactness, convexity along rays, the round trip, and H drift.
- **`preview.py`** makes the matplotlib previews in `cache/preview/`.
- **`NOTES.md`** has the state, resume commands, key numbers, decisions and risks.

## Commands (from `art/invariant-tori/`, `OMP_NUM_THREADS=4`)

```bash
P=/home/fzeng/ml/research/art/.venv/bin/python
$P compute_orbits.py all        # eps0 3 s, eps05 5 s, sweep 12 s
$P compute_chart.py all         # pole 40 s, stereo 6 s, density 25 s, membrane 33 s
$P -m pytest -q -p no:cacheprovider test_chart.py   # 5 passed
$P preview.py all               # ~10 s
```

The cache is about 1.2 GB and gitignored. `logs/` is ignored by the repo-root `.gitignore`, so the logs are local only.

## Key numbers

### Pole
- **p** = (−0.151754, 0.693158, 0.514107, −0.481868).
- **Minimum angle to all 23.6 M stored orbit points: 11.85°.** Measured against the chords between consecutive samples it is 11.84°, so sampling does not hide a closer pass.
- **What sets the angle:** regular ε = 0.5 orbit kam 38, the corner-island torus. The pole sits at that island's core, at strategies x = (0.18, 0.12, 0.70), y = (0.18, 0.70, 0.12).
- **Distance from every other group:**

  | group | min angle to pole |
  |---|---|
  | ε = 0 tori | ≥ 30.4° |
  | chaotic orbit | ≥ 32.2° |
  | the other 7 regular orbits | ≥ 20.9° |
  | sweep | ≥ 30.4° |
- **Alternatives:** two equally good poles lie 120° away (the R→P→S cyclic symmetry), at 11.845°. Candidates for the M3 second-pole plate are in `cache/pole.json` under `runner_up`.
- **Chart distortion:** the scale |dX|/|dq| over stored points ranges from 0.50 to 46.9 (99th percentile 4.2), and the largest |X| is 9.6.

### H drift (max |H(t) − H(0)|, recomputed from stored float64 logits; the test requires < 1e-8)
| orbits | drift |
|---|---|
| ε = 0 tori | 6.5e-10 |
| chaotic, T = 2e5 | **3.7e-9** |
| ε = 0.5 regular | 1.1e-9 |
| sweep (all 312) | 3.2e-10 |

Start energies are within 1.2e-14 of 2.8.

### Lyapunov exponents λ (finite-time)
- **ε = 0 tori (T = 2e4):** 4.56, 4.05, 4.30, 4.43, 4.37, 4.25, 4.09, 4.43, 4.07, 4.24, 4.07, 4.18 (×10⁻⁴). All are regular.
- **ε = 0.5 chaotic (T = 2e5):** λ = 0.0246. On the five 2·10⁵ continuation chunks it was 0.0228, 0.0229, 0.0182, 0.0226, 0.0226. The dip in chunk 3 suggests the orbit stuck near an island for a while.
- **ε = 0.5 regular, kam indices 38, 2, 59, 84, 121, 201, 243, 347 (T = 2e4):** 3.91, 3.94, 4.05, 3.81, 3.60, 3.44, 3.81, 3.24 (×10⁻⁴). All are ≤ 5e-3.
- **Sweep, number of the 12 starts with λ > 5e-3, ε = 0 → 0.5:** 0 0 1 1 2 1 4 4 5 4 7 6 6 7 7 10 7 8 7 9 7 7 10 6 7 8.

### Tests (5 passed)
- **Convexity:** H(tq) is strictly increasing on all 10⁴ random rays at 601 steps out to 1.5·t\*. The smallest increment is 3.5e-6, and the analytic derivative is > 0 for all t > 0. t\* ranges from 1.83 to 2.40.
- **Round trip** on 10⁴ orbit points drawn from all 29 orbit groups: max strategy error **5.6e-16** using each point's own H. Using h = 2.8 instead it is 1.0e-9, which is integrator drift, not chart error.
- **H drift:** see the table above.
- **Exactness:** stereographic inverse error < 1e-12, and the energy formula matches −⅓Σlog x − ⅓Σlog y to < 1e-12.

### Voxels (`cache/density_eps05.npz`)
- **Grid:** 256³ over the per-axis 0.5–99.5 percentile box of the chaotic orbit's chart coordinates: lo (−1.850, −1.963, −1.847), hi (1.973, 1.953, 1.797).
- **Samples:** 24 M, from T = 1.2·10⁶ (see Decisions); 96.5% fall inside the box.
- **Fill:** 30.9% of voxels are occupied, with a mean of 4.5 counts per occupied voxel and a maximum of 333.
- **Convergence:** the two halves of the run agree voxel by voxel with correlation r = 0.71.

### Membrane (`cache/membrane.npz`)
- **Grid:** 128³, cell-centred, over the same box. The inverse chart lands exactly on the energy level set: |H − 2.8| ≤ 8.9e-16.
- **Size:** 24,692 voxels lie within half a voxel of g = 0; 13,266 of those also have dg/dt > 0 at ε = 0.5.
- **Agreement with the section dots:** 94.1% of the 30,384 in-box ε = 0.5 dots fall in a membrane voxel, and 100% fall in one or a face neighbour.
- **How NaN is assigned:** the stored `g`, signed distance `sdist` = g/|∇g|, `gdot_eps0` and `gdot_eps05` are complete fields. In `membrane_both`, g is NaN wherever |sdist| > ½ voxel. In `membrane_eps05`, g is also NaN where dg/dt ≤ 0 at ε = 0.5.

### ε = 0 seeds
- The elliptic fixed point of the return map is x = y = (0.46538, 0.46538, 0.06924), with residual 2e-14.
- The ray runs toward the upper-right tip of the plate along the diagonal, on section root 1. The section fold is at r = 0.3448, and the seeds are at r = fold·(i+1)/13.

## Previews (all inspected)
- `/home/fzeng/ml/research/art/invariant-tori/cache/preview/tori_eps0_slices.png`: thin slabs through the innermost torus in three orientations show **clean nested closed curves**, including the two-lobed cut of a doughnut through its core. This is the main evidence for acceptance.
- `/home/fzeng/ml/research/art/invariant-tori/cache/preview/tori_eps0_each.png`: each of the 12 orbits alone. Each is a doughnut with a hole, the winding lines are visible on the outer ones, and they grow from inner to outer.
- `/home/fzeng/ml/research/art/invariant-tori/cache/preview/tori_eps0_views.png`: three views of all 12. The outer tori hide the inner ones, as expected for a 3D line plot.
- `/home/fzeng/ml/research/art/invariant-tori/cache/preview/sea_eps05_views.png`: three views of the grey chaotic points with the 8 regular orbits (t ≤ 1200) clipped to the box. It is busy, but torus kam 121 reads clearly as a doughnut.
- `/home/fzeng/ml/research/art/invariant-tori/cache/preview/sea_eps05_slices.png`: slices through the density volume (log density, magma) with the g = 0 contour, membrane voxels and regular orbits. The sea has sharp black holes, and the regular island orbits sit exactly inside them. The membrane contour cuts through the sea. It is the best picture of "sea + islands".

## Decisions (also in NOTES.md)
1. **Integrator copy stores logits.** The chart needs exact logits, and chained runs are then bit-identical.
2. **ε = 0 seeds** lie on the ray from the core fixed point to the section fold, on section root 1. This ray spans the nested family from one core circle to the other without crossing another island.
3. **Chaotic orbit** = game-chaos kam_eps0.50 orbit 190, which has the largest λ there.
4. **The 8 regular orbits** are kam indices [38, 2, 59, 84, 121, 201, 243, 347].
   - They were picked by farthest-point sampling, with distance = 1 − IoU of hole-filled section masks.
   - Candidates had λ ≤ 2.5e-3 at T = 4e4 and filled section area between 0.003 and 0.05.
   - The result covers the corner island, both central islands, a period-3 island chain and four rim resonance chains.
5. **Integration lengths:** T = 2e4 for the tori and regular orbits, 2e5 for the chaotic orbit, 1e4 for the sweep. The sweep's chart coordinates are cached as float32, but its logits stay float64. This keeps the cache around 1.2 GB.
6. **One pole for all scenes**, so the film keeps a single chart. The search used 4e5 random candidates, then Nelder–Mead refinement of the 24 best distinct candidates.
7. **Longer density run.** The density adds 5 bit-identical continuations, reaching T = 1.2e6. At T = 2e5 only 12% of voxels were hit, with about 2 counts each. The box is still the percentile box of the stored T = 2e5 segment. The density is time fraction per chart voxel, so it includes the chart's volume distortion; M2 captions must say so.
8. **Membrane definition.** Off the level set means first-order distance > ½ voxel. The upward half is the dg/dt > 0 mask at ε = 0.5. The full fields are kept so M2 can render an iso-surface instead.
9. **Round-trip test uses each point's own H.** This isolates chart error from integrator drift; the h = 2.8 round trip is also asserted to be < 1e-7.

## Open risks
- **Orbit kam 38 dominates uncropped views.** It passes 11.85° from the pole, so the chart magnifies it up to 47× and it reaches |X| ≈ 9.6, far outside the box. M2 should crop to the box, or keep kam 38 only for the second-pole plate.
- **The spec's narrative does not match the sweep.** The spec says outer tori dissolve while islands survive inside. In the data, the **innermost** seed is the first to go chaotic (at ε = 0.04), and outer ones follow. M3's film caption must follow the data.
- **The 256³ fog is noisy.** Split-half r = 0.71 at voxel level, and there are 4.5 counts per occupied voxel. M2 may need a declared smoothing or a 128³ variant, with a 2× resolution check.
- **The section is 2-to-1 in (x_R, y_P).** Any slice plate in M2 that compares against game-chaos's Poincaré plates must project both sheets the same way those plates did, i.e. plot all upward crossings in (x_R, y_P).
- **The shared renderer `art/_shared/r3d/` was not ready**, so nothing in M1 depends on it.
