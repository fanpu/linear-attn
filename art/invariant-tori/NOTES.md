# Invariant Tori — running notes

Spec: `art/ml-art-3d.md` §5 (+§0). Plan: `docs/superpowers/plans/2026-09-15-3d-pieces.md` §5.
Source piece: `art/game-chaos/` (read only; code copied with source comments).

## State (M3 done, 2026-09-15)

M1 = chart + orbits + voxels + membrane + previews (report `docs/superpowers/plans/reports/invariant-tori-M1.md`).
M2 = renders with `art/_shared/r3d/` into `gallery/` (report `.../invariant-tori-M2.md`). M3 (films, STL, README) next.

**M1 numbers below the M2 section are for the FIRST pole; the current pole is the M2 one.**


## M3 (controller fixes + films, STL, README)

- Fix 1: sheet shown only on the fog support: weight clip(2 * G_sigma=2vox([count>0]), 0, 1).
- Fix 2: density box = full support of the stored T=2e5 chaotic segment + 2% per side (was 0.5-99.5 percentile):
  lo (-3.383,-2.169,-2.710) hi (3.596,2.122,2.392); 99.95% of 24M samples inside; 30.8% voxels occupied, mean 23.2;
  split-half r = 0.61. fields256 and membrane recomputed (dots in membrane voxel 96.9%, +neighbour 100%).
- Fix 3: stereo +-1.5 deg.
- Fix 4: `tori_plaster_cutaway.png` renamed `tori_plaster.png`; new `tori_slice.png` = exact meridional crossings
  (compute_slice.py, T=1e5, 163,955 crossings, plane spanned by doughnut axis a and b from render_lib.tori_frame()).
  Scene frame now defined once in render_lib.tori_frame() (raw dt=0.01 samples of torus 0, every 4th, t<=1000).
- Film sweep: stereo_sweep_fine.npz (dt=0.01, t<=1000, float32); HOLD 10 + TWEEN 6 crossfade frames, 24 fps,
  camera spin 60 deg about the torus axis; chaotic (lambda>5e-3 at T=1e4) pale cyan, regular plasma(index).
- Turntable: cutaway with camera-facing clip, 240 frames, step 0.008, 1080^2.
- Second pole plate: render_secondpole.py; eps0 tori 29.0 deg/0.50-8.0 (current) vs 30.5/0.50-7.2 (M1 pole);
  regular eps0.5 23.6/0.50-11.9 vs 20.9/0.50-15.1.
- STL: make_stl.py 201 -> gallery/torus_woven_kam201.stl, t<=70 (self-avoidance rule with 2.2 r), r = 0.6 mm,
  80 x 72 x 27 mm, 162,816 faces, watertight, 1079 mm^3. Decision: tube_mesh with the self-avoidance cap rather than
  a fused union (tried a 300^3 EDT union of kam 121 t<=300: watertight but lumpy at printable voxel size; dropped).
- Break-up: index 0 chaotic at eps 0.04, 0.06, 0.08 (with 1 at 0.08); at 0.10 only index 1 -> finite-time noise.
- commit.sh prints a harmless `stat` error for deleted files (they are still removed correctly).
- Batch: `./render_m3.sh` (CPU stills, then gpu1.sh: sea stills, film sweep, turntable; frames checkpointed in cache/).

## M2 (controller rulings applied)

- **Pole re-chosen** (`repole.py`): excluding kam 38, the pole is p = (0.67218, -0.25691, -0.66643, -0.19506),
  min angle **23.64 deg** (23.63 deg to sample chords), set by regular orbits kam 59 and 347; eps=0 tori >= 28.98,
  chaotic >= 31.50. Chart scale over stored points **0.50-11.9** (p99 4.18), max |X| 4.78.
  Pole strategies x = (0.123, 0.712, 0.165), y = (0.693, 0.121, 0.186).
- kam 38 replaced by **kam 32** (period chain in the central sea + rim chain), 25.79 deg from the new pole,
  lambda 4.2e-4. Rule: candidates with lambda <= 2.5e-3, section area in (0.003, 0.05), min angle >= the pole's
  own min angle (so adding it leaves the pole optimal: re-run gave the identical pole), farthest in 1-IoU from
  the 7 kept orbits. Old pole kept in `cache/pole_first.json` (for the M3 second-pole plate); `cache/pole_no38.json`.
- Tests re-run with the new pole: see M2 report.
- Density extended to **T_total = 6e6** (30 segments of 2e5): 52% voxels occupied, mean 13.0 per occupied
  voxel, max 2492; split-half r = 0.53 (the orbit sticks near an island for ~4e5 time units in segments 17-18,
  lambda 0.004 / 0.006 there, so the halves differ by a real sticky episode, not only Poisson noise).
- `fields256.npz`: g, dg/dt(eps=0.5), sdist on the density grid itself (|H-2.8| <= 8.9e-16).
- Render copies `orbits_*_fine.npz` at dt = 0.01, t <= 2000 (identical trajectories; dt=0.1 chords reached
  0.7 chart units).
- **Break-up order (ruling 2):** innermost seed chaotic first (eps = 0.04); captions must say so.

Decisions (M2):
- Decision: raw 256^3 fog (T = 6e6), no blur — at 2400 px the cut face shows island holes crisply; blur128
  also rendered in tests and softened the hole rims. Declared in captions.
- Decision: nested tori view along 52 deg from the doughnut axis (smallest-variance PCA direction of torus 0),
  wedge cutaway of half-angle 50 deg about that axis, identical for all tori; t <= 1000 of each orbit.
- Decision: plaster tint = 0.55 white + 0.45 viridis(torus index); AO/shadow from a 400^3 voxel occupancy of the
  tube centres (form only); one raking light.
- Decision: glow colour = plasma(0.15 + 0.8 i/11); additive Gaussian hairlines, exposure 0.22, wedge applied.
- Decision: sea fog TF = cmc.oslo(0.15 + 0.85 x), x = log1p(count)/q99.9, extinction 12 x^2 (exterior) or 80 x^2
  (cutaway); sheet = ivory, extinction 3 * exp(-(sdist/1.5 voxel)^2) * [dg/dt > 0]; tubes copper, t <= 150,
  radius 0.009, cropped to the density box; dots: chaotic crossings red, regular crossings dark copper.
- Decision: stereo = rotation stereo az -/+ 2.5 deg (orthographic cameras have no translation parallax), same clip.
- Decision: slice plate = game-chaos ink plate | same 1.77 M crossings in plate axes coloured by rank of x_R + y_P
  (cmc.batlow) | the same coloured crossings in the chart, framed on the density box; ink coverage 1-exp(-g hits),
  gain 3x on the chart panel.
- Decision: splatting on CPU (r3d CUDA splat bug, controller note); volumes on CUDA inside gpu1.sh.

## Files

| file | role |
|---|---|
| `replicator_c.py` | copy of game-chaos C RK4 integrator; only change: trajectory buffer stores float64 logits |
| `section.py` | `section_seed` copied from game-chaos `compute_poincare.py` (+ `root=` choice), first-return helper |
| `chart.py` | logits -> u in R^4 -> S^3 -> stereographic R^3, and the inverse (bisection + Newton on H(tq)=h) |
| `compute_orbits.py` | eps=0 tori, eps=0.5 chaotic + regular, eps sweep -> `cache/orbits_*.npz`, `cache/sweep/` |
| `compute_chart.py` | pole, stereo coords, 256^3 density, 128^3 membrane -> `cache/` |
| `preview.py` | matplotlib previews -> `cache/preview/` |
| `repole.py` | M2 ruling 1: pole without kam 38, replacement choice |
| `render_lib.py`, `render_tori.py`, `render_sea.py`, `render_plates.py`, `render_m2.sh` | M2 renders -> `gallery/` |
| `compute_slice.py`, `render_film.py`, `render_secondpole.py`, `make_stl.py`, `render_m3.sh`, `README.md` | M3 |
| `test_chart.py` | pytest: convexity along rays, round trip, H drift, basis/stereo exactness |

## Resume commands (from this directory; ~3 min total, 4 threads)

```bash
P=/home/fzeng/ml/research/art/.venv/bin/python; export OMP_NUM_THREADS=4
$P compute_orbits.py all          # ~20 s
$P compute_chart.py all           # ~2 min (pole search 40 s, membrane 33 s)
$P -m pytest -q -p no:cacheprovider test_chart.py   # 5 tests, ~11 s
$P preview.py all
$P repole.py                      # M2: needs cache/pole_first.json copy of the M1 pole
$P compute_orbits.py fine && $P compute_chart.py stereo fields256
./render_m2.sh                     # tori + plate on CPU, sea through gpu1.sh
```

## Key numbers (M1)

- Pole p = (-0.15175, 0.69316, 0.51411, -0.48187); min angle to all 23.6 M stored orbit points
  **11.85 deg** (11.84 deg to the chords between samples). Set by regular eps=0.5 orbit kam 38 (the
  corner island torus); eps=0 tori >= 30.4 deg, chaotic orbit >= 32.2 deg, sweep >= 30.4 deg.
  The pole sits inside the corner island (strategies x=(0.18,0.12,0.70), y=(0.18,0.70,0.12)).
  Two other poles, 120 deg away (the R->P->S cyclic symmetry), are equally good (11.845 deg).
- Chart scale |dX|/|dq| over stored points 0.50 - 46.9 (p99 4.2); max |X| 9.6.
- H drift (max |H(t) - H(0)| from stored float64 logits): eps=0 6.5e-10; chaotic (T=2e5) 3.7e-9;
  regular 1.1e-9; sweep 3.2e-10.
- lambda: eps=0 tori 3.2-4.6e-4 (T=2e4); chaotic 0.0246 (T=2e5; 0.018-0.023 on 5 continuation
  chunks of 2e5); regular eps=0.5 orbits 3.2-4.1e-4 (T=2e4).
- Tests: min increment of H along 1e4 rays 3.5e-6 > 0; round trip 5.6e-16 (own h), 1.0e-9 (h=2.8).
- Density: box lo (-1.850,-1.963,-1.847) hi (1.973,1.953,1.797); T_total 1.2e6, 24 M samples, 96.5%
  inside, 30.9% voxels occupied, mean occupied count 4.5, split-half voxel correlation 0.71.
- Membrane: |H - 2.8| on grid 8.9e-16; 24,692 band voxels, 13,266 with dg/dt>0 at eps=0.5; 94% of
  the eps=0.5 section dots fall in a membrane voxel, 100% in it or a face neighbour.
- Sweep: chaotic (lambda > 5e-3) count per eps rises 0 -> ~7-10 of 12; **the innermost seed (0) is the
  first to become chaotic, at eps = 0.04**.

## Decisions

- Decision: copy `replicator_c.py` and store logits instead of probabilities — the chart needs exact
  float64 logits, and chained runs (density continuation) are then bit-identical.
- Decision: eps=0 seeds on the ray from the elliptic fixed point of the return map (x = y =
  (0.46538, 0.46538, 0.06924), i.e. section point (x_R, y_P) = (0.46538, 0.46538), residual 2e-14,
  section root 1 = large-x_P branch) toward the upper-right tip of the plate, radii fold*(i+1)/13, fold r = 0.3448 — this ray
  runs from one core circle of the nested family to the other without crossing another island.
  The section in (x_R, y_P) is 2-to-1 (two roots of section_seed); root 1 is the branch containing
  the nested curves around the plate's centre.
- Decision: chaotic eps=0.5 orbit = game-chaos kam_eps0.50 orbit 190 (largest lambda there).
- Decision: 8 regular orbits = game-chaos kam_eps0.50 indices [38, 2, 59, 84, 121, 201, 243, 347], by
  farthest-point sampling (distance 1 - IoU of hole-filled section masks, 160^2 over [0,0.8]^2) among
  orbits with lambda <= 2.5e-3 at T=4e4 and filled area in (0.003, 0.05) — gives the corner island,
  both central islands, a period-3 chain and four rim resonance chains.
- Decision: T and sampling: eps=0 and regular T=2e4 at dt=0.1; chaotic T=2e5 at dt=0.05; sweep T=1e4
  at dt=0.2 (sweep stereo cached as float32, logits float64) — keeps cache ~1.2 GB.
- Decision: one pole for every scene (eps=0, eps=0.5, sweep) so the film keeps one chart; pole search
  = 4e5 random candidates on S^3, 24 distinct best holes refined by Nelder-Mead on the KD-tree
  nearest distance; best kept.
- Decision: density counts extend the stored chaotic orbit with 5 bit-identical continuations to
  T_total = 1.2e6 (at T=2e5 only 12% of voxels were hit, mean ~2 per hit voxel); the box stays the
  0.5-99.5 percentile box of the stored T=2e5 segment. Density = time fraction per chart voxel, so it
  includes the chart's volume distortion (declare in M2 captions).
- Decision: membrane grid = the density box (so they composite), cell-centred 128^3. "Off the level
  set" = first-order distance |g|/|grad g| > half the largest voxel side -> NaN (`membrane_both`);
  `membrane_eps05` additionally NaN where dg/dt <= 0 at eps=0.5 (the plates use upward crossings).
  Full `g`, `sdist`, `gdot_eps0`, `gdot_eps05` are stored for iso-surface rendering instead.
- Decision: inverse-chart round-trip test uses each point's own H (isolates chart numerics from the
  integrator's 1e-9 drift); the h=2.8 round trip is asserted < 1e-7 (measured 1.0e-9).

## Open risks for M2/M3

- Orbit kam 38 passes 11.85 deg from the pole: it is magnified up to 47x and extends to |X| ~ 9.6,
  far outside the density box; it will dominate uncropped views. Consider cropping to the box or
  dropping it from the hero (keep it for the second-pole plate).
- Fog density at 256^3 is Poisson-noisy (mean 4.5 per occupied voxel; split-half r = 0.71).
- The spec's "outer tori dissolve while islands survive inside" is not what the sweep shows: the
  innermost seed goes chaotic first (eps=0.04). The film caption must follow the data.
