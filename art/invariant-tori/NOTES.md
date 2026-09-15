# Invariant Tori — running notes

Spec: `art/ml-art-3d.md` §5 (+§0). Plan: `docs/superpowers/plans/2026-09-15-3d-pieces.md` §5.
Source piece: `art/game-chaos/` (read only; code copied with source comments).

## State (M1 done, 2026-09-15)

M1 = chart + orbits + voxels + membrane + previews, CPU only. M2 (renders with `art/_shared/r3d/`)
not started. Report: `docs/superpowers/plans/reports/invariant-tori-M1.md`.

## Files

| file | role |
|---|---|
| `replicator_c.py` | copy of game-chaos C RK4 integrator; only change: trajectory buffer stores float64 logits |
| `section.py` | `section_seed` copied from game-chaos `compute_poincare.py` (+ `root=` choice), first-return helper |
| `chart.py` | logits -> u in R^4 -> S^3 -> stereographic R^3, and the inverse (bisection + Newton on H(tq)=h) |
| `compute_orbits.py` | eps=0 tori, eps=0.5 chaotic + regular, eps sweep -> `cache/orbits_*.npz`, `cache/sweep/` |
| `compute_chart.py` | pole, stereo coords, 256^3 density, 128^3 membrane -> `cache/` |
| `preview.py` | matplotlib previews -> `cache/preview/` |
| `test_chart.py` | pytest: convexity along rays, round trip, H drift, basis/stereo exactness |

## Resume commands (from this directory; ~3 min total, 4 threads)

```bash
P=/home/fzeng/ml/research/art/.venv/bin/python; export OMP_NUM_THREADS=4
$P compute_orbits.py all          # ~20 s
$P compute_chart.py all           # ~2 min (pole search 40 s, membrane 33 s)
$P -m pytest -q -p no:cacheprovider test_chart.py   # 5 tests, ~11 s
$P preview.py all
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
