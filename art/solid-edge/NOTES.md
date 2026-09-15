# solid-edge NOTES (handoff)

Spec: `art/ml-art-3d.md` §0, §1, §11. Plan: `docs/superpowers/plans/2026-09-15-3d-pieces.md` §1.
Source piece (read-only): `art/trainability-fractal/` (`tfractal.py`, `boxcount.py`, `common_render.py` imported via sys.path).
M1 report: `docs/superpowers/plans/reports/solid-edge-M1.md`.

## Files
- `se_engine.py` batched trainer: `net2` (source net, reuses `tfractal._make_step` lossgrad; optional init scale sigma),
  `net3` (two hidden layers, three lrs), `quad3` (null: frozen-feature quadratic of net3, three lrs).
  Early exit as in the source, but compaction only to power-of-two buckets (fixed shape set). Grid helpers
  `overview_lr_axis` (every 16th pixel centre of the 1024^2 overview) and `sigma_axis`.
- `toys_compute.py` one toy volume, per-chunk checkpoints in `cache/toys/<name>/`, assembled `cache/toys/<name>.npz`.
- `run_m1.sh` GPU chain: toy a -> null -> toy a sigma=1 plane in float64 -> toy b.
- `se_analysis.py` 3D boundary (6-neighbour), 2x2x2 edge cells, 3D box counting. `test_engine.py`, `test_analysis.py` (CPU).
- `spacetime.py` space-time label volume from the source cache + README checks -> `cache/spacetime_*`, `cache/preview/spacetime_*.png`.
- `analyze_toys.py` toys + null structure tests, slice agreement -> `cache/toys_summary.json`, `cache/preview/toy_*.png`.

## Resume commands
```
cd /home/fzeng/ml/research/art/solid-edge
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest -q test_engine.py test_analysis.py
OMP_NUM_THREADS=4 ../.venv/bin/python spacetime.py
setsid nohup /home/fzeng/ml/research/art/_shared/gpu1.sh bash /home/fzeng/ml/research/art/solid-edge/run_m1.sh > logs/m1_toys.log 2>&1 < /dev/null &   # resumes per chunk
OMP_NUM_THREADS=4 ../.venv/bin/python analyze_toys.py
```

## Decisions
- Decision: sign convention converged <-> negative measure — verified: `tfractal.train_chunk` returns `-sum v` when converged, measure_T[T=1000] signs equal `measure`, and the 16 lowest-eta1 rows are 100% "negative" at T=1000 while the 16 highest are 0%.
- Decision: toy (a) eta axes use overview pixel centres i = 16j+8 (j = 0..63), values computed exactly as `tfractal.log_grid` in float64 then cast — so the sigma = 1 plane samples the float64 overview's own learning rates.
- Decision: toy (a) sigma axis log10 sigma_k = (k-26)*6/64, i.e. [-2.44, 3.47], sigma = 1 exactly on plane k = 26 — mirrors the source sigma x eta plate's six decades [-2.5, 3.5] and keeps the small-sigma (ragged) side.
- Decision: toy (b) "depth-2 width-16 tanh net with three learning rates" = two hidden layers h0 = phi(X W0/sqrt n), h1 = phi(h0 W1/sqrt n), y = h1 W2 / n (source mean-field scalings extended by one layer), N = #params = 528 data points, all three axes on the same overview eta grid — three lrs need three weight matrices; this is the smallest net that has them with the source's data recipe.
- Decision: null = quadratic of y = F0 a + F1 b + F2 c with F0 = X/sqrt n, F1 = h0/n, F2 = h1/n frozen at net (b)'s init, init a = W0[0], b = W1[0], c = W2, one lr per block on the net (b) grid, float32, same trainer/early exit/measure — the source's quadratic null (`train_chunk_quadratic`) extended to three blocks.
- Decision: chunk 32768 for toy (a) and the null, 16384 for toy (b) (its (P,N,n) activations are ~2x larger); compaction buckets are powers of two >= 1024 so a run only ever sees <= 6 shapes.
- Decision: 3D box counting uses 2x2x2 edge cells (cell holds both labels; the 3D analogue of his `extract_edges`), not the two-sided 6-neighbour voxel set — the latter is two layers thick, doubles the b = 1 count and biased a plane's b = 1-16 slope to 2.20 in `test_analysis.py`. The 6-neighbour count is still reported as "boundary voxels".
- Decision: slice agreement "near the boundary" = an edge cell of the float64 overview within Chebyshev distance 16 overview px (one toy voxel) of the sampled pixel.
- Decision: added a float64 recompute of the 64^2 sigma = 1 plane (4096 nets, ~1 min) to the chain, to separate float32 flips from sampling/kernel effects in the slice agreement.

- Decision: axis set for the 128^3 volume = toy (a) (log eta0, log eta1, log sigma) — (1) rougher boundary: D3(b=1-16) 2.12 +- 0.01 vs toy (b) 2.07 +- 0.02 and null 1.99 +- 0.02 (only (a) clears null + 0.1); (2) sigma is not an extrusion: 21 % of sigma-lines change label and 25 % of all label changes are across sigma (toy (b): eta0-lines 11 %, eta1-lines 24 %; its body is essentially one wall normal to eta2); (3) it contains the sigma = 1 plane, which reproduces the float64 overview (99.71 %, all misses near the boundary) and is what the M3 cutaway needs; (4) 1.5x cheaper per voxel (1553 vs 1026 px/s).
- Decision (recommendation for M2, not yet acted on): the overview window is 0.14 decade/voxel, where the edge is a near-flat wall; consider an (eta0, eta1) window centred on the seam (e.g. the 10^1 zoom window of steps_zoomA2 / zoomA kf2, whose sigma = 1 plane is an existing plate) so 128^3 resolves the rough part.

## M2 (in progress)
- Files: `vol_run.py` (f32 -> iterated f64 shell -> 1 % f64 audit -> final -> f64 plane; per-chunk checkpoints under
  cache/vol/<name>/), `choose_windows.py` (candidates | null | B), `analyze_m2.py` (D tables, 12 oblique slices,
  resolution doubling, plane checks, previews `cache/preview/m2_*.png`), `run_m2.sh` (single queued chain).
- Resume: `setsid nohup ../_shared/gpu1.sh bash run_m2.sh > logs/m2.log 2>&1 < /dev/null &` (skips finished chunks/stages).
- Decision: MEASURE FIX — non-finite loss -> v = 1e6 (clamp ceiling) instead of min(1e6/l0, 1e6). Found when the first M2 chain's
  quad2 null audit flipped 3.5 % of interior voxels, all at sigma >= 10^3.28: l0 > 1e6 there, float32 overflows to inf
  (never reaching the 1e100 early exit), and his rule scored those runs converged (float64 said diverged, measure ~1.0005).
  Identical labels whenever l0 <= 1e6 (all of sigma = 1, all source plates). The aborted log is logs/m2_aborted_measurebug.log;
  cache/vol was wiped. M1 toy (a) (sigma up to 10^3.47, float32) was affected in its large-sigma planes: recomputed as
  cache/toys/toy_a_64_float32_fix.npz at the head of run_m2.sh. Toy (b) and the null (no sigma, l0 ~ 1) are unaffected.
- Decision: probe candidates are data-driven: the 3 densest 8^3 edge-cell blocks of the fixed M1 toy (a) (>= 2 blocks apart),
  each a window of +-1 decade on every axis at 64^3 f32 (`choose_windows.py candidates`); B = max D3(b=1-16) - D3(null probe).
- Decision: the null for the sigma volumes is the source's own quadratic null (`quad2` = tfractal.train_chunk_quadratic) with sigma
  scaling both inits; its window (same +-1 decade size) is centred on the densest edge-cell region of its own 128^3 volume
  (the source centred its null zoom on its own boundary too). A three-lr quadratic has no sigma axis.
- Decision: "within one voxel of a label change" = the 3x3x3 neighbourhood holds both labels; the f64 shell is iterated
  (up to 6 rounds) on voxels that become adjacent to a change after f64 labels land.

- Controller rule (06:40): no queued GPU job may run > ~20 min. `vol_run.py --max_seconds 1080` stops at a chunk checkpoint and exits 3;
  `run_m2_loop.sh` (setsid nohup, CPU-side) requeues gpu1.sh segments until each volume's stages finish (B256, then B256_sub64).
  The single-job chain was stopped right after chunk 92/512 of B256 f32 (93 chunk files + times verified intact). Segmented ==
  unsegmented output checked bit-for-bit on a CPU smoke run. Resume: `setsid nohup bash run_m2_loop.sh >> logs/m2.log 2>&1 < /dev/null &`.
- Controller ruling for M3 (noted): render the space-time solid with a declared log10(T) vertical axis (above T ~ 200 it is a pure extrusion).
- 1-ulp floor on probe shells (float64, 400 shell voxels each, `ulp_check.py`): c1 3.75 %, c2 5.75 %, c3 10 %, null 0 % flips.
  (see next decision for the B choice).
- Decision: B window = c2 (small-sigma intrusion; centre log10 (eta0, eta1, sigma) = (-0.18, 3.19, -1.36), +-1 decade), overriding the
  letter of the ruling (max D - D_null picked c1). Probe D3(1-16): c1 2.54, c2 2.45, c3 2.33, null 2.03. c1's excess comes from a
  largely 2D (eta0, sigma) stripe/dust pattern extruded along eta1 (eta1 = 10^-3.4..10^-1.4, where the output layer barely moves):
  only 12 % of its label changes cross eta1 (isotropic = 33 %); c2 is isotropic (33/33/34 %), shows nested diagonal bands and
  filaments in every axis plane (cache/preview/m2_probes.png), and joins the main eta1 ~ 10^2-10^4 seam. A mostly-extruded skin is
  what the spec warns against. Cost: 93/512 f32 chunks of c1 (~34 GPU min) kept in cache/vol/B256_c1_partial/;
  cache/vol/B_window_c1_rule.json holds the rule's pick.
- M1 toy (a) recompute with the measure fix: 0 label flips vs the M1 volume (M1 numbers stand).

## M1 results (2026-09-15)
- Space-time: trainable 64.80 / 49.29 / 48.94 % at T = 10/100/1000; per-T D (b=2-32) 1.046/1.270/1.367/1.420/1.414/1.412 at T = 10/30/100/250/500/1000 (README 1.05/1.27/1.37/1.42/1.41/1.41). 6-neighbour boundary voxels 300 990. Body stops changing above T ~ 250 (top 75 % of the linear T axis is nearly an extrusion: M3 risk).
- Toy a (net2, f32): conv 58.22 %, boundary voxels 11 543, edge cells 8420, D3(1-16) 2.122 +- 0.013; 175 s, 1494 px/s (1553 excl. compile chunk).
- Toy b (net3, f32): conv 54.83 %, boundary voxels 10 825, edge cells 7501, D3(1-16) 2.066 +- 0.019; 273 s, 959 px/s (1026).
- Null (quad3, f32): conv 15.00 %, boundary voxels 6305, edge cells 3433, D3(1-16) 1.994 +- 0.019; 13 s.
- Slice sigma = 1 vs f64 overview at pixels (16i+8,16j+8): 4084/4096 = 99.71 % agree; 12 misses, all near the boundary (12/214 = 5.6 % of near-boundary px), 0/3882 far. f64 recompute of the plane: 100.00 % vs overview (so all 12 are float32 flips); 236 px/s f64 (4096 nets incl. compile).
- GPU chain wall 03:37:27-03:45:35 (8 min), shared with another process (62 % util at start).

## Log
- 2026-09-15 03:3x: engine + tests pass (net2 matches `tfractal.train_chunk` to rtol 1e-9 with compaction; net3/quad3 grads match autograd).
- Space-time checks reproduce the source README exactly (see report).
- 03:37 GPU chain started (queued ~3 min behind ribbon); ALL_DONE 03:45. analyze_toys.py run, previews inspected. M1 committed.
