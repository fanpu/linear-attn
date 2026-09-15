# Ribbon (§4) — M1 report

**Status: DONE.** Stage A was analysed and fails the decision rule. Stage B (exact rerun) and Stage C (canyon) both finished and pass their checks. No packages were installed.

Directory: `/home/fzeng/ml/research/art/ribbon/`. Running log and resume commands: `NOTES.md`.

## 1. What the cache actually contains (`art/edge-of-stability/cache/main4.npz`)

Model index 1 is 2/η = 80 (η = 0.025). Per-step arrays have shape (6000, 4[, …]).

| key | shape | meaning (checked in `eos_batch.py`) |
|---|---|---|
| `x, y, z` | (6000,4) | ⟨θ_t − θ̄_t, u_j(t)⟩. θ̄ is the 21-step centred mean; u_j(t) is the **current, moving** eigenvector (sign-tracked step to step). |
| `dtheta_u1` | (6000,4) | ⟨θ_t − θ_{t−1}, u₁(t)⟩ |
| `sketch` | (6000,4,1024) | count-sketch of θ_t − θ₀ (hash: torch CPU generator seed 1234) |
| `u1_sketch` | (6000,4,1024) | count-sketch of u₁(t) |
| `u1_gram` | (4,60,60) | exact \|⟨u₁(a), u₁(b)⟩\| at `snap_t` = 0, 100, …, 5900 |
| `evals, resid, g_u` | (6000,4,3) | top-3 λ (1 warm subspace iteration per step), Rayleigh–Ritz residuals, gradient along u_j |
| `loss, acc, gnorm, u1_overlap_prev, wdist, eig_fresh` | (6000,4) | per-step scalars |
| `checks_t/v` | (12,)/(12,4,3) | cold-start audits every 500 steps |
| `invs, eta, meta, …` | | run metadata (seed 0, float32, n = 5000, P = 656,810) |

**The cache holds no exact projection onto a fixed vector.** The "exact projection series" is x / dtheta_u1 along the moving u₁(t). The "sketches" are 1024-bin count-sketches of θ_t − θ₀ and of u₁(t).

## 2. Stage A (CPU, `stage_a.py`, t_ref = 3200)

The chart follows the plan:
- **axis 1:** ⟨S(θ_t − θ_ref), S u₁(t_ref)/|S u₁(t_ref)|⟩.
- **axes 2–3:** top-2 PCs of the 21-step centred mean of the sketch trajectory over the EoS phase, orthogonalised against axis 1. They carry 94.5 % and 4.7 % of the smoothed variance.
- The count-sketch is declared as an approximate inner-product-preserving random projection.

Validation ran only on steps where u_ref ≈ u₁(t). The per-step overlap is the sketch cosine, which tracks the exact Gram at the 60 snapshots with r = 0.991 (mean |Δ| 0.021). x and dtheta_u1 were sign-aligned by the sign of that cosine.

| window | EoS steps | coverage of EoS | r (detrended coord) | sign (coord) | r (chord Δ) | sign (chord) |
|---|---|---|---|---|---|---|
| single ref, \|cos\| ≥ 0.8 | 28 | 0.50 % | 0.846 | 0.714 | 0.837 | 0.714 |
| **single ref, \|cos\| ≥ 0.9** | **18** | **0.32 %** | **0.902** | **0.778** | **0.905** | **0.778** |
| single ref, \|cos\| ≥ 0.95 | 14 | 0.25 % | 0.914 | 0.857 | 0.917 | 0.857 |
| piecewise refs every 100 steps, ≥ 0.9 | 1380 | 24.7 % | 0.859 | 0.780 | 0.854 | 0.780 |

Any single reference is close (≥ 0.9) to u₁(t) on only 0.07–3 % of the EoS phase. The main4 Gram at lag 100 is typically about 0.1, because u₁ labels swap within the near-degenerate top subspace.

**Decision: FAIL. Stage A is not the source for the ribbon curve.**
- Sign agreement is 0.78, below the 0.95 threshold.
- Validated windows cover 0.3 % of the EoS phase (25 % with piecewise references), so no claim "over the EoS phase" is possible.
- The rule is logged in NOTES as r ≥ 0.9, sign ≥ 0.95 for both the coordinate and the chord, and coverage ≥ 50 %.

## 3. Stage B (GPU, `stage_b.py`, one gpu1.sh job, `logs/stage_b.log`)

Stage B was queued as soon as Stage A's shape was clear. It follows the main4 protocol with M = 1:
- 2/η = 80, 6000 steps, eigenvalues every step, cold audit every 500 steps, seed 0, float32.
- Model and data code are copied into `eosnet.py` with a source comment.
- The sketch hash is identical to main4's.

Saved outputs:
- **Bank:** top-3 eigenvectors every 250 steps, refined with 30 extra iterations, 24×3×P float32 (189 MB).
- **θ every 10 steps:** float32 (1.58 GB), so the replay is exact.
- **θ̄ every 10 steps:** float16 (788 MB, declared).
- **Per-step logs:** λ, residuals, moving-u projections and sketches.
- **Checkpoint** every 500 steps; the resume path was tested in a 40-step smoke run.
- **Replay pass:** rebuilds ⟨θ_t − θ₀, bank vector⟩ for **every** step and all 72 vectors (`replay.npz`).

Wall time: 187 s training plus 22 s replay, versus the ~2 h estimate. The main4 run needed 7611 s for M = 4 under 8-slot contention. A GPU smoke test (13 s) came first.

Results (`cache/stageB_verify.json`):
- **λ₁·η/2 vs main4:** pointwise median relative difference **1.63 %** (all steps) and **1.86 %** (after t_edge + 200); p95 6.1 %. **PASS** (≤ 5 %).
  - Hover median 1.0632 (B) vs 1.0617 (main4); p5–p95 1.017–1.099 vs 1.016–1.098.
  - t_edge = 405 in both runs; final loss 0.0992 vs 0.0944.
- **Trajectory identity:** sketches agree to 2e-6 up to t_edge, then differ by 1–2 %. This is float32 nondeterminism amplified at the edge. The statistics match, but the burst-by-burst detail is not the main4 trajectory.
- **Replay:** anchor mismatch after 10 replayed steps is exactly 0.0 (bit-exact).
- **Bank quality:** residual/λ medians are 0.15 / 0.41 / 1.4 %.
- **Rotation:** bank u₁ lag-250 overlap median 0.17; top-3 subspace overlap 0.30.
- **Stage A's method tested against exact fixed-vector projections** (Stage B's own sketch, u_ref = bank u₁ at 3250, whole EoS phase):
  - detrended coordinate: r **0.997**, sign **0.967**;
  - chord: r **0.997**, sign **0.976**;
  - **raw axis 1 (not detrended): r 0.33**, rms error 0.024 vs rms signal 0.014.

  So the sketch recovers the zig-zag well, but the slow part of axis 1 is swamped by cross-talk from the large drift (|θ_t − θ_ref| up to several units, error ~|v|/√1024). Stage A failed on main4 because no exact fixed-frame series existed to validate against, plus this slow-axis error. The sketch is not bad for the oscillation.
- **Axes** (`cache/stageB/axes.npz`, orthonormal to 1e-6):
  - axis 1 is bank u₁ at t_ref = 3250 (λ 87.77), the bank step nearest mid-EoS;
  - axes 2–3 are top-2 PCs of θ̄ over the EoS phase (558 rows) with u_ref removed, carrying 94.4 % and 4.9 % of the smoothed variance;
  - the exact share of smoothed variance along u_ref is 3.8e-7 (the sketch reported 0.13 %, which is cross-talk).

## 4. Stage C (GPU, `stage_c.py`, `logs/stage_c.log`)

Stage C waited 30 min in the queue behind rough-skin, then ran in 39 s. Outputs:
- **Coordinates:** exact per-step values on the three axes, replayed (`cache/stageC/coords.npz`, including the 21-step mean path).
- **Canyon grids:** two 32³ loss volumes centred at θ(3250) over the three unit directions, on 5000 images, float32 forward with float64 sums.
  - `canyon_global.npz`: the plan's span, ±1.5 × max |coord| over the EoS phase = ±(0.062, 4.64, 1.81). Loss 0.101–1.165.
  - `canyon_local.npz`: ±1.5 × max |coord| over t_ref ± 200 = ±(0.058, 0.300, 0.074). Loss 0.162–0.382.
  - Loss(θ_ref) = 0.1775.
- **Checks:**
  - The fused first-layer path agrees with the plain forward to 3.2e-8.
  - A quadratic fit to the local canyon along u_ref gives curvature **87.6**, vs bank λ₁(3250) = **87.77** (0.2 %).
  - The fitted valley floor sits at −0.0089, which is where the axis-1 oscillation is centred (≈ −0.01).

  The ribbon does bounce inside the valley that the curvature predicts.

## 5. Previews (all opened and inspected)

- `/home/fzeng/ml/research/art/ribbon/cache/preview/stageA_validation_ref3200.png`: overlap trace and the two validation scatters (18 points, two offset branches).
- `/home/fzeng/ml/research/art/ribbon/cache/preview/stageA_chart_ref3200.png`: sketch-chart 3D scatter; detrended sketch axis 1 vs exact moving-u₁ x (they disagree outside ±9 steps).
- `/home/fzeng/ml/research/art/ribbon/cache/preview/stageB_verify.png`:
  - λ₁·η/2 overlay of B and main4;
  - exact vs sketch detrended axis 1 (visually indistinguishable);
  - raw axis 1, where the sketch drifts from +0.03 to −0.03 while the exact value stays flat.
- `/home/fzeng/ml/research/art/ribbon/cache/preview/stageC_chart.png`:
  - exact 3D chart with even and odd strands, and the EoS phase coloured by λ₁η/2;
  - local and global canyon slices with the trajectory overlaid;
  - 1D canyon cuts (steep along u_ref, gentle along the PCs).

## 6. Decisions (full list in NOTES.md)

- **Validation windows:** sketch |cos| ≥ 0.9 (0.8 and 0.95 also reported). Validated against x and dtheta_u1, because no fixed-frame series exists.
- **Rule:** requires the coordinate and the chord sign, plus ≥ 50 % coverage of the EoS phase.
- **Stage A:** not the M2 source. Stage B exact projections are the source for all axes.
- **Stage B:** queued regardless, since the canyon needs exact directions.
- **θ anchors:** float32 rather than float16, so every step can be replayed onto any direction; θ̄ is float16.
- **Bank refinement:** does not feed back into the per-step chain, so the λ trace stays comparable to main4.
- **Sketch in Stage B:** logged with main4's hash, to test the Stage A method against exact projections.
- **t_ref:** 3250, the bank step nearest mid-EoS.
- **Canyon grids:** both a global (plan) grid and a local grid; M2 chooses.

## 7. Risks and open items for M2

- **Axis 1 is only locally meaningful.** The top-3 subspace overlap is 0.30 over 250 steps. For long windows, use the spec's piecewise-fixed frames (sign + Procrustes) from the 24-entry bank. `proj_bank` already covers every step; new directions replay in about 25 s of GPU.
- **Scales are extremely anisotropic.** Axis-1 oscillation is ~0.03; pc1 drift spans −2…+3. M2 must declare per-axis scaling.
- **The global canyon wastes volume.** It is symmetric about θ_ref, but the trajectory's pc1–pc2 bounding box fills ~18 % of the cross-section. An asymmetric box reruns in ~10 s.
- **Stage B is a sibling trajectory of main4,** statistically matched but not identical after t_edge. Captions must not cite main4 burst times as the same run.
- **Cache size:** `art/ribbon/cache` is ~2.8 GB (under the 5 GB limit). `ckpt.pt` (315 MB) could be deleted after M3.

## 8. Commits

- `23fb5da` art/ribbon: M1 Stage A sketch validation, Stage B exact rerun and verification, Stage C script
- `fdc1e28` art/ribbon: M1 Stage C canyon grids, exact chart preview, NOTES
