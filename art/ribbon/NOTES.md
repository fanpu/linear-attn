# ribbon NOTES (running log + resume)

Piece §4 of `art/ml-art-3d.md`; plan `docs/superpowers/plans/2026-09-15-3d-pieces.md` §4.
Source: `art/edge-of-stability/` (read-only). Model/data code copied into `eosnet.py` with a source comment.

## State (M1)
- Stage A (CPU, `stage_a.py`) analysed: **fails the decision rule** -> Stage B is the M2 source.
- Stage B (GPU, `stage_b.py` via `run_stage_b.sh`) **finished**: 187 s training + 22 s replay (not ~2 h:
  M = 1 on an uncontended GPU; the source's 7611 s was M = 4 under 8-slot contention).
- `verify_b.py` -> `cache/stageB_verify.json`, `cache/stageB/axes.npz`. λ₁·η/2 matches main4 (median 1.9 %).
- Stage C (GPU, `stage_c.py`) queued/running; log `logs/stage_c.log`; outputs `cache/stageC/`.

## main4.npz inventory (what the keys are)
Per step, shape (6000, 4[, …]), model index 1 = 2/η = 80:
- `x, y, z`: ⟨θ_t − θ̄_t, u_j(t)⟩, θ̄ = 21-step centred mean, u_j(t) the CURRENT (moving, sign-tracked)
  eigenvectors. NOT a fixed-frame projection. `dtheta_u1` = ⟨θ_t − θ_{t−1}, u₁(t)⟩.
- `sketch` (…,1024) count-sketch of θ_t − θ₀; `u1_sketch` count-sketch of u₁(t). Hash: torch CPU generator
  seed 1234 (idx then sign) -> reproducible, identical in Stage B.
- `u1_gram` (4,60,60) exact |⟨u₁(a), u₁(b)⟩| at `snap_t` = 0,100,…,5900. `evals, resid, g_u` (…,3).
- There is **no exact projection onto a fixed vector** in the cache; the plan's validation had to use
  x / dtheta_u1 on steps where u₁(t) ≈ u₁(t_ref).

## Key numbers
Stage A (main4, t_ref = 3200, cache/stageA_verify.json):
- sketch cosine vs exact Gram at snapshots: r = 0.991, mean |Δ| = 0.021.
- |cos(u₁(t), u₁(3200))| ≥ 0.9 on only 18 EoS steps (0.32 % of the EoS phase); r_coord 0.902,
  sign 0.778; r_chord 0.905, sign 0.778. Piecewise refs every 100 steps: coverage 25 %, r 0.86, sign 0.78.
- Any single reference covers 0.07–3 % of the EoS phase (u₁ labels swap; main4 lag-100 Gram ~0.1).
Stage B (cache/stageB_verify.json):
- t_edge 405 (= main4). λ₁·η/2 pointwise median rel diff 1.63 % (all) / 1.86 % (after t_edge+200), p95 6.1 %.
  hover median 1.0632 (B) vs 1.0617 (main4); p5–p95 1.017–1.099 vs 1.016–1.098.
- Trajectories identical to 1e-6 (sketch) up to t_edge, then diverge to ~1–2 % (float32 nondeterminism at EoS).
- Replay from float32 anchors is bit-exact (anchor mismatch 0.0).
- Bank (refined, 30 extra iterations): residual/λ medians 0.15 / 0.41 / 1.4 %. Bank u₁ lag-250 overlap
  median 0.17; top-3 subspace overlap 0.30 -> the top subspace itself rotates over 250 steps.
- Stage A METHOD vs EXACT fixed-u_ref projection (Stage B's own sketch, t_ref 3250, whole EoS phase):
  detrended r 0.997, sign 0.967; chord r 0.997, sign 0.976; **raw (not detrended) r 0.33**, rms err 0.024
  vs rms signal 0.014: sketch cross-talk from the large drift swamps the slow part of axis 1.
- Axes: axis 1 = bank u₁ at 3250 (λ 87.77); PCs of θ̄ (558 rows, EoS) capture 94.4 % + 4.9 % of smoothed
  variance; exact axis-1 share of smoothed variance 3.8e-7 (sketch said 0.13 % = cross-talk).

## Decisions
- Decision: validate Stage A against x_t and dtheta_u1_t on steps with sketch |cos(u₁(t), u_ref)| ≥ 0.9 — the cache has no exact fixed-vector series; sketch cosine tracks the exact Gram (r 0.991), and 0.8/0.95 are also reported.
- Decision: the rule needs r ≥ 0.9 AND sign ≥ 0.95 for both the detrended coordinate and the chord (step) sign, AND validated windows covering ≥ 50 % of the EoS phase — "over the EoS phase" cannot be claimed from 18 steps.
- Decision: Stage A is NOT the source for the ribbon curve (fails coverage and sign). Stage B exact projections are the M2 source for all three axes.
- Decision: Stage B runs even so and was queued immediately (canyon needs exact directions).
- Decision: store θ every 10 steps in float32 (1.58 GB), not float16 — anchors must be exact so every step can be replayed (≤ 9 gradient steps) onto any later direction; replay verified bit-exact.
- Decision: θ̄ (21-step centred mean) every 10 steps stored float16 (788 MB); rows 0 and 599 are zeros (incomplete window). Quantisation std per coordinate ~2e-5, noise norm ~0.016 vs drift extent of several units.
- Decision: bank eigenvectors get 30 extra warm subspace iterations that do not feed back into the per-step chain, so the λ trace keeps the main4 method (1 iteration/step) and stays comparable.
- Decision: Stage B also logs the main4-hash sketch per step, so the Stage A method is tested against exact fixed-vector projections.
- Decision: t_ref = 3250, the bank step nearest mid-EoS ((405 + 6000)/2); Stage A used snapshot step 3200.
- Decision: axes 2–3 = top-2 PCs of θ̄ over t ≥ t_edge in full weight space, orthogonalised against axis 1.
- Decision: Stage C computes two 32³ grids: "global" (±1.5 × max |coord| over the EoS phase, the plan's span) and "local" (±1.5 × max |coord| over t_ref ± 200), because the global span along pc1 is set by 5600 steps of drift and may not resolve the oscillation-scale valley. M2 picks.
- Decision: Stage C first layer uses the exact linear decomposition X(W0_ref + Σ c_j W0_j)ᵀ; checked against the plain forward (assert rel diff < 1e-4).

## Risks for M2
- A single fixed axis 1 is only locally meaningful (top-3 subspace overlap 0.30 over 250 steps). The spec's piecewise-fixed frames (sign + Procrustes) are available from the bank (24 frames) and proj_bank covers every step.
- Axis scales are very anisotropic (axis 1 oscillation ~0.03, pc1 drift ~several units): M2 must declare per-axis scaling.

## Files
- `cache/stageA.npz`, `cache/stageA_verify.json`; `cache/stageB/{log.npz, replay.npz (proj_bank 6000×72), bank_vecs.npy (24×3×P), theta_f32.npy, thetabar_f16.npy, ckpt.pt, axes.npz}`; `cache/stageC/{coords.npz, canyon_local.npz, canyon_global.npz, meta.json}`.
- Previews: `cache/preview/stageA_validation_ref3200.png`, `stageA_chart_ref3200.png`, `stageB_verify.png`, `stageC_chart.png`.

## Resume
```bash
cd /home/fzeng/ml/research/art/ribbon; P=/home/fzeng/ml/research/art/.venv/bin/python
OMP_NUM_THREADS=4 $P stage_a.py 3200
setsid nohup ../_shared/gpu1.sh ./run_stage_b.sh > logs/stage_b.log 2>&1 < /dev/null &   # resumes from ckpt
OMP_NUM_THREADS=4 $P verify_b.py
setsid nohup ../_shared/gpu1.sh env OMP_NUM_THREADS=4 $P stage_c.py > logs/stage_c.log 2>&1 < /dev/null &  # resumes per slab
OMP_NUM_THREADS=4 $P preview_c.py
```
