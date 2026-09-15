# ribbon NOTES (running log + resume)

Piece §4 of `art/ml-art-3d.md`; plan `docs/superpowers/plans/2026-09-15-3d-pieces.md` §4.
Source: `art/edge-of-stability/` (read-only). Model/data code copied into `eosnet.py` with a source comment.

## State (M1)
- Stage A (CPU, `stage_a.py`) analysed: **fails the decision rule** -> Stage B is the M2 source.
- Stage B (GPU, `stage_b.py` via `run_stage_b.sh`) **finished**: 187 s training + 22 s replay (not ~2 h:
  M = 1 in share mode, GPU otherwise lightly loaded; the source's 7611 s was M = 4 under 8-slot contention).
- `verify_b.py` -> `cache/stageB_verify.json`, `cache/stageB/axes.npz`. λ₁·η/2 matches main4 (median 1.9 %).
- Stage C (GPU, `stage_c.py`) **finished** (queued 30 min behind rough-skin; ran 39 s): coords for every step + two 32³ canyons.
  fused-vs-plain forward max rel diff 3.2e-8. local span ±(0.058, 0.300, 0.074), loss 0.162–0.382; global span
  ±(0.062, 4.64, 1.81), loss 0.101–1.165; loss(θ_ref) = 0.1775. Quadratic fit of the local canyon along u_ref:
  curvature 87.6 vs bank λ₁(3250) = 87.77 (0.2 %), vertex −0.0089 = centre of the axis-1 oscillation (≈ −0.01).
  Preview `cache/preview/stageC_chart.png`: the ribbon bounces inside the u_ref valley; loss falls along pc1.

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
- Global canyon is symmetric about θ_ref but the EoS trajectory spans pc1 −2…+3 and pc2 0…+1.2: its bounding box fills only ~54 % × 33 % ≈ 18 % of the pc1–pc2 cross-section; M2 may want an asymmetric box (rerun is 10 s).
- Stage B diverges from main4 in detail after t_edge (float32 nondeterminism); the ribbon is Stage B's run, so M2 captions must not cite main4 braid details (burst times) as the same trajectory.
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

## M2 (renders) — log
- Controller rulings: hero = local window + fixed u_ref + local canyon; context = full EoS in piecewise frames + tightened global canyon; strands (splat_spheres) + chords (splat_additive); colour λ₁η/2 diverging at 1; honesty panel; canyon volume with cutaway; captions cite Stage B only.
- `prep_m2.py` -> cache/m2/{hero.npz, context.npz, boxes.json, prep_info.json}.
  hero window 3050–3449: u_ref captures 36 % of the detrended top-3 (bank 3250) oscillation energy; λ₁η/2 1.011–1.116 (all above the edge); u₁(t) swaps (overlap < 0.95) on 34 % of its steps.
  piecewise frames (context): principal oscillation direction per bank frame captures median 41 % of top-3 energy; |⟨v_i, v_{i−1}⟩| median 0.39, min 0.006.
- `canyon_box.py` (CPU, 4 threads; the gpu1 queue was 4 jobs deep, so the queued GPU job was cancelled and the grid ran on CPU): hero32 144 s (loss 0.162–0.279), context32 149 s (0.102–0.434), hero64 (2× check). fused-vs-plain ≤ 1.4e-8.
- `render_ribbon.py` plates: hero, honesty, stereo, context (GPU job `run_m2_renders.sh`, log logs/m2_renders.log), plotter + check64 (CPU).
- Decision: hero window 3050–3449 — centred on t_ref, three clear bursts, highest u_ref capture (0.36) among 400-step windows centred near t_ref.
- Decision: canyon boxes = trajectory range ± 30 % margin each side (hero), ± 10 % in pc1/pc2 and ±1.3 × max |axis 1| (context).
- Decision: exaggeration hero u_ref ×4, pc2 ×4 relative to pc1; context axis 1 ×20, pc2 ×2. pc2 ×4 in the hero so the slow pc2 drift reads as depth (×2 looked planar).
- Decision: colour Spectral_r with TwoSlopeNorm 0.90 | 1.00 | 1.12, same for every plate.
- Decision: canyon TF = upper 65 % of cmc.oslo over the grid's loss range; opacity = 6 Gaussian shells (σ 0.018 in normalised loss) at 0.08…0.88; density 6/box-extent (hero), 1.2 (context); trilinear, 64³ check plate.
- Decision: half cutaway removes pc2 < median pc2 of the window; chords are split at the cut plane so chords behind it are fogged and chords in front are added after the volume (correct ordering without a chord depth buffer).
- Decision: honesty panel replaces only the oscillation part of axis 1 with the stored moving-frame x_t = ⟨θ − θ̄, u₁(t)⟩ (Stage B log); slow part, pc axes, canyon and camera are identical.
- Decision: stereo = rotation stereo (az ± 2.5°), because parallel-shift stereo gives no parallax with orthographic cameras.
- Decision: plotter = hidden-line GD path (visible_runs vs a thin tube depth), loss contours of the far face pc2 = hi at the six TF levels, box wireframe; three Inkscape layers.
- Final renders (GPU job run_m2_renders.sh, queued 04:52, started 06:48, 50 s total): hero_glow 2400² 12 s, honesty 3200×1600 10.6 s, stereo 2800×1400 8 s, context 3000×1860 8.5 s. CPU: check_canyon_32_vs_64 (1024², 48 s), plotter_hero.svg + proof PNG.
- 32³ vs 64³: trilinear(32³) vs 64³ max rel diff 6.1e-4, median 3.6e-4; plates visually identical.
- Observation: strand colours alternate between even and odd steps: λ₁ itself oscillates with period 2 (measured, not a render artefact).
- Open for M3: the context plate's tubes (R 0.0035) dominate the chords at 5600 steps; a thinner-tube variant would read more as a ribbon. The hero's bottom ~20 % is empty (camera framing).

## M3 (re-frame, canyon TF, film, README) — log

- `render_ribbon.py` refactored to be importable without side effects: CLI parsing moved into `_parse_cli()`,
  called only under `if __name__ == "__main__":`; module-level `DEV`/`S`/`DT`/`args` keep safe defaults so
  `render_film.py` can `import render_ribbon as rr` and reuse `Chart`, `ribbon_layers`, `canyon_layer`,
  `composite`, `lam_rgb`, `crop_to_content` directly (setting `rr.DEV`/`rr.DT`). `camera()`'s `w`/`h` defaults
  changed from `=S` (bound at def time, so they froze at the import-time value) to `=None` with an
  at-call-time fallback to the current `S` — a latent bug the refactor would otherwise have introduced.
  Verified with a size-96 smoke render after the refactor: output matches pre-refactor behaviour.
- Decision: hero re-frame is a pure post-composite **crop**, not a camera change. `crop_to_content()` finds
  the row bounding box of non-background pixels (eps 0.03) in the *unlabelled* composite, then keeps a 6%
  top pad and a 7% bottom pad (room for the caption), before `annotate()` runs on the cropped canvas. Camera
  az/el/radius/zoom, chart, axes and exaggeration are all unchanged. Old hero_glow.png was 2400×2400 with the
  bottom ~21% empty (measured: content rows 114–1895 of 2400, i.e. top margin 4.75%, bottom margin 21%.
  New hero_glow.png is 2400×2012 (aspect 1.19), content centred.
- Decision: canyon transfer function replaced. M2's 6 equal-weight Gaussian shells read as flat stacked
  sheets (loss is nearly quadratic along u_ref alone, so equal-loss shells are nearly flat planes). New:
  `opacity_floor(x) = exp(-x / FLOOR_TAU)`, x = normalised loss (0 at the grid minimum), FLOOR_TAU = 0.10 —
  chosen by comparing 0.10/0.14/0.22/0.30 on the hero window (`cache/preview` not kept, inspected directly):
  0.10 gives the most legible bright floor with visible falloff; 0.30 reads as generic fog. This is a
  monotone re-weighting of the same measured scalar field (declared in every canyon caption), not new
  structure — checked against the loss profile along each axis (`cache/m2/canyon_hero32.npz`): along u_ref
  the floor slice loss goes 0.266 (lo) -> 0.1746 (min, at u_ref ~ -0.0086, matching Stage C's fitted vertex
  -0.0089) -> 0.261 (hi); along pc1 at the floor slice loss only spans 0.1627-0.1873 across the whole ±0.30
  range (the canyon's long, nearly flat trench direction). Applied uniformly via `canyon_layer()`'s default
  `opacity_floor`, so hero, honesty, stereo, context and check64 all use it (re-rendered together for
  consistency; none of these needed a data change, only the TF).
  Old TF_DOC / SHELLS constant is kept only for the plotter's contour levels (unrelated to opacity).
- Decision: context plate tubes thinned R 0.0035 -> 0.0018 so the additive chord glow shows between strands
  over the full 5585-step EoS phase (open item from M2). Also fixed the context caption's overlap number:
  it quoted "top-3 subspace overlap 0.30" (a Stage-B bank-rotation statistic, cache/stageB_verify.json),
  which is not the context plate's own quantity; replaced with "median adjacent-frame overlap 0.39"
  (`frame_align_median` in cache/m2/prep_info.json = 0.3935, the actual per-frame alignment statistic
  computed for this plate in prep_m2.py) per controller ruling.
- Decision: `canyon_layer()` gained a `step_div` parameter (default 400, unchanged for existing stills) and
  an in-process cache of the loaded loss grid, both film-only optimisations. `step_div=120` (coarser ray
  march) was checked side by side against 400 at hero-window frames and is visually indistinguishable,
  because the new floor TF is smooth/broad (unlike M2's narrow width-0.018 shells, which needed the finer
  step). This alone was a ~2.5x speedup (S=500: 6.4s/frame -> 2.5s/frame).
- Film design (`render_film.py`): camera target = the 21-step centred mean (`uniform_filter1d`, mode
  "nearest") of the hero window's own fixed-frame coordinates (the "central flow" of the window, oscillation
  removed), az/el/radius/zoom constant at the hero's values -> a pure translation alongside the path
  (declared: not an orbit). At output frame i (one per window step, 400 total) the ribbon draws only steps
  [max(0, i-149), i] (trailing 150-step window, growing from the start of the window); the canyon is the
  same static hero32 box/TF as the hero plate, re-rendered every frame (parallax as the camera moves), never
  cut away. No new data: reads only cache/m2/hero.npz and cache/m2/canyon_hero32.npz (already computed at
  M1/M2), so the film needed no queued compute job, only rendering.
- Perf check: at S=720 (final choice, <= 1080²), step_div=120, no clip plane: ~5.1s/frame measured on
  frames 200-202 -> 400 frames ~= 34 min, fps=16 -> 25 s film (in the 20-30s range). Frames are skipped if
  their PNG already exists (`cache/film_frames/frame_%05d.png`), so a rerun resumes; this doubled as the de
  facto checkpoint for the pause below.
- Re-rendered for the new canyon TF (crop only for hero): hero_glow.png 149.6s, context_eos_glow.png 114.0s
  (thinner tubes + caption fix), honesty_fixed_vs_moving.png 133.8s, stereo_crosseye.png 102.0s,
  check_canyon_32_vs_64.png 57.5s (all CPU, 4 threads). plotter_hero.svg/_proof.png untouched (no canyon TF
  in that plate; only contour levels from the raw grid).

## PAUSED (M3) — 2026-09-15, controller pause request

**Stopped cleanly at the controller's request.** No corruption: the killed process (`render_film.py`) writes
one complete PNG per frame and checks `os.path.exists` per frame, so partial progress is safe to resume from.

**Done:**
- Hero re-framed (crop_to_content), committed.
- Canyon valley-floor TF (FLOOR_TAU=0.10) designed, applied to hero/honesty/stereo/context/check64, all
  re-rendered and committed.
- Context plate thinned tubes + caption fix, re-rendered and committed.
- `render_ribbon.py` refactored to be safely importable (see above), committed.
- `render_film.py` written and validated on individual test frames (frames 0-2, 200-202, 399 all inspected
  and correct), committed.
- Film full run started (`logs/film_render.log`, 720², fps 16, step_div=120): **killed at frame ~22/400**
  (`cache/film_frames/` has partial frames; `cache/` is gitignored, nothing to commit there).

**Not done (remaining M3 work):**
1. Finish the film: rerun the exact command below (resumes from existing frames), then check the MP4 size
   (<=20MB; re-encode crf<=27 if not — `render_film.py` already does this automatically) and look at a few
   frames plus the assembled MP4/GIF.
2. Write `README.md` (does not exist yet) per the plan's M3 spec: hero + film, "The phenomenon" (EoS
   Cohen et al. ICLR 2021 arXiv:2103.00065; central flows Cohen/Damian/Talwalkar/Kolter/Lee ICLR 2025
   arXiv:2410.24206), "The pieces" (per-image captions: hero, honesty, stereo, plotter, context, check64,
   film), "What was computed" (wall-time table, numbers are in this file and in
   `docs/superpowers/plans/reports/ribbon-M1.md` / `ribbon-M2.md`), "Verification" (Stage A failure +
   sketch-drift-leakage reason, Stage B lambda1*eta/2 vs main4 median ~1.9% [1.86% after t_edge+200, 1.63%
   all steps], bit-exact replay, canyon curvature 87.6 vs lambda1 87.77, 32³ vs 64³ max rel diff 6.1e-4),
   "Caveats" (fixed axis only local: top-3 subspace overlap 0.30 after 250 steps for hero/honesty/stereo,
   median adjacent-frame overlap 0.39 for context; Stage B != main4 trajectory after t_edge, do not cite
   main4 burst times; stereo pairs rotate rather than shift; alternating strand colour is measured, lambda1
   itself oscillates period 2), "References". Run the link check before committing (command in the plan and
   in `art/ml-art-3d.md` §0 shared brief item 5).
3. Write the M3 report to `docs/superpowers/plans/reports/ribbon-M3.md`.
4. Final `commit.sh ribbon "..."`.

**Resume commands:**
```bash
cd /home/fzeng/ml/research/art/ribbon
P=/home/fzeng/ml/research/art/.venv/bin/python
OMP_NUM_THREADS=4 $P render_film.py --size 720 --device cpu --fps 16 > logs/film_render.log 2>&1   # resumes
tail -f logs/film_render.log   # ~5.1s/frame, ~34 min total for the frames not yet on disk
# then: write README.md, write the M3 report, commit.sh ribbon "<msg>"
```
