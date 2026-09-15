# number-knot NOTES (handoff)

Spec: art/ml-art-3d.md §6. Plan: docs/superpowers/plans/2026-09-15-3d-pieces.md §6. M1 report:
docs/superpowers/plans/reports/number-knot-M1.md.

## Files
- `common.py` templates, token assertions (id-level prompts: prefix_ids + [target_id], state read at target).
- `extract.py` forward passes -> `cache/hs/<set>_t<k>.npy` (N, L+1, D) float16, resumable per file.
  Hooks on embed + every block => last entry is the residual BEFORE the final norm.
- `fits.py` pure numpy fits (helix, Fourier, calendar circle); `test_fits.py` synthetic tests.
- `analyze.py` -> `cache/fits_numbers_{100,1000}.npz`, `cache/fits_qwen_{days,months}.npz`,
  `cache/tables_M1.md`, `cache/summary_M1.json`.
- `preview.py` -> `cache/preview/*.png` (matplotlib, M1 only).

## Resume commands
```
cd art/number-knot
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest -q test_fits.py
OMP_NUM_THREADS=4 ../.venv/bin/python extract.py --device cpu --tiny   # 20 s smoke test
setsid nohup env OMP_NUM_THREADS=4 ../.venv/bin/python extract.py --device cpu > logs/extract_cpu.log 2>&1 < /dev/null &
OMP_NUM_THREADS=4 ../.venv/bin/python analyze.py
OMP_NUM_THREADS=4 ../.venv/bin/python preview.py
```

## Decisions
- Decision: OLMo-2 templates `{a}` (after BOS), `The number {a}`, `x = {a}`, `Output ONLY a number. {a}` — the plan's `{a}+` is causally identical to `{a}` at the number token (the `+` comes later), so K&T's GPT-J prefix replaces it.
- Decision: prepend OLMo-2's BOS `<|endoftext|>` (id 100257) to every OLMo prompt — its tokenizer adds none, and it is the pretraining document separator; `{a}` alone would otherwise put the number at position 0 (attention-sink position).
- Decision: keep the separate space token (id 220) before numbers in `The number {a}` etc. — the OLMo-2 pretokeniser always splits " 42" into " " + "42", so this is the natural in-distribution form; the number itself is asserted single-token.
- Decision: Qwen3 has no BOS; no prefix token is added. Targets are " <Day>"/" <Month>" with the leading space; the full-string tokenisation is asserted equal to prefix + [target] for every template/word.
- Decision: random-token null = 1000 OLMo-2 vocabulary tokens that decode to >= 2 ASCII letters, no leading space, round-trip to one token, drawn with seed 0 (27277 candidates), placed at the number slot of the same id-level templates; labels 0..999 in random draw order (never sorted by id, which tracks BPE merge order).
- Decision: forward passes on CPU (fp32, 4 threads, ~12 min) instead of the GPU queue — the GPU was held by autonomous/ jobs back to back; the brief allows a declared < 45 min CPU run. fp32 compute, float16 storage (max |h| asserted < 6e4).
- Decision: residual stream captured with forward hooks (pre-final-norm at the last layer) instead of HF `hidden_states`, whose last entry is post-norm.
- Decision: helix R² is variance-weighted over the top-100 PCs (K&T); the circle's contribution is ΔR²_T = R²([1,a,cos,sin]) − R²([1,a]); sin(πa) ≡ 0 is dropped for T = 2 (so T=2 adds one column, parity).
- Decision: nulls use the same 200 permutations for every layer/period so a family-wise (max over layers) 99th percentile is valid. "Clear margin" = best-layer ΔR²_T ≥ 2 × max(family-wise 99% shuffle, family-wise 99% random-token) and ≥ 0.01.
- Decision: calendar plane = mean-difference (top-2 PCs of the class means), not LDA (1024 dims vs 140–240 points makes LDA need a shrinkage choice). Circle regression P = c + A[cos, sin] (A free 2×2, so ellipses allowed). Primary score = held-out-template R² (fit on even templates, score odd, and swap). Nulls: 200 class-order shuffles and 200 point-label shuffles through the identical pipeline.
- Decision: add two comparators the plan did not name — ΔR²([1,a,one-hot(a mod m)]) (all residue-class structure; an isotropic cluster set gives each circle a share of 2/(m−1)) and ΔR²([1,a,a²,a³]) (smooth non-periodic, same column count as a circle). Without them a shuffled-label null cannot tell a circle from clusters or from curvature.
- Decision: exact order null for days (all 360 cyclic orders up to rotation/reflection) — random order shuffles hit rotations/reflections of the true order (identical R² because A is a free 2×2).
- Decision: primary number cell for M2 = template 1 `The number {a}`, layer 1 (output of block 0 = K&T's h^0), fit on 0–999. Rule: largest geometric mean of the family-wise margin ratios for T=10 and T=100 (the knot's two periods) over templates and layers ≥ 1: L1 7.96 vs L12 6.99 (L12 has the marginally larger min ratio, 6.74 vs 6.71). Strongest period = T=100 (9.4× family-wise null, 16% of all mod-100 structure vs 2% isotropic).
- Decision: calendar layers for M2 previews = argmax held-out R² minus the larger null 99%: days L12, months L13; the depth tower uses all 29.

## State (M1 done 2026-09-15)
- Extraction: CPU fp32, 526 s total (logs/extract_cpu.log). The GPU queue attempt was cancelled before it started (logs/extract_gpu_queue_cancelled.log).
- Analysis: 146 s (logs/analyze.log). Tables: cache/tables_M1.md. Summary: cache/summary_M1.json. Previews: cache/preview/.
- Numbers 0–999: every K&T period beats both family-wise nulls by 5–10× at every layer ≤ 14, but ΔR² per circle is only 0.015–0.05 of the 100-PC variance. Fourier over a: sharp peaks at T=100, 50, 25(,33) and at T=10, 5, 2, 2.5 (3.3 weakest).
  Mod-10 structure is near-isotropic (share T=10 0.24–0.27, T=5 0.27, T=2 0.14–0.21 vs 0.22/0.22/0.11) — residue clusters, only mildly circular. Mod-100: T=100 share 0.12–0.16 vs isotropic 0.02 — genuinely circular.
- Numbers 0–99 (K&T range): T=2/5/10 do not clear 2× family-wise null (0.9–1.8×). T=100 clears (3.2–3.5×) but equals the poly3 comparator (0.126 vs 0.126) — curvature, not a circle.
- Qwen3 days: held-out R² 0.92 at L12 (order null 99% 0.69; point null 99% −0.05); true order ranks 1/360 at 27 of 29 layers; cyclic angular order exact at 28/29 layers, already in the embedding (L0 0.87).
  Months: 0.90 at L13 (order null 99% 0.44, max of 200 0.51), cyclic at 18/29 layers (not L1–2, L4–10, L27–28 — Jul/Aug/Sep crowd together).
- Next (M2, needs r3d): helix from cache/fits_numbers_1000.npz `coords[1, 1, COORD_T.index(100)]`, knot from the T=100 and T=10 coords at the same cell, depth towers from `proj` in cache/fits_qwen_*.npz and coords over layers; null panels via preview.helix's shuffled-label fit.

## M2 (renders) — 2026-09-15
Files: `geometry.py` (CPU, 20 s -> cache/geom_M2.npz/.json), `render_m2.py` (r3d; glow, plaster, plotter; 2400 px on CPU,
~9 min; logs/render_m2_2400.log, gallery/timings_2400.json), `render_atlas.py` (matplotlib slice atlases, 10 s).
Resume: `../.venv/bin/python geometry.py && ../.venv/bin/python render_atlas.py && setsid nohup env OMP_NUM_THREADS=4 ../.venv/bin/python render_m2.py --size 2400 --device cpu > logs/render_m2_2400.log 2>&1 < /dev/null &`
- Decision: helix/knot beads = orthogonal projection of the measured PCA-100 state onto an orthonormal frame built from the fit (e1 = cos direction, e2 = sin ⟂ e1, e3 = linear ⟂ both), in residual-stream units — not the M1 pseudo-inverse decode, which divides by the fitted amplitude and inflates the null by ~5×, so measured and null share one metric scale.
- Decision: the fitted K&T curve C·B(a) is drawn as a thin opaque line (declared "fit"); consecutive-integer connectors are glow hairlines only (opaque tubes through noisy beads hid the structure). Plaster and plotter omit the hairlines.
- Decision: helix beads coloured by a mod 100 (the period drawn), knot beads by last digit (its minor period T=10) — plan said last digit for the helix; with T=100 the last-digit colouring is confetti that hides the winding.
- Decision: knot minor-radius scale s = 0.35 (measured median minor radius 1.04 ≈ major 1.28 would self-intersect); knot fit line drawn once (0 ≤ a < 100).
- Decision: towers — per-layer orthogonal Procrustes (reflection allowed) of the class means onto the calendar angles, per-layer scale to unit RMS bead radius, height = layer index × 0.2 (numbers 0.32). Null tower = point-label shuffle (numbers: shuffled a) through the identical pipeline including Procrustes and scaling.
- Decision: cyclic palettes: colorcet `cyclic_rygcbmr_50_90_c64_s25` (L* 55–84) on dark, cmcrameri `romaO` pigment for plaster; light in the camera frame (glow key upper-left front; plaster raking upper-left).
- Decision: all splatting on CPU (controller: r3d CUDA splat bug), orthographic cameras only.
- Decision: slice atlases (per-layer 2-D planes, same coordinates as towers) are the §0.4 companion for every tower.

## M3 (films, README) — 2026-09-15
- Controller rulings applied: fit line thinner (helix r 0.006, knot 0.009) and lower-contrast (×0.45 towards black on dark, towards white on plaster); `tower_numbers_plotter.svg` dropped (render_m2 skips plotter for the numbers tower); knot kept as secondary plate with its weakness stated.
- Films (`films.py`, glow, CPU): film_turntable_{days,months} (720 frames, 1024², 24 s, ~590 s each while sharing CPU), film_sweep_helix (750 frames, 1080×720, 25 s, 525 s). Frames in cache/frames/<film>/ (resume skips existing).
- Decision: turntable MP4s re-encoded at CRF 27 preset slow (write_film uses CRF 18 → 37–47 MB, over commit.sh's 20 MB) — 10–14 MB, visually checked.
- Decision: sweep holds each layer 1 s, 0.5 s linear morphs (declared interpolation); camera frames the central 99 % of beads over all layers; both panels scaled by the measured layer's RMS in-plane radius. Known flaw: the last layer (16) spills out of frame.
- README written (shape of edge-of-stability); link check clean. References verified by arXiv abstract pages: K&T, Engels (SAE), Zhou (title/authors), Štefánik (abstract does not mention OLMo 2).
- Resume: `OMP_NUM_THREADS=4 ../.venv/bin/python films.py --only sweep` (or days/months); `--test <layer>` renders one sweep frame to /tmp.
- Polish (controller): sweep film re-rendered with a declared per-layer zoom (frame fitted to all beads of the layer in both panels, interpolated in morphs, printed as "zoom"); layer 16 and the null tail now fit. 573 s CPU; MP4 9.5 MB (write_film CRF 18), GIF 3.9 MB. README out-of-frame note removed.
