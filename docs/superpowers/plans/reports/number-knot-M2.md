# Number Knot (§6) — M2 report: renders

**Status: DONE.** Commit `f2b3297` (art/number-knot: `geometry.py`, `render_m2.py`, `render_atlas.py`, 28 gallery files, NOTES).
No r3d bugs hit in my paths. Per the controller's heads-up, all splatting ran on CPU with orthographic cameras only, and no `lut_tf` was used.

## What was done
1. **`geometry.py`** (CPU, 20 s) computes every bead position once, measured and null, into `cache/geom_M2.npz` and `.json`. The renders read only this file.
2. **`render_m2.py`** (r3d) draws three styles per scene:
   - **glow:** `splat_spheres` beads, Lambert key light, and depth-tested `splat_additive` hairlines on the connectors;
   - **plaster:** matte pigment, a raking light, and `ambient_occlusion` against an `iso_occluder` on a sampled union-of-balls field (256³);
   - **plotter:** `visible_runs` plus bead circles, then `write_svg`.

   Numbers renders are diptychs: the measured scene beside its shuffled-label null, with one shared camera and scale.
3. **`render_atlas.py`** (matplotlib 2-D plates) makes slice atlases, one panel per layer for every tower, measured and null. This is the spec §0.4 companion to the exterior towers.

Commands (from `art/number-knot/`):
```
OMP_NUM_THREADS=4 ../.venv/bin/python geometry.py                       # 20 s
OMP_NUM_THREADS=4 ../.venv/bin/python render_atlas.py                   # 10 s
setsid nohup env OMP_NUM_THREADS=4 ../.venv/bin/python render_m2.py --size 2400 --device cpu > logs/render_m2_2400.log 2>&1 < /dev/null &   # 8.5 min
```

## Timings (2400 px, CPU, 4 threads; `gallery/timings_2400.json`)
glow 1–5 s, plotter 0.2–0.9 s, plaster 39–59 s per image (most of it the KD-tree field plus 24-ray AO). The full set took 8.5 min.

## Images (all opened and inspected; `art/number-knot/gallery/`)
**Shared declarations (apply to every caption):**
- Chart: towers use layer index as height (declared), and each layer's plane is scaled to unit RMS bead radius (declared per-layer normalisation).
- Light shows form only.
- Colour: cyclic colorcet `cyclic_rygcbmr_50_90_c64_s25` (L* 55–84) on the dark ground; cmcrameri `romaO` pigment at 55 % colour + 45 % white on plaster; single ink `#1f1d1b` on `#f3efe6` paper for the plotter.
- Hardware: GB10 Grace CPU, torch 2.14.0+cu130, fp32 geometry, 2026-09-15.

| File(s) | Measured | Declared |
|---|---|---|
| **HERO** `hero_tower_days_{glow,plaster}.png` (1600×2400), `hero_tower_days_plotter.svg` | Qwen3-0.6B residual stream at each ` <Day>` token, 24 templates × 7 days × 29 layers (emb + 28 blocks), projected onto the layer's supervised mean-difference plane. Held-out-template circle R² is 0.77–0.92 across layers 0–26 (best 0.92 at L12, order-null 99 % 0.69). The true order ranks 1 of 360 at 27/29 layers. | Each layer rotated/reflected by orthogonal Procrustes so the class means face the calendar angles; per-layer RMS scaling; layer spacing 0.2. Rings join the 7 class means of one layer in calendar order; threads join one day's means across layers. Camera az 30°, el 40°, orthographic. |
| **HERO** `hero_tower_months_{glow,plaster}.png`, `hero_tower_months_plotter.svg` | Same pipeline for 12 months: held-out R² 0.70–0.92 (best 0.90 at L13, null 99 % 0.44). Exact cyclic order at 18/29 layers; L0–2 and L27–28 visibly deform (the flare at the bottom right is the embedding and first blocks). | Same as the days hero. |
| `tower_days_{glow,plaster}.png` (3200×2400), `tower_days_plotter.svg` | Measured tower beside the point-label-shuffle null (labels permuted across all 168 points, then plane, Procrustes and scaling through the identical pipeline). The null's means collapse to the axis and its beads are an isotropic column. | Same as the hero; el 28°. |
| `tower_months_{glow,plaster}.png`, `tower_months_plotter.svg` | As above, for months. | As above. |
| `tower_numbers_{glow,plaster}.png`, `tower_numbers_plotter.svg` | OLMo-2-0425-1B, `The number {a}`, a = 0–999, 17 layers. Each layer's measured states are orthogonally projected onto that layer's fitted T=100 (cos, sin) plane (PCA-100 frame; ΔR²_T100 0.026–0.048 by layer). Null uses shuffled a. | The fitted frame fixes the phase (no Procrustes); per-layer RMS scaling; spacing 0.32. Colour = a mod 100. Rings join the 100 residue-class means per layer. The bright core of the null is its collapsed ring of means. |
| `helix_{glow,plaster}.png` (3600×2400), `helix_plotter.svg` | OLMo-2 L1, `The number {a}`, 0–999. Beads are orthogonal projections of the measured PCA-100 states onto an orthonormal frame from the K&T fit, in residual-stream units (same units as the null): e1 = fitted cos100, e2 = sin100 ⟂ e1, e3 = linear ⟂ both. ΔR²_T100 = 0.048 of the 100-PC variance (PCA-100 = 47.8 % of the layer's variance); the null gives 0.002. Linear coordinate vs a: r = 0.975. Median in-sample angular error 10.8° (null 65°). | The fitted curve C·B(a) is drawn as a thin opaque line and labelled as the fit, not data. The glow hairlines joining consecutive integers are declared connectors. Colour = a mod 100 (the period drawn). el 35°. |
| `knot_{glow,plaster}.png` (4800×2400), `knot_plotter.svg` | Same cell. θ and ρ are the measured angle and radius in the T=100 plane; φ and r are the measured angle and radius in the T=10 plane (ΔR²_T10 0.031, null 0.0025; median angle error 12.5°, null 66°). **T=100 is a genuine ring; T=10 is an ordered ring of clusters** (M1: T=10 holds 24–27 % of all last-digit structure vs 22 % for cluster geometry). | Torus composition ((ρ + s·r·cos φ) cos θ, (ρ + s·r·cos φ) sin θ, s·r·sin φ) with s = 0.35 (declared; the measured minor radius ≈ major would self-intersect). The fitted (1,10) torus knot is drawn once (0 ≤ a < 100) as the declared fit line. Colour = last digit. el 50°. |
| `atlas_{days,months,numbers}_{measured,null}.png` | Slice atlas (inside of the towers): each layer's plane on its own, in the tower's coordinates, titled with held-out R² and the cyclic flag (numbers: ΔR²). | 2-D matplotlib plate; light lines join the class means. |

What the images show:
- The calendar towers read as a coloured cylinder with the days/months in order around it, and the null is a structureless column.
- The helix is legible: 10 turns of the fitted curve with the measured beads clustered along it, against a blob for the null.
- The knot is the weakest image. The measured ring and its hole are visible, but the beads' spread (12° angle error plus radial noise) hides the minor winding, which shows mainly through the fit line.
- `tower_numbers_plotter.svg` is a dense stipple with little structure: 17k circles. The glow and plaster versions and the atlas show the ring far better.

## Decisions (logged in `NOTES.md`)
- **Beads as metric projections.** Beads are orthogonal projections onto a fit-derived orthonormal frame, not the M1 pseudo-inverse decode, which divides by the fitted amplitude and inflated the null ~5×. Measured and null now share one metric scale.
- **Fit line added.** The fitted K&T curve is drawn as a thin "fit" line. Consecutive-integer connectors are glow hairlines only; opaque tubes through noisy beads hid everything. Plaster and plotter omit the hairlines.
- **Helix colour.** Helix beads are coloured by a mod 100; the plan said last digit, which becomes confetti at T=100. The knot keeps last digit.
- **Knot scale.** Minor-radius scale is s = 0.35, and the fit knot is drawn once.
- **Tower normalisation and null.** Per-layer Procrustes and RMS scaling; layer spacing 0.2 (numbers 0.32); point-label-shuffle null through the identical pipeline.
- **Colour.** Cyclic colorcet map on dark (chosen for near-constant lightness, L* 55–84); romaO pigment on plaster.
- **Light.** Defined in the camera frame.
- **Bead sizes.** Towers use smaller beads (days 0.032, months 0.026) so the rings inside stay visible. Connector sampling spacing is ≤ radius/2; the first 2400 pass showed a string-of-pearls artefact, which was fixed and re-rendered.
- **Plaster AO grid.** 192³ for proofs, 256³ for heroes.

## Risks and open issues
- **Tower shape is partly normalisation.** Per-layer scaling hides the real growth of the residual norm (days RMS 0.3 at L0 → 45 at L28). Captions must say so; the atlas titles give the fits. Procrustes can reflect a layer, which is declared.
- **Plaster AO is coarse.** The 256³ grid has h ≈ 0.023 against a bead radius of 0.026–0.04, so the AO sees slightly blobby spheres. It is form-only, but at 100 % zoom the AO grain is visible.
- **Knot and numbers-tower plotter are weak.** Candidates for dropping in M3 or replacing with a T=10-plane small-multiple.
- **Hero AO/splat ran on CPU,** because of the r3d CUDA splat bug the controller flagged. M3 films can move to GPU after that fix; turntable frames on CPU cost ~1–5 s in glow but ~45 s in plaster.
- **Report files are not committed.** `commit.sh number-knot` only covers the piece directory.
