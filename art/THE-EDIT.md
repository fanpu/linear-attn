# The Edit — 15 works

*Companion to [display-directions.md](display-directions.md). That document says the edit is the
highest-leverage move available. This is the edit.*

## Method

I looked. Contact sheets of all 55 stills the README nominates as highlights, a second sheet of
every render ≥5000 px wide (the only ones that can be printed at wall scale, several of which the
README does **not** nominate), and representative frames pulled from twelve films. Then clustered
the highlights by ground and palette to check a suspicion. Selections below are by eye; the
reasoning is stated so you can overrule it.

---

## The finding that shaped the edit

The archive has a **palette monoculture**, and the README's own picks make it worse.

Clustering all 55 highlight stills by median lightness and hue distribution:

| ground | palette | n |
|---|---|---:|
| dark | warm | 16 |
| light | warm | 10 |
| dark | warm + cool | 7 |
| light | warm + cool | 6 |
| light | **neutral (ink on cream)** | **5** |
| dark | cool | 3 |
| dark | mixed | 3 |
| dark | neutral | 2 |
| light | cool | 2 |
| light | mixed | 1 |

**39 of 55 are warm-dominant.** Orange-into-teal on a dark ground is the house reflex, and it is
applied to gd-bifurcation, trainability-fractal, signal-propagation, game-chaos, outcome-basins,
depth-roughness, diffusion-basins, lattice and precision-divergence alike. Individually each is
handsome. **Hung together they cancel.** A viewer walking the room would read nine different
phenomena as one wallpaper, which is the precise opposite of the Becher effect — the grid only
reveals difference when everything else is held constant, and here the *subject* varies while the
*treatment* is constant, so the eye sorts by treatment and sees nothing.

Meanwhile the rarest register — **ink and neutral on cream**: `grokking/specimen_plotter`,
`float-ruler/specimen_engraved`, `ouroboros/fern_sheet_sepia_ink`, `rough-skin/diptych`,
`attention-textile/overlay` — is **5 of 55**, and it is by a wide margin the most distinctive
work in the repo. It looks like nothing else in computational art: it reads as a specimen sheet,
a botanical plate, an engraver's proof. The spectral fractals look like every other GPU-rendered
fractal on the internet, however truthfully computed.

**So the edit deliberately inverts the ratio.** Two spectral fractals, not nine. The cream/ink
register promoted from curiosity to spine. This is the single decision that turns a folder of
good images into a show with a position.

---

## The 15

Print ceilings use the 8,600 px/m rule from `display-directions.md` §3.2 — the width at which the
image still holds true detail at a 40 cm read, with no interpolation.

### Movement I — Laws (ink on cream, hung small and tight)

The room that teaches the viewer to look closely, so Movement II can land.

| # | Work | File | Max print | Why |
|---|---|---|---|---|
| 1 | **The Ruler** | `hardware/float-ruler/gallery/specimen_engraved.png` | 0.84 m | The best single object in the repo. Six number formats as physical combs of ticks on cream — int4, int8, FP4 E2M1, FP8 E5M2, float16, bfloat16. Reads instantly as a specimen sheet; rewards a minute; is exactly true. **Engrave it in brass** and it stops being an image. |
| 2 | **Circuit Formation** | `art/grokking/gallery/specimen_plotter.png` | 0.42 m | Rows of circles becoming star polygons {113/k}, ordered by k. Pure Becher typology, already executed. The one case in ML where a network converges on a shape a Greek geometer would recognise, and it is drawn like a botanical plate. **Plot it.** |
| 3 | **Filix ouroborum, Pl. VII** | `art/ouroboros/gallery/fern_sheet_sepia_ink.png` | 0.43 m | A fern disintegrating across generations of self-training, presented as a 19th-century botanical plate with a plate number. The conceit and the content are the same joke and it is a good one. Model collapse has never looked like this. |
| 4 | **Specimen: 64 Bayer tiles** | `hardware/posterize/gallery/bayer_specimen8_riso.png` | 0.66 m | Pink and blue halftone tiles in a 64-cell grid. **Must be risograph** — a piece about quantisation, printed by a quantiser. The material argument is free here. |
| 5 | **Every orbit ends in a cycle** | `hardware/precision-divergence/gallery/graph_float16_plotter.png` | 0.98 m | The plotter variant, not the fire one. Branching river systems in pencil-grey converging on a small closed polygon — the 40-cycle every float16 orbit eventually falls into. Quiet, strange, and the largest true-detail render in the repo. |
| 6 | **Where the heads look** | `art/attention-textile/gallery/overlay_sierpinski-doubling_indigo.png` | — | Induction-head attention on a self-similar sequence, drawn as a triangular text field in indigo on cream. Typographic, almost a concrete poem. Small, hung at 1 m. |

### Movement II — Edges (the wall-scale pieces)

Three works, three walls. Unglazed, dibond-mounted, bench at 40 cm. One law element each.

| # | Work | File | Max print | Why |
|---|---|---|---|---|
| 7 | **Cascade** | `art/gd-bifurcation/gallery/hero_dark_fire.png` | 0.93 m | The bifurcation cascade of gradient descent in red and gold on black. This is the emotional centre of the show and the best pure image in the archive — it reads as a form at 10 m and dissolves into filament at 30 cm. Re-render at 13,000 px for a 1.5 m hang. |
| 8 | **Cascade (riso)** | `art/gd-bifurcation/gallery/hero_riso_lyapunov.png` | 0.97 m | The *same computation* in blue on cream with the pink Lyapunov trace beneath. Hung directly opposite #7. This pairing is the LeWitt argument made visible: one instruction, two executions, neither the original. It also quietly proves the colour work in `color-research/` was not decoration. |
| 9 | **Not Whether, But Which** | `art/trainability-fractal/gallery/hero_deep_swirl_spectral_print.png` | **0.24 m** ⚠ | The sunset-over-ocean plane where every pixel is a separate network trained 500 steps. Best one-sentence reveal in the repo. **Currently 2048 px — this is a 24 cm object.** It is in the show only if it is re-rendered at ~13,000 px. That is the one piece of compute the show requires. |

### Movement III — Forms (objects, and the dark room)

| # | Work | File | Max print | Why |
|---|---|---|---|---|
| 10 | **Same weights, one activation apart** | `art/rough-skin/gallery/diptych_heaviside_L1_vs_relu_cutaway.png` | 0.48 m | Two solids side by side: one crumpled to D = 2.426 ± 0.031 against theory 2.5, one a smooth piecewise-linear fold. **Cast both from the STLs.** As objects on a plinth this is the most immediately legible piece in the show — you can see the difference with your hands. |
| 11 | **Terminal Phase** | `art/neural-collapse/gallery/stl/terminal_phase_{I,II,III,IV,V}.stl` | — | Five printed solids, epochs 0 → 250, a cloud resolving into a regular tetrahedron. A five-object Becher typology in three dimensions, and it already exists as five STLs nobody has printed. |
| 12 | **One Basin, by width** | `art/invariant-tori/gallery/tori_glow.png` + `film_eps_sweep.mp4` | 0.28 m | Nested tori as glowing filament on black; the film's vortex frame is extraordinary. Show as a small print beside a looping projection — the only moving image in the room. |
| 13 | **Filter-Normalized** | `art/loss-landscape/gallery/hachure_dark_resnet56_noshort_g101.png` | 0.30 m | The no-shortcut ResNet-56 loss surface as a hachured survey engraving — concentric, moiré, genuinely odd. **Mill it as a relief** from `gallery/stl/resnet56_noshort_g101.stl` and hang the print beside the object. |
| 14 | **Dither** | `hardware/dither/gallery/audio/*.flac` + `hero_sd_idle_zoom_riso.png` | 0.48 m | A dark alcove, good speakers, int2 → int4 undithered → int4 TPDF on a loop. The only work in the repo that is complete in a non-visual medium and has never been played to anyone. The radiating-line riso print is the wall piece; the sound is the work. |
| 15 | **Progressive Sharpening** | `art/edge-of-stability/gallery/plotter_eta80.svg` | — | **A plotter running for the duration of the show**, in the entrance, drawing 40,000 steps of oscillation against the single straight line at 2/η. The oscillation is genuinely at pen-stroke frequency. This is the certificate executing, and it is the piece people will photograph. |

---

## What got cut, and why

**Eight spectral fractal planes.** `signal-propagation/frontier`, `game-chaos/lyap_zoom_sequence`,
`outcome-basins/zoom_zX`, `depth-roughness/poster_hammer`, `diffusion-basins/zoom_iter_ring8`,
`lattice/lattice_spectral_fp32`, `gd-bifurcation/print_lyapplane_AB`, `precision-divergence/hero_float16_c1_rivers`.
Every one is a good image. All eight are the same image at the level a viewer perceives. Keeping
two is what makes those two land. *(`lattice` is the most painful cut — as woven textile it has a
genuinely different reading, and it is the first work I would add back at #16.)*

**Everything that is a figure.** `scaling-dimension/agree_dark`, `roofline_*`, `pulse_*`,
`staircase_paper`, `mode-connectivity/width_planes`, `ouroboros/phase_ring`,
`fingerprint/divergence_raster`, `hessian-spectrum/plates_exact_emission`. Axes, legends, error
bars, captions inside the frame. These are excellent *content* — they belong in the catalogue,
the wall text and the feed — but on a wall they read as a conference poster, and one poster in a
room re-labels everything around it as posters. `hessian-spectrum` hurts to cut: *Bulk and
Outliers* as spectroscopic plates is a lovely conceit undermined by the C = 2…10 row labels and
the density curves. **Strip the apparatus and it is straight back in.**

**Second-best variants.** `hero_paper_ink` (loses to #8 on the same wall), `nautilus_sheet_*`
(loses to #1), `decode-map/story_tp256_glass` — which is a genuinely beautiful stained-glass
plane, and my most uncertain cut. At 2924 px it is a 34 cm print; at that size it competes with
Movement I's ink works and loses on strangeness. Re-render it at 8,000 px and it displaces #12.

**The whole of `color-research`.** It is the method, not the work. It belongs in the catalogue as
the essay, where it is very strong.

**`weight-spectrum/Unknown Pleasures of SGD`.** Witty, and a superb *feed* post. On a wall the
Joy Division quote does the talking and the ridgelines stop being about ESDs.

---

## Print feasibility: the honest tally

| status | works |
|---|---|
| printable at ≥0.8 m today | #1, #5, #7, #8 |
| printable at 0.4–0.7 m today | #2, #3, #4, #6, #10, #14 |
| **object, no print needed** | #11, #13, #15 |
| **needs re-render before it can hang** | **#9 only** (2048 → ~13,000 px) |

So the show is **one compute job away from being physically possible**, plus a plotter, a
resin printer, a riso run and a mill. That is a much smaller gap than 1,474 files suggests.

---

## Next

Labels. Fifteen tombstones and fifteen 50–100 word texts, per `display-directions.md` §3.4 —
each of which is simultaneously the wall label, the alt text, the post caption and the catalogue
entry. Say the word and I'll draft them.
