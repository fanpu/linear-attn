# Critique — three targets, looked at rather than read about

*Run with `anthropic-skills:ml-art-critique` against [THE-EDIT.md](THE-EDIT.md). That document
selected the 15. This one asks whether the selection holds as **images**, in the language used
for abstract painting. The premise of the exercise: the viewer meets the picture before the
idea, and no amount of correct computation rescues a picture that does not hold.*

Evidence is measured, not asserted. Two statistics recur:

- **key commitment** = |mean luminance − 0.5|. High means the image commits to being dark or to
  being light. Low means it sits in the middle, which is where value structure goes to die.
- **boundary visibility (bvis)** = mean |L(boundary pixel) − L(local surround)| for the *measured*
  converge/diverge boundary. It asks whether the thing the work claims to be about is actually
  the thing you can see.

---

## Target 1 — the two kept spectral fractals against the eight cut ones

**What it is.** Eight full-bleed rectangles, each a rank-normalised Spectral split: red-to-cream
on one side of a phenomenon boundary, blue-to-green on the other. `signal-propagation` is four
panels of a magnifying sequence; `outcome-basins` six; `diffusion-basins` six flat-shaded basin
plates; `lattice` a pixel checkerboard; `gd-bifurcation/lyapplane` one 6144² diagonal sweep.
Against them, the two keeps: `hero_dark_fire`, red and gold filament on near-total black, and
`hero_riso_lyapunov`, blue on cream with a pink trace along the bottom edge.

**Working.** The edit's verdict is right, and the eight really are one image at the level a
viewer perceives. But the reason given — palette monoculture — is the symptom. The cause is
**value structure**, and it is measurable. The two keeps sit at opposite extremes of key
(0.07 and 0.88; commitment 0.43 and 0.38). Every one of the cut spectral planes sits in the
middle: signal-propagation 0.48, diffusion-basins 0.43, outcome-basins 0.40, lattice 0.57,
lyapplane 0.65 — commitment 0.02 to 0.15. Across the whole archive the pattern holds: of 244
gallery files whose names carry a scientific colormap, **46.3% sit at commitment < 0.15**,
against **13.5%** of the 533 plotter/riso/ink files. The keeps are not kept for being less warm.
They are kept for committing to a key, which is the decision the spectral family never makes.

The mechanism is one function. `common_render.cdf_img` histogram-equalises each phase onto a
`linspace`, which hands **equal area to every rank by construction**. A flat histogram is a
guarantee that no tonal mass can dominate, and composition is nothing but weighting. This is a
faithful reproduction of Sohl-Dickstein's figure colouring, and as reproduction it is correct —
but it is a colouring built to make data legible in a paper, and it has been carried
unexamined into every print in the repo.

**Not working.** One cut is a mistake. `precision-divergence/hero_float16_c1_rivers` is not a
member of this family at all: key 0.03, 98.9% of the frame below quarter-luminance, a single
pale-blue 40-gon at the optical centre with yellow rivers draining into it across black. It is
focal where the others are fields, it commits harder to low key than `hero_dark_fire` does, and
at thumbnail it is the most legible image in the entire archive — a ring with tributaries. It
was cut as one of the eight spectral planes. It is not one of them.

This matters because the edit kept the *cream plotter variant* of the same computation as #5.
That version is two grey blobs afloat on cream with its most legible element, the red 40-gon,
parked in the lower-left as an afterthought. The spine argument chose cream; the image argument
chooses black. Both cannot be right, and the black one is the better picture.

**The one change.** Stop rank-normalising within phase. Replace `cdf_img` with a gamma on the
speed rank so that slow-near-the-boundary goes dark, which makes the boundary the darkest thing
in the frame instead of the seam where two ends of a colormap happen to meet.

**Next moves.**
1. Swap #5 for `hero_float16_c1_rivers`, or hang both as a black/cream pair the way #7 and #8
   are hung — same computation, two executions, which is the LeWitt argument the edit already
   makes once and could make twice.
2. Re-cut `lattice` with a committed key. It is the cut the edit calls most painful, and its
   problem is not that it is spectral, it is that it is mid-grey at 0.57.
3. Apply the gamma mapping to the two survivors and check they do not get worse; `hero_dark_fire`
   is at 94.7% dark already and may need nothing.

**Verdict.** The edit is right for a wrong reason, and one cut should be reversed.

---

## Target 2 — *Not Whether, But Which* before its re-render

**What it is.** A square, full bleed, no ground. The top two thirds are laminar horizontal bands
of red, orange and cream that lens and swirl toward the upper right. The bottom third is blue
going to green, the same laminar banding. Between them a roughly horizontal break at about 55%
height, ragged, with a thin darker fringe. No frame, no anchor, no mark of scale.

**Working.** The horizon. The value break between the warm upper field and the cool lower one is
the only compositional event, it survives the thumbnail test, and it is not decorative — it *is*
the converge/diverge boundary, the exact subject of the work. Composition and claim coincide,
which is rare and is why this piece was selected at all.

**Not working.** Three seconds gives you a sunset over an ocean. That is the problem, stated
plainly. The piece reads as an agate slice or a stock gradient, which lands it in exactly the
decorative-data-art trap that `display-directions.md` Part II spends a page positioning against.
Nothing in the frame says one million neural networks.

The cause is again `cdf_img`. Measured on the cached field: key 0.572, commitment **0.072** —
this work is tonally a member of the *cut* family, not the kept one. Only 0.2% of the frame is
below quarter-luminance. And the boundary, which the README calls "the dark seam where the two
ends of Spectral meet," scores **bvis 0.026** — the least visible element in the picture is the
one the work is about. The laminar banding elsewhere is higher contrast than the subject.

**The one change.** Make the measured boundary the darkest thing in the frame.

**Next moves.** All seven were rendered from the same cached 1024² field and scored
(`mapping_probe.py`, sheet at `trainability-fractal/gallery/study_mapping_probe.png`):

| mapping | commitment | bvis |
|---|---:|---:|
| current spectral | 0.072 | 0.026 |
| `dark_fire_ice` (exists, unused) | 0.047 | 0.037 |
| `dark_magma` | 0.079 | 0.047 |
| gamma hot/ice 0.55 | 0.081 | 0.078 |
| gamma ember 0.70 | 0.102 | 0.097 |
| gamma hard 1.6 | 0.169 | 0.026 |
| **two-ink + black edge** | **0.168** | **0.237** |

Two-ink is 9× the current mapping on boundary visibility and reads as a print rather than a
render, which also joins it to the cream/ink spine. Its weakness is that salmon and blue-grey sit
close in value, so it is soft at thumbnail (sd 0.189); push the two inks further apart in value,
not in hue. `dark_magma` is the best of what already exists and costs nothing to adopt.

**Verdict.** A study worth pushing, not a finished piece — and the re-render was about to make a
sunset bigger.

> **The compute finding.** THE-EDIT says the show is "one compute job away". It is not. The cached
> 1024² float64 field took **17,757 s (4.93 h)**; float32 at the same resolution took 5,354 s.
> Time scales with pixels, so the proposed 13,000² grid is ~161× the work: **≈ 33 days** at
> float64, ≈ 10 days at float32, on a GB10 that runs one job at a time. A 1.5 m hang is not
> available at any sane budget. What *is* available is 2048² in about 20 h, which at 8,600 px/m
> is a **24 cm** object — and at 40 cm viewing distance that puts roughly one network per
> acuity cell. The print is then exactly as large as the computation, and the label can say so.
> `display-directions.md` §I.1 already sanctioned this: "hang it small and let it be intimate."

---

## Target 3 — the cream/ink spine, works #1–#6

**What it is.** Six light-ground sheets. Four are specimen sheets with titles, captions and
ruled panels: *The Ruler* (seven format rows on a shared log₂ axis), *Circuit Formation* (46
star polygons in a 7-column grid), *Filix ouroborum* (three rows of ferns disintegrating),
*Specimen: 65 Bayer Tiles* (65 halftone squares in pink and blue). Two are not: the float16
cycle graph, two grey branching masses on cream, and the attention triangle, a right triangle of
small indigo and madder cells.

**Working.** The spine holds, and it holds for the reason the edit could not see from the
clustering: ink on cream is a **two-value system**, so notan is forced before any aesthetic
decision is made. Measured commitment is 0.40, 0.41 and 0.42 for works 1, 2 and 5 — the same
decisiveness as the two kept fractals, arrived at by constraint rather than by taste. This is
also why they look unlike other computational art: almost nobody working in this medium accepts
a constraint that removes value as a free variable.

*Circuit Formation* is the strongest thing in the show. It is a true Becher typology — one
variable (k), 46 instances, everything else held rigid, ordered so the eye does the comparison
and watches a circle become a sunburst. It needs nothing.

**Not working.** The six are not one register, and hung together the eye sorts them into
"labelled specimen sheet" and "not", which is the precise failure the Becher principle warns
about: the grid reveals difference only when everything except the subject is held constant.
Here the *treatment* varies — four sheets carry typography and ruled panels, two do not.

Work #6, the attention triangle, is the weakest and should be looked at hard before it hangs.
At thumbnail it is a grey triangle. Its whole claim is the agreement between madder outline
(theory) and indigo fill (measurement) at AUC 0.81 — but 0.81 means a fifth of it disagrees, and
at any normal viewing distance neither the agreement nor the disagreement is visible. The proof
element is real and invisible, which is the same defect as #9 in a different register.

Work #5's composition is discussed above: it loses to its own black variant.

**The one change.** Give the six a single shared frame — same margin, same caption block, same
typeface, same stock — so the register is constant and only the subject varies.

**Next moves.**
1. Strip or standardise the apparatus. Either all six carry a title and a caption rule, or none
   do and the labels go on the wall.
2. For #6, print the disagreement: a third ink, or outline-without-fill left unfilled, so the
   19% is a visible event rather than a number in the caption.
3. `hessian-spectrum` was cut for having axes and row labels. If the spine standardises its
   apparatus anyway, the cut should be revisited — the edit itself says "strip the apparatus and
   it is straight back in."

**Verdict.** Finished as individual works; not yet a series.

---

## What this changes in THE-EDIT

1. **Swap #5 for the black variant**, or hang both as a pair. The cut was wrong.
2. **#9 is not one compute job away.** 1.5 m is ~33 GPU-days. 24 cm is 20 hours and is the
   honest object. Running now.
3. **`cdf_img` is the house reflex to retire**, not Spectral. Palette was the symptom; equal-area
   rank normalisation is the cause, and it is one function used everywhere.
4. **The spine needs one shared frame** before it is a series rather than six good sheets.
