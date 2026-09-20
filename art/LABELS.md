# Labels — fifteen tombstones and fifteen texts

*Companion to [THE-EDIT.md](THE-EDIT.md) and [display-directions.md](display-directions.md) §3.4.
That document sets the convention: tombstone (title, year, medium, dimensions), then interpretive
text of 50–100 words, hard cap 120, active voice, sentences under 25 words, three sentences in
shape — the phenomenon, what you are looking at, the one number.*

**Each text below is written once and used four times: wall label, alt text, post caption,
catalogue entry.** That is the whole reason to write them carefully.

**The medium line is the opportunity.** Its second row is this project's "oil on canvas" — the
grid size, the precision, the machine. It is true, it is checkable, and it is more impressive
than any adjective. The dimensions line is *derived* from it at 8,600 px per metre
(§3.2), so the physical size of every object is itself a declared measurement. Nothing here is
interpolated up to fill a wall.

**Every work carries one proof element** — a law line, a null panel, a control, a stated grid
size — named in the last line of its entry. In the room that is what separates this from
decoration; in the feed it is the entire differentiator.

---

## Movement I — Laws

### 1. The Ruler

> **The Ruler**, 2026
> Engraved brass; archival pigment print on Hahnemühle Photo Rag (edition variant)
> Seven number formats, every positive finite value decoded by bit pattern and cross-checked against torch/numpy; 7200 × 4560 px
> 84 × 53 cm

Every number format is a ruler with its own ticks. Each row is one format's complete set of
positive finite values on a shared log₂ axis: evenly spaced inside an octave, doubling from one
octave to the next, so the pattern repeats exactly at every scale. Seven formats, int4 through
bfloat16, decoded one bit pattern at a time. float16 carries 1,024 values per octave. int4 has
seven values in total, and you can count them from across the room.

*Proof element: the linear-axis strip beneath, where the same seven rulers are redrawn 0 … 8 and
the log-axis claim becomes checkable by eye.*

### 2. Circuit Formation

> **Circuit Formation**, 2026
> Pen plotter on rag paper
> 46 key-frequency embeddings from 12 training seeds, one-layer transformer, (a + b) mod 113; 3600 × 4292 px
> 42 × 50 cm

A one-layer transformer learning (a + b) mod 113 memorises its training set in about 200 steps,
then sits at chance on everything else for tens of thousands of steps before it suddenly
generalises. What it built in the meantime is a circle: each number sits at angle 2πka/113, so
joining a to a+1 draws the star polygon {113/k}. Every star found by twelve seeds is here,
ordered by k — a near-circle at k = 1, a dense sunburst at k = 56. Forty-six stars, no selection.

*Proof element: k and 113 − k give the same star, stated on the sheet and visible in the ordering.*

### 3. Filix ouroborum, Pl. VII

> **Filix ouroborum, Pl. VII**, 2026
> Archival pigment print on cream rag
> Barnsley's fern under 200 generations of self-training, 64-component Gaussian mixture, n = 4096 per generation; 3680 × 3860 px
> 43 × 45 cm

Fit a model to data, sample from it, fit the next model to those samples, and repeat. The top row
is a fern learned once and then retrained only on its own output: by generation 200 the plant is
a handful of splinters. The middle row keeps a quarter of the real samples each generation and
the fern survives all 200 intact. Sliced Wasserstein distance from the true fern climbs 0.021 →
0.122 along the top row and stays at 0.021 along the middle one.

*Proof element: the middle and bottom rows are the controls — same model, same generations, only
the diet changes.*

### 4. Specimen: 65 Bayer Tiles

> **Specimen: 65 Bayer Tiles**, 2026
> Two-colour risograph on cream
> Every tone of the 8 × 8 ordered-dither matrix, each tiled 4 × 4; exact float64 thresholding; 5680 × 2670 px
> 66 × 31 cm

Ordered dithering replaces a grey tone with a fixed threshold pattern. The 8 × 8 Bayer matrix
holds 64 distinct thresholds, so it can render exactly 65 tones — all of them here, tiled four by
four, empty to full. Nothing is approximated: a cell is on wherever the matrix entry falls below
the tone. The print is a risograph, a press that genuinely halftones and misregisters, so the
piece about quantisation is made by a quantiser.

*Proof element: count the tiles. 65, not 64 — the endpoints are included, and that is the whole
arithmetic of the matrix.*

### 5. Every Orbit Ends in a Cycle

> **Every Orbit Ends in a Cycle**, 2026
> Pen plotter on rag paper, two pens
> All 15,361 float16 values in [0,1] under x ↦ round(4x(1−x)); trees rooted on their terminal cycles; 8400 × 4200 px
> 98 × 49 cm

A float has finitely many states, so iterating any map on it must eventually repeat itself. Every
one of the 15,361 float16 values in [0,1] is a node here, joined to wherever the logistic map
sends it, with the trees hanging outward from the cycle they drain into. There are only three
destinations in the whole format. A 40-step cycle catches 66.1% of random real seeds, the fixed
point at zero catches 33.8%, and a two-value cycle at 0.75 catches 0.07%.

*Proof element: the red polygon is the 40-cycle itself, drawn closed. The percentages sum to 100.*

### 6. Where the Heads Look

> **Where the Heads Look**, 2026
> Archival pigment print on cream rag, indigo and madder
> Qwen3-0.6B top-4 induction heads on a 255-letter self-similar sequence, 8 random tokens per letter, 4 draws; 2120 × 2230 px
> 25 × 26 cm

Some attention heads do one specific thing: on a sequence that repeats, they look back to
whatever followed the last copy of the current token. Here Qwen3-0.6B reads a 255-letter
self-similar string, and every cell of the triangle is attention from a later letter to an
earlier one. The madder outlines are where an ideal induction head should look. The indigo fill
is where the model's four strongest induction heads actually looked. The two agree at AUC 0.81.

*Proof element: the madder outline is the theory and the indigo is the measurement, printed in
separate inks on the same cells. Disagreement would be visible as outline without fill.*

---

## Movement II — Edges

### 7. Cascade

> **Cascade**, 2026
> Archival pigment print on Hahnemühle Photo Rag, unglazed, dibond-mounted
> 16,000 step sizes η ∈ [0.48, 0.99]; 8,192 gradient-descent iterates per column after 20,000 discarded; float64, NVIDIA GB10; 8000 × 2000 px
> 93 × 23 cm

Turn the learning rate up on a four-parameter network and gradient descent stops converging on an
answer. It cycles between two, then four, then eight, then everything. Each vertical column is a
single step size: 8,192 successive iterates of the network's output, after 20,000 have been
thrown away, so what you see is where the training run actually ends up living. Brightness is how
often it visits. The dark windows inside the chaos are step sizes where order comes back.

*Proof element: the stated grid. Sixteen thousand independent columns, no smoothing between them
— the vertical striping is real structure, not rendering.*

### 8. Cascade (riso)

> **Cascade (riso)**, 2026
> Two-colour risograph on cream, blue and fluorescent pink
> The same computation as *Cascade*; pink trace is the measured Lyapunov exponent of the same runs; 8340 × 2930 px
> 97 × 34 cm

The same computation as the work opposite, printed in two spot inks rather than rendered on
black. Blue carries the cascade. The pink trace beneath is the Lyapunov exponent of the very same
runs — negative where the iteration converges, positive where it is chaotic. Every downward spike
in the pink is a window of order, and every one of them lines up with a gap in the blue above it.
Nothing was recomputed for this version.

*Proof element: the registration between the two inks. One instruction, two executions, neither
of them the original.*

### 9. Not Whether, But Which

> **Not Whether, But Which**, 2026
> Archival pigment print on Hahnemühle Photo Rag, unglazed, dibond-mounted
> 1024 × 1024 independent tanh networks, 500 training steps each, float64, NVIDIA GB10; 10² magnification into the trainable boundary
> 12 × 12 cm as computed — see note

Every pixel is a separate neural network, trained for 500 steps. The two axes are the learning
rates of its two layers. Colour says not merely whether the network learned but how fast, and the
dark seam is the exact boundary between training and blowing up. Box counting on that seam gives
a fractal dimension of 1.67 ± 0.04 over two decades of scale. One million networks are in this
frame, and it is already a hundredfold magnification into an edge that stays jagged at every
magnification anyone has yet paid for.

*Proof element: the seam is measured, not drawn. It is where the two ends of the colour map meet,
so its position is not an aesthetic choice.*

> **Note on size.** The gallery file is 2048 px, but `render_hero.py` writes it as an integer
> nearest-neighbour upscale of a native 1024² computation. No structure is invented, but none is
> added either: the true-detail ceiling is **0.12 m**, not the 0.24 m implied by the file. This is
> the one work in the edit that cannot hang at scale until it is recomputed. Target 13,000 px for
> a 1.5 m wall — roughly 161× the pixels of the present grid.

---

## Movement III — Forms

### 10. Same Weights, One Activation Apart

> **Same Weights, One Activation Apart**, 2026
> Cast plaster, two solids on a shared plinth
> Zero set of a width-4096 random network on a 0.5 rad cube of S³, 256³ voxels; measured box-counting dimension 2.426 ± 0.031 against theory 2.5
> Each solid approx. 48 × 31 cm; print 48 × 31 cm

Take one random network of width 4,096 and ask where its output is zero. With a ReLU the answer
is a smooth folded sheet — an ordinary surface, dimension 2. Change nothing except the activation
to a Heaviside step and the same weights give a crumpled object of measured dimension 2.426 ±
0.031, against a theoretical 2.5. Both solids are cut away on the same quadrant, so you can see
that each is a skin and not a fill. You can feel the difference with your hands.

*Proof element: the ReLU solid is the null. Identical weights, identical code path, one activation
apart.*

### 11. Terminal Phase

> **Terminal Phase, I–V**, 2026
> Five 3-D printed solids, rods and hubs; ideal circumradius 50 mm
> Measured last-layer class means of a ResNet-18 on four CIFAR-10 classes at epochs 0, 2, 16, 60, 250
> Five objects, each approx. 10 cm across

Train a classifier long past the point where it stops making mistakes and something keeps
happening. The last-layer class means drift apart until they are as far from one another as four
points can get, which is the regular tetrahedron. These five solids are those four means at
epochs 0, 2, 16, 60 and 250 — measured positions, not idealised ones, which is why the final
shape still misses the perfect tetrahedron slightly. For four classes the three-dimensional
picture is exact: nothing is lost in projection.

*Proof element: the ghost tetrahedron. The measured rods visibly do not reach it, and the residual
misfit is stated on the plinth.*

### 12. Invariant Tori

> **Invariant Tori**, 2026
> Archival pigment print on rag, with looping projection
> Twelve orbits of two replicator learners at energy H = 2.8, RK4 at dt = 0.01, float64; stereographic chart into ℝ³, exact to 5.6 × 10⁻¹⁶
> 28 × 28 cm; projection 17 s, looped

Two players learning rock–paper–scissors against each other never settle down. They conserve a
quantity, so each run is trapped on a three-dimensional surface — one that maps into ordinary
space exactly, with nothing lost to projection. Twelve starting strategies wind around twelve
nested tori and never cross. Perturb the payoff for a tie and the tori break; the innermost one
goes chaotic first, at ε = 0.04, which is the opposite of what was expected.

*Proof element: the round trip. The inverse chart recovers the original strategies to 5.6 ×
10⁻¹⁶, so the shape in space is the dynamics and not a rendering choice. The bright knot is
strands seen end-on, not a maximum in the data.*

### 13. Filter-Normalized

> **Filter-Normalized**, 2026
> CNC-milled relief, with archival pigment print on cream rag
> Training cross-entropy of ResNet-56 without shortcuts on a 101 × 101 grid of filter-normalised weight perturbations, CIFAR-10
> Print 30 × 35 cm; relief approx. 30 × 30 cm

A loss surface has 853,018 dimensions. This is a two-dimensional slice through one, taken along
two random directions normalised filter by filter so that distance on the page means something.
The heights are the training loss of a ResNet-56 built without shortcut connections — the version
famous for being hard to train. The strokes are hachures, the way a survey office draws a
mountain: each follows the fall line, weighted by steepness. The cross marks the trained network.

*Proof element: the stated honesty of the slice. Non-convexity here implies non-convexity
globally; smoothness here implies nothing at all, and the sheet says so.*

### 14. Dither

> **Dither**, 2026
> Sound, dark alcove, two speakers; with two-colour risograph on cream
> 1-bit first-order sigma-delta idle tones, DC input 0.30–0.42, 2¹⁴ samples per row, exact float64; audio at 2, 4 and 4-bit TPDF, lossless
> Print 48 × 49 cm; audio looped, approx. 3 min

Round a signal to a few levels and the error is not noise. It is an exact lattice of harmonics
that fold back at Nyquist — the rays and crossings on this sheet are a one-bit modulator's idle
tones, and the star is where its rotation number is exactly two thirds. Add the right random
noise before rounding and the lattice melts. You can hear it here: two bits, four bits
undithered, four bits with triangular dither. The last one hisses more and distorts not at all.

*Proof element: measured error power. Undithered spectral flatness is 0.06; with triangular
dither it is 0.980, and the error power is 0.25 at every input level rather than varying with it.*

### 15. Progressive Sharpening

> **Progressive Sharpening**, 2026
> Pen plotter drawing continuously for the duration of the exhibition; A2 rag paper, two pens
> 6,000 full-batch gradient-descent steps at η = 2/80, one vertex per step, 200 steps per row, boustrophedon; top Hessian eigenvalue refreshed every step
> 42 × 59.4 cm, redrawn daily

Gradient descent with step size η is stable only while the curvature of the loss stays below 2/η.
Training pushes the curvature up until it reaches exactly that value — and then it stays there,
oscillating across the line rather than settling beneath it. The plotter is drawing one such run
as a single continuous line, one vertex per training step. The line it cannot escape is 2/η.

*Proof element: that hairline. It is the only straight line in the drawing, and the only part of
it that is theory rather than measurement. Everything else crashed into it and stayed.*

---

## Notes on the tombstones

**Three corrections to [THE-EDIT.md](THE-EDIT.md), found by looking at the files rather than the
text.** All three are now fixed there.

1. **The Ruler shows seven formats, not six.** int4, int8, FP4 E2M1, FP8 E5M2, **FP8 E4M3FN**,
   float16, bfloat16. The E4M3FN row was omitted from the count.
2. **The Bayer specimen is 65 tiles, not 64.** An 8 × 8 matrix has 64 thresholds and therefore
   65 reachable tones, 0/64 … 64/64 inclusive. The sheet labels them; the count matters because
   the off-by-one *is* the arithmetic of ordered dithering.
3. **Work #12 was titled "One Basin, by width."** That is `mode-connectivity`'s title and belongs
   to a different piece, which the edit cuts. Retitled **Invariant Tori**.

**And one number that moves.** `trainability-fractal/hero_deep_swirl_spectral_print.png` is
2048 px on disk but `render_hero.py:41` writes it as a 2× integer nearest-neighbour upscale of a
native 1024² grid. Its honest print ceiling is 0.12 m, not 0.24 m. This does not change the
edit's conclusion — #9 was already the one work requiring compute before it can hang — but the
gap is 12.7× in width, not 6.3×.

## Reuse

- **Wall label**: tombstone in bold, text beneath at 50–100 words, proof line in italic.
- **Alt text**: the text alone, unchanged. It is read aloud and indexed, and it is already
  written to the same length rule.
- **Post caption**: the text alone, with the proof line as the second tweet in the thread —
  per [display-directions.md](display-directions.md) §4.3, the proof step is the one everybody
  else skips.
- **Catalogue**: tombstone, text, proof line, then the project README as the long form.
