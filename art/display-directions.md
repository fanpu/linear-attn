# Display Directions: how to present this work

*Companion to [ml-art-directions.md](ml-art-directions.md). That document governs what is
true. This one governs what is seen. Target venues: a physical show (prints and objects)
and the feed (X / Instagram).*

---

## 0. The diagnosis

`ml-art-directions.md` opens with a governing constraint on **truth**: every visual decision
is either a faithful rendering of a real quantity or an explicitly declared aesthetic choice.
That rule has done its job — 34 projects, all defensible.

There is no equivalent constraint on **display**, and it shows. The pipeline currently ends at
*"PNG written to `gallery/`"*. Everything after that — selection, sequence, scale, title, label,
material, framing, the decision about what a viewer meets first — is unexecuted. So:

- **1,474 PNGs is an archive, not a gallery.** No human will ever see it. The curatorial act,
  which is the act of *throwing away*, has not happened.
- **Filenames are doing the work of titles.** `hero_deep_swirl_spectral_print.png` is a build
  artifact. Meanwhile the READMEs contain genuinely excellent titles — *Not Whether, But Which*;
  *One Basin*; *Four over d*; *The Ruler* — that exist nowhere near the images.
- **Most of the archive is physically unprintable at exhibition scale.** Measured across all
  gallery PNGs:

  | render width | count |
  |---|---:|
  | under 1500 px | 180 |
  | 1500–2999 px | 806 |
  | 3000–4999 px | 393 |
  | 5000–7999 px | 84 |
  | 8000+ px | 11 |

  Only **95 of 1,474** are ≥5000 px. A 2048 px render is a 24 cm print (see §3.2). The bulk of
  the archive was rendered for a screen and silently inherited a screen's ambitions.

The good news: the hard part is done. Nobody else has this material. What follows is what the
art world actually knows about making people feel something, and how it maps onto what is
already in these directories.

---

## Part I — Why art moves people: seven mechanisms

Not vibes. These are specific, repeatable, and every one of them has a direct analogue here.

### 1. Scale calibrated to the body, not to the wall

Rothko specified **18 inches** as the viewing distance for canvases two metres tall. The point
was never "big." The point is that at the intended distance the work **exceeds peripheral
vision**, so the viewer stops being an observer standing in front of an image and becomes an
occupant standing inside one. Scale used psychologically rather than monumentally. He also
specified the benches, the barriers, and the light in the Rothko Chapel, because the conditions
of encounter are part of the work.

> **Mapping.** This is computable here rather than guessed — see §3.2. And it cuts both ways:
> `trainability-fractal`'s hero is 2048 px, which means it is honestly a 24 cm object. Either
> re-render it at 12,000 px or hang it small and let it be intimate. Do not interpolate: that
> would violate rule §0 in the most literal way, by inventing structure.

### 2. Seriality — the typology grid

Bernd and Hilla Becher photographed water towers, blast furnaces and gas holders from a fixed
camera position, always on overcast days, always in the same frame, and hung them in grids of
9, 12 or 15. Uniformity is the instrument: because everything else is held constant, the **only
thing the eye can see is the difference**. Bernd had trained as a typographer, and described the
piping on a furnace as its serifs and the grey sky as its kerning.

> **Mapping.** This is the single most under-used move in the repo, because the material is
> already typological: `block-quant`'s bitplanes, `float-ruler`'s specimens, `hessian-spectrum`'s
> plates, `lattice`'s shape maps, `neural-collapse`'s five terminal-phase stages, `ouroboros`'s
> generations. Currently these are rendered as *contact sheets* — many small panels composited
> into one PNG. The Becher move is different and much stronger: **identical individual prints,
> identically framed, hung as a block with tight even gaps**, so the wall reads as one work but
> the eye does the comparison. A contact sheet is a figure. A grid of framed objects is a piece.

### 3. The instruction exhibited alongside the execution

Sol LeWitt: *"The idea becomes a machine that makes the art."* His wall drawings are sold as
**certificates** — written instructions. The wall is painted over when the show ends; the
instruction is the work, and no execution is the original.

> **Mapping.** The code is the certificate, and this project is unusually well-positioned to
> make that claim honestly, because rule §0 already requires every decision to be declarable.
> Almost all ML/AI art hides its machinery; exhibiting the generating procedure as a wall panel —
> plain language, ten lines, next to the print — is both truthful and, in 2026, genuinely
> distinctive. `edge-of-stability`'s plotter variant goes further: the plotter drawing *in front
> of the audience* is LeWitt's re-execution, live.

### 4. Micro/macro — a work that reads at 10 m and rewards at 30 cm

The reason Ikeda's rooms work is not only scale but **density**: they are legible as a field
from across the room and resolve into individual numbers up close. Two artworks in one object,
and the trip between them is the experience.

> **Mapping.** Native. A fractal *is* this property. `gd-bifurcation`'s 6144² Lyapunov plane and
> `trainability-fractal`'s zooms are built out of it. Practical consequences: print big, print at
> full resolution, and **do not put it behind glass** — glass stops people approaching, and the
> approach is the point. Face-mounted acrylic looks luxurious and kills this. Matte paper,
> unglazed or with museum glass at most.

### 5. The law line — one element that is checkably true

`edge-of-stability` already invented this: a hairline at 2/η, *"dead straight, the only straight
line in the image."* Everything else in the frame is measurement; that one line is theory; and
the measurement crashes into it and stays.

> **Mapping.** Promote this from one project to a house rule. Every exhibited work carries one
> element a sceptical viewer can verify with their eyes: a law line, a null panel, a scale bar,
> a shuffled-label control (`number-knot` already ships one). This is the entire defence against
> §II below, and it should be visible, not buried in a README.

### 6. Pacing — intensity alternating with rest

Exhibition design is choreography before it is decoration: sightlines, a centre line around
145–150 cm, narrative rhythm, and deliberate **decompression zones** so the visitor does not
fatigue. A room where everything shouts is a room where nothing is heard.

> **Mapping.** 34 projects hung evenly would be an unreadable wall of coloured rectangles. The
> edit in §6 gives most works a modest size and two or three works an entire wall each.

### 7. Duration — the viewer's budget is 27 seconds

The measured mean time a museum visitor spends in front of a work is **27.2 seconds** (median
17). Slow Art Day exists because that number is so low. In a feed the equivalent budget is
roughly **one to two seconds**.

> **Mapping.** Design to the budget explicitly. In the room: one thing must land in 3 seconds
> (silhouette, colour, scale), one thing must reward 30 seconds (the structure), one thing must
> reward 3 minutes (the detail, the label, the law line). In the feed: only the first two exist,
> and the second one is a tap.

---

## Part II — The trap, and why rule §0 is already the antidote

The dominant mode of "AI art" in museums right now is Refik Anadol's: monumental, gorgeous,
data-derived, immersive. It is also the most criticised work of its kind. Jerry Saltz called
*Unsupervised* at MoMA **"a fancy lava lamp."** Others: "screensaver," "empty cathedrals where
algorithms replace thought," work that does not let the viewer *feel* anything and expresses
very little. Dataland, his standalone AI art museum, opened in 2025–26 to the same split
reception: enormous crowds, unconvinced critics.

The structural problem is precise and worth naming, because it is not snobbery: **nothing is at
stake and nothing can be checked.** The data is a texture source. Swap the dataset and the
image would look approximately the same. There is no claim, so there is nothing to be right or
wrong about, so there is nothing to feel.

This work is the exact inverse, and that is its entire competitive position:

| | decorative data art | this project |
|---|---|---|
| role of the data | texture source | the claim itself |
| swap the dataset | looks the same | looks completely different |
| can a viewer check it? | no | yes — that is rule §0 |
| failure mode | beautiful, empty | true, but looks like a figure |

So the correct move is **not to become more spectacular**. Competing with Anadol on immersion is
a losing game and lands in the same critical trap. The move is to make the *specificity* the
visible, felt thing: this pixel is one neural network; this line is a law; this is what a
6144×6294 grid of separate gradient-descent runs actually looks like when you do not smooth it.

The other failure mode — the one `ml-art-directions.md` §0 warns about — is the diagram. Between
"empty spectacle" and "matplotlib with a nicer colormap" is the target, and everything in Parts
III and IV is about hitting it.

---

## Part III — The room

### 3.1 The edit

**Cut to 12–18 works.** At most one per project, and most projects contribute nothing. This is
the hardest and highest-value step, and it is entirely unstarted. A show is defined by what is
absent from it.

Rough selection principle: keep pieces with **intrinsic geometry** (a circle, a simplex, a braid,
a comb, a lattice) and pieces with **micro/macro depth**. Drop anything that is fundamentally a
line graph with good taste applied.

### 3.2 Print size is a computable quantity, not a preference

Human acuity is about **1 arcminute**. At distance *D* the finest resolvable feature is
*D* × 2.9×10⁻⁴.

| viewing distance | finest visible feature | ppi at that distance |
|---|---|---|
| 0.4 m (nose to paper) | 0.12 mm | ~220 |
| 1 m | 0.29 mm | ~87 |
| 2 m | 0.58 mm | ~44 |
| 3 m | 0.87 mm | ~29 |

To fill ~60° of horizontal field — Rothko's "you are inside it" condition — the print width
*W* ≈ 1.15 × *D*.

Combining the two gives a single usable rule:

> **A print that still holds true detail at a 40 cm read needs ≈ 8,600 px per metre of width
> (~220 ppi).**

| print width | pixels required | your renders that qualify |
|---|---:|---|
| 25 cm | 2,150 | almost everything |
| 50 cm | 4,300 | 488 files |
| 1.0 m | 8,600 | 11 files |
| 1.5 m | 12,900 | none |
| 2.0 m | 17,200 | none |

This is why §0's histogram matters: **to make wall-scale prints, the selected works must be
re-rendered, not resized.** For the fractal pieces this is a compute decision (a 12,900 px grid
is 4× the pixels of 6144²), and it is worth spending the GPU time on 12 images rather than
re-rendering 1,474.

And there is an elegant consequence worth putting on the label: since interpolation would
manufacture structure that was never computed, **the print can be exactly as large as the
computation was, and no larger.** The physical size of the object is itself a declared,
honest measurement. Very few artists can say that.

### 3.3 Material — the substrate is a claim

The strongest available move, and almost free, because the material can be made to *agree with
the phenomenon*:

| work | material | why it is not arbitrary |
|---|---|---|
| `dither`, `posterize` | **risograph / screenprint** | the process genuinely halftones and misregisters; the piece is *about* quantisation, printed by a quantiser |
| `float-ruler` | **engraved brass or letterpress** | a ruler should be an object with physical ticks you can feel |
| `edge-of-stability`, `combed`, `number-knot`, `game-chaos`, `ribbon`, `invariant-tori` (15 SVGs exist) | **pen plotter on rag paper** | the oscillation is at pen-stroke frequency; the drawing time is the training time; live plotting is a performance |
| `rough-skin`, `neural-collapse`, `loss-landscape`, `invariant-tori` (10 STLs exist) | **cast plaster / 3D print / CNC relief** | a zero set and a loss surface are surfaces; a simplex is a solid |
| `dither` (14 FLACs exist) | **a dark room and good speakers** | it is already audio. This is the only piece in the repo that is complete in a non-visual medium and it has never been played to anyone |
| fractal planes | **matte rag, unglazed, dibond-mounted** | micro/macro requires approach; glass and acrylic forbid it |

Note how much of this is *already rendered* — 15 plotter SVGs, 10 STLs, 14 FLACs sitting unused
in `gallery/` folders. The objects exist as files and have never been made.

### 3.4 Labels

Convention, which is worth following exactly because it signals seriousness:

- **Tombstone**: title, year, medium, dimensions. The medium line is the opportunity —
  *"Archival pigment print on Hahnemühle Photo Rag. 6144 × 6294 grid; each pixel one
  gradient-descent trajectory, float64, NVIDIA GB10."* That is this project's "oil on canvas,"
  it is true, and it is more impressive than any adjective.
- **Interpretive text: 50–100 words. Hard cap 120.** Active voice, sentences under 25 words.
  Three sentences is the ideal shape: *(1) the phenomenon in plain language, (2) what you are
  looking at, (3) the one number.*
- Anything longer goes in a handout or the catalogue, never on the wall.

The README summaries are already close but run 2–4× too long and too technical. They need one
editing pass each, not a rewrite.

### 3.5 Titles

Promote the README titles to the works, and give every exhibited piece a real title. The good
ones already exist: *Not Whether, But Which*. *One Basin*. *Four over d*. *The Ruler*.
*Circuit Formation*. Titles like these are doing serious work — they state the claim in three
words. Filenames stay filenames.

---

## Part IV — The feed

Opposite venue, opposite constraints. The room gives the viewer a body; the feed gives them a
thumbnail and 1.5 seconds. `color-research/PIPELINE.md` is already a better piece of empirical
work on this than most artists will ever do — it measures codec damage against chroma-carried
contrast at r = 0.88. What it does not yet do is govern *composition*.

### 4.1 Format facts (2026)

- **X**: 16:9 at **1600×900** gets the largest uncropped in-timeline preview on both desktop and
  mobile. 1:1 also renders clean. Everything else gets cropped by the platform's crop, not yours.
- **Instagram**: **1080×1350 (4:5)** occupies the most vertical screen in a mobile feed.
- The in-timeline serve is ~440 px wide (per `PIPELINE.md`). **That is the real canvas.**

### 4.2 The composition rule the pipeline study implies but does not state

> A feed image must have a **readable silhouette at 440 px** and a **separate reward at full
> resolution**.

Most of these renders fail the first test, because they were composed as full-resolution objects
where every part of the frame is equally busy. At 440 px a uniformly dense fractal becomes
noise-coloured mush. The fix is compositional, not technical: **crop for a shape.** One dominant
form, one high-contrast boundary, generous dark ground. The full-res version, one tap away,
delivers the density.

Add this as a measurable to `twitter_pipeline.py` alongside `detail_retained`: something like
low-frequency structure energy at 440 px — the existing harness already computes everything
needed.

### 4.3 Four formats that are natively strong here

1. **The deep zoom.** The most reliably scroll-stopping video format that exists, and this work
   has genuinely infinite honest detail. 45–60 s, continuous, a magnification counter in the
   corner, never blurry because it never interpolates. `trainability-fractal`,
   `gd-bifurcation`, `precision-divergence`, `signal-propagation` all already have zoom films.
   The counter is the addition, and it is the thing that converts "pretty" into "wait, what."
2. **The 4-tile grid as a Becher typology.** X's four-image grid is a form, not a dumping
   ground: hold everything constant, change one variable across four tiles, state the variable
   in the post. Depth 1→4. Precision fp32→fp16→fp8→fp4. Generation 1→10→100→1000.
3. **The one-sentence reveal.** The repo's best asset is not an image, it is a sentence.
   *"Every pixel is a separate neural network trained for 500 steps."* *"Send one fixed row
   through a kernel at batch size 1 versus 512 and the model writes a different sentence."*
   One image plus one sentence like that outperforms any amount of production value.
4. **Thread as exhibition sequence.** hook image → the claim in one line → **the proof** (the
   law line, the null panel) → the method in two lines → the link. The proof step is the one
   everybody else skips and it is the reason to follow this account rather than an AI-art one.

### 4.4 Two cheap disciplines

- **Alt text is a wall label**, subject to the same 50–100 word rule. It is also read aloud, and
  indexed.
- **A consistent frame.** A fixed dark margin, a fixed corner mark or chop, a fixed caption
  typeface. Twenty posts that share a frame read as a body of work; twenty that do not read as
  twenty screenshots. This is the cheapest single upgrade available and it is pure design system.

---

## Part V — What both venues actually share

Only three things, and they are the things to invest in first, because they pay twice:

1. **The edit.** Which ~15 images are the work. Needed for the room; needed for a coherent feed.
2. **The titles and the 50–100 word labels.** Wall label = alt text = post caption = catalogue
   entry.
3. **The proof element.** The law line, the null, the scale bar, the stated grid size. In the
   room it is what separates this from decoration; in the feed it is the entire differentiator
   against an infinite supply of prettier, emptier images.

---

## Part VI — Concrete proposals

### 6.1 A show: *Not Whether, But Which*

~15 works, three movements, one wall each for three of them.

**I. Laws** — things that are exactly true, hung small and precisely, Becher-style grids.
`float-ruler` (engraved, as an object), `hessian-spectrum` plates (grid of 12), `staircase`,
`roofline`. Quiet, dense, rewards approach. This room teaches the viewer to look closely, which
is what makes Movement II work.

**II. Edges** — the wall-scale fractals. `trainability-fractal` and `gd-bifurcation`, each at
1.5–2 m, re-rendered at 13,000–17,000 px, unglazed, with a bench at 40 cm. One law line each.
This is the emotional centre and the room where someone stands for three minutes.

**III. Forms** — the objects. `rough-skin` and `neural-collapse` cast; `loss-landscape` as a
milled relief; `combed` and `number-knot` as plotter drawings; `invariant-tori` printed.
Plus a small dark room with the `dither` audio.

A plotter running `edge-of-stability` live in the entrance, drawing for the duration of the show.
That is the LeWitt certificate, executing.

### 6.2 A posting system

One post a week, in the fixed frame, rotating through the four formats in §4.3. Each post is a
finished work that also happens to be a label test. After ten weeks, the posts *are* the edit —
whichever ten survived the feed are the ten that go on the wall, and the feed did the curating
empirically. That is a very good use of a venue that is otherwise just distribution.

---

## Part VII — Cheapest first moves, in order

1. **Cut to 15.** No compute. Highest leverage of anything on this list.
2. **Write 15 tombstones + 15 labels** at 50–100 words. A day's writing. Immediately reusable as
   alt text and captions.
3. **Plot the SVGs and print the STLs.** 25 objects that already exist as files and have never
   been made. A plotter and a printer, no re-rendering.
4. **Play the FLACs to someone.** The only complete non-visual piece in the repo.
5. **Add a 440 px silhouette metric** to `twitter_pipeline.py` — the harness is already built.
6. **Re-render the 3–4 wall pieces** at 13,000+ px. The only expensive item, deliberately last,
   and cheap once the edit means it is four images instead of 1,474.

---

## Sources

Modern/contemporary practice: [Rothko Chapel, scale and the conditions of encounter](https://www.pacegallery.com/journal/atmospheric-pressure-pamela-g-smart-rothko-chapel/) · [the Rothko effect and color-field scale](https://www.beyondeveryart.com/the-rothko-effect-and-how-large-color-fields-create-emotional-response/) · [Becher typologies](https://fraenkelgallery.com/portfolios/bernd-and-hilla-becher-typologies) · [the Bechers and typographic grids](https://www.artsy.net/article/artsy-editorial-photographer-couple-turned-industrial-architecture-fine-art) · [Becher: Landscape/Typology, MoMA](https://www.moma.org/calendar/exhibitions/95) · [Sol LeWitt wall drawings and certificates](https://publicdelivery.org/sol-lewitt-wall-drawings/) · [LeWitt at Dia](https://www.diaart.org/exhibition/exhibitions-projects/sol-lewitt-exhibition) · [Ryoji Ikeda, data-verse](https://brooklynrail.org/2025/07/artseen/ryoji-ikeda/) · [data-verse at the High Museum](https://high.org/ryoji-ikeda-data-verse-brochure/) · [Vera Molnár: Possibilities, Kunstmuseum Basel 2026](https://kunstmuseumbasel.ch/en/exhibitions/2026/vera-moln%C3%A1r) · [Molnár, Mohr, Nake and plotter drawing](https://www.artforum.com/features/zsofi-valyi-nagy-vera-molnar-552381/)

The decorative-data-art critique: [Anadol's Dataland reviewed](https://news.artnet.com/art-world/refik-anadol-dataland-review-2-2781630) · ["fancy lava lamp" and the Unsupervised reception](https://nouaiart.substack.com/p/review-why-refik-anadols-unsupervised) · [Artforum on Anadol](https://www.artforum.com/events/refik-anadol-250940/) · [the "empty cathedrals" critique](https://www.artcritic.com/en/refik-anadol-the-illusionist-of-empty-data/)

Display mechanics: [exhibition design, sightlines and pacing](https://www.andacademy.com/resources/blog/interior-design/exhibition-design-guide/) · [gallery wall labels](https://customrubontransfers.com/gallery-wall-labels/) · [label writing 101, NCMA](https://learn.ncartmuseum.org/resources/exhibition-planning-and-label-writing-101-top-tips/) · [writing wall labels, Harvard Art Museums](https://harvardartmuseums.org/article/writing-on-the-wall) · [27.2 seconds: Slow Art Day](https://news.artnet.com/art-world/slow-art-day-2019-1508566) · [time spent looking at art](https://www.artsy.net/article/artsy-editorial-long-people-spend-art-museums) · [giclée paper, dibond and viewing distance](https://www.brooklyneditions.com/services/fine-art-printing) · [resolution vs print size](https://www.giftlyartprint.co.uk/blogs/about-art/resolution-and-file-preparation-for-giclee-printing)

Feed: [X media specs 2026](https://www.heyorca.com/blog/x-twitter-media-specs-best-practices-2026) · [X image sizes 2026](https://influencermarketinghub.com/twitter-image-size/) · [social image sizes, Sept 2026](https://blog.hootsuite.com/social-media-image-sizes-guide/)

Scene: [Feral File and the FF1 art computer](https://feralfile.substack.com/p/first-impressions-of-the-ff1-art) · [Casey Reas, Feb 2026](https://caseyreas.substack.com/p/reas-001-19-february-2026)
