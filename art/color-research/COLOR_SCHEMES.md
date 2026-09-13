# Colour schemes for the ML / hardware art projects

A practical catalogue of 73 colour schemes (32 sequential, 9 diverging, 4 cyclic, 2 multi-sequential, 14 categorical,
12 spot-ink sets; 15 of them wrap existing ColorBrewer/Crameri/CET maps) and 15 split-at-the-boundary pairings, all importable from
[`palettes.py`](palettes.py). Previews on real data are in [`gallery/`](gallery/) (see [README](README.md)).

**How the numbers below were computed (by `palettes.metrics`, no external colour package):**

- sRGB -> CIELAB (D65) for L*, and sRGB -> CIECAM02 -> **CAM02-UCS** (Luo et al. 2006, colorspacious default viewing
  conditions) for lightness J' and perceptual step size. Sanity check: viridis and magma come out with a speed CV of
  0.01, as expected for maps designed in that space.
- **Speed CV** = std/mean of the CAM02-UCS distance between 256 successive samples (0 = perfectly uniform).
  **Spike** = largest step / median step over the central 92% (> 3 means a visible false band or hard seam).
  **J' reversals** = direction changes of the lightness profile larger than 3 units. A diverging map should have 1, a
  sequential map 0. More than that means the map contains lightness bands that can look like structure.
- **Deuteranopia retention** = UCS path length after Machado et al. (2009) deuteranopia simulation (severity 1) divided
  by the normal path length. Near 1 means little is lost. **Greyscale retention** = |dJ'| path / total path: near 1
  means the map is carried by lightness and survives greyscale printing.
- Categorical and ink sets: minimum pairwise CIEDE2000 (verified against Sharma et al. 2005 test pairs), normal
  and under simulated deuteranopia/protanopia. Below ~10 the two colours are hard to tell apart as small patches.
- Hand-written stop lists are interpolated in CIELAB, then **re-parameterised to constant CAM02-UCS speed**. That is
  why most curated ramps show a speed CV of about 0.00. Uniform speed does *not* guarantee monotone lightness, so check
  the J' reversals.

**Honesty labels.** Hexes from ColorBrewer/Crameri/CET come from the libraries. Riso inks come from stencil.wiki via
`mattdesl/riso-colors` (saved in `sources/riso-colors.json`). Japanese colours come from nipponcolors.com's own
data endpoint (all 250 saved in `sources/nipponcolors.tsv`). MetBrewer palettes come from `BlakeRMills/MetBrewer`
`PaletteCode.R` (saved). Star colours come from Mitchell Charity's table. **Everything else (film stocks, Rothko,
Morandi, Klimt gold ramp, aurora, CRT, kente, Bauhaus, Memphis...) is a curated evocation I wrote, not a measured
standard.** Treat it as a declared aesthetic choice.

---

## Why the Sohl-Dickstein Spectral split works

His colab (`github.com/Sohl-Dickstein/fractal`) does `cdf_img`: sort all values; map negatives by rank onto
[-1, -0.25] and non-negatives onto [0.25, 1]; negate; show with `cmap='Spectral'`, `vmin=-1, vmax=1`.
`palettes.render_split(M, 'sd_spectral', near_boundary='large')` reproduces it to within 4/255 per channel
(`test_palettes.py`). Five things make it work:

1. **The seam is where the physics is.** The slowest runs (largest |M|) sit at the trainability boundary. Rank
   normalisation puts them at the two ends of Spectral, deep red `#9e0142` (L* 33) and purple `#5e4fa2` (L* 39). The
   boundary is therefore drawn *twice* in dark: a dark red line kissing a dark purple line. The fractal edge becomes a
   dark filament whose thickness follows how slowly training fails or succeeds next to it. You get an automatic
   line drawing with no edge detection.
2. **Pastel interiors.** Spectral's lightness rises from both ends to a pale-yellow peak (L* 98). Far from the boundary
   everything is light and low contrast, so the eye goes to the seam. The `buffer=0.25` stops each side at 0.375/0.625
   of the map (L* ~ 80-90), which keeps the two pastels distinguishable (pale orange vs pale green) instead of merging in
   the yellow.
3. **Hue codes the side, lightness codes distance.** Each half is a clean sequential ramp in lightness, and the sides
   differ by hue family (warm vs cool). Two independent visual channels carry two independent facts.
4. **Rank normalisation removes the heavy tails.** |M| spans ~10 decades (diverged sums up to 1e8). Linear or even log
   scaling would put all colour in a few pixels. Rank/CDF scaling uses the full ramp *per side*, whatever the
   distribution. The price: colour is only ordinal, and two different images are not colour-comparable. For zooms, pass
   a pooled `ref=` array.
5. **Speckle reads as texture, not noise.** Near a fractal boundary, neighbouring pixels flip sides. Because both sides
   are dark there, the flips become fine dark grain instead of red/green confetti.

Measured caveats of Spectral itself: not colour-blind safe (red/green ends; deuteranopia retention 0.85). The speed CV
is 0.38 with a spike of 2.1: the green-yellow segment is fast and the orange segment slower, so equal rank steps are
not equal visual steps. It has one lightness peak by design.

**Generalisation used here.** Any two ramps that start dark (the seam) and end pale, with *different hues* and *similar
lightness ranges*, reproduce the effect. `split_cmap(neg, pos)` builds the map and `signed_rank_normalize(x,
near_boundary='small'|'large', pastel=0.75)` builds the normalisation. `seam='light'` reverses both ramps, giving a
glowing boundary on dark interiors. Rules of thumb from the sheets: (a) keep the seam colours within ~15 L* of
each other and below L* 25 so the seam reads as one line; (b) make the two pastel ends differ in hue by > 90 deg,
or the far interiors merge; (c) prefer a blue-vs-orange/gold axis if CVD matters.

### Recommended alternative pairings (see `gallery/split_*.png`)

| pairing | direction | why it works / when to use |
|---|---|---|
| `aurora_ember` | nature, dark and luminous | Near-black navy/oxide seam; green vs orange pastel ends are far apart in hue. Most "Spectral-like" in energy, with more drama. Its light-seam variant glows. |
| `indigo_madder` | textile / natural dyes | Navy-vs-oxblood seam is almost black; blue vs terracotta survives deuteranopia by the b* axis. Looks like printed cloth. |
| `hubble_sho` | astronomy | Teal vs gold is the most CVD-robust pair here. The seam is dark brown-teal, and it reads like Pillars of Creation. |
| `klimt_lapis` | painting | Ultramarine vs gold leaf; the lapis side stays saturated, so interiors look jewel-like rather than pastel. |
| `verdigris_copper` | nature / metal | Earthy, low-key; excellent on Lyapunov planes (patina "veins" through copper). |
| `cyanotype_vandyke` | print, archival | Low chroma, prints in greyscale; the most "observatory plate" option. |
| `synthwave` | digital | Cyan vs magenta with a violet-black seam; loud but coherent. Best on dark mounts. |
| `cinestill` | film | Night teal vs halation red; very high seam contrast. |
| `hokusai_sunset` | ukiyo-e (MetBrewer) | Prussian blue vs Hiroshige terracotta; softer than Spectral, and the closest in mood. |
| `malachite_rhodochrosite` | minerals | Green vs rose. Beautiful, but red-green: not CVD safe. |
| `morandi` | muted painting | The seam is weak on purpose (umber). Use when the structure should whisper. |
| `crt_phosphor` | digital | Green vs amber on black glass; the saturated green overwhelms. It works better as an overlay than as a full field. |
| `riso_pink_blue` | print (contone) | Continuous stand-in for a two-drum riso print; both ends meet at cream paper. |
| `crameri_bukavu` | counter-example | A real topographic multi-sequential map. Its sea-level seam is *light* (coastline), which shows why dark seams matter. |

---

## Scientific / cartographic

### `brewer_spectral` - ColorBrewer Spectral
- **Type:** diverging (library map `Spectral`; stops sampled).
- **Stops:** `#9e0142` `#dd4a4c` `#f98e52` `#fed481` `#ffffbe` `#d6ee9b` `#86cfa5` `#3d95b8` `#5e4fa2`
- **Measured diagnostics:** L* 33 -> 39 (range 33-99); J' reversals 1; CAM02-UCS speed CV 0.38, spike 2.1; deuteranopia retention 0.85; greyscale retention 0.68.
- **Caveats:** Not CVD-safe (red/green ends); hue-heavy, mid pale-yellow is the lightness peak.
- **Suits:** signed two-sided fields when split at the boundary (see PAIRINGS['sd_spectral']).
- **Lineage:** Cynthia Brewer's ColorBrewer (2002), 11-class diverging scheme; matplotlib 'Spectral'. The map Sohl-Dickstein uses for trainability fractals (split at the boundary). Sources: <https://colorbrewer2.org>, <https://github.com/Sohl-Dickstein/fractal>

### `brewer_brbg` - ColorBrewer BrBG
- **Type:** diverging (library map `BrBG`; stops sampled).
- **Stops:** `#543005` `#995d13` `#cfa256` `#f1dfb3` `#f4f5f5` `#b4e2db` `#58b0a7` `#0c7169` `#003c30`
- **Measured diagnostics:** L* 24 -> 22 (range 22-96); J' reversals 1; CAM02-UCS speed CV 0.21, spike 1.2; deuteranopia retention 0.95; greyscale retention 0.86.
- **Caveats:** Colour-blind safe per ColorBrewer; pale centre hides small |x|.
- **Suits:** symmetric signed fields around zero (random-field level sets, dither error).
- **Lineage:** ColorBrewer diverging brown-blue-green; soil/water cartography. Sources: <https://colorbrewer2.org>

### `brewer_ylgnbu` - ColorBrewer YlGnBu
- **Type:** sequential (library map `YlGnBu`; stops sampled).
- **Stops:** `#ffffd9` `#edf8b1` `#c6e9b4` `#7ecdbb` `#40b5c4` `#1d90c0` `#225da8` `#243392` `#081d58`
- **Measured diagnostics:** L* 99 -> 13 (range 13-99); J' reversals 0; CAM02-UCS speed CV 0.27, spike 1.3; deuteranopia retention 0.95; greyscale retention 0.74.
- **Caveats:** Only 9 native classes; lightness compresses in the yellow end.
- **Suits:** densities on paper-white ground (bifurcation densities, histograms).
- **Lineage:** ColorBrewer multi-hue sequential. Sources: <https://colorbrewer2.org>

### `crameri_batlow` - Crameri batlow
- **Type:** sequential (library map `cmc.batlow`; stops sampled).
- **Stops:** `#011959` `#114360` `#226061` `#4d734d` `#828231` `#c09036` `#f29d6d` `#fdb4b6` `#faccfa`
- **Measured diagnostics:** L* 12 -> 87 (range 12-87); J' reversals 0; CAM02-UCS speed CV 0.09, spike 1.2; deuteranopia retention 0.94; greyscale retention 0.58.
- **Caveats:** Perceptually uniform, CVD-friendly, readable in greyscale.
- **Suits:** all-round sequential; spectrograms, escape times
- **Lineage:** Fabio Crameri, Scientific colour maps (2018-2023), designed for perceptual uniformity and colour-vision-deficiency readability. Sources: <https://www.fabiocrameri.ch/colourmaps/>, <https://doi.org/10.1038/s41467-020-19160-7>

### `crameri_oslo` - Crameri oslo
- **Type:** sequential (library map `cmc.oslo`; stops sampled).
- **Stops:** `#010101` `#0e1e2e` `#15395b` `#26578c` `#507bbc` `#7d99ca` `#a3b1ca` `#cfd2d8` `#ffffff`
- **Measured diagnostics:** L* 0 -> 100 (range 0-100); J' reversals 0; CAM02-UCS speed CV 0.44, spike 1.2; deuteranopia retention 1.00; greyscale retention 0.85.
- **Caveats:** Uniform, near-monochrome blue; very CVD-safe.
- **Suits:** dark-ground densities; convergence time; engraved plates
- **Lineage:** Fabio Crameri, Scientific colour maps (2018-2023), designed for perceptual uniformity and colour-vision-deficiency readability. Sources: <https://www.fabiocrameri.ch/colourmaps/>, <https://doi.org/10.1038/s41467-020-19160-7>

### `crameri_lajolla` - Crameri lajolla
- **Type:** sequential (library map `cmc.lajolla`; stops sampled).
- **Stops:** `#191900` `#372411` `#67342a` `#a64644` `#d9604e` `#e58851` `#edae54` `#f7da74` `#fffecb`
- **Measured diagnostics:** L* 8 -> 99 (range 8-99); J' reversals 0; CAM02-UCS speed CV 0.14, spike 1.3; deuteranopia retention 0.82; greyscale retention 0.75.
- **Caveats:** Uniform; reverse for dark ground.
- **Suits:** warm densities on light ground; heat/energy
- **Lineage:** Fabio Crameri, Scientific colour maps (2018-2023), designed for perceptual uniformity and colour-vision-deficiency readability. Sources: <https://www.fabiocrameri.ch/colourmaps/>, <https://doi.org/10.1038/s41467-020-19160-7>

### `crameri_bukavu` - Crameri bukavu
- **Type:** multiseq (library map `cmc.bukavu`; stops sampled).
- **Stops:** `#1a3333` `#235786` `#3f92c8` `#7ac7cc` `#014026` `#4b6d1a` `#9c7e43` `#cfbba0` `#ededfc`
- **Measured diagnostics:** L* 19 -> 94 (range 19-97); J' reversals 2; CAM02-UCS speed CV 4.35, spike 96.9; deuteranopia retention 0.96; greyscale retention 0.83.
- **Caveats:** Two sequential halves with a jump at 0.5: intended to be used with a centred norm.
- **Suits:** topography-like fields with a meaningful zero (sea level)
- **Lineage:** Fabio Crameri, Scientific colour maps (2018-2023), designed for perceptual uniformity and colour-vision-deficiency readability. Sources: <https://www.fabiocrameri.ch/colourmaps/>, <https://doi.org/10.1038/s41467-020-19160-7>

### `crameri_berlin` - Crameri berlin
- **Type:** diverging (library map `cmc.berlin`; stops sampled).
- **Stops:** `#9eb0ff` `#519fd3` `#286886` `#14303e` `#190c09` `#411201` `#7d341e` `#be6f63` `#ffadad`
- **Measured diagnostics:** L* 73 -> 79 (range 4-79); J' reversals 1; CAM02-UCS speed CV 0.19, spike 1.7; deuteranopia retention 0.92; greyscale retention 0.81.
- **Caveats:** Dark-centre diverging, uniform; centre is near black so small |x| disappears on dark ground.
- **Suits:** signed fields with a DARK centre: boundary as a dark seam
- **Lineage:** Fabio Crameri, Scientific colour maps (2018-2023), designed for perceptual uniformity and colour-vision-deficiency readability. Sources: <https://www.fabiocrameri.ch/colourmaps/>, <https://doi.org/10.1038/s41467-020-19160-7>

### `crameri_vik` - Crameri vik
- **Type:** diverging (library map `cmc.vik`; stops sampled).
- **Stops:** `#001261` `#034481` `#307da6` `#94bed2` `#ece5e0` `#dbaa8d` `#c27041` `#912d06` `#590008`
- **Measured diagnostics:** L* 11 -> 16 (range 11-92); J' reversals 1; CAM02-UCS speed CV 0.18, spike 1.7; deuteranopia retention 0.95; greyscale retention 0.86.
- **Caveats:** Uniform, CVD-friendly blue/brown.
- **Suits:** signed fields, light centre (random fields, dither error)
- **Lineage:** Fabio Crameri, Scientific colour maps (2018-2023), designed for perceptual uniformity and colour-vision-deficiency readability. Sources: <https://www.fabiocrameri.ch/colourmaps/>, <https://doi.org/10.1038/s41467-020-19160-7>

### `crameri_romaO` - Crameri romaO
- **Type:** cyclic (library map `cmc.romaO`; stops sampled).
- **Stops:** `#733957` `#8b4433` `#aa752f` `#cfbc66` `#cbe1b3` `#8bcbcf` `#5393bf` `#595891` `#723959`
- **Measured diagnostics:** L* 32 -> 32 (range 32-87); J' reversals 1; CAM02-UCS speed CV 0.17, spike 1.3; deuteranopia retention 0.91; greyscale retention 0.60.
- **Caveats:** Uniform cyclic; hue-based so check under CVD.
- **Suits:** cyclic phase (gradient direction, residues)
- **Lineage:** Fabio Crameri, Scientific colour maps (2018-2023), designed for perceptual uniformity and colour-vision-deficiency readability. Sources: <https://www.fabiocrameri.ch/colourmaps/>, <https://doi.org/10.1038/s41467-020-19160-7>

### `cet_fire` - colorcet fire
- **Type:** sequential (library map `cet_fire`; stops sampled).
- **Stops:** `#000000` `#4b0100` `#7e0200` `#b40600` `#ed1500` `#ff6900` `#ffa701` `#ffdb0a` `#ffffff`
- **Measured diagnostics:** L* 0 -> 100 (range 0-100); J' reversals 0; CAM02-UCS speed CV 1.27, spike 2.0; deuteranopia retention 0.81; greyscale retention 0.58.
- **Caveats:** Uniform (CET L3), black->white through red/yellow.
- **Suits:** dark-ground densities, glowing filaments
- **Lineage:** Peter Kovesi, 'Good Colour Maps: How to Design Them' (2015), CET maps; packaged by colorcet (HoloViz). Sources: <https://colorcet.com>, <https://arxiv.org/abs/1509.03700>

### `cet_bmy` - colorcet bmy
- **Type:** sequential (library map `cet_bmy`; stops sampled).
- **Stops:** `#000c7d` `#0016a8` `#7e0896` `#bd0086` `#f11a75` `#ff5d5c` `#ff9a38` `#ffc81d` `#fff123`
- **Measured diagnostics:** L* 14 -> 94 (range 14-94); J' reversals 0; CAM02-UCS speed CV 0.49, spike 3.0; deuteranopia retention 0.72; greyscale retention 0.52.
- **Caveats:** Uniform blue-magenta-yellow.
- **Suits:** dark-ground spectrograms, synthwave-adjacent
- **Lineage:** Peter Kovesi, 'Good Colour Maps: How to Design Them' (2015), CET maps; packaged by colorcet (HoloViz). Sources: <https://colorcet.com>, <https://arxiv.org/abs/1509.03700>

### `cet_cbtl1` - colorcet CET_CBTL1
- **Type:** sequential (library map `cet_CET_CBTL1`; stops sampled).
- **Stops:** `#111111` `#550d0c` `#8b0214` `#be0c21` `#f21c2f` `#e57b74` `#69c5de` `#89e7fd` `#f9f9f9`
- **Measured diagnostics:** L* 5 -> 98 (range 5-98); J' reversals 0; CAM02-UCS speed CV 0.58, spike 4.9; deuteranopia retention 0.79; greyscale retention 0.49.
- **Caveats:** Designed for protan/deutan viewers.
- **Suits:** CVD-safe sequential for mixed audiences
- **Lineage:** Peter Kovesi, 'Good Colour Maps: How to Design Them' (2015), CET maps; packaged by colorcet (HoloViz). Sources: <https://colorcet.com>, <https://arxiv.org/abs/1509.03700>

### `cet_c2` - colorcet CET_C2
- **Type:** cyclic (library map `cet_CET_C2`; stops sampled).
- **Stops:** `#ef55f2` `#fcb0a2` `#f1ee35` `#97d410` `#32ad28` `#3e729b` `#2e22ea` `#9139fb` `#ed53f3`
- **Measured diagnostics:** L* 63 -> 62 (range 33-92); J' reversals 2; CAM02-UCS speed CV 0.41, spike 2.5; deuteranopia retention 0.84; greyscale retention 0.50.
- **Caveats:** Four-colour cyclic; bright, strong hue cycle.
- **Suits:** cyclic phase
- **Lineage:** Peter Kovesi, 'Good Colour Maps: How to Design Them' (2015), CET maps; packaged by colorcet (HoloViz). Sources: <https://colorcet.com>, <https://arxiv.org/abs/1509.03700>

### `cet_glasbey` - colorcet glasbey_dark
- **Type:** categorical (library map `cet_glasbey_dark`; stops sampled).
- **Stops:** `#d70000` `#8c3cff` `#028800` `#00acc7` `#e7a500` `#ff7fd1` `#6c004f` `#583b00` `#005759`
- **Measured diagnostics:** min pairwise dE2000 14 (deuteranopia 1, protanopia 5); min dL* 0.
- **Caveats:** Max-distinct categorical; loud, not 'designed'.
- **Suits:** many-class basins
- **Lineage:** Peter Kovesi, 'Good Colour Maps: How to Design Them' (2015), CET maps; packaged by colorcet (HoloViz). Sources: <https://colorcet.com>, <https://arxiv.org/abs/1509.03700>

### `hypsometric` - Hypsometric tints (atlas)
- **Type:** multiseq.
- **Stops:** `#0b2a4a` `#1f5f8b` `#6fa8d6` `#cfe7f3` `#4f8a55` `#9cc27a` `#e7dc97` `#d7a15a` `#a36e45` `#7a6353` `#f4f2ee`
- **Measured diagnostics:** L* 17 -> 96 (range 17-96); J' reversals 4; CAM02-UCS speed CV 2.68, spike 62.8; deuteranopia retention 0.97; greyscale retention 0.91.
- **Caveats:** Non-monotone lightness by design (dark sea floor, light coast, dark land then white peaks): reads as terrain, not as magnitude.
- **Suits:** signed fields with a physical-feeling zero: LIGHT seam at the coastline (x=0).
- **Lineage:** Layer tinting of elevation (Hauslab 1840s; Imhof's Swiss relief school): sea blues deepen with depth, land runs green -> yellow -> brown -> snow. Curated stops. Sources: <https://en.wikipedia.org/wiki/Hypsometric_tints>, <https://www.shadedrelief.com/hypso/hypso.html>

### `ironbow` - Thermal 'ironbow'
- **Type:** sequential.
- **Stops:** `#000004` `#1d0b5a` `#6a1a8c` `#b4306f` `#e25a2f` `#f89b14` `#fdd65c` `#fffbe8`
- **Measured diagnostics:** L* 0 -> 98 (range 0-98); J' reversals 0; CAM02-UCS speed CV 0.02, spike 1.0; deuteranopia retention 0.89; greyscale retention 0.57.
- **Caveats:** Close to inferno but more saturated in the magenta; minor lightness wobble near orange.
- **Suits:** energy/heat-like densities on dark ground; loss magnitude; Hessian spectra.
- **Lineage:** Evocation of the FLIR 'Ironbow' thermal-camera palette (black->indigo->magenta->orange->white). Sources: <https://www.flir.com/discover/industrial/picking-a-thermal-color-palette/>

### `landsat_false` - Landsat NIR false colour
- **Type:** sequential.
- **Stops:** `#07142b` `#27435a` `#6b8f9b` `#a67f82` `#c9474f` `#e3182f` `#f6b9b5`
- **Measured diagnostics:** L* 7 -> 81 (range 7-81); J' reversals 2; CAM02-UCS speed CV 0.02, spike 1.0; deuteranopia retention 0.81; greyscale retention 0.59.
- **Caveats:** Hue flips cool->warm mid-ramp: can read as a two-class split; not CVD-safe.
- **Suits:** sequential 'vigour' quantities (trainability speed, fraction trained).
- **Lineage:** Evocation of NASA/USGS Landsat colour-infrared composites (NIR->red): water near black, bare ground cyan-grey, vegetation red. Curated stops. Sources: <https://earthobservatory.nasa.gov/features/FalseColor>

## Print traditions

### `riso_fluopink_blue` - Riso Fluo Pink + Blue
- **Type:** inks. Ground/paper `#f4efe3`.
- **Stops:** `#ff48b0` `#0078bf`
- **Measured diagnostics:** min pairwise dE2000 42 (deuteranopia 24, protanopia 4); min dL* 12.
- **Caveats:** Two inks + paper = 4 printable tones incl. overprint; screen density carries the magnitude.
- **Suits:** the canonical zine pair; converge/diverge phase maps
- **Lineage:** Risograph soy inks as listed by stencil.wiki (hex are screen approximations of the drum inks). Sources: <https://stencil.wiki/colors>, <https://github.com/mattdesl/riso-colors>

### `riso_federalblue_sunflower` - Riso Federal Blue + Sunflower
- **Type:** inks. Ground/paper `#f4efe3`.
- **Stops:** `#3d5588` `#ffb511`
- **Measured diagnostics:** min pairwise dE2000 27 (deuteranopia 26, protanopia 28); min dL* 16.
- **Caveats:** Two inks + paper = 4 printable tones incl. overprint; screen density carries the magnitude.
- **Suits:** strong value contrast; line-art over a density
- **Lineage:** Risograph soy inks as listed by stencil.wiki (hex are screen approximations of the drum inks). Sources: <https://stencil.wiki/colors>, <https://github.com/mattdesl/riso-colors>

### `riso_teal_brightred` - Riso Teal + Bright Red
- **Type:** inks. Ground/paper `#f4efe3`.
- **Stops:** `#00838a` `#f15060`
- **Measured diagnostics:** min pairwise dE2000 40 (deuteranopia 27, protanopia 18); min dL* 8.
- **Caveats:** Two inks + paper = 4 printable tones incl. overprint; screen density carries the magnitude.
- **Suits:** complementary two-sided fields; overprint gives near-black
- **Lineage:** Risograph soy inks as listed by stencil.wiki (hex are screen approximations of the drum inks). Sources: <https://stencil.wiki/colors>, <https://github.com/mattdesl/riso-colors>

### `riso_aqua_orange` - Riso Aqua + Orange
- **Type:** inks. Ground/paper `#f4efe3`.
- **Stops:** `#5ec8e5` `#ff6c2f`
- **Measured diagnostics:** min pairwise dE2000 28 (deuteranopia 28, protanopia 23); min dL* 12.
- **Caveats:** Two inks + paper = 4 printable tones incl. overprint; screen density carries the magnitude.
- **Suits:** light, poster-like; spectrogram overlays
- **Lineage:** Risograph soy inks as listed by stencil.wiki (hex are screen approximations of the drum inks). Sources: <https://stencil.wiki/colors>, <https://github.com/mattdesl/riso-colors>

### `riso_burgundy_mint` - Riso Burgundy + Mint
- **Type:** inks. Ground/paper `#f4efe3`.
- **Stops:** `#914e72` `#82d8d5`
- **Measured diagnostics:** min pairwise dE2000 24 (deuteranopia 17, protanopia 11); min dL* 13.
- **Caveats:** Two inks + paper = 4 printable tones incl. overprint; screen density carries the magnitude.
- **Suits:** muted, bookish; basins on cream
- **Lineage:** Risograph soy inks as listed by stencil.wiki (hex are screen approximations of the drum inks). Sources: <https://stencil.wiki/colors>, <https://github.com/mattdesl/riso-colors>

### `riso_indigo_melon` - Riso Indigo + Melon
- **Type:** inks. Ground/paper `#f4efe3`.
- **Stops:** `#484d7a` `#ffae3b`
- **Measured diagnostics:** min pairwise dE2000 27 (deuteranopia 25, protanopia 27); min dL* 17.
- **Caveats:** Two inks + paper = 4 printable tones incl. overprint; screen density carries the magnitude.
- **Suits:** evening palette; Lyapunov planes
- **Lineage:** Risograph soy inks as listed by stencil.wiki (hex are screen approximations of the drum inks). Sources: <https://stencil.wiki/colors>, <https://github.com/mattdesl/riso-colors>

### `riso_trio_pink_yellow_blue` - Riso Fluo Pink + Yellow + Medium Blue
- **Type:** inks. Ground/paper `#f4efe3`.
- **Stops:** `#ff48b0` `#ffe800` `#3255a4`
- **Measured diagnostics:** min pairwise dE2000 27 (deuteranopia 24, protanopia 13); min dL* 3.
- **Caveats:** Yellow on cream has very low contrast (dL* ~ 7); use yellow only for areas, never lines.
- **Suits:** three-class basin maps with overprint mixtures at shared boundaries.
- **Lineage:** Three-drum 'CMY-ish' riso set (stencil.wiki hexes). Sources: <https://stencil.wiki/colors>, <https://github.com/mattdesl/riso-colors>

### `cyanotype` - Cyanotype (Prussian blue)
- **Type:** sequential.
- **Stops:** `#07172c` `#0e2f55` `#1b4f82` `#3c77a8` `#86aecd` `#d3e0e6` `#f3f0e6`
- **Measured diagnostics:** L* 8 -> 95 (range 8-95); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 1.00; greyscale retention 0.88.
- **Caveats:** Monochrome: very CVD-safe, uniform after Lab interpolation.
- **Suits:** sequential density on paper; single-hue line-art ground; boundary maps (white lines on blue).
- **Lineage:** Herschel 1842; Anna Atkins' 'Photographs of British Algae' (1843). Ferric ammonium citrate + potassium ferricyanide -> Prussian blue (pigment ~#003153). Curated tonal ramp. Sources: <https://en.wikipedia.org/wiki/Cyanotype>, <https://www.metmuseum.org/art/collection/search/285381>

### `vandyke` - Van Dyke brown print
- **Type:** sequential.
- **Stops:** `#1b0f08` `#3d2415` `#664226` `#95704f` `#c6a98a` `#ecdfcb` `#f6f0e4`
- **Measured diagnostics:** L* 5 -> 95 (range 5-95); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.99; greyscale retention 0.96.
- **Caveats:** Monochrome; the top end is paper, so keep a margin to avoid clipping into the ground.
- **Suits:** sequential density; pairs with cyanotype as the second half of a split (see PAIRINGS).
- **Lineage:** Van Dyke brown (kallitype family) iron-silver print, named after the pigment used by Anthony van Dyck. Curated. Sources: <https://en.wikipedia.org/wiki/Van_Dyke_brown>

### `platinum` - Platinum/palladium print
- **Type:** sequential.
- **Stops:** `#1f1b18` `#3e3732` `#665c54` `#948779` `#c1b4a2` `#e3d9c8` `#f5efe2`
- **Measured diagnostics:** L* 10 -> 95 (range 10-95); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 1.00; greyscale retention 0.99.
- **Caveats:** Low chroma: structure must be carried by lightness alone (which is perceptually honest).
- **Suits:** archival 'observatory plate' renders of densities and relief; greyscale-safe.
- **Lineage:** Pt/Pd printing (Willis 1873): long tonal scale, warm neutral blacks, matte paper. Curated. Sources: <https://en.wikipedia.org/wiki/Platinum_print>

### `sepia` - Sepia toning
- **Type:** sequential.
- **Stops:** `#24160b` `#4b2e14` `#704214` `#9c6d3c` `#c9a275` `#ecdcc2`
- **Measured diagnostics:** L* 9 -> 88 (range 9-88); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.99; greyscale retention 0.94.
- **Caveats:** Monochrome, uniform; slightly compressed highlights.
- **Suits:** sequential density; nostalgia plates.
- **Lineage:** Sepia-toned silver prints (sodium sulphide toning). #704214 is the conventional 'sepia'. Sources: <https://en.wikipedia.org/wiki/Sepia_(color)>

### `blueprint` - Blueprint (diazo/cyanotype)
- **Type:** inks. Ground/paper `#1d3f78`.
- **Stops:** `#eef3f8`
- **Measured diagnostics:** min pairwise dE2000 61 (deuteranopia 63, protanopia 59); min dL* 68.
- **Caveats:** Single ink: magnitude only via line weight/hatching.
- **Suits:** line-art: boundaries, level sets, contour stacks; one light ink on dark ground.
- **Lineage:** Engineering blueprint (cyanotype reprographics, 1870s-1940s): white lines on Prussian-blue ground. Sources: <https://en.wikipedia.org/wiki/Blueprint>

### `letterpress` - Letterpress vermilion + black
- **Type:** inks. Ground/paper `#f3eee2`.
- **Stops:** `#d63a2a` `#1c1b1a`
- **Measured diagnostics:** min pairwise dE2000 41 (deuteranopia 36, protanopia 29); min dL* 39.
- **Caveats:** Red/black are separable under deuteranopia only by lightness (L* 51 vs 10): fine.
- **Suits:** line-art plates: black for structure, vermilion for the measured boundary.
- **Lineage:** Two-colour letterpress (rubrication tradition: black text, red initials) on cotton stock. Curated. Sources: <https://en.wikipedia.org/wiki/Rubrication>

## Painting

### `hokusai_wave` - Hokusai blues (MetBrewer Hokusai2)
- **Type:** sequential.
- **Stops:** `#0a3351` `#134b73` `#2f70a1` `#4692b0` `#72aeb6` `#abc9c8` `#f2ece0`
- **Measured diagnostics:** L* 20 -> 94 (range 20-94); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.97; greyscale retention 0.86.
- **Caveats:** MetBrewer lists Hokusai2 as colourblind-friendly; cyan middle slightly compresses L*.
- **Suits:** sequential density; cool plates; ocean-like spectrograms.
- **Lineage:** MetBrewer 'Hokusai2' (sampled from 'The Great Wave off Kanagawa', c.1831, Prussian-blue woodblock) + an added paper end. Sources: <https://github.com/BlakeRMills/MetBrewer>, <https://www.metmuseum.org/art/collection/search/36491>

### `hokusai_categorical` - Hokusai1 (MetBrewer)
- **Type:** categorical.
- **Stops:** `#6d2f20` `#b75347` `#df7e66` `#e09351` `#edc775` `#94b594` `#224b5e`
- **Measured diagnostics:** min pairwise dE2000 14 (deuteranopia 6, protanopia 9); min dL* 2.
- **Caveats:** MetBrewer flags NOT colourblind-safe; three adjacent warm oranges are close.
- **Suits:** basin classes with an ukiyo-e warmth.
- **Lineage:** MetBrewer 'Hokusai1'. Sources: <https://github.com/BlakeRMills/MetBrewer>

### `hiroshige` - Hiroshige (MetBrewer)
- **Type:** diverging.
- **Stops:** `#e76254` `#ef8a47` `#f7aa58` `#ffd06f` `#ffe6b7` `#aadce0` `#72bcd5` `#528fad` `#376795` `#1e466e`
- **Measured diagnostics:** L* 58 -> 29 (range 29-92); J' reversals 1; CAM02-UCS speed CV 0.01, spike 1.0; deuteranopia retention 0.90; greyscale retention 0.70.
- **Caveats:** MetBrewer lists as colourblind-friendly; the orange end is lighter than the blue end (asymmetric L*).
- **Suits:** symmetric signed fields; a softer Spectral-like diverging map; split halves work too.
- **Lineage:** MetBrewer 'Hiroshige' (Utagawa Hiroshige prints): sunset orange to indigo. Sources: <https://github.com/BlakeRMills/MetBrewer>

### `nippon_beni` - Beni reds (Nippon colors)
- **Type:** sequential.
- **Stops:** `#3f2b36` `#64363c` `#9f353a` `#cb1b45` `#e87a90` `#f8c3cd` `#fedfe1`
- **Measured diagnostics:** L* 20 -> 91 (range 20-91); J' reversals 0; CAM02-UCS speed CV 0.01, spike 1.0; deuteranopia retention 0.84; greyscale retention 0.74.
- **Caveats:** KURENAI is very saturated: a chroma spike mid-ramp can look like a band on smooth data.
- **Suits:** warm sequential; the diverged half of a split; dark-seam reds.
- **Lineage:** KUROBENI 黒紅, KUWAZOME 桑染, ENJI 臙脂, KURENAI 紅, USUBENI 薄紅, TAIKOH 退紅, SAKURA 桜 (nipponcolors.com). Sources: <https://nipponcolors.com>

### `nippon_categorical` - Nippon traditional set
- **Type:** categorical.
- **Stops:** `#cb1b45` `#ffb11b` `#1b813e` `#005caf` `#592c63` `#ca7a2c` `#86a697` `#1c1c1c`
- **Measured diagnostics:** min pairwise dE2000 18 (deuteranopia 4, protanopia 7); min dL* 3.
- **Caveats:** KURENAI vs TOKIWA collapse under deuteranopia (check table); separate them spatially.
- **Suits:** basin classes; Edo-textile look on SHIRONERI (#fcfaf2) paper.
- **Lineage:** KURENAI 紅, YAMABUKI 山吹, TOKIWA 常磐, RURI 瑠璃, MURASAKI 紫, KOHAKU 琥珀, SABISEIJI 錆青磁, SUMI 墨. Sources: <https://nipponcolors.com>

### `rothko` - Rothko colour field (maroon->orange)
- **Type:** sequential.
- **Stops:** `#16080a` `#3f0d12` `#7a1b16` `#b3371b` `#df6a25` `#f0a54a` `#f5d59a`
- **Measured diagnostics:** L* 3 -> 87 (range 3-87); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.89; greyscale retention 0.80.
- **Caveats:** Very close to lajolla/inferno in L*; distinctive by its oxblood low end.
- **Suits:** dark-ground densities with a glowing core; soft-edged fields (smooth Lyapunov interiors).
- **Lineage:** Evocation of Mark Rothko's warm fields (e.g. 'Orange and Yellow' 1956; Seagram murals' maroons). Curated. Sources: <https://www.moma.org/artists/5047>

### `morandi` - Morandi muted
- **Type:** sequential.
- **Stops:** `#4a4744` `#716c64` `#958c7f` `#b3a898` `#cdc2b2` `#e2dacd` `#f1ede5`
- **Measured diagnostics:** L* 30 -> 94 (range 30-94); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 1.00; greyscale retention 0.99.
- **Caveats:** Low chroma, L* 30->94: fine, but faint differences vanish on screens with poor gamma.
- **Suits:** quiet sequential plates where form (relief, hachure) carries the structure.
- **Lineage:** Giorgio Morandi still lifes: dusty greys, putty, bone. Curated. Sources: <https://www.moma.org/artists/4102>

### `morandi_categorical` - Morandi muted set
- **Type:** categorical.
- **Stops:** `#a39e93` `#c9b1a0` `#8d9b8f` `#b8a3a8` `#7f8a91` `#d8cfc4`
- **Measured diagnostics:** min pairwise dE2000 9 (deuteranopia 3, protanopia 2); min dL* 3.
- **Caveats:** Min pairwise dE2000 is small by design: use with outlines.
- **Suits:** gentle basin maps with few classes; overlays under black line-art.
- **Lineage:** Morandi-style desaturated set. Curated. Sources: <https://www.moma.org/artists/4102>

### `klimt_gold` - Klimt gold
- **Type:** sequential.
- **Stops:** `#120d05` `#3a2a0c` `#6f5316` `#a8862c` `#d4b24f` `#ecd68c` `#faf0cf`
- **Measured diagnostics:** L* 4 -> 95 (range 4-95); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 1.00; greyscale retention 0.90.
- **Caveats:** Olive mid-tones can look dirty on low-quality displays; uniform in L*.
- **Suits:** dark-ground densities (bifurcation atlas, spectrogram harmonics) that should shimmer.
- **Lineage:** Gustav Klimt's golden phase (The Kiss, 1907-08; Adele Bloch-Bauer I, 1907): gold leaf on dark. Curated; MetBrewer 'Klimt' (#df9ed4 #c93f55 #eacc62 #469d76 #3c4b99 #924099) for accents. Sources: <https://github.com/BlakeRMills/MetBrewer>, <https://en.wikipedia.org/wiki/The_Kiss_(Klimt)>

### `klimt_categorical` - Klimt (MetBrewer)
- **Type:** categorical.
- **Stops:** `#df9ed4` `#c93f55` `#eacc62` `#469d76` `#3c4b99` `#924099`
- **Measured diagnostics:** min pairwise dE2000 20 (deuteranopia 10, protanopia 2); min dL* 6.
- **Caveats:** Not colourblind-safe per MetBrewer.
- **Suits:** jewel-like basin classes on dark or gold ground.
- **Lineage:** MetBrewer 'Klimt'. Sources: <https://github.com/BlakeRMills/MetBrewer>

### `tam` - Tam (MetBrewer)
- **Type:** sequential.
- **Stops:** `#341648` `#62205f` `#9f2d55` `#bb292c` `#de4f33` `#ef8737` `#ffb242` `#ffd353`
- **Measured diagnostics:** L* 14 -> 86 (range 14-86); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.91; greyscale retention 0.68.
- **Caveats:** Colourblind-friendly per MetBrewer; L* speeds up in the orange section.
- **Suits:** dark-ground sequential with synthwave warmth; escape times.
- **Lineage:** MetBrewer 'Tam' (reversed to run dark->light). Sources: <https://github.com/BlakeRMills/MetBrewer>

### `isfahan` - Isfahan (MetBrewer Isfahan1)
- **Type:** diverging.
- **Stops:** `#4e3910` `#845d29` `#ae8548` `#e3c28b` `#4fb6ca` `#178f92` `#175f5d` `#054544`
- **Measured diagnostics:** L* 26 -> 26 (range 26-80); J' reversals 1; CAM02-UCS speed CV 0.02, spike 1.0; deuteranopia retention 0.97; greyscale retention 0.79.
- **Caveats:** Jump from #e3c28b to #4fb6ca at the centre: a hard hue seam (useful as a boundary).
- **Suits:** signed fields; almost a ready-made split (dark ends, light middle) in bronze/turquoise.
- **Lineage:** MetBrewer 'Isfahan1' (Safavid tilework: turquoise glaze and ochre). Sources: <https://github.com/BlakeRMills/MetBrewer>

### `okeeffe` - O'Keeffe (MetBrewer OKeeffe1)
- **Type:** diverging.
- **Stops:** `#6b200c` `#973d21` `#da6c42` `#ee956a` `#fbc2a9` `#f6f2ee` `#bad6f9` `#7db0ea` `#447fdd` `#225bb2` `#133e7e`
- **Measured diagnostics:** L* 24 -> 27 (range 24-96); J' reversals 1; CAM02-UCS speed CV 0.02, spike 1.0; deuteranopia retention 0.98; greyscale retention 0.84.
- **Caveats:** Colourblind-friendly per MetBrewer.
- **Suits:** symmetric signed fields on white; random-field level sets.
- **Lineage:** MetBrewer 'OKeeffe1' (Georgia O'Keeffe desert reds and sky blues). Sources: <https://github.com/BlakeRMills/MetBrewer>

## Design

### `bauhaus` - Bauhaus primaries
- **Type:** categorical.
- **Stops:** `#d4282d` `#f2c21a` `#1f4e9a` `#141414` `#e9e2d0`
- **Measured diagnostics:** min pairwise dE2000 24 (deuteranopia 23, protanopia 25); min dL* 10.
- **Caveats:** Red/black similar-ish under protanopia in lightness terms; yellow/cream weak contrast.
- **Suits:** 3-4 class basins with geometric composition; poster-style plates.
- **Lineage:** Itten/Kandinsky colour-form correspondences (yellow triangle, red square, blue circle; 1923 questionnaire). Curated. Sources: <https://en.wikipedia.org/wiki/Bauhaus>, <https://www.moma.org/artists/2981>

### `swiss` - Swiss style (Muller-Brockmann)
- **Type:** inks. Ground/paper `#f5f3ee`.
- **Stops:** `#e2231a` `#111111`
- **Measured diagnostics:** min pairwise dE2000 44 (deuteranopia 38, protanopia 31); min dL* 44.
- **Caveats:** Spot colours only; keep red for one semantic role.
- **Suits:** line-art/small multiples: black data, red boundary or single highlighted curve.
- **Lineage:** Swiss International Style posters (Josef Muller-Brockmann, Tonhalle 'musica viva' series 1950s-70s): red + black on white, grid and Akzidenz-Grotesk. Curated. Sources: <https://en.wikipedia.org/wiki/Josef_M%C3%BCller-Brockmann>

### `memphis` - Memphis Group
- **Type:** categorical.
- **Stops:** `#f7a6c0` `#18a4a0` `#f7d23e` `#2b3990` `#ee4c3a` `#8ccf5a` `#111111`
- **Measured diagnostics:** min pairwise dE2000 24 (deuteranopia 10, protanopia 6); min dL* 0.
- **Caveats:** Deliberately clashing; several pairs collide under CVD. Declared aesthetic only.
- **Suits:** playful categorical basins; confetti-like scatter plots of attractors.
- **Lineage:** Memphis Group (Ettore Sottsass, Milan 1981): candy pastels + primaries + black squiggles. Curated. Sources: <https://en.wikipedia.org/wiki/Memphis_Group>

## Film and photography

### `kodachrome` - Kodachrome
- **Type:** categorical.
- **Stops:** `#c1272d` `#f2b500` `#1f5aa6` `#2f7d3b` `#e8d6b3` `#2a1d16`
- **Measured diagnostics:** min pairwise dE2000 20 (deuteranopia 7, protanopia 14); min dL* 4.
- **Caveats:** Red/green pair collapses under deuteranopia.
- **Suits:** saturated categorical basins on dark ground.
- **Lineage:** Kodachrome (1935-2009) dye-coupler reversal film: dense blacks, saturated primaries. Curated evocation. Sources: <https://en.wikipedia.org/wiki/Kodachrome>

### `portra` - Kodak Portra 400
- **Type:** sequential.
- **Stops:** `#2e3539` `#4f5f61` `#7c7f76` `#a88f7c` `#d3aa91` `#efcfb9` `#f8ece2`
- **Measured diagnostics:** L* 22 -> 94 (range 22-94); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.96; greyscale retention 0.89.
- **Caveats:** Hue flips cool->warm with little L* help mid-ramp; subtle, not for fine magnitude reading.
- **Suits:** soft sequential plates; smooth interiors (Lyapunov stable regions).
- **Lineage:** Portra: low-contrast colour negative with lifted cool shadows and warm skin highlights. Curated evocation. Sources: <https://en.wikipedia.org/wiki/Kodak_Portra>

### `cinestill` - CineStill 800T (tungsten + halation)
- **Type:** sequential.
- **Stops:** `#061520` `#0c3342` `#1c5e6b` `#4f9a9a` `#b9d8cf` `#fff4e0` `#ff8a5c` `#e4322b`
- **Measured diagnostics:** L* 6 -> 51 (range 6-96); J' reversals 1; CAM02-UCS speed CV 0.39, spike 1.9; deuteranopia retention 0.89; greyscale retention 0.84.
- **Caveats:** Deliberately non-monotone at the top (red is darker than cream): the peak reads as a halo, not as 'more'.
- **Suits:** dark-ground densities where the very brightest values should 'bleed' red (bifurcation branches).
- **Lineage:** CineStill 800T (Kodak Vision3 500T with remjet removed): tungsten-balanced teal nights and red halation around highlights. Sources: <https://cinestillfilm.com/products/800tungsten-high-speed-color-film-35mm-135-36exp>

### `technicolor2` - Technicolor two-strip (Process 3)
- **Type:** inks. Ground/paper `#f7f1e6`.
- **Stops:** `#e2553b` `#2e9a8e`
- **Measured diagnostics:** min pairwise dE2000 36 (deuteranopia 30, protanopia 21); min dL* 3.
- **Caveats:** Only two primaries: no true blue or yellow; overlap gives a brown-black.
- **Suits:** two-sided fields as two dye layers; flesh-and-sea look.
- **Lineage:** Technicolor Process 2/3 (1922-1932): two dye-imbibition records, orange-red and blue-green, cemented/imbibed. Curated. Sources: <https://en.wikipedia.org/wiki/Technicolor>, <https://filmcolors.org/timeline-entry/1244/>

### `autochrome` - Autochrome Lumiere
- **Type:** inks. Ground/paper `#1a1712`.
- **Stops:** `#d0583a` `#5c963e` `#584aa0`
- **Measured diagnostics:** min pairwise dE2000 35 (deuteranopia 3, protanopia 13); min dL* 4.
- **Caveats:** Additive on dark: render as random grain assignment, not as a smooth map.
- **Suits:** stochastic 3-colour dithering of densities; grainy additive renders.
- **Lineage:** Autochrome (Lumiere 1907): potato-starch grains dyed orange-red, green, blue-violet as a random additive mosaic. Curated. Sources: <https://en.wikipedia.org/wiki/Autochrome_Lumi%C3%A8re>

## Nature

### `aurora` - Aurora borealis
- **Type:** sequential.
- **Stops:** `#03081a` `#081f3a` `#0b4a52` `#11845f` `#35c27b` `#9cf0a6` `#effbe0`
- **Measured diagnostics:** L* 2 -> 97 (range 2-97); J' reversals 0; CAM02-UCS speed CV 0.01, spike 1.0; deuteranopia retention 0.91; greyscale retention 0.80.
- **Caveats:** Green mid-ramp is bright and saturated: very legible, but red/green CVD viewers lose some separation vs ember.
- **Suits:** dark-ground densities, spectrograms, cool half of a split.
- **Lineage:** Auroral oxygen green line (557.7 nm) over night sky; red 630 nm and N2+ violet above. Curated. Sources: <https://en.wikipedia.org/wiki/Aurora#Colours>

### `bioluminescence` - Bioluminescent sea
- **Type:** sequential.
- **Stops:** `#010409` `#031a33` `#054b70` `#0791a8` `#35d0d0` `#a8f6ee` `#f2fffd`
- **Measured diagnostics:** L* 1 -> 99 (range 1-99); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.91; greyscale retention 0.84.
- **Caveats:** Uniform and CVD-safe (blue axis). Greyscale-honest.
- **Suits:** sparse bright structures on black: bifurcation branches, harmonic lattices.
- **Lineage:** Dinoflagellate (Noctiluca) blue-cyan glow, emission ~475 nm. Curated. Sources: <https://en.wikipedia.org/wiki/Bioluminescence>

### `verdigris` - Verdigris / copper
- **Type:** diverging.
- **Stops:** `#3a1a0c` `#7c3d1b` `#b8703f` `#e3b48a` `#eee7da` `#9fd3c2` `#43a58f` `#24936e` `#0f4a3e`
- **Measured diagnostics:** L* 14 -> 28 (range 14-92); J' reversals 1; CAM02-UCS speed CV 0.05, spike 1.1; deuteranopia retention 0.93; greyscale retention 0.88.
- **Caveats:** Copper/patina differ mainly in hue: deuteranopes see brown vs grey-blue (still separable by b*).
- **Suits:** symmetric signed fields; as split halves see PAIRINGS['verdigris_copper'].
- **Lineage:** Oxidised copper: metal (copper #b87333) to patina (ROKUSYOH 緑青 #24936E, Japanese verdigris pigment). Curated + nipponcolors. Sources: <https://en.wikipedia.org/wiki/Verdigris>, <https://nipponcolors.com/#rokusyoh>

### `malachite` - Malachite
- **Type:** sequential.
- **Stops:** `#04221a` `#0b4430` `#136a47` `#23925f` `#5bbb86` `#aee0bd` `#eaf6ee`
- **Measured diagnostics:** L* 11 -> 96 (range 11-96); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.93; greyscale retention 0.90.
- **Caveats:** Single hue, uniform.
- **Suits:** sequential density; banded mineral renders of level sets.
- **Lineage:** Malachite banding (Cu2CO3(OH)2): concentric green bands. Curated. Sources: <https://en.wikipedia.org/wiki/Malachite>

### `lapis` - Lapis lazuli + pyrite
- **Type:** categorical.
- **Stops:** `#0f1a4a` `#26619c` `#6d8fcb` `#d4af37` `#efe6d2`
- **Measured diagnostics:** min pairwise dE2000 19 (deuteranopia 19, protanopia 18); min dL* 14.
- **Caveats:** Blue/gold is the safest CVD axis; the two blues need lightness separation (they have it).
- **Suits:** few-class basins; gold for the rare class.
- **Lineage:** Lapis lazuli (lazurite blue, pyrite gold flecks, calcite white); ultramarine pigment source. Curated. Sources: <https://en.wikipedia.org/wiki/Lapis_lazuli>

### `deep_sea` - Deep sea (photic zones)
- **Type:** sequential.
- **Stops:** `#000510` `#001733` `#00305c` `#0b5487` `#2e86b0` `#7cc0d8` `#d2f0f7`
- **Measured diagnostics:** L* 1 -> 93 (range 1-93); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.99; greyscale retention 0.91.
- **Caveats:** Very close to cmocean 'ice'/Crameri 'oslo': uniform, CVD-safe.
- **Suits:** depth-like sequential values, dark ground.
- **Lineage:** Ocean light attenuation: midnight/twilight/sunlit zones. Curated. Sources: <https://oceanexplorer.noaa.gov/facts/light-travel.html>

### `ember` - Ember / lava
- **Type:** sequential.
- **Stops:** `#070202` `#300806` `#6e140a` `#b3300e` `#e5641c` `#f9a444` `#ffe2a8`
- **Measured diagnostics:** L* 1 -> 91 (range 1-91); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.90; greyscale retention 0.80.
- **Caveats:** Near 'cet_fire'/'inferno' but redder; uniform.
- **Suits:** dark-ground densities; warm half of aurora/ember split.
- **Lineage:** Blackbody-like glow of cooling lava. Curated. Sources: <https://en.wikipedia.org/wiki/Black-body_radiation>

### `verdigris_cyclic` - Copper patina (cyclic)
- **Type:** cyclic.
- **Stops:** `#6b3419` `#c07a45` `#e9d3b4` `#7cc2ad` `#24936e` `#1c4f52` `#4a2a3a`
- **Measured diagnostics:** L* 29 -> 29 (range 22-86); J' reversals 2; CAM02-UCS speed CV 0.02, spike 1.0; deuteranopia retention 0.84; greyscale retention 0.70.
- **Caveats:** Lightness not constant: one bright pole at the cream.
- **Suits:** cyclic phase (full 2pi) with a warm/cool hemisphere split.
- **Lineage:** Copper -> patina -> oxidised dark cycle. Curated. Sources: <https://en.wikipedia.org/wiki/Verdigris>

## Digital

### `synthwave` - Synthwave sunset
- **Type:** sequential.
- **Stops:** `#0d0221` `#2b0a4f` `#6a1270` `#b3246f` `#ee4d63` `#ff8e53` `#ffd76e`
- **Measured diagnostics:** L* 2 -> 87 (range 2-87); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.87; greyscale retention 0.69.
- **Caveats:** Uniform-ish but chroma very high; magenta section speeds up in dE.
- **Suits:** dark-ground densities, escape times, neon heroes.
- **Lineage:** 1980s retro-futurist sunset gradients (Outrun/synthwave album art). Curated. Sources: <https://en.wikipedia.org/wiki/Synthwave>

### `vaporwave` - Vaporwave
- **Type:** categorical.
- **Stops:** `#ff71ce` `#01cdfe` `#05ffa1` `#b967ff` `#fffb96`
- **Measured diagnostics:** min pairwise dE2000 18 (deuteranopia 11, protanopia 4); min dL* 8.
- **Caveats:** All light (L* 70-97): needs a dark ground; poor as sequence.
- **Suits:** loud categorical basins on #1b1035 ground.
- **Lineage:** Vaporwave/aesthetic palette as circulated on colour sites (2010s): neon pink, cyan, mint, violet, pale yellow. Sources: <https://en.wikipedia.org/wiki/Vaporwave>, <https://www.color-hex.com/color-palette/10223>

### `crt_green` - CRT P1 green phosphor
- **Type:** sequential.
- **Stops:** `#000400` `#002a06` `#00590f` `#00911c` `#20d13a` `#8cff9a` `#e6ffe9`
- **Measured diagnostics:** L* 1 -> 98 (range 1-98); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.91; greyscale retention 0.80.
- **Caveats:** Monochrome and uniform; highly readable. Add gaussian bloom for authenticity (declared).
- **Suits:** glowing line-art and densities on black (bitplanes, spectrograms, oscilloscope traces).
- **Lineage:** P1 phosphor (peak ~525 nm), monochrome terminals (IBM 3270, Apple II monitors). Curated. Sources: <https://en.wikipedia.org/wiki/Phosphor#Standard_phosphor_types>

### `crt_amber` - CRT P3 amber phosphor
- **Type:** sequential.
- **Stops:** `#050200` `#2a1500` `#5c3000` `#9a5600` `#e08a00` `#ffb000` `#ffe7b5`
- **Measured diagnostics:** L* 1 -> 92 (range 1-92); J' reversals 0; CAM02-UCS speed CV 0.01, spike 1.0; deuteranopia retention 0.99; greyscale retention 0.84.
- **Caveats:** Monochrome, uniform.
- **Suits:** as crt_green; warmer. Quantization bitplanes.
- **Lineage:** P3 amber phosphor (~602 nm), 1980s terminals (DEC, IBM 5151 amber). Curated. Sources: <https://en.wikipedia.org/wiki/Phosphor#Standard_phosphor_types>

### `eink16` - E-ink 16-level greyscale
- **Type:** categorical.
- **Stops:** `#1f1f1f` `#2d2d2c` `#3b3a39` `#494846` `#575654` `#656461` `#73726e` `#81807b` `#8f8e89` `#9e9c96` `#acaaa3` `#bab8b0` `#c8c6be` `#d6d4cb` `#e4e2d8` `#f2f0e6`
- **Measured diagnostics:** min pairwise dE2000 3 (deuteranopia 3, protanopia 3); min dL* 5.
- **Caveats:** Bands are real (16 levels) and intended; don't use where false contours would mislead.
- **Suits:** honest quantized sequential (16 declared bands) - fits int4 quantization projects literally.
- **Lineage:** E Ink Carta 4-bit greyscale waveforms: 16 levels, paper-like off-white. Used as a quantized sequential map. Sources: <https://www.eink.com/tech/detail/How_it_works>

## Astronomy

### `hubble_sho` - Hubble palette (SHO)
- **Type:** diverging.
- **Stops:** `#0f2a33` `#155e63` `#2e9a93` `#8fd1c2` `#f4ecd2` `#f0c064` `#d98a26` `#9c4f16` `#3c1a08`
- **Measured diagnostics:** L* 15 -> 14 (range 14-93); J' reversals 1; CAM02-UCS speed CV 0.01, spike 1.0; deuteranopia retention 0.94; greyscale retention 0.86.
- **Caveats:** Teal/gold is roughly along the CVD-safe blue-yellow axis: good for deutan/protan viewers.
- **Suits:** signed fields: gold vs teal; very popular split pair.
- **Lineage:** Narrowband 'Hubble palette': [S II, H-alpha, O III] -> (R, G, B) as in 'Pillars of Creation' (1995). Processed images read teal (O III) vs gold (S II/H-alpha). Curated. Sources: <https://esahubble.org/images/heic1501a/>, <https://en.wikipedia.org/wiki/Pillars_of_Creation>

### `star_classes` - Stellar spectral classes OBAFGKM
- **Type:** categorical.
- **Stops:** `#9bb0ff` `#aabfff` `#cad7ff` `#f8f7ff` `#fff4ea` `#ffd2a1` `#ffcc6f`
- **Measured diagnostics:** min pairwise dE2000 4 (deuteranopia 5, protanopia 4); min dL* 1.
- **Caveats:** All very light (L* 72-97) and low chroma: only works on a dark ground; tiny steps between A/F/G.
- **Suits:** bright points on black (attractor scatter coloured by a sequential 'temperature'); gentle sequential.
- **Lineage:** Mitchell Charity, 'What color are the stars?' (blackbody sRGB, D65 white) for O B A F G K M. Sources: <http://www.vendian.org/mncharity/dir3/starcolor/>

## Textiles and natural dyes

### `nippon_aizome` - Aizome indigo dips (Nippon colors)
- **Type:** sequential.
- **Stops:** `#08192d` `#0f2540` `#0b346e` `#006284` `#33a6b8` `#81c7d4` `#a5dee4` `#fcfaf2`
- **Measured diagnostics:** L* 8 -> 98 (range 8-98); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.95; greyscale retention 0.83.
- **Caveats:** Hue drifts navy->cyan; uniform after Lab interpolation, CVD-safe (blue axis).
- **Suits:** sequential density; indigo textile renders; the cool half of a split.
- **Lineage:** Japanese traditional colour names for successive indigo dips, dark to light: KACHI 褐, KON 紺, RURIKON 瑠璃紺, HANADA 縹, ASAGI 浅葱, MIZU 水, KAMENOZOKI 瓶覗, SHIRONERI 白練. Hexes from nipponcolors.com. Sources: <https://nipponcolors.com>, <https://en.wikipedia.org/wiki/Aizome>

### `morris` - William Morris 'Strawberry Thief'
- **Type:** categorical.
- **Stops:** `#1d2b45` `#a8322d` `#d9a441` `#5e7f4a` `#ede3cf` `#6b4a32`
- **Measured diagnostics:** min pairwise dE2000 18 (deuteranopia 8, protanopia 1); min dL* 5.
- **Caveats:** Red and green are close in L*: fails under deuteranopia unless outlined.
- **Suits:** categorical basins on indigo ground; ornament-like renders of intricate boundaries.
- **Lineage:** William Morris 'Strawberry Thief' (1883), indigo-discharge block print with madder red and weld yellow. Curated. Sources: <https://collections.vam.ac.uk/item/O78889/strawberry-thief-furnishing-fabric-morris-william/>

### `madder` - Madder root dye
- **Type:** sequential.
- **Stops:** `#2a0b0b` `#5a1a17` `#8f2a22` `#bd4a36` `#d9806a` `#ecb9a6` `#f7e6dc`
- **Measured diagnostics:** L* 7 -> 92 (range 7-92); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 0.93; greyscale retention 0.87.
- **Caveats:** Uniform; pairs with indigo along a red-blue axis that survives deuteranopia by lightness.
- **Suits:** warm sequential; the diverged half of indigo/madder.
- **Lineage:** Madder (Rubia tinctorum, alizarin/purpurin) on wool: Turkey red, pink to brick with mordant. Curated. Sources: <https://en.wikipedia.org/wiki/Rubia_tinctorum>

### `weld` - Weld yellow dye
- **Type:** sequential.
- **Stops:** `#2b2408` `#5c4c0f` `#8e7a1e` `#bfa434` `#dcc760` `#efe29d` `#faf4d8`
- **Measured diagnostics:** L* 14 -> 96 (range 14-96); J' reversals 0; CAM02-UCS speed CV 0.00, spike 1.0; deuteranopia retention 1.00; greyscale retention 0.89.
- **Caveats:** Olive low end; uniform.
- **Suits:** warm-green sequential; accent category.
- **Lineage:** Weld (Reseda luteola, luteolin), the brightest historic European yellow; overdyed with woad for Lincoln green. Curated. Sources: <https://en.wikipedia.org/wiki/Reseda_luteola>

### `kente` - Kente (Asante/Ewe)
- **Type:** categorical.
- **Stops:** `#f2b705` `#0a7b3e` `#c8102e` `#141414` `#1f4e9a`
- **Measured diagnostics:** min pairwise dE2000 30 (deuteranopia 9, protanopia 14); min dL* 3.
- **Caveats:** Red/green collapse under deuteranopia; black separates everything by lightness.
- **Suits:** bold categorical basins; strip/weave compositions of small multiples.
- **Lineage:** Kente strip-woven silk/cotton (Ghana): gold, green, red, black, blue with documented symbolic meanings. Curated. Sources: <https://en.wikipedia.org/wiki/Kente_cloth>, <https://africa.si.edu/exhibits/kente/>

### `shibori_cyclic` - Shibori indigo (cyclic)
- **Type:** cyclic.
- **Stops:** `#0f2540` `#2e5c8a` `#a9c7de` `#f3f1e8` `#8fb3cf` `#1f4a78`
- **Measured diagnostics:** L* 14 -> 14 (range 14-95); J' reversals 1; CAM02-UCS speed CV 0.02, spike 1.0; deuteranopia retention 1.00; greyscale retention 0.90.
- **Caveats:** Two-fold symmetric lightness cycle: angle theta and theta+pi look similar - use only for axial (mod pi) data.
- **Suits:** cyclic phase where orientation +-pi should look the same (gradient direction mod pi). 
- **Lineage:** Indigo resist-dye repeats (arashi/itajime); dark->white->dark cycle. Curated using KON 紺. Sources: <https://en.wikipedia.org/wiki/Shibori>

## Split-at-the-boundary pairings (`PAIRINGS`)

Each side is listed from the seam (boundary) outward. Use `P.render_split(x, name, near_boundary=...)`.

| name | x < 0 side (seam -> pastel) | x > 0 side (seam -> pastel) | idea |
|---|---|---|---|
| `sd_spectral` | `#5e4fa2` `#3f97b7` `#89d0a4` `#d8ef9b` `#ffffbe` | `#9e0142` `#dd4a4c` `#f98e52` `#fed481` `#ffffbe` | Sohl-Dickstein Spectral (reference): matplotlib Spectral, split at 0.5; converged=purple half, diverged=red half; rank-normalized per side. |
| `verdigris_copper` | `#0e3a33` `#1c6b5b` `#3f9a84` `#86c8b2` `#d5ece2` | `#3d1a0b` `#7e3a18` `#b86f3c` `#deaa7c` `#f5e3cc` | Verdigris / copper: Patina (ROKUSYOH) against bare metal; both sides earthy, seam = dark oxide. |
| `indigo_madder` | `#08192d` `#0b346e` `#2f6aa3` `#7fb0d2` `#dcebf2` | `#2a0b0b` `#7a1f1d` `#b8473a` `#e19a82` `#f6e2d8` | Indigo / madder: The two great natural dyes (aizome KACHI->KAMENOZOKI vs Rubia tinctorum). Morris textiles. |
| `aurora_ember` | `#03122a` `#0b4a52` `#1b9a70` `#76dca0` `#e3fbdc` | `#1c0503` `#6e140a` `#c4401a` `#f59a3e` `#ffe6b0` | Aurora / ember: Cold night-sky oxygen green vs blackbody lava glow. |
| `cyanotype_vandyke` | `#07172c` `#153f6e` `#3c74a6` `#9cc0dd` `#eef3f6` | `#1b0f08` `#4a2e1c` `#7e573a` `#bc9a78` `#f1e6d6` | Cyanotype / Van Dyke: Two iron-process prints; a darkroom diptych. Low chroma, archival. |
| `hubble_sho` | `#061c22` `#0f4e58` `#2a8c8c` `#8fd0c2` `#eaf6ef` | `#221003` `#6e3a0c` `#c0761f` `#edbb5e` `#fff2cc` | Hubble SHO teal / gold: O III teal against S II/H-alpha gold (Pillars of Creation processing). |
| `klimt_lapis` | `#0b0f2e` `#1e2a6e` `#3e55a8` `#8e9fd6` `#e6eaf7` | `#1c1405` `#5e4410` `#a77f24` `#ddbf5e` `#f8edc4` | Lapis / Klimt gold: Ultramarine against gold leaf (Byzantine mosaic, Klimt's golden phase). |
| `synthwave` | `#060a2a` `#10307a` `#1f7fc0` `#5fd0ea` `#d8f8ff` | `#1a0626` `#5b0f6b` `#b0247e` `#f2629a` `#ffd8e6` | Synthwave cyan / magenta: Neon grid vs sunset; VHS-era chromatic aberration. |
| `cinestill` | `#04121a` `#0f3c4a` `#2b7a86` `#9ccac4` `#eef7f2` | `#1d0403` `#6b0f0b` `#c9281c` `#f28a6a` `#ffe3d6` | CineStill tungsten / halation: Tungsten-balanced night teal vs the red halation ring of 800T. |
| `hokusai_sunset` | `#0a2e57` `#295384` `#5a97c1` `#95c9c3` `#e7f0e2` | `#3a160c` `#6d2f20` `#b75347` `#e09351` `#f5e2b8` | Hokusai wave / Hiroshige sunset: MetBrewer Hokusai3 blues vs Hokusai1/Hiroshige warm ramp. |
| `crt_phosphor` | `#000400` `#005a10` `#15b030` `#8cff9a` `#eaffec` | `#050200` `#6a3a00` `#d98400` `#ffc24a` `#fff0cc` | CRT green / amber phosphor: Two terminal phosphors (P1 vs P3) on a black glass seam. |
| `malachite_rhodochrosite` | `#04221a` `#0f5a3c` `#2f9a66` `#9bd8b0` `#eef8f0` | `#2a0914` `#6e1c35` `#b44a6a` `#e79bb0` `#fbe6ec` | Malachite / rhodochrosite: Two banded minerals: copper-carbonate green vs manganese-carbonate rose. |
| `morandi` | `#3e4744` `#62716b` `#8e9c94` `#bfc8c0` `#eef0ea` | `#4a3c38` `#77605a` `#a58d84` `#cdb9ae` `#f3ebe4` | Morandi sage / clay: Muted still-life pair: the seam is a soft umber, interiors powdery. |
| `crameri_bukavu` | `#014026` `#78c5cc` `#3d90c7` `#235582` `#1a3333` | `#014026` `#4b6d1a` `#9c7e43` `#cfbba0` `#ededfc` | Crameri bukavu (light 'coastline' seam): Crameri's multi-sequential topographic map split at sea level; seam is where both halves are lightest->darkest. |
| `riso_pink_blue` | `#123a78` `#3255a4` `#62a8e5` `#bcdcf4` `#f4efe3` | `#6a1742` `#c02c7e` `#ff48b0` `#f9b0d6` `#f4efe3` | Riso blue / fluo pink (contone): Continuous-tone version of the two-ink zine pair, both halves ending at cream paper. |
