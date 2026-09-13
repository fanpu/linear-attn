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
