# Invariant Tori (§5), milestone M2 report

**Status: DONE_WITH_CONCERNS.** Every image in the M2 list exists in `art/invariant-tori/gallery/`, and I looked at each one (downscaled, plus full-resolution crops). The concerns are small: see Risks.

**Commits:**
- `087df81`: re-chosen pole, render scripts, tori images and slice plate.
- `7adaeeb`: sea-and-islands renders, stereo pair, anaglyph.

**Compute:** splatting, AO and plates ran on CPU at 4 threads, with no job longer than 36 s. The four sea volume renders ran through `gpu1.sh` with `render_volume` on CUDA. That job waited 45 min in the art queue, then took 66 s.

## Rulings applied

### 1. Pole (`repole.py`)
- **New pole:** p = (0.67218, −0.25691, −0.66643, −0.19506), chosen without orbit kam 38.
- **Minimum angle to all 23.6 M stored points: 23.64°** (23.63° to the chords between samples). The limiting orbits are kam 59 and kam 347; the ε = 0 tori stay ≥ 28.98° away and the chaotic orbit ≥ 31.50°.
- **Chart scale over stored points: 0.50–11.9** (99th percentile 4.18), with max |X| = 4.78. The first pole gave 0.50–46.9 and 9.6.
- **Replacement for kam 38: kam 32**, at 25.79° from the new pole, λ = 4.2e-4. It is an island chain in the central sea plus a rim chain.
  - Candidates had to meet the M1 filter and lie at least as far from the pole as the pole's own minimum angle, so adding one cannot move the pole. Among those, I took the one least like the 7 orbits kept.
  - Re-running the pole search with kam 32 included returned the identical pole.
- The first pole is kept as `cache/pole_first.json` for the M3 second-pole plate.
- **Re-checks with the new pole:** tests 5/5 pass; inverse-chart round trip 5.6e-16; H drift max 3.7e-9 (regular orbits 9.2e-10).
- **Membrane on the new chart:** 95.8% of the section dots fall in a membrane voxel, and 99.99% in it or a face neighbour. Grid energy error |H − 2.8| ≤ 8.9e-16.

### 2. Break-up order
The first orbit to turn chaotic is the innermost ε = 0 seed, at ε = 0.04. NOTES.md records this, and every caption below that mentions break-up follows it.

### 3. Fog
- I extended the chaotic orbit to **T = 6·10⁶** (24 M samples in the stored segment's box). 52% of voxels are occupied, with a mean of 13.0 counts per occupied voxel.
- Decision: render the **raw 256³ grid, with no blur**. A full-resolution crop of the cutaway shows sharp island holes and only fine grain. The 128³ + σ = 1 variant (tried at 512²) softened the hole rims. It can still be rendered with `--fog blur128`.
- **Split-half correlation dropped to r = 0.53, but not from noise.** The orbit stays near an island for about 4·10⁵ time units (continuation chunks 17–18, λ = 0.004 and 0.006 there), so the two halves differ by a real sticky episode.

## Images (all orthographic)

Every caption names the chart as: *energy level set H = 2.8 → radial projection to S³ in zero-mean logits → stereographic projection from p (min angle 23.6°, scale 0.5–11.9)*.

| file | size | measured | declared |
|---|---|---|---|
| `tori_plaster_cutaway.png` | 2400² | 12 ε = 0 orbits (t ≤ 1000 of the dt = 0.01 copies) as tubes; the nesting | tube radius 0.008; view at 52° from the doughnut axis (smallest-variance direction of torus 0); wedge cutaway of half-angle 50° about that axis, the same for all tori; tint = 0.55 white + 0.45 viridis(torus index); AO from a 400³ voxel occupancy of the tubes plus one raking light (form only) |
| `tori_glow.png` | 2400² | the same 12 orbits, same wedge and view | additive Gaussian hairlines, colour = plasma(0.15 + 0.8·i/11), exposure 0.22·(size/1024)^1.2, no occlusion, so brightness is where lines pile up in projection |
| `tori_plotter.svg` (+ `tori_plotter_raster.png` preview) | 2400² | the same orbits, hidden lines removed against the tube depth buffer | one pen per torus = viridis·0.42, strokes split at the wedge |
| `sea_islands_hero.png` | 2400² | fog = time density of the chaotic ε = 0.5 orbit (256³, T = 6e6); 8 regular orbits (λ ≤ 5e-3) as tubes; section crossings as dots | transfer function: x = log1p(count)/q99.9, colour cmc.oslo(0.15 + 0.85x), extinction 12x²; tubes copper, t ≤ 150, radius 0.009, cropped to the box; sheet = ivory, extinction 3·exp(−(sdist/1.5 voxel)²)·[dg/dt > 0]; chaotic dots red, regular dots dark copper |
| `sea_islands_cutaway.png` | 2400² | the same fields; the island holes in the sea on the cut face, with island tubes passing through them | same transfer function with extinction 80x², so the cut face reads as a slice; vertical clip plane through the box centre |
| `sea_stereo_crosseye.png` | 4900×2400 | the cutaway scene | rotation stereo: eyes at azimuth ∓2.5° (orthographic cameras have no translation parallax), same clip plane for both eyes, right-eye view on the left |
| `sea_anaglyph_redcyan.png` | 2400² | the same pair | red = left-eye luminance, green and blue = right-eye luminance |
| `plate_section_flat_vs_membrane.png` | 6166×2000 | left: game-chaos `poincare_ink_eps0.50.png`, unchanged. Middle: the same 1.77 M crossings (395 orbits) in the plate's axes. Right: those crossings mapped through the chart, i.e. where they sit on the membrane | colour key = rank of x_R + y_P through cmc.batlow, identical in middle and right, so a point can be matched between panels; ink coverage 1 − exp(−g·hits), with g three times higher on the right; right panel framed on the fog box, view along the sheet's least-variance direction, tilted |

**How the slice plate checks itself:** the middle panel reproduces the left plate island for island. The right panel shows the same two central tear-drop islands and the surrounding sea, bent by the chart. The dots lie on the rendered membrane: 95.8% hit a membrane voxel.

## Render timings

| render | size | time | device |
|---|---|---|---|
| tori plaster | 2400 | 36 s | CPU |
| tori glow | 2400 | 1.1 s | CPU |
| tori plotter | 2400 | 2.8 s | CPU |
| slice plate | 2000 per panel | about 5 s including loading | CPU |
| sea hero | 2400 | 12.6 s | CUDA volume, CPU splats |
| sea cutaway | 2400 | 14.4 s | CUDA volume, CPU splats |
| stereo eyes | 2400 each | 15.7 s + 14.5 s | CUDA volume, CPU splats |

Times are logged in `art/invariant-tori/logs/render_times.log`, and the batch script is `render_m2.sh`.

## New compute
- **Render copies of the orbits** (`compute_orbits.py fine`) at dt = 0.01, t ≤ 2000. They are the identical trajectories: the difference from the stored samples is 0.0. They were needed because dt = 0.1 samples jumped up to 0.7 chart units in one step.
- **`fields256.npz`:** g, dg/dt (ε = 0.5) and signed distance on the density grid itself, which took 117 s on CPU.
- **Pole search and chart re-run:** about 2 min.

## Decisions
All are logged in NOTES.md:
- the pole rule and the replacement rule (Ruling 1);
- raw 256³ fog at T = 6e6;
- wedge cutaway and view axis;
- tint and lighting;
- plasma glow;
- the fog/sheet transfer function;
- tube time windows;
- rotation stereo;
- the three-panel slice plate with a rank colour key;
- splatting kept on CPU because of the r3d CUDA splat bug.

## Risks and concerns
- **Stereo separation is strong** (±2.5°). Disparity near the cut face is large at 2400 px, and the anaglyph may be tiring to view. ±1.5° would be gentler. It is a one-line change plus a 30 s GPU re-render, left for M3.
- **The sea hero is busy.** Copper arcs cross the whole box, and the island structure reads clearly only in the cutaway and stereo images, which are the stronger plates. One long, nearly straight tube segment in the hero is a real orbit segment through a high-scale part of the chart: samples there are ≤ 0.07 units apart, so it is not a chord artifact.
- **The section sheet is cropped by the fog box.** At grazing view angles its straight edges look like faint grey panels at the box boundary, as the captions must say.
- **Additive glow saturates** where tori are seen edge-on, near the core. Brightness there means line pile-up, not data.
- **Membrane panel of the slice plate:** the sheet folds away from the viewer, leaving a large empty region. The chart scale (up to 11.9×) makes the rim chains dominate the outer ring.
- **r3d:** no new bugs found. I worked around the known CUDA splat crash by splatting on CPU. I did not use `lut_tf`, so the linear-interpolation label issue does not apply.
- **M1 cached products were regenerated** for the new pole: stereo, density, membrane. The old M1 numbers in NOTES.md are labelled as belonging to the first pole.
