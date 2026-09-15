# Invariant Tori (§5), milestone M3 report

**Status: DONE_WITH_CONCERNS.** All four gate fixes and all M3 deliverables are done, and the README link check passes. The only concern of substance is the STL, which is sparse (see Concerns).

**Commits:**

| commit | contents |
|---|---|
| `3defa43` | the four fixes, second-pole plate, STL, README draft |
| `c11b2aa` | films, re-rendered sea stills, final README |

The notes for this milestone are part of `c11b2aa`.

## Gate fixes
1. **Grey section-sheet panels.** The sheet is now multiplied by the fog's support: clip(2·G_σ=2 voxel([count > 0]), 0, 1). The flat grey panels at the box faces are gone in the exterior, cutaway, stereo and turntable renders.
2. **Flat bottom cut.** The density box is now the full support of the stored chaotic segment plus 2% per side, replacing the 0.5–99.5 percentile clip.
   - Box: lo (−3.383, −2.169, −2.710), hi (3.596, 2.122, 2.392); 99.95% of 24 M samples fall inside.
   - Occupancy: 30.8% of voxels, mean 23.2 counts per occupied voxel; split-half r = 0.61.
   - Downstream products were recomputed on the new box: `fields256`, and the membrane (96.9% of dots in a membrane voxel, 100% counting neighbours, grid |H − 2.8| ≤ 1.3e-15).
   - Result: the fog has no flat cuts left. The only straight edge is the cutaway's declared clip plane.
3. **Stereo.** Both the pair and the anaglyph now use ±1.5°.
4. **Plaster cutaway.** Renamed to `tori_plaster.png`, with a real slice companion, `tori_slice.png`.
   - The slice is exact: 163,955 crossings through the meridional plane containing the doughnut axis, over t ≤ 1e5, interpolated between dt = 0.01 steps.
   - It shows two sets of 12 nested closed curves. One torus is near-resonant and appears as a dotted curve.
   - The tori scene frame (axis and camera) is now defined once, in `render_lib.tori_frame()`. Plaster, glow and plotter were re-rendered with it, and I looked at them again.

## Deliverables (all images and sample film frames viewed)

### Films
- **ε-sweep film:** `gallery/film_eps_sweep.mp4` (17 s, 1080², 6.9 MB) plus a GIF (12.7 MB).
  - **Measured:** 26 keyframes, drawing t ≤ 1000 at dt = 0.01 per keyframe (`compute_orbits.py sweep_fine`).
  - **Declared:** each keyframe is held 10 frames, then cross-faded to the next over 6 frames (opacity only). The camera spins 60° about the torus axis over the film. Chaotic orbits are pale cyan, regular ones plasma(index).
  - **Label:** ε, the chaotic count, and the measured break-up order.
  - **Frames viewed:** ε = 0, 0.02, 0.04, 0.24, 0.36, 0.50.
  - **Rendering:** 69 s on GPU. The first encode was 24 MB, so I re-encoded at crf 27 to get under the 20 MB commit limit.
- **Turntable:** `gallery/film_turntable_sea.mp4` (10 s, 240 frames, 1080², 7.4 MB) plus a GIF (7.2 MB).
  - It is the cutaway with a clip plane that faces the camera (declared).
  - Island holes show from every side.
  - Rendering took 594 s on GPU, after about 1.8 h waiting in the shared `gpu1.sh` queue.

### Second-pole plate
`gallery/plate_second_pole.png` is a 2×2 glow plate: current pole versus the M1 pole.

| orbits | current pole | M1 pole |
|---|---|---|
| ε = 0 tori | 29.0°, scale 0.50–8.0 | 30.5°, scale 0.50–7.2 |
| ε = 0.5 regular | 23.6°, scale 0.50–11.9 | 20.9°, scale 0.50–15.1 |

### STL
`gallery/torus_woven_kam201.stl` (8.1 MB) is built with `r3d.tube_mesh`. It is declared "a torus woven from its own orbit".
- **Orbit:** regular ε = 0.5 orbit kam 201 (central island), window t ≤ 70.
- **Size:** 80 × 72 × 27 mm, tube radius 0.6 mm.
- **Mesh:** 162,816 faces, watertight, volume 1079 mm³.
- **How the window was chosen:** a declared self-avoidance rule takes the longest window with no tube-to-tube approach closer than 2.2r.
- **Preview:** `torus_woven_kam201_preview.png`.

### README
`art/invariant-tori/README.md` follows the plan's shape. Hero = `tori_glow` + `tori_plaster`, with the sweep film underneath. It contains:
- the chart derivation (convexity → star-shaped level set → S³ → stereographic);
- pole angles and scale ranges;
- H-drift and λ tables;
- the measured break-up order;
- references (SAF 2002, game-chaos, r3d).

Every gallery link resolves.

## Other re-renders
- **Sea stills** (hero, cutaway, stereo, anaglyph) at 2400² on GPU, 13–17 s each.
- **Slice plate:** the membrane panel is now framed on the 5–95 percentile box of the crossings (declared).

## Key numbers
| quantity | value |
|---|---|
| pole | min angle 23.64°, chart scale 0.50–11.9 |
| H drift | ≤ 3.7e-9 |
| λ, ε = 0 tori | (3.2–4.6)e-4 |
| λ, chaotic orbit | 0.0246 |
| tests | 5/5 pass |
| round trip | 5.6e-16 |
| sweep chaotic counts | 0, 0, 1, 1, 2, 1, 4, 4, 5, 4, 7, 6, 6, 7, 7, 10, 7, 8, 7, 9, 7, 7, 10, 6, 7, 8 |

**Break-up order:** the innermost torus (index 0) is chaotic at ε = 0.04 to 0.08. At ε = 0.10 only index 1 is chaotic, so the finite-time classification is noisy near the threshold. The README says this.

## Decisions (also in NOTES.md)
- **Density grid:** kept at 256³ over the larger box (voxel 0.027 chart units), with no blur.
- **Film colours:** chaotic = pale cyan, regular = plasma(index), as in the glow stills.
- **Film tween:** 60° camera spin over the film; constant exposure.
- **Turntable cut:** the clip plane turns with the camera.
- **STL construction:** tube_mesh with the self-avoidance cap, not a fused union. I tried a union (300³ EDT, kam 121, t ≤ 300): it was watertight but lumpy at a printable voxel size, and I dropped it.
- **Plate framing:** the membrane panel is framed on the dense core of the crossings.

## Concerns
- **The STL reads as a loose coil (about 5 windings), not a woven torus.** Neighbouring windings of these thin island tori come within about 0.02 chart units by t ≈ 80, so a self-avoiding printable tube cannot be longer. A fused-union object would need a finer grid (≥ 600³) to look smooth. That is feasible if you want it.
- **Sweep film content:** at ε = 0.02, one regular torus near the section fold already jumps to a large ring. This is measured, not an artifact, but the eye may read it as the first break-up. The label gives the λ count.
- **Exterior sea hero:** there is still a faint ivory glow near the upper right where the sheet meets the fog edge. The support fade removed the panels, but not all sheet brightness.
- **`commit.sh` bug:** it prints a `stat: cannot statx` error for files deleted in the working tree (line 22). The deletion is still committed. This is a bug in the shared script, not in this piece.
- **Queue latency:** each GPU batch waited 45 min to 1.8 h behind other art jobs.
