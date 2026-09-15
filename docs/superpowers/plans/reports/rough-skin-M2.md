# Rough Skin (§7) — M2 report: renders

**Status: DONE.** Every requested render exists at ≥ 2048 px, and I looked at each one. The two extra width-4096 draws are measured. The plate-reproduction check against `depth-roughness` passes: 3 and 27 of 10⁶ pixels differ in sign. L = 2 reads as foam from outside, so the slice atlas carries it; the details are below.

Commits: `9320f45` (casts, atlases, fog, diptych, code), and `7d52594` (draw statistics, plate check, notes; the report file itself lives outside the piece directory and is not committed by commit.sh).

## 1. Draw statistics (ruling 1)

The main draw is seed 7; seeds 8 and 9 were queued at the start (`run_draws4096.sh`, 300 s of GPU each; the rate was 5.7e4 voxels/s on the shared GPU).

Measurement settings:
- width 4096, 256³ grid, level = volume median;
- 3D box counting over b = 2–64 voxels (1.5 decades), calibrated with the window protocol at k = 1;
- 12 oblique slices, calibrated with the same protocol.

| act | L | theory | 3D raw per draw | **3D calibrated, mean ± sd (3 draws)** | systematic: k = 0 | systematic: periodic protocol | 1 + slice raw | **1 + slice calibrated, mean ± sd** |
|---|---|---|---|---|---|---|---|---|
| Heaviside | 1 | 2.5 | 2.443, 2.475, 2.496 | **2.426 ± 0.031** | 2.463 | 2.358 | 2.405 | **2.448 ± 0.067** |
| Heaviside | 2 | 2.75 | 2.871, 2.788, 2.736 | **2.774 ± 0.076** | 2.809 | 2.721 | 2.636 | **2.769 ± 0.040** |
| Heaviside | 3 | 2.875 | 2.960, 2.967, 2.955 | out of range (no claim) | – | – | 2.816 | – |
| Heaviside | 4 | 2.9375 | 2.988, 2.985, 2.992 | out of range (no claim) | – | – | 2.921 | – |
| ReLU | 1–4 | 2 | 2.007–2.045 | below the power-law range; the smooth null reads 2.03–2.06 | | | 2.09–2.11 | |

- **Heaviside L = 1:** the three-draw mean is within 0.074 of 2.5 in 3D and 0.052 via slices.
- **Heaviside L = 2:** within 0.024 in 3D and 0.019 via slices. The high M1 value (2.856) was draw scatter.
- **Systematics:** both are about as large as the draw sd.
  - The window-versus-periodic calibration protocol shifts D by −0.07.
  - The sub-voxel roughness assumption (k = 0 vs k = 1) shifts it by +0.04.
- **L ≥ 3:** the estimator saturates, so the README (M3) will make no D claim there.
- **Source:** full table in `art/rough-skin/cache/dims_tables.md`, from `measure_dims.py`.

## 2. Plate reproduction (§0.7, §11.3)

The great 2-sphere {x₄ = 0} ⊂ S³ is S², and a width-n net on S³ restricted to it is a width-n net on S² with input weights W₀[:, :3]. `verify_plate.py` feeds depth-roughness's own n = 4096, seed-11 weights (its `nets.py`, imported, unmodified) through this piece's `forward_all`, with inputs embedded as (x, y, z, 0). It compares the result with `depth-roughness/cache/nets_patch.npz` on its 1024² Lambert patch.

| L | pixels | sign disagreements at the plate median | max abs field difference |
|---|---|---|---|
| 1 | 1,048,576 | **3** | 0.056 |
| 2 | 1,048,576 | **27** | 0.30 |

- **Cause:** the differences are single hidden units flipping at ties (float32 here vs float64 there, and `>` vs `>=`). Each flip moves the output by √(2/n)|v_i| ≈ 0.02|v_i|.
- **Image:** `gallery/verify_plate_depth_roughness.png` shows, left to right, depth-roughness's plate, this code, and the disagreement mask in red. It is viewed and indistinguishable.

## 3. Renders (ruling 2)

All use one chart: the exp map of the 0.5 rad cube, world = tangent coordinates / 0.25 rad, orthographic camera at az −35°, el 32°. All show the solid {T ≤ u} of the main draw (seed 7), in plaster with one raking light (0.45, −1, 0.75), a hard shadow and 32-ray AO. Darker plaster marks cut faces: the cube boundary and the cutaway planes.

| file | what | render |
|---|---|---|
| `gallery/cast_heaviside_L1_voxel.png` | Heaviside L = 1, crisp voxels, exterior (hero) | CPU 51 s |
| `gallery/cast_heaviside_L1_voxel_cutaway.png` | same, with a quadrant cutaway | 81 s |
| `gallery/cast_heaviside_L2_voxel.png` | Heaviside L = 2, exterior | 56 s |
| `gallery/cast_heaviside_L2_voxel_cutaway.png` | L = 2 cutaway | 94 s |
| `gallery/cast_relu_L1_voxel.png`, `..._cutaway.png` | ReLU twin through the identical voxel path (null) | 100 s, 118 s |
| `gallery/cast_relu_L1_iso.png`, `..._cutaway.png` | ReLU linear isosurface (allowed for ReLU only) | 137 s, 131 s |
| `gallery/diptych_heaviside_L1_vs_relu.png`, `..._cutaway.png` | L = 1 against ReLU: same weights, same camera, same voxel path and light; captioned (4136×2638) | composed |

What the images show:
- **Heaviside L = 1:** a rough, crumbling skin over a block, with overhangs and loose voxel islands. Crisp voxels are visible at 1:1. Faint straight streaks are the finite-width great-sphere facets (≈ 4/n), a real property of the finite net, not a render artefact.
- **ReLU voxel:** one smooth sheet with regular voxel terraces. The null stays smooth through the identical path, so the renderer is not the fractal.
- **ReLU iso:** one smooth sheet with faint straight creases, the kinks of a piecewise-linear network.
- **L = 2 looks like foam from outside** (noted per the ruling). The exterior is a porous cube, and the cutaway's section faces show more foam rather than a readable interface. Its roughness is carried by the slice atlas and the fog.

## 4. Slice atlas (ruling 3)

The cuts are the planes z = const of the chart (z-voxels round(linspace(8, 247, 12))). The coastline is T = volume median, drawn along voxel edges between opposite signs, with no interpolation. The style is depth-roughness plotter: its paper, ink and C059 helpers are imported from its `render_common.py`.
- `gallery/atlas_heaviside_L1.png` … `atlas_heaviside_L4.png` and `gallery/atlas_relu_L1.png`: 12 cuts each, 4526×3732.
- `gallery/atlas_depth.png` (5358×4360): rows Heaviside L = 1–4 and ReLU L = 1; columns are 6 of the 12 cuts.

This plate reads the dial best: L = 1 is coastline islands, L = 2 lace, L = 3–4 near-noise. ReLU shows a single smooth curve, and it does not cross the lowest cut.

## 5. Spectral split fog (ruling 4)

`gallery/fog_split_heaviside_L1_paper.png`, `fog_split_heaviside_L2_paper.png`, `fog_split_relu_L1_paper.png` (2048², CPU about 6 s each).

- **Slab:** z-voxels 116–139 (24 voxels, 0.047 rad).
- **Declared transfer function:**
  - Each side of the median is rank-normalised on its own.
  - The below side runs pale yellow → purple #5e4fa2 at the level; the above side runs deep red #9e0142 at the level → pale yellow.
  - Opacity is 0.10 + 0.90(1 − |q|)^24, with extinction 25 per world unit.
  - Sampling is nearest-voxel with step h/2, jittered; paper ground; orthographic camera at el 58°.
- **The ReLU null goes through the identical function.**

What the fog shows:
- **L = 1:** a lace of dark seams around pastel islands, with facet lines visible.
- **L = 2:** fine haze.
- **ReLU:** one soft seam band. Rank normalisation makes the "within 3 % of the level" band wide for a smooth field, which is a property of the declared TF.

A dark-ground variant was tried and dropped: low-opacity emission reads muddy on black.

## Decisions (all logged in `art/rough-skin/NOTES.md`)

- **Solid side:** solid = {T ≤ u}, viewed from above. For this draw {T > u} sits mostly in the upper half for both Heaviside L = 1 and ReLU, so the skin is the top of a block. The other side would hide the skin inside a closed box.
- **Quadrant cutaway instead of the planned octant:** the cut removes x > 0, y < 0 through the full height. A top octant misses most of the L = 1 skin, which lies near z ≈ 0. This deviates from the plan's wording.
- **ReLU rendered twice:** in voxel mode as the null and for the diptych, and as an iso for its own plate. The iso cutaway is a CSG min with the quadrant's signed distance, which rounds the cut edge within one voxel (declared).
- **Casts and fog rendered on CPU (deviation from "heroes through gpu1").** The art GPU lock was held by solid-edge's B256 volume (≈ 3 h, with four jobs waiting). r3d renders 2048² from 256³ in 50–140 s on 4 CPU threads. The GPU still did the extra draws and the plate check through gpu1.
- **Draws are statistics only:** extra draws skip the float64 slab, which is checked on the main draw only. Renders use the main draw.

## Risks / open

1. **Readability:** L = 2 is foam from outside, and L = 3–4 are not rendered in 3D (per the ruling). The M3 film has to lean on slices for depths ≥ 2.
2. **AO darkness:** AO makes the rough skin darker than the flat section faces. Light is declared as form-only, but viewers may still read darkness as depth into pores, which it partly is.
3. **ReLU atlas:** the ReLU surface is nearly planar across the cube, so some ReLU atlas cuts contain no coastline. That is faithful but sparse.
4. **No README yet (M3):** the D table there must use §1's three-draw statistics with the systematics and the 1.5-decade range.
5. **Cache size:** 2.6 GB, including three 536 MB fields. Chunk checkpoints of finished draws were deleted.
