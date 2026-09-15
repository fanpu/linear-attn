# §8 Combed — M2 report (renders)

**Status:** DONE. 38 PNGs plus 4 SVGs were rendered and every image was viewed. Six styles:

1. Dark-ground glow hair
2. Matte plaster tubes
3. Dark tubes coloured by t
4. Plotter hidden-line SVG
5. Cross-eye stereo
6. Crisp-voxel basin with its null and a slice atlas

A box-counting verification plate is included. The controller's M1 rulings are applied:

- Every render uses the run to t = 1 − 10⁻⁶.
- Every printed memorised fraction appears with the fresh-knot null for that N.
- Width and step budget are fixed and declared in the captions.

## What was computed
| Step | Where | Wall clock |
|---|---|---|
| `compute_m2.py dense` renders dense RK4 trajectories for 10 fields: 257 uniform states on [0, 1−10⁻³] + 12 of 96 tail steps geometric in (1 − t) to 1−10⁻⁶, float64 integration stored as float32 | GPU via `gpu1.sh` for N ≥ 64; the N = 16 files were made on CPU (device is recorded in the npz) | closed-form 2 s (N = 64) to 117 s (N = 4096); MLP about 35 s each |
| Consistency check | — | the stored state at 1−10⁻³ matches the M1 CPU `end256` to ≤ 6e-8, which is float32 storage rounding |
| `compute_m2.py basin`: N = 16 closed-form basin labels on a 128³ and a 256³ grid of noise in [−2.5, 2.5]³, stopped at 1−10⁻⁶; null = Voronoi cell of the start point | GPU via `gpu1.sh` | 83 s (128³), 656 s (256³) |
| `compute_m2.py boxcount`, `plot_verify.py` | CPU | seconds |
| SVGs (`render_tubes.py svg`) | CPU | 3–4 s each |
| Heroes: `logs/cpu_m2_heroes.sh` | CPU, 4 threads (controller-approved) | 689 s total: tube set 8 s per (N, field); basin 9 s / 17 s; diptych 68–107 s at 2048 px per panel |

Queue history: the GPU render job `logs/gpu_m2_render.sh` waited 05:02–06:35 behind about 9 art jobs. The controller approved CPU rendering, and I killed only my own `gpu1.sh` process and its `flock` waiter, before it took the lock.

## Numbers
- **Memorised fraction at the render stop t = 1−10⁻⁶** (raw endpoint, d1 < d2/3), for N = 16 / 64 / 256 / 1024 / 4096:

  | | 16 | 64 | 256 | 1024 | 4096 |
  |---|---|---|---|---|---|
  | closed-form | 1.000 | 1.000 | 1.000 | 0.998 | 0.998 |
  | MLP | 0.704 | 0.434 | 0.183 | 0.049 | 0.005 |
  | null (fresh points on the knot) | 0.263 | 0.367 | 0.313 | 0.333 | 0.345 |

- **Basin, N = 16:**
  - Label counts are 48k–261k voxels per point at 128³.
  - Only 76.6% of voxels agree with the Voronoi-of-x0 null. The boundaries are curved sheets, not planes (see the slice atlas).
  - **Box counting of the label boundary** (boxes of side ε meeting a voxel with a differing 6-neighbour):

    | Grid | Fit range (ε, voxels) | Basins D | Null D |
    |---|---|---|---|
    | 256³ | 1–16 | 2.13 | 2.10 |
    | 128³ | 1–8 | 2.16 | 2.13 |

  - The basins' local slopes track the planar null to within 0.03–0.07 at every ε. Both rise toward 3 at large ε, where boxes saturate.
  - Boundary voxels 256³ / 128³ = 4.11 (a surface gives about 4).
  - Conclusion: a surface. **No fractal claim.** The fit spans only 1.2 decades, because larger boxes saturate; this is declared on the plate.

## Images (all in `art/combed/gallery/`; each caption strip is burned into its image)
| File(s) | Measured | Declared |
|---|---|---|
| `hair/hair_diptych_N{16,64,256,1024,4096}.png` (4104×2048 + strip). Left: closed-form v*; right: trained MLP | all 20k trajectories (RK4 states to 1−10⁻⁶), endpoints, memorised fractions with the null | identity chart, orthographic, az −60 el 40, ortho height 3.4. Hue = density-weighted mean t via colorcet bmy. Brightness = (log1p(W)/log1p(150))^0.9, where W = hair length per 2048-px-equivalent pixel. Endpoint glow σ 1.6 px, tonemap 1−exp(−0.05x), screen blend. Black ground. |
| `stereo/hair_stereo_crosseye_closed_N64.png`, `stereo/hair_stereo_crosseye_mlp_N1024.png` (1600 px per eye) | same as the diptych | cross-eye layout (right-eye view on the left); rotation stereo at azimuth −60 ∓ 2.5°; the same glow mapping |
| `tubes/tubes_{plaster,dark}_{closed,mlp}_N{64,1024}[_cutaway].png` (2400 px; 16 images) | the first 400 seeds' trajectories; endpoints at 1−10⁻⁶; memorised fraction and null over all 20k | tube r 0.0075, endpoint spheres r 0.022. Plaster: matte off-white tubes, terracotta endpoints, grey ground, one raking Lambert light (form only; no AO or shadow). Dark: hue = t (bmy), warm-white endpoints. Cutaway: every sample on the camera side of the plane through the origin facing the camera is removed. |
| `plotter/plotter_{closed,mlp}_N{64,1024}.svg` (2000 px, about 7.6 MB each) plus 1000 px PNG previews | 2,000 trajectories, t ≥ 0.5 | declared time window t ≥ 0.5. Hidden lines against a tube depth buffer, r = max(0.004, 2.5 px), eps = 10 r. Runs shorter than 3 px dropped. Stroke #1f1d1b 0.44 px on #f3efe6. Orthographic, ortho height 2.6. |
| `basin/basin_R{128,256}[_cutaway].png` (2400 px) | basin label per noise voxel; training-point positions | crisp voxels (`render_voxels`, no interpolation). Palette colorcet glasbey_category10 in training-index order. Lambert at 0.5 ambient (form only). Cutaway removes the octant x>0, y<0, z>0; training points shown as spheres in their basin colour. |
| `basin/basin_null_voronoi_R{128,256}[_cutaway].png` | null labels (Voronoi cell of x0) | the identical render path |
| `basin/basin_slices_R{128,256}.png` | 6 axis slices through labels (top row) and null (bottom row), one block per voxel | palette as above |
| `verify/basin_boxcount.png` | box counts and local slopes, basins vs null, at 128³ and 256³ | matplotlib plate |

Suggested M3/README heroes:

- `hair/hair_diptych_N64.png`: tufts vs the knot forming.
- `hair/hair_diptych_N1024.png`: dotted vs smooth knot.
- `tubes/tubes_dark_mlp_N1024.png`
- `basin/basin_R256_cutaway.png`, with its null.

## Decisions (all logged in `art/combed/NOTES.md`)
1. **Render from dense RK4 states, not the 64 cached M1 points.** Linear interpolation of the 64 points misses the true states by up to 1.6e-2 (p99.9 ≈ 7–10e-3, about 5 px at 2048 px).
2. **MLP renders also use the 1−10⁻⁶ run**, so the diptych stays matched. The ruling mandates this only for the closed form; MLP endpoints move by less than 1e-3 between the two stops.
3. **Log-density brightness with mean-t hue for the glow.** A plain additive tonemap either saturated the core or hid the outer hairs.
4. **SVG shows t ≥ 0.5 only.** With full-length hairs the SVG is a uniform starburst. Hidden-line eps = 10 r, because curves heading along the view axis otherwise self-occlude into dashes.
5. **Rotation stereo.** `r3d.stereo_pair` shifts an orthographic camera sideways, which gives no parallax. This is worth a note in the r3d README.
6. **No AO on the tubes.** r3d has no occluder for splatted tubes. The plaster ground was darkened so that the white tubes read.
7. **Box-counting fit range ε = 1..R/16**, because larger boxes saturate.
8. **Hero N = 64 (tufts) and 1024 (knot)** for tubes and SVG; diptychs at all five N (they feed the M3 film).
9. **CPU heroes instead of the GPU queue**, approved by the controller.

## Risks and open issues
1. **Caption strips are small** on the 2400 px tube and basin images (about 13–17 px text). They are readable at full size but will need re-flowing or README captions in M3.
2. **Stack line on the diptychs.** Its "GB10 … CUDA 13.0" refers to the stack. The dense trajectories for N ≥ 64 were computed on CUDA, while the images themselves were rasterised on CPU. M3 README captions should say so.
3. **Box-counting range is 1.2 decades, below §11's two decades.** Acceptable because no fractal claim is made and the planar null matches; a 512³ basin (about 90 min GPU) would extend it if wanted.
4. **The MLP's low memorised fraction at N ≥ 256 is below the null** (M1 risk 1 still stands). The diptych captions print the null beside it.
5. **Tube images show only 400 of the 20k seeds**, so any tuft structure in them is a subsample. The glow diptychs use all 20k.
6. **Box-counting range in captions.** The render captions quote D with its fit range, so they must be regenerated if `basin_boxcount.json` changes.

## Commits
- `d0c2c3a`: M2 compute, render scripts, SVGs, box counting.
- `858afa4`: final M2 commit (heroes, NOTES, cleanup of superseded 1024 px previews).
