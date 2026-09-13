# Posterize

*Rounding a smooth image leaves two different fingerprints: the pixel grid aliases the signal, and the quantizer's error, which is full of harmonics, is aliased again by the same grid. When you print in one bit, the halftone's threshold matrix adds a third fingerprint, and Bayer's is recursive.*

<p align="center">
<img src="gallery/zone_hero_ws_int3_crop2048_spectral.png" width="49%">
<img src="gallery/bayer_digits_dark.png" width="49%">
</p>
<p align="center"><img src="gallery/halftone_zone_paper.png" width="98%"></p>

## 1. The phenomenon

**Zone plate** f(x, y) = cos(k r²), r in pixels. Its local radial frequency is 2kr rad/px, which reaches Nyquist (π) at R<sub>N</sub> = π/2k. Two separate things happen to a quantized zone plate:

1. **Sampling aliasing (signal).** Beyond R<sub>N</sub> the pixel grid aliases f itself into ghost zone plates. The ghosts are centred every 2R<sub>N</sub> px, and odd ones carry a (−1)<sup>x</sup> checkerboard (checked: max deviation 0 at x = 1024).
2. **Amplitude quantization error.** e = Q(f) − f = g(φ) is a periodic function of the phase φ = kr², so it expands as Σ c<sub>m</sub> cos(mkr²). The m-th harmonic has local frequency 2mkr and aliases beyond R<sub>N</sub>/m. The error therefore aliases much closer to the centre than the signal does. That is the moiré you see even in a "well sampled" image.

The grid separates them. Row 1 is f<sub>point</sub> − f<sub>band-limited</sub> (sampling only). Row 2 is the continuous band-limited error Σ c<sub>m</sub> taper(2mkr/π) cos(mkr²) (quantization only). Row 3 is Q(f<sub>point</sub>) − f<sub>point</sub>, what the tensor actually holds (combined). Row 4 is row 3 − row 2, i.e. error harmonics folded by the grid.

**FP8 product.** (Q(Q(x)Q(y)) − xy)/xy. Input rounding gives rectilinear cells whose width doubles every octave, and output rounding gives hyperbolae. Because Q(2x) = 2Q(x) in the normal range, the field is exactly periodic in log₂x and log₂y.

**Bayer matrix.** M<sub>1</sub> = [[0,2],[3,1]], M<sub>2k</sub> = [[4M<sub>k</sub>, 4M<sub>k</sub>+2],[4M<sub>k</sub>+3, 4M<sub>k</sub>+1]]. Equivalently, M = Σ<sub>b</sub> d<sub>b</sub>·4<sup>n−1−b</sup> with digits d<sub>b</sub> = 2(x<sub>b</sub> XOR y<sub>b</sub>) + y<sub>b</sub>. The *finest* coordinate bits set the *most significant* digits, so neighbours get distant thresholds. That is why M looks like noise even though it is exactly recursive; read the same digits coarse-first and the nested quadrants appear (`bayer_digits`).

## 2. Gallery

### Recursive Bayer

| | |
|---|---|
| <img src="gallery/bayer_digits_dark.png"> *The same 8 base-4 digits: fine-first (the dither matrix, looks like noise), coarse-first (nested quadrants), and the two interleaved bit streams y and x XOR y ("munching squares").* | <img src="gallery/bayer_selfsim_paper.png"> *The top-left 2<sup>m</sup>×2<sup>m</sup> corner of order 8 is exactly order m (checked). Top row: coarse-first digits; bottom row: Bayer.* |
| <img src="gallery/bayer_genealogy_dark.png"> *M<sub>2</sub> … M<sub>256</sub> at one print size, with ranks written out up to 8×8.* | <img src="gallery/bayer_equation_riso.png"> *The substitution rule M<sub>256</sub> = 4·tile(M<sub>128</sub>) + expand(M<sub>2</sub>), asserted bit-exactly.* |
| <img src="gallery/bayer_crosshatch_crops_paper.png"> *Cross-hatch up close: constant grays 1/8 … 5/8 at orders 4–256, 1 dot = 6 px.* | <img src="gallery/bayer_crosshatch_riso.png"> *0→1 ramp per order. Riso rows alternate blue/pink ink (declared).* |
| <img src="gallery/bayer_bitplanes_riso.png"> *The 16 bit planes of M<sub>256</sub>: checkerboards and stripes, one scale per bit pair.* | <img src="gallery/bayer_specimen8_paper.png"> *Specimen sheet: all 65 tones of the 8×8 matrix.* |

<img src="gallery/bayer_sweep.gif" width="45%"> [MP4](gallery/bayer_sweep.mp4). *Threshold sweep on a 2×2 tiling of M<sub>256</sub>. Geometric steps first, so the lattice stages (1, 4, 16, … dots per tile, spacing 256/2<sup>k</sup>) are visible.*

All Bayer plates come in `_dark`, `_paper` and `_riso` versions.

### Zone-plate error fields (sampling vs quantization)

| | |
|---|---|
| <img src="gallery/zone_grid_ws_spectral.png"> *Well sampled (R<sub>N</sub> = 2897 px > image corner). Row 1 is identically zero in the window, but row 3 is full of ghosts: those come from the quantizer's harmonics, not the signal. Sohl-Dickstein Spectral (`cdf_img`, sign-preserving rank normalisation, ported exactly from his notebook).* | <img src="gallery/zone_grid_al_seam.png"> *Aliased (R<sub>N</sub> = 1024 px). Row 1 now carries the signal's own ghosts. Split-at-zero Spectral: each sign rank-normalised, dark ends meet at zero.* |
| <img src="gallery/zone_grid_ws_diverge.png"> *Metric version: linear cmcrameri vik, symmetric limits at the 99.5th percentile.* | <img src="gallery/zone_grid_al_riso.png"> *Riso: pink = rounded up, blue = rounded down, 1-bit FS halftone of \|e\|.* |

Heroes at 1 px = 1 sample: `zone_hero_ws_int3{,_crop2048}_*`, `zone_hero_al_fp4_crop2048_*` and `zone_hero_ws_fp8_crop1536_*`, each in spectral, seam, diverge and riso. Grid panels are native-resolution crops x, y ∈ [2048, 3072) (image centre at the panel's top-left corner). **Downsampling a zone plate for display creates new aliasing, so thumbnails look muddy; view at 100%.**

### FP8 product

| | |
|---|---|
| <img src="gallery/product_fp8_loglog_spectral.png"> *Log-log axes 2<sup>−4</sup>–2<sup>4</sup>: one octave cell, tiled. Near the origin (bottom-left) the subnormal range breaks the tiling.* | <img src="gallery/product_fp8_linear_spectral.png"> *Linear axes (0, 8]: cells double in width every power of two.* |

### Halftones (Floyd–Steinberg vs Bayer vs blue noise)

| | |
|---|---|
| <img src="gallery/halftone_basin_riso.png"> *A fractal from `art/gd-bifurcation` (basin_wide_dark luminance, read-only) printed three ways, with exact-dot crops.* | <img src="gallery/halftone_zone_paper.png"> *Zone plate in 1 bit: Bayer's period-4 and period-8 components create ghost zone plates at the predicted offsets 2R<sub>N</sub>m/P = 724m and 362m px.* |
| <img src="gallery/halftone_spectra_dark.png"> *Spectra of flat grays: Bayer gives isolated lines, blue noise a dark low-frequency disc, white noise is flat. At rational grays FS locks into periodic worms.* | <img src="gallery/halftone_overprint_riso.png"> *Overprint: blue-noise halftone (blue) over Bayer-256 (pink), shifted (3, 2) px.* |

## 3. What was computed

Exact float64 numpy on CPU; FS error diffusion is a small C loop (ctypes). No GPU.

- **Formats:** int-b symmetric abs-max with 2<sup>b</sup>−1 levels, clipped. FP4 E2M1 and FP8 E4M3FN are enumerated from bit patterns, round-to-nearest with ties-to-even, and saturating. This is the same code as `hardware/dither`, where the FP8 grid was verified bit-exact against `torch.float8_e4m3fn`.
- **Zone plates:** 4096², k = π/(2·2897) (ws) and π/(2·1024) (al). Harmonic coefficients c<sub>m</sub>, m ≤ 2048, come from an FFT of g(φ) with 2<sup>18</sup> samples. The series residual (RMS, relative) is 3% (int2) to 13% (FP8). The unresolved tail has m > 2048 and aliases at r > R<sub>N</sub>/2048 ≈ 1.4 px, so it belongs in the "aliased" part anyway. The band-limit taper is a raised cosine over the last 10% below Nyquist (declared).
- **Halftones:** blue noise is Ulichney void-and-cluster, 128², σ = 1.5, seed 0. The white-noise null is a random rank matrix. Spectra use 1024² flat grays.
- **Wall time:** fields ~2 min, Bayer and halftones seconds, renders ~15 min.

```
cd hardware/posterize; PY=/home/fzeng/ml/research/art/.venv/bin/python; export OMP_NUM_THREADS=4
$PY compute_fields.py; $PY compute_bayer.py; $PY compute_halftones.py
$PY render_bayer.py; $PY render_zone.py; $PY render_product.py; $PY render_halftones.py
```

## 4. Verification

- **Bayer:**
  - The recursion equals the closed form for orders 1–8, and each matrix is a permutation.
  - Level-set self-similarity holds for **all 87 388 (m, c) pairs** at order 8: {M<sub>8</sub> < c·4<sup>8−m</sup>} = tile({M<sub>m</sub> < c}).
  - The first 4<sup>k</sup> dots form a lattice with minimum spacing 128, 64, 32, 16, 8 for k = 1–5.
  - This is exact self-similarity by construction. It is not a fractal-dimension claim: every level set has dimension 2, so no box counting was done.
- **Zone plate, sampling vs quantization:**
  - Signal aliasing is 0.059 relative RMS for ws (all of it in the taper zone) and 0.90 for al.
  - The aliased share of the pixel error RMS is 0.94 (int2), 0.98 (int3), 0.99 (int4), 0.98 (FP4) and 0.995 (FP8) **even in the well-sampled image**. Within r < 256 px it is still 0.43 (int2), 0.75 (int3) and 0.96 (FP8).
  - So nearly all visible moiré in a quantized, well-sampled zone plate is quantization harmonics aliased by the grid, not sampling aliasing of the signal. Finer grids (FP8) have more high harmonics and alias more.
- **FP8 product:** shifting one octave in x leaves every normal-range pixel bit-identical (fraction **1.0000**; 98.5% of all pixels, the rest subnormal or saturated).
- **Halftones:**
  - Low-frequency (< 0.1 cyc/px) energy share at gray 1/3: FS 0.0001, Bayer-8 0, Bayer-256 0.0008, blue noise 0.0006, white noise 0.031 (equal to its area share 0.031, as expected).
  - Mean tone error is ≤ 3e-4, except Bayer-8 at 1/3 (−0.005): 8×8 has only 65 tones.

## 5. Caveats

- Spectral, seam and riso colourings are declared aesthetic mappings. The rank normalisation hides magnitude, so use the `diverge` plates for amplitudes.
- Thumbnails of zone plates alias on your screen. Halftone "full" panels are 2×2 averages (a filter with its own weak moiré); the dot crops are exact.
- The basin source image is an 8-bit sRGB render from another project; its luminance is taken without gamma decoding (declared).
- PNGs over the 20 MB commit cap were saved as 256-colour palette PNGs (`zone_hero_ws_int3_*`, `zone_grid_*`). No file was skipped.

## 6. Ideas explored / not pursued

Built: recursive Bayer (digits, self-similar corners, cross-hatch crops, sweep), the zone-plate decomposition, the FP8 product, halftone triptychs and spectra, and the riso overprint.

Not pursued (scope cut at the end for rate limits):
- Floyd–Steinberg of the error fields themselves, per format.
- A zoom movie of the log-log product tiling.
- Block-scaled NVFP4 zone plates.
- Dot-gain / ink-spread simulation for the riso prints.
- Blue-noise masks at several σ.
- Box-count comparison of halftone dot sets (all dimension 2, so uninformative).

## References

- B. E. Bayer, "An optimum method for two-level rendition of continuous-tone pictures," *IEEE ICC* 1973.
- R. W. Floyd, L. Steinberg, "An adaptive algorithm for spatial greyscale," *Proc. SID* 17(2), 1976.
- R. Ulichney, "The void-and-cluster method for dither array generation," *Proc. SPIE* 1913, 1993.
- S. P. Lipshitz, R. A. Wannamaker, J. Vanderkooy, "Quantization and Dither: A Theoretical Survey," *JAES* 40(5), 1992 (see `hardware/dither`).
- J. Sohl-Dickstein, "The boundary of neural network trainability is fractal," 2024; colour mapping `cdf_img` from github.com/Sohl-Dickstein/fractal.
- P. Micikevicius et al., "FP8 Formats for Deep Learning," arXiv:2209.05433.
