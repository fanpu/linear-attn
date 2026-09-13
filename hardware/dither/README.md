# Dither

*Round a sine to a few levels and the error is not noise. It is an exact lattice of harmonics that fold at Nyquist. Add the right noise first and the lattice melts. Feed the rounding error back, one bit at a time, and the lattice becomes a Farey fan.*

<p align="center">
<img src="gallery/hero_sd_idle_zoom_riso.png" width="49%">
<img src="gallery/hero_int3_native_magma_glow.png" width="49%">
</p>

<p align="center"><img src="gallery/sd_zoomseq_golden_deep_dark.png" width="100%"></p>

<p align="center">
<video src="gallery/video_scroll_int3.mp4" controls width="80%"></video><br>
<em>Scrolling spectrogram with its own audio. You hear 30 s of a 3-bit chirp rounded, then the same chirp with TPDF dither. The picture is computed from the same samples as the soundtrack.</em>
</p>

---

## 1. The phenomenon

**Quantization error of a deterministic signal is deterministic.** Round x(t) = A sin φ(t) to a grid of step Δ, and the error e = Q(x) − x = g(φ(t)) is a periodic function of the phase. For a symmetric mid-tread quantizer it has an exact Fourier series in odd harmonics only:

  e(φ) = Σ<sub>k odd</sub> b<sub>k</sub>(A) sin kφ,  b<sub>k</sub>(A) = Σ<sub>n≥1</sub> 2(−1)<sup>n</sup> J<sub>k</sub>(2πnA/Δ) / (πn)

(from the sawtooth expansion round(x) − x = Σ (−1)<sup>n</sup> sin(2πnx)/(πn) and the Jacobi–Anger identity). The strongest harmonics sit near k ≈ 2πA/Δ. For a 4-bit sine that is k ≈ 44. In a sampled system, harmonic k of a chirp lives at the folded frequency fold(k·f(t)) = |((k f + f<sub>s</sub>/2) mod f<sub>s</sub>) − f<sub>s</sub>/2|, so each harmonic climbs, reflects off Nyquist, falls, reflects off 0, and so on. Together they weave the lattice. The textbook "white noise of power Δ²/12" model is false here.

**Dither** (Lipshitz, Wannamaker & Vanderkooy, *J. Audio Eng. Soc.* 40(5):355–375, May 1992; citation checked against the paper) adds a random ν before rounding. For the total error ε = Q(x+ν) − x:

| scheme | E[ε \| x] | E[ε² \| x] | note |
|---|---|---|---|
| none | sawtooth −x … | x² (0 … Δ²/4) | harmonics |
| RPDF, 1 LSB peak-to-peak (±½ LSB), nonsubtractive | **0** | **depends on x**: 0 … Δ²/4 | no distortion, but *noise modulation* |
| TPDF, 2 LSB p-p (±1 LSB) = sum of two RPDFs | **0** | **Δ²/4, constant** | higher moments still depend on x; optimal (least power) of all dithers that fix the first two moments |
| subtractive RPDF (ν removed again after quantizing) | 0 | Δ²/12 | the error is uniform and statistically *independent* of x, not just moment-independent; needs ν at the decoder |

ML aside: stochastic rounding (floor(x + U[0,1))) is RPDF nonsubtractive dither. It is unbiased, but its variance depends on where x sits between the levels.

**Floating-point grids** are logarithmic: Q(2x) = 2Q(x) outside the subnormal range, so the error pattern relative to the signal repeats every octave of level, while a uniform grid loses 6 dB per 6 dB.

**Sigma-delta modulators** turn 1-bit rounding into high-resolution audio by feeding the error back: y[n] = sign(w[n]), w[n] = u[n] − Σ c<sub>k</sub> e[n−k], so Y = U + (1 − z<sup>−1</sup>)<sup>L</sup> E. For a constant input u, the first-order modulator is an exact circle rotation with rotation number ρ = (1+u)/2 (the density of +1s). Its spectrum is a line spectrum at fold(kρ), and the rays for all k meet wherever ρ is rational. That structure is a Farey fan, and zooming toward an irrational such as 1/φ reveals its continued-fraction convergents (Fibonacci ratios) at every scale.

---

## 2. Gallery

### 2a. The sigma-delta idle-tone fan (extra piece, the strongest result)

<p><img src="gallery/sd_zoomseq_golden_deep_dark.png" width="100%"><br>
<em>Five-times zooms (×1 … ×3125) toward ρ = 1/φ. Labels are the rationals in each window with the smallest denominator. They are Fibonacci ratios 3/5, 5/8, 8/13 … 144/233 …, plus a few neighbours (e.g. 18/29). Panels ×1–×25 use N = 2<sup>14</sup> samples per input. Panels ×125–×3125 use N = 2<sup>20</sup>, which moves the resolution floor (~1/N ≈ 1e-6 in ρ) out of the window. Measured: output spectrum (dB). Declared: magma, [−52, −20] dB, γ 0.55, 4× max-pool in frequency.</em></p>

<p><img src="gallery/sd_zoomseq_golden_dark.png" width="100%"><br>
<em>The same zoom at N = 2<sup>14</sup> everywhere, kept as the honest "before". At ×125 and ×625 the window is narrower than the 1/N ≈ 6e-5 floor, so these panels show the finite-length kernel, not number theory.</em></p>

<table><tr>
<td width="33%"><img src="gallery/hero_sd_idle_zoom_riso.png"><br><em>Riso engraving (2 inks, 1-bit Floyd–Steinberg, 2 px misregistration): u ∈ [0.30, 0.42], ρ ∈ [0.65, 0.71]; the star is ρ = 2/3.</em></td>
<td width="33%"><img src="gallery/hero_sd_idle_zoom_dark.png"><br><em>Same data, magma on black.</em></td>
<td width="33%"><img src="gallery/hero_sd_idle_zoom_paper.png"><br><em>Same data, single ink on paper.</em></td>
</tr></table>

<table><tr>
<td width="50%"><img src="gallery/sd_zoomseq_twothirds_dark.png"><br><em>Zooming at the rational ρ = 2/3: the star keeps its shape until the finite-length floor turns it into a phase-slip kernel (last two panels).</em></td>
<td width="50%"><img src="gallery/sd_zoomseq_twothirds_order2_dark.png"><br><em>The same windows for the second-order modulator: mostly noise, with one star at 2/3. A second-order loop with a constant input is not a pure rotation.</em></td>
</tr><tr>
<td><img src="gallery/sd_idle_order1_riso.png"><br><em>Full range u ∈ [−1, 1], first order. Rays meet at rationals; low frequencies are darkened by 20 dB/decade noise shaping.</em></td>
<td><img src="gallery/sd_idle_order2_dark.png"><br><em>Full range, second order (|u| ≤ 0.6, stable). The black wedge at low frequency is 40 dB/decade shaping.</em></td>
</tr></table>

<p><img src="gallery/sd_zoom_golden.gif" width="45%"> &nbsp; <a href="gallery/sd_zoom_golden.mp4">MP4 (864², 24 fps)</a><br>
<em>Continuous exponential zoom toward 1/φ (half-width 0.38 → 5e-5, ×7600) with a p/q ticker. N = 2<sup>17</sup> per input (floor 7.6e-6).</em></p>

<p><img src="gallery/hero_sd_idle_zoom_spectral_variant.png" width="40%"><br>
<em>Stylistic variant: Sohl-Dickstein Spectral with rank-normalised dB (his `readout='probe_point'` mapping). dB is sequential, so the hue bands in the noise floor are colour-map artefacts; only the dark-red rays are tones.</em></p>

Paper and riso versions: `sd_zoomseq_*_paper.png`, `sd_idle_*_{dark,paper,riso}.png`, `sd_idle_zoom_*.png`. In the deep panels (×125–×3125) the frequency axis is cropped to f ∈ [0.372, 0.392]·f<sub>s</sub> around fold(1/φ); at full band, max-pooling hundreds of line-filled bins per column paints vertical stripes (first attempt, rejected).

### 2b. The lattice: bit-depth series

<table><tr>
<td width="34%"><img src="gallery/bitseries_column_magma.png"><br><em>2, 3, 4, 6, 8, 12 bits, undithered. Each plate has its own black point (3rd percentile, 70 dB span; declared).</em></td>
<td width="33%"><img src="gallery/bitseries_column_sonograph.png"><br><em>"Sona-graph" paper print, ink density = dB.</em></td>
<td width="33%"><img src="gallery/bitseries_column_shared.png"><br><em>One absolute scale [−115, −5] dB for all plates. This is the honest "fainter as bits increase" version.</em></td>
</tr></table>

<p><img src="gallery/bitseries_grid_riso.png" width="70%"><br>
<em>Riso grid, 8 bit depths plus the cure (4-bit + TPDF). Blue ink carries all dB values; pink carries only the loudest lines.</em></p>

<table><tr>
<td width="50%"><img src="gallery/hero_int3_native_magma_glow.png"><br><em>Hero, 3-bit, native STFT resolution (5610 × 2049). Black point at the 65th percentile of dB (declared), so the noise floor goes black.</em></td>
<td width="50%"><img src="gallery/hero_int3_logf_magma_glow.png"><br><em>Same data on a log-frequency axis (250 Hz–24 kHz). Unaliased harmonics k·f(t) become parallel copies of the fundamental; aliased ones become arches.</em></td>
</tr><tr>
<td><img src="gallery/hero_int3_native_magma_full.png"><br><em>Full dynamic range (3rd percentile floor), for honesty.</em></td>
<td><img src="gallery/hero_int3_native_sonograph.png"><br><em>Sonograph ink version. The 4-bit equivalents are `hero_int4_*`.</em></td>
</tr></table>

<p><img src="gallery/bitsweep.gif" width="60%"> &nbsp; <a href="gallery/bitsweep.mp4">MP4</a><br>
<em>Continuous bit depth: the step shrinks smoothly from peak = 1 LSB to 512 LSB (1.6 → 10 effective bits), on a fixed colour scale.</em></p>

<p><img src="gallery/key_int3_paper.png" width="100%"><br>
<em>Key plate. The measured 3-bit spectrogram (16–26 s) with the predicted fold(k f(t)) drawn for odd k ≤ 13. Every predicted curve lands on a measured line except k = 5, which is genuinely missing: at A = 3 LSB its Bessel sum nearly vanishes (−48 dB, vs −18 to −27 dB for its neighbours). Dark version: `key_int3_dark.png`.</em></p>

### 2c. Undithered vs RPDF vs TPDF

<p><img src="gallery/triptych_int3_dark.png" width="100%"><br>
<em>Row 1: spectrograms on one shared scale. Row 2: conditional mean and power of the error vs the input's position inside an LSB. Lines are exact (integration over the dither pdf); dots are measured on the 30 s chirp above. Row 3: noise modulation on a decaying 220 Hz tone at 6 bits. The RPDF error follows the signal down into the tail; the TPDF error stays flat.</em></p>

<table><tr>
<td width="50%"><img src="gallery/triptych_int3_paper.png"><br><em>Paper.</em></td>
<td width="50%"><img src="gallery/triptych_int3_riso.png"><br><em>Riso (blue halftone spectrogram, pink curves).</em></td>
</tr><tr>
<td><img src="gallery/triptych_int4_dark_quad.png"><br><em>4-bit, with subtractive RPDF as a fourth column.</em></td>
<td><img src="gallery/diptych_int3_undithered_tpdf_glow.png"><br><em>Artwork diptych: the lattice above, its dithered twin below, one shared dB scale, no axes. Sonograph version: `diptych_int3_undithered_tpdf_sonograph.png`.</em></td>
</tr></table>

### 2d. Floating-point grids

<table><tr>
<td width="34%"><img src="gallery/fpgrid_dark.png"><br><em>int4 / FP4 E2M1 / int8 / FP8 E4M3FN at 0, −12, −24 dB. Each plate is divided by its amplitude, so colour is error relative to the signal. Bottom: SINAD vs level.</em></td>
<td width="33%"><img src="gallery/fpgrid_paper.png"><br><em>Paper.</em></td>
<td width="33%"><img src="gallery/fpgrid_riso.png"><br><em>Riso.</em></td>
</tr></table>

### 2e. Harmonic amplitude maps (extra piece)

<table><tr>
<td width="50%"><img src="gallery/harmonics_int_dark.png"><br><em>|b<sub>k</sub>(A)| for a uniform grid, A = 0–24 LSB, odd k ≤ 301. The rays k ≈ 2πA and the Bessel interference are exact (FFT of the sampled period, M = 2<sup>16</sup>; matches the Bessel series to ~1e-5).</em></td>
<td width="50%"><img src="gallery/harmonics_fp8_dark.png"><br><em>FP8 E4M3, relative harmonics over 12 octaves of level: one octave, twelve times (octave-shift difference, median 3e-5 dB).</em></td>
</tr><tr>
<td><img src="gallery/harmonics_fp4_dark.png"><br><em>FP4: only 3 normal octaves, then everything rounds to zero.</em></td>
<td><img src="gallery/harmonics_int8log_dark.png"><br><em>Control: a uniform 255-level grid on the same log axis does not repeat.</em></td>
</tr></table>

Paper and riso versions: `harmonics_*_{paper,riso}.png`.

### 2f. DSD64 (extra)

<table><tr>
<td width="50%"><img src="gallery/sd_dsd64_order2_dark.png"><br><em>A 20 Hz–20 kHz chirp through a 1-bit second-order modulator at 3.072 MHz (the DSD64 rate), log-frequency 400 Hz–1.5 MHz. The audio band stays clean while noise rises toward 1.5 MHz.</em></td>
<td width="50%"><img src="gallery/sd_dsd64_order1_riso.png"><br><em>First order, riso. Paper/dark/riso versions of both are in the gallery.</em></td>
</tr></table>

### 2g. Audio (lossless FLAC; codes stored exactly; −6 dB)

| | |
|---|---|
| 2-bit undithered chirp | <audio controls src="gallery/audio/int2.flac"></audio> |
| 3-bit undithered | <audio controls src="gallery/audio/int3.flac"></audio> |
| 4-bit undithered | <audio controls src="gallery/audio/int4.flac"></audio> |
| 4-bit RPDF | <audio controls src="gallery/audio/int4_rpdf.flac"></audio> |
| 4-bit TPDF | <audio controls src="gallery/audio/int4_tpdf.flac"></audio> |
| 6-bit / 8-bit / 12-bit undithered | <audio controls src="gallery/audio/int6.flac"></audio> <audio controls src="gallery/audio/int8.flac"></audio> <audio controls src="gallery/audio/int12.flac"></audio> |
| FP4 E2M1 / FP8 E4M3 (24-bit FLAC, exact) | <audio controls src="gallery/audio/fp4_00dB.flac"></audio> <audio controls src="gallery/audio/fp8_00dB.flac"></audio> |
| decaying 220 Hz tone, 6-bit: undithered / RPDF / TPDF (listen to the tail: RPDF noise fades with the note, TPDF hiss stays) | <audio controls src="gallery/audio/decay_int6_undithered.flac"></audio> <audio controls src="gallery/audio/decay_int6_rpdf.flac"></audio> <audio controls src="gallery/audio/decay_int6_tpdf.flac"></audio> |
| video soundtrack (3-bit undithered, then TPDF) | <audio controls src="gallery/audio/video_int3_undithered_then_tpdf.flac"></audio> |

---

## 3. What was computed

Everything is exact float64 math in numpy on CPU (tiny C loops via ctypes for Floyd–Steinberg and long sigma-delta runs; both are checked bit-exact against the numpy reference). No GPU was used and no timing was measured.

- **Signal:** log chirp 20 Hz → 24 kHz (Nyquist), 30 s, f<sub>s</sub> = 48 kHz, peak 1.
- **int-b:** ML-style symmetric grid, Δ = peak/(2<sup>b−1</sup> − 1), 2<sup>b</sup> − 1 levels. round-half-even, **no clipping** (declared; dither can exceed the peak code by 1–2 codes).
- **FP4 E2M1 / FP8 E4M3FN:** all bit patterns enumerated; round-to-nearest, ties-to-even mantissa, saturating at ±max. The FP8 grid (253 values) and rounding were verified identical to `torch.float8_e4m3fn` on 10⁶ random values and on every tie point.
- **Dither:** RPDF U[−½, ½) LSB; TPDF = sum of two; numpy PCG64 with fixed seeds (2026, 99, 12345).
- **STFT:** 4096 Blackman-Harris (declared; it sets line sharpness and sidelobes), hop 256, dB re a full-scale sine's peak bin. For display, power is averaged into pixels (energy-honest). Max-pooling is used only in the sigma-delta maps, and is declared.
- **Sigma-delta:** error-feedback form, NTF (1 − z⁻¹)<sup>L</sup>. DC maps use 4096 inputs × 2<sup>14</sup> samples after 4096 warm-up. The golden zoom uses 2048 inputs per panel, N = 2<sup>14</sup> (×1–×25) and N = 2<sup>20</sup> (×125–×3125). The movie uses 1080 inputs × 2<sup>17</sup>.
- **Wall time:** about 1 min for all chirp spectrograms and audio, 30 s for the harmonic and DC maps, a few minutes for the deep zoom panels, ~25 min for the N = 2<sup>17</sup> movie, and ~10 min for all renders. GPU time: **0**.

```
cd hardware/dither
PY=/home/fzeng/ml/research/art/.venv/bin/python; export OMP_NUM_THREADS=4
$PY verify.py                  # every numerical claim below -> cache/verify.json, cache/moments.npz
$PY compute_spectrograms.py    # chirp spectrograms + FLAC
$PY compute_moments.py; $PY compute_fpcurves.py; $PY compute_bitsweep.py
$PY compute_extras.py A B1 B2  # harmonic maps, DC idle-tone maps, DSD64 (builds cache/sd.so)
$PY compute_sdzoom.py static movie_c; $PY compute_sdzoom_deep.py
$PY render_bitseries.py; $PY render_triptych.py 3; $PY render_triptych.py 4 dark --quad; $PY render_fpgrid.py
$PY render_key.py; $PY render_extras.py; $PY render_sdzoom.py; $PY render_bitsweep.py; $PY render_video.py
```

---

## 4. Verification (numbers from `cache/verify.json` and friends)

**Harmonics and aliasing (undithered).** 4-bit sine at 3001.3 Hz, N = 2<sup>16</sup>:
- The ±4-bin neighbourhoods of the first 201 folded harmonics hold **90.6%** of the error power while covering **5.5%** of bins. With TPDF the same bins hold 5.8%, i.e. what coverage predicts.
- The 40 strongest error peaks all match a folded odd harmonic (40/40); **37 of them aliased past Nyquist** (k up to 45).
- Odd harmonics exceed even ones by **53.5 dB** (half-wave symmetry).
- The same 40/40 match holds at 2, 3, 4, 6 and 8 bits. At 12 bits the test is impossible: ~13 000 significant harmonics fold into 32 768 bins. The lattice is denser than the frequency resolution, which is why the 12-bit plate looks like noise.

**Bessel series.** b<sub>k</sub> from the FFT matches Σ<sub>n≤20000</sub> 2(−1)<sup>n</sup>J<sub>k</sub>(2πnA)/(πn) to ≤ 3e-5 (A = 1.3, 3.7, 7.25; k = 1–21).

**Dither moments (exact integration, LSB units).**
- RPDF: mean exactly 0; power ranges **0 … 0.25**.
- TPDF: mean 0 (±1e-16); power **0.25 at every x**. The third moment (±0.048) and fourth moment (0.0625 … 0.25) still depend on x, as the paper says.
- Subtractive: mean 0; power 1/12. Monte Carlo KS test vs U[−½, ½] over 41 input positions gives a minimum p = 0.047 (expected minimum of 41 uniform p-values is about 0.024).
- Measured on the actual chirp: RPDF error power varies **0.020 … 0.249** with input position; TPDF **0.247 … 0.254**.

**Noise modulation.** On the decaying 6-bit tone, RMS error drops from 0.41 to 0.21 LSB into the tail with RPDF, stays at **0.50 → 0.50** with TPDF, and falls from 0.29 to 0.066 undithered (signal truncated to silence).

**Whiteness.** Spectral flatness of the error (4-bit sine) is 0.06 undithered, 0.979 RPDF and 0.980 TPDF. RPDF and TPDF look the same in any spectrogram: noise modulation is a time-domain effect (see row 3 of the triptych).

**FP grids.** FP8 SINAD stays at **32.9–33.2 dB** from 0 to −60 dB; int8 falls from 50 to 0 dB. Q(x/2<sup>j</sup>) = Q(x)/2<sup>j</sup> holds exactly for every sample in the normal range (j = 1–8). FP8 SINAD at a·2<sup>−j</sup> is identical to 3 decimals for j = 0–10. FP4 has 3 normal octaves and collapses below ~−20 dB.

**Sigma-delta.** Measured noise-floor slopes are **17.4 dB/decade** (order 1, theory 20) and **39.3** (order 2, theory 40); order 1 is contaminated by idle tones even with a median estimator. The strongest idle tone sits at fold(ρ) for **33/37** DC inputs; in the 4 misses a harmonic line is stronger than the fundamental. The C modulator is bit-exact against numpy.

**Self-similarity claims.**
- FP8 harmonic map: exact discrete scale invariance (one octave), measured to 3e-5 dB.
- Sigma-delta fan: number-theoretic self-similarity. Rays meet at every rational, and zooming at an irrational shows its convergents. This is **not** a claim of a fractal dimension, so no box counting was done; the set of rays is dense and the image has dimension 2.
- The zoom is limited by observation length: inputs whose ρ differ by less than ~1/N are indistinguishable, and this was checked by rerunning the deep panels at 64× the length.

## 5. Caveats

- The lattice depends on amplitude relative to Δ (see the k = 5 dropout) and on the window (declared).
- The no-clipping convention is a simplification. At 2–3 bits, a real converter with TPDF dither would clip at the rails.
- The MP4 soundtrack is AAC (lossy). Use the FLACs for listening to low-level noise.
- PNG files that would exceed the 20 MB commit cap are saved as 256-colour palette PNGs (colormaps are 256-entry LUTs anyway), and one is downscaled 0.85×: `diptych_int3_undithered_tpdf_sonograph.png`.
- "Quantization noise is white" appears in no caption here except for dithered cases, where it was measured.
- Sigma-delta error-feedback modulators are textbook forms, not a model of any specific DAC chip; SACD/DSD encoders use higher-order loops.

## 6. Ideas explored / not pursued

Brainstorm (extras):
1. **1-bit sigma-delta idle tones:** built. It became the strongest piece: Farey fan, golden zoom and DSD64.
2. **Error vs amplitude map:** built as the harmonic-amplitude maps (Bessel structure; exact octave repeat for FP8).
3. Noise-shaped vs flat dither (error-feedback requantizer with an F-weighted filter): not built. It is visually a tilted haze next to a flat haze, which the DSD64 plates already show more strikingly.
4. μ-law 8-bit (telephone companding) vs FP8: not built. It is the same story as FP8 (logarithmic grid) with a smooth rather than piecewise-linear compander.
5. "Listening to the error alone" (e(t) as audio): not built. It duplicates the harmonic audio.
6. Stochastic rounding in ML training as RPDF dither: kept as a caption and analysis point, not a separate image.
7. Limit cycles of the second-order loop as an (e1, e2) phase portrait: not pursued for time.

Negative results and surprises:
- The first attempt at a chirp-lattice mask test failed: too many harmonics (k ≤ 400) cover the whole time-frequency plane. The pure-tone peak identification replaced it.
- The first-order idle-tone map rendered with a Hann window and mean-pooling looked like noise. The lines are there, but their 1/k amplitudes spread into a dense line forest; Blackman-Harris plus max-pooling and 4096 rows fixed it.
- At 3 bits the 5th harmonic is absent (Bessel near-zero).

## References

- S. P. Lipshitz, R. A. Wannamaker, J. Vanderkooy, "Quantization and Dither: A Theoretical Survey," *J. Audio Eng. Soc.* 40(5), 355–375, May 1992. Checked: RPDF 1-LSB p-p makes the first moment input-independent; TPDF 2-LSB p-p makes the first and second moments independent with total error power Δ²/4 and is the unique minimum-power such dither; subtractive dither gives uniform, input-independent error.
- B. Widrow, I. Kollár, *Quantization Noise*, Cambridge UP, 2008.
- P. Micikevicius et al., "FP8 Formats for Deep Learning," arXiv:2209.05433 (E4M3FN convention).
- R. M. Gray, "Spectral analysis of quantization noise in a single-loop sigma-delta modulator with dc input," *IEEE Trans. Commun.* 37(6), 1989 (idle-tone line spectrum of the first-order loop).
- S. R. Norsworthy, R. Schreier, G. C. Temes (eds.), *Delta-Sigma Data Converters*, IEEE Press, 1997.
