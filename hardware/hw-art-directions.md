# Art from the Machine Itself — Project Directions

*Companion to `art/ml-art-directions.md` and `art/ml-art-fractals.md`. The subject here is not a model but the substrate: the number formats, kernels, memory, and silicon of the GB10 this all runs on.*

---

## 0. The governing constraint

The same rule as the ML-art docs: **every visual decision is either a faithful rendering of a measured or exactly computed quantity, or an explicit, declared aesthetic choice.**

Hardware adds two constraints of its own:

1. **Every measurement is a measurement of *this* machine.** Record the software stack in every caption: GB10 (Grace Blackwell, `sm_121a`), driver 580.173.02, CUDA 13.0, torch 2.14.0+cu130, dtype, and date. A performance map is a portrait of a specific chip + driver + library version, not of "GPUs". That specificity is the point, not a limitation.
2. **Contention ruins measurements.** `misc/burst/task.md` already documents a 62% throughput drop that turned out to be another job on the GPU. Timing-based pieces (§5, §6, §8, §9) must run on an **idle** machine: no art agents, no training. Log `nvidia-smi` before and after each run, and discard contaminated data. Pieces built from exact math (§1–§4) have no such constraint.

**Two families:**

- **Part I — exact math.** Number formats and quantization. Computed, not measured, so they are perfectly reproducible anywhere.
- **Part II — measured on the GB10.** Kernels, memory, power, sound. Unique to this box.

---

# Part I — Number formats and quantization (exact)

## 1. *Dither* — the spectrogram of a quantized chirp

### The phenomenon
Quantizing a signal to b bits is often modelled as adding white noise of power Δ²/12 (Δ = step size). For a deterministic signal like a sine, **that model is false**. The quantization error of a pure tone is itself periodic, so its energy concentrates in harmonics of the input frequency. In a sampled system, harmonics above the Nyquist frequency **alias**, folding back into the band. Sweep the input frequency (a chirp) and each harmonic traces a line that climbs, hits Nyquist, reflects, falls, hits zero, reflects again: a lattice of zig-zag lines.

**Dither** fixes this: add a small random signal before quantizing. Triangular-PDF (TPDF) dither of ±1 LSB makes the error's first two moments independent of the signal, and the harmonic lattice melts into a flat noise floor. That's the entire theory of dither (Lipshitz, Wannamaker & Vanderkooy, JAES 1992) as one before/after diptych.

### Why it's beautiful
A crosshatched, woven lattice of reflecting lines on a dark field, with finer and denser weave at lower bit depth, next to its dithered twin: an even grey haze. Order and its dissolution, and the order is the *error*. There's an audio companion too: listen to 4-bit undithered vs dithered sweeps.

### The piece
- **Bit-depth series:** spectrograms of the same log chirp quantized at 2, 3, 4, 6, 8, 12 bits. The lattice gets denser and fainter as bits increase. Print as a tall column of plates.
- **Diptych:** undithered vs TPDF-dithered at 4 bits.
- **Bonus:** the same chirp stored in FP4 (E2M1) and FP8, where the non-uniform step sizes make the harmonic lattice amplitude-dependent, unlike int quantization.
- **Sound:** WAV files of each, embedded in the README.

### How to compute it
numpy only: a chirp at 48 kHz, quantize, STFT with a long window (e.g. 8192, Blackman-Harris) for clean lines, log magnitude. Seconds of compute.

### Pitfalls
- Window choice changes line sharpness and sidelobe texture. That's a declared aesthetic choice; state it.
- Check the amplitude: a sine that doesn't span many quantization levels behaves differently from one that does. Plot the amplitude as a second axis variant.

---

## 2. *The Ruler* — representable numbers as a self-similar comb

### The phenomenon
A floating-point format represents numbers as sign × 2^exponent × mantissa. **Within one exponent** the representable values are evenly spaced; **each step up in exponent**, the same pattern repeats at twice the scale. So the set of floats is a ruler that is linear locally and logarithmic globally: a self-similar comb, with density ∝ 1/|x|.

The formats that matter for ML:

| Format | Bits (sign/exp/mantissa) | Notes |
|---|---|---|
| FP4 E2M1 | 1/2/1 | 16 bit patterns, 15 distinct values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} |
| FP8 E4M3 (fn) | 1/4/3 | max 448, no ∞ |
| FP8 E5M2 | 1/5/2 | IEEE-like, max 57344 |
| fp16 | 1/5/10 | |
| bf16 | 1/8/7 | fp32's range, coarse mantissa |
| int4 / int8 | — | uniform: the null model, a boring ruler |

### Why it's beautiful
An engraved slide rule. Tick marks thicken toward zero in a pattern that repeats at every octave. FP4 is almost comically sparse (15 ticks), E4M3 is a delicate comb, and bf16's coarse mantissa gives long even stretches with sudden density changes at each power of two.

### The piece
- **Format series:** one long horizontal ruler per format on a shared log axis, stacked as a type specimen sheet. Tick height encodes mantissa position, so the repeating octave pattern is visible.
- **Zoom:** an octave of E4M3 magnified to show the internal linear spacing, then the next octave, identical up to scale. A true self-similarity piece with an exact statement.
- **Plane version:** representable (x, y) pairs in FP8 as a 2D point lattice, a non-uniform grid that is dense near the axes and sparse far out. Riso or plotter.
- **Block-scaled formats:** show how a per-block scale *slides* the whole FP4 comb. NVFP4 uses 16-element blocks with an FP8 E4M3 scale per block (plus a per-tensor scale); MXFP4 uses 32-element blocks with a power-of-two E8M0 scale. The E8M0 scale can only move the comb by octaves; the E4M3 scale moves it continuously.

### How to compute it
Enumerate every bit pattern (≤ 2¹⁶ for fp16/bf16; trivially few for FP4/FP8) and decode. Exact.

---

## 3. *Posterize* — quantization error fields

### The phenomenon
Quantize a smooth 2D function f and plot the error Q(f) − f. For slowly varying f you get flat terraced bands (posterization). For a function whose frequency rises, like a zone plate cos(k(x² + y²)), the error aliases into **moiré**: phantom ring systems at positions set by the step size and sampling rate, with nested sub-systems at higher frequencies.

### Why it's beautiful
Moiré is the most visually intricate honest pattern there is, because it is an exact interference between two lattices: the sampling grid and the quantization levels.

### The piece
- **Zone plate** under int2/int3/int4/FP4/FP8 quantization, as an error field with a diverging perceptual colour map. Each format leaves a different fingerprint.
- **x·y product field** in FP8 E4M3: the rounding error of multiplication itself, laid over the plane. It shows the exponent octave structure as a rectilinear grid of changing texture.
- **Error diffusion:** Floyd–Steinberg dither of the same fields, printed as a 1-bit risograph. The halftone is the art, and it links to §1's dither story.

### How to compute it
numpy/torch on a 4096² grid. Exact.

---

## 4. *Bitplanes* and *Block Error* — real LLM weights under quantization

### The phenomenon
Real weight matrices are not uniformly distributed. They have outlier channels (whole rows or columns with large magnitude), heavy-tailed distributions, and structure from training. Block-scaled 4-bit formats (NVFP4, MXFP4) handle outliers by giving each block its own scale. Where a block contains an outlier, the scale is large and every other value in that block gets crushed onto few levels.

### Why it's beautiful
Rendered as an image, the per-element quantization error of a real matrix is a **woven texture**: the 16-element block grid is the weave, outlier channels are stripes, and the fine grain is the rounding of individual weights. It's a textile made by the format meeting the model.

### The pieces
- **Block Error:** take matrices from Qwen3-0.6B (local HF cache). Implement NVFP4 and MXFP4 quantization yourself and render |Q(W) − W| per element at full resolution, with zoomed crops. Also render the **per-block scale map**, a coarser mosaic that reveals the outlier structure.
- **Spectral lines:** the histogram of one block's values with its 15 FP4 levels drawn as spectral lines, repeated for many blocks as a stacked "spectrograph plate". Outlier blocks have widely spaced lines; ordinary blocks have tight ones.
- **Bitplanes:** decompose a bf16 weight matrix into its 16 bitplanes and render each as a 1-bit image in a 4×4 grid. The sign plane shows model structure, the exponent planes show smooth magnitude contours, and the low mantissa bits are pure noise. It's a gradient from signal to entropy, read top-left to bottom-right, and a very striking object.
- **Layer atlas:** the same analysis across all 28 layers, arranged as a grid. Attention vs MLP matrices should differ visibly.

### How to compute it
Minutes on CPU or GPU. No timing involved, so it can run alongside anything.

### Pitfalls
- NVFP4 and MXFP4 each have spec details: scale computation, rounding mode, the per-tensor second-level scale for NVFP4. Follow the published spec (NVIDIA's NVFP4 docs; OCP Microscaling Formats spec v1.0), and state any simplification.
- Error maps depend on rounding mode (nearest-even vs stochastic). Show both, or declare one.

---

## 5. *Divergence* — the same computation in different precisions

### The phenomenon
In a chaotic map, a small difference doubles roughly every step. For the logistic map at r = 4 the Lyapunov exponent is exactly ln 2, so an initial rounding error of ε grows as ε·2ⁿ. A float with m mantissa bits starts with relative error about 2⁻ᵐ, so its orbit should separate from the true orbit after roughly **m steps**: about 7 for bf16, 10 for fp16, 23 for fp32, 52 for fp64. Each extra mantissa bit buys one more step of truth.

### Why it's beautiful
A braid of orbits that runs as one strand, then splits into rivers at exactly predictable points. The exact prediction (steps ≈ mantissa bits) is the educational payload, and it takes one glance to check.

### The piece
- **The braid:** logistic-map orbits from the same seed in bf16, fp16, fp32, fp64, plus an arbitrary-precision reference (`mpmath`), overlaid as 5 inks. A vertical rule marks the predicted divergence step per format.
- **Precision floor of a fractal:** the Mandelbrot set (or the art project's trainability fractal) rendered at the same deep zoom in fp32 vs fp64. At depth, fp32 turns to pixelated blocks while fp64 still resolves. The failure is a visible texture with a precise cause.

### How to compute it
Trivial for the braid. The fractal version reuses `art/trainability-fractal/`.

### Pitfalls
- Verify m-step divergence empirically, because the prefactor depends on the seed. Report measured divergence steps vs predicted.
- On GPU, check whether torch actually computes in the dtype you think; fp16 ops can be upcast.

---

# Part II — Measured on the GB10

## 6. *Lattice* — the matmul performance map

### The phenomenon
`torch.matmul` dispatches to cuBLAS/cuBLASLt, which chooses among many kernels by heuristics over shape, dtype, and alignment. Throughput as a function of matrix shape is therefore **not smooth**:
- a fine grid at multiples of 8/16/64, where dimensions align with the hardware's preferred block sizes;
- sharp boundaries where the heuristic switches algorithms;
- cliffs where a matrix stops fitting in fast memory.

### Why it's beautiful
A 2D map with fine lattice texture overlaid by large mosaic cells, i.e. the look of the decoding map in `art/decode-map/` but produced by a hardware library's decision boundaries. It's a portrait of this chip, this driver, this library version, and no one has made it.

### The piece
- **Throughput map:** sweep (m, n) with k fixed, e.g. m, n ∈ [1, 2048] (4M shapes is too many; start at every integer in [1, 512] and a sparser sweep beyond). Colour by GFLOPS, in bf16, fp32, and fp16 as three plates.
- **Algorithm map:** colour by *which kernel ran*. It's detectable without profiler access through (a) bitwise output fingerprints (different algorithms round differently: hash C for fixed A, B) or (b) `torch.profiler` kernel names. That gives a categorical mosaic of dispatch regions.
- **Alignment close-up:** a zoom on [1, 128]² showing the multiple-of-8 grid.
- **Diptych across dtypes**, or across `torch.backends.cuda.matmul.allow_tf32` settings.

### How to compute it
- Reuse `day2/` benchmark hygiene and `misc/strassen/strassen_forensics.py`'s rounding-error fingerprinting, which already detects algorithm identity from rounding patterns.
- Per shape: warmup, then median of several timed runs with `torch.cuda.synchronize()`.
- Hours of wall time on an **idle** GPU.

### Pitfalls
- Timing noise and thermal drift. Randomize sweep order so drift doesn't paint fake gradients. Measure a reference shape periodically and normalize, or plot drift.
- The heuristic may also depend on workspace size and library version. Record both.
- Contention (§0) is fatal here.

---

## 7. *Fingerprint* — kernel nondeterminism

### The phenomenon
The same input can produce bitwise-different outputs depending on batch size, because the kernel chosen (and its reduction order) changes with shape. This is why LLM inference is nondeterministic even at temperature 0. The Thinking Machines Lab (2025, "Defeating Nondeterminism in LLM Inference") traced it to a lack of **batch invariance** in kernels like matmul, RMSNorm, and attention.

### The piece
- Run one fixed row through matmul / RMSNorm / attention as part of batches of size 1…512. Render a 2D bitmap: rows = batch size, columns = output element, colour = number of differing bits vs the batch-1 result (or the log |difference|).
- Expected look: bands where kernels switch, with speckle inside. Pair it with a Qwen3-0.6B run where one prompt decoded at different batch sizes produces diverging text, showing the token position where each diverges.

### Pitfalls
`art/decode-map/` is fighting exactly this. Share findings and don't duplicate work.

---

## 8. *Staircase* — the memory hierarchy

### The phenomenon
Random-access latency climbs a staircase as the working set exceeds each cache level. From `lscpu` on this machine, the Grace CPU has 10 Cortex-X925 + 10 Cortex-A725 cores; L1d 1.3 MiB total across 20 instances, L2 25 MiB across 20 instances, L3 24 MiB across 2 instances. The per-core-type sizes differ, so pinning the benchmark to an X925 core vs an A725 core should give **two different staircases**.

### The piece
- **Staircase plate:** latency vs working-set size (log-log) for random reads, one line per core type, with each cache level annotated by where the step lands vs where `lscpu` says it should.
- **Heartbeat:** latency of a fixed microbenchmark repeated 10⁶ times, as a long strip or folded raster (a row per second). Periodic OS interrupts, timer ticks, and throttling appear as regular texture.

### Pitfalls
Pin threads (`taskset`), disable frequency scaling if possible, and state the governor. Hugepages vs 4 KB pages change the TLB step.

---

## 9. *Roofline* — the machine's constellation

### The data
`day2/roofline.json` already holds measured ceilings: bf16 matmul ≈ 95.5 TFLOP/s measured (118.8 peak), memory-copy ≈ 222 GB/s measured (273 peak), ridge ≈ 429 FLOP/byte.

### The piece
A night-sky rendering of the roofline model. The memory-bandwidth roof and compute ceiling are drawn as a horizon line, and each benchmarked kernel (prefill, decode, attention variants, the linear-attention kernels from day1/day2) is a star at its measured (operational intensity, throughput) position. Label the constellation.

### Pitfalls
`misc/burst/task.md` flags that the roofline baseline is in doubt: the committed vs working-tree values differ by 62%, likely due to contention. **Re-measure on an idle GPU before making a print.**

---

## 10. *Pulse* — power and thermal rhythm

### The idea
Sample GPU power and temperature at the highest rate available during distinct workloads: idle, prefill, decode, matmul burst, the linear-attention kernels. Plot the traces and their spectrograms. During decode, the power trace may carry a line at the token rate.

### Honest uncertainty
NVML power readings on many GPUs are averaged over a window (often around 1 s). Check whether `nvidia-smi --query-gpu=power.draw.instant --loop-ms=...` gives real instantaneous samples on GB10. If the sensor bandwidth is below the token rate, the rhythm piece is impossible and only the slow thermal-soak piece (`misc/burst/` territory) remains. Test this first; it's a 10-minute check.

---

## 11. *Whine* — sonification from coil noise (needs a microphone)

### The idea
Power-delivery inductors vibrate audibly under changing load (coil whine), modulated by the workload. Record the machine during decoding with a phone held near the chassis and compute a spectrogram. If the whine is audible, the model's rhythm (token boundaries, possibly layer structure) may be visible as modulation lines: the sound of a transformer thinking.

### Status
Speculative; it depends on whether this unit whines at all. It's cheap to try: record 30 s idle vs 30 s decoding. If there's a visible difference, it's the most striking piece in this document. Pair the spectrogram with the software's own token timestamps to prove the correspondence rather than assume it.

---

## 12. *Heat* — thermal camera timelapse (needs a thermal camera)

A thermal timelapse of the chassis across workloads: idle, burst, sustained. It's the least ML-specific idea here, but visual, and it pairs with §10's sensor traces as ground truth.

---

## Media

| Medium | Fits | Notes |
|---|---|---|
| Large archival print | Lattice, Block Error, Bitplanes, zone-plate moiré | Fine texture needs resolution; view up close |
| Type specimen / engraved plate | The Ruler | Hairlines, letterpress idiom |
| Risograph 1-bit | Error diffusion, Bitplanes | 1-bit data in a 1-bit medium |
| Audio + spectrogram | Dither, Whine | The only pieces with sound |
| Pen plotter | Divergence braid, Staircase, Roofline | Line-native |
| Video | Precision-floor zoom, chirp spectrogram scrolling with sound | |

Colour rules are the same as the ML docs: perceptual maps for sequential data, a declared categorical palette for kernel-identity maps, never jet.

---

## Suggested sequencing

1. **Now, alongside the running art agents** (exact math, no timing): §1 *Dither*, §2 *The Ruler*, §4 *Block Error / Bitplanes*. §3 and §5 are also cheap.
2. **10-minute feasibility checks:** §10 power sensor bandwidth; §11 record 30 s of audio.
3. **Once the GPU is idle** (after the art wave finishes): §6 *Lattice* (strongest measured piece), §7 *Fingerprint*, §9 re-measured *Roofline*, §8 *Staircase*.

The top three: **Dither** (pure math, audio companion), **Lattice** (unique to this machine), and **Block Error / Bitplanes** (beautiful and directly relevant to low-bit inference work).

---

## Blind spots

1. **Quantization-noise models.** "Quantization error is white noise" is only approximately true with dither or busy signals. §1 exists to show where it's false. Don't caption any other piece as if it were true.
2. **A map of performance is a map of software.** Most of §6's structure comes from cuBLAS heuristics, not silicon. Caption it as "GB10 + cuBLAS version X", not "the GB10".
3. **Timing is fragile.** Contention, thermal state, and frequency scaling all paint fake structure. Randomize order, repeat, and show the noise.
4. **Formats have fine print.** NVFP4 vs MXFP4 scale encoding, FP8 E4M3 vs E4M3FN, rounding modes. Get them from the spec, not from memory.
5. **Sensor claims need correlation, not coincidence.** For §10/§11, align with software timestamps before claiming any rhythm corresponds to tokens or layers.

---

## References

- Lipshitz, Wannamaker & Vanderkooy, "Quantization and Dither: A Theoretical Survey", *J. Audio Eng. Soc.* 40(5), 1992
- Widrow & Kollár, *Quantization Noise*, Cambridge University Press, 2008
- Micikevicius et al., "FP8 Formats for Deep Learning", 2022 — arXiv:2209.05433
- OCP Microscaling Formats (MX) Specification v1.0, 2023
- Rouhani et al., "Microscaling Data Formats for Deep Learning", 2023 — arXiv:2310.10537
- NVIDIA NVFP4 documentation (Transformer Engine / TensorRT Model Optimizer)
- Thinking Machines Lab (He et al.), "Defeating Nondeterminism in LLM Inference", 2025
- Williams, Waterman & Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", *CACM* 52(4), 2009
- Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", *ACM Computing Surveys*, 1991
- Miller, "Computational complexity and numerical stability", *SIAM J. Comput.* 4(2), 1975 (cited in `misc/strassen/`)
- In-repo: `day2/roofline.json`, `day2/bench_throughput.py`, `misc/burst/`, `misc/strassen/strassen_forensics.py`
