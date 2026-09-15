# Art in Three Dimensions — Project Directions

*Companion to `ml-art-directions.md`, `ml-art-fractals.md` and `../hardware/hw-art-directions.md`. Nearly everything in the gallery so far is a plane. The exceptions are the tetrahedron films in `neural-collapse/`, the globes in `depth-roughness/` and one mplot3d preview in `loss-landscape/render_extra.py`. This document proposes eight pieces whose object is three-dimensional in its own right: a volume, a curve in space, or a surface in space.*

---

## 0. The governing constraint, in three dimensions

The rule is unchanged: **every visual decision is either a faithful rendering of a measured or exactly computed quantity, or an explicit, declared aesthetic choice.** Three dimensions add new ways to break it, so each piece here also follows these rules:

1. **No height maps of planes.** A 2D slice raised into a mountain is the rainbow surface plot that `ml-art-directions.md` §4 rules out. The third axis must be a real third coordinate of the phenomenon.
2. **Declare the chart.** Every 3D picture maps something into ℝ³: three hyperparameters, three weight-space directions, a stereographic projection of a 3-sphere, or a fitted subspace of a residual stream. Name the map and its distortion in the caption, and prefer coordinates that mean something over PCA of a high-dimensional cloud. When PCA is unavoidable, print the variance it captures.
3. **Light is not data.** Shading, shadows and ambient occlusion show form and nothing else. Colour and position carry data. Never let a viewer read a brightness gradient caused by the lighting as a gradient in the quantity.
4. **Occlusion hides the interior.** Every exterior hero ships with a cutaway or a slice atlas. Claims about the inside are made from slices, never from the outside view.
5. **Interpolation invents smoothness.** Trilinear interpolation and marching cubes make surfaces smoother than the grid that measured them. Categorical and fractal data are drawn at voxel resolution (nearest-neighbour, crisp faces), or the interpolation is declared and a 2× resolution check is shown.
6. **Transfer functions are colormaps.** In volume rendering, the map from value to opacity and colour is an aesthetic choice with the same power to invent structure as jet. Declare it, and render the null model (§11) through the same function.
7. **Slices must agree with the plane they came from.** Where a volume contains a plane that an existing piece already measured, the slice has to reproduce that plate. This is the cheapest correctness test available and every volume piece uses it.

---

# Part I — Volumes

## 1. *Solid Edge* — trainability as a body

### The phenomenon
`trainability-fractal/` colours the (η₀, η₁) plane of learning rates for a width-16 tanh network by whether full-batch GD converges in 500 steps. The boundary has box-counting D = 1.67 ± 0.04 in the 1024² float64 deep hero, and 1.36–1.68 across the zoom plates. Sohl-Dickstein (2024) notes that these fractals are "naturally defined in three or more dimensions" but only draws 2D slices. A search of his paper, blog, repository and both follow-ups found no volumetric rendering, so this appears to be open ground.

### Why it's beautiful
It is a Mandelbulb that nobody designed. The converged set becomes a solid with a crumpled skin, and cutting it open should show fjords and islands running through the interior.

### The piece
- **Space-time solid, free from cache.** `trainability-fractal/cache/windows/steps_zoomA2_384.npz` holds `measure_T` of shape (100, 384, 384): the convergence measure at T = 10, 20, …, 1000 over the 10¹ zoom window. Stacking T as the vertical axis gives a solid whose skin starts as a smooth sheet (D = 1.05 at T = 10) and frays as it rises (D = 1.41 by T = 250). The time axis is declared as time, not as a hyperparameter. This is the first object to build, because it tests the renderer on real data at zero compute cost.
- **Hyperparameter volume.** Axes are (log η₀, log η₁, log σ), where σ is the init scale. The σ × η plate (D = 1.21) showed a ragged edge at small σ and a smoother one at large σ, so the body's skin may change texture along one axis. A second candidate is a depth-2 width-16 net with axes (η₀, η₁, η₂). A 64³ toy of each decides which to scale up. The σ axis may turn out to be an uninformative extrusion (the σ × η plate found "the learning rate decides trainability almost alone"), and the toy exists to find that out cheaply.
- **Renders:**
  - an exterior hero in plaster-cast style: matte white, ambient occlusion, a single raking light (declared);
  - a Spectral-split volume in which the converged and diverged sides are rank-normalised separately, the boundary is left transparent, and speed is shown as coloured fog;
  - a cutaway plate that slices the body at σ = 1 to reveal the existing overview plate as a cut face;
  - a turntable film.
- **Object:** a resin print of the converged set at 128³, with the diverged set subtracted.

### How to compute it
Reuse `tfractal.py`'s batched step (each network is one row of a (P,16,16) tensor, chunks of 32,768, `torch.compile`), adding one axis to the grid.
- **Adaptive refinement:** compute 128³ in float32, which agrees with float64 on 99.80% of labels in the existing check. Then recompute in float64 only the voxels within one voxel of a label change, because float32 flips 29% of boundary pixels.
- **Audit:** recompute a random 1% of "interior" voxels in float64 and report their flip rate, since adaptive refinement can miss islands smaller than a cell.
- **Cost:** existing throughput under ~8-way GPU contention was 63 px/s (float64) and ~196 px/s (float32).
  - 64³ toy in float32: ≈ 25 min.
  - 128³ in float32 plus the float64 shell: ≈ 4–5 h.
  - 256³ over a zoom sub-volume: ≈ 13 h, and only if 128³ shows structure. Idle-GPU rates are unmeasured, so the toy measures them.

### Pitfalls
- **The slice reproduction** (§0.7) should agree with the 1024² overview everywhere except near the boundary. The existing 1-ulp perturbation test flips ≤ 0.2% of boundary labels, so that is the expected size of disagreement. Kernel choice can differ with chunk shape (`hardware/fingerprint/`), so compare labels, not bits.
- **The 3D dimension claim** needs 3D box counting (§11) on boundary voxels, plus a null quadratic loss with three learning rates, whose boundary should measure D ≈ 2.
- **The deflationary reading still applies.** "Complex Fractal Trainability Boundary Can Arise from Trivial Non-Convexity" (fractals doc §14) is about this exact object.

---

## 2. *Shells* — the loss landscape in three directions

### The phenomenon
Li et al. (2018) drew loss over two filter-normalised random directions. Their code supports only 1D curves and 2D slices; the 3D surfaces in the paper are those 2D slices rendered in ParaView. No three-direction volumetric loss landscape was found in the literature, and losslandscape.com also shows 2D planes. Take three directions instead of two, and the loss becomes a scalar field on a cube whose level sets are nested closed shells around the minimum.

### Honest expectation, stated up front
`loss-landscape/` measured all four models as smooth single valleys. That includes ResNet-56 without shortcuts, trained 40 epochs to 81.8% train accuracy, which does *not* reproduce Li et al.'s chaotic picture. The 3D version of those checkpoints will almost certainly be smooth, nearly ellipsoidal onions. The new content is therefore their **anisotropy**, their **evolution over training**, and the **training path threading through them**, not chaos. The crumpled version is optional §2b.

### The piece
- **Onion:** 6–8 nested translucent isosurfaces at declared loss levels, cut away along one octant. The shortcut and no-shortcut models are set side by side as two cast objects.
- **Tightening:** a film across the cached checkpoints (epochs 0–40). The shells contract and change shape while the network trains, and the camera stays fixed.
- **Thread:** the three directions are the top-3 PCA directions of the training trajectory, computed from the 17 cached checkpoints on CPU. The path of those checkpoints is drawn as a thin wire piercing the shells. This is honest, because the directions are defined by that path, and the caption says so.
- **§2b, optional:** train ResNet-56-noshort for Li et al.'s 300 epochs (≈ 7.5× the existing run), then check with a 2D slice first. Build the 3D version only if the 2D slice crumples.

### How to compute it
Everything in `loss-landscape/common.py` carries over: the directions, the 1000-image training subset, BN in eval mode, and fp32 with TF32 off. The cost is the point count.
- **Cost at existing contended rates:**
  - ResNet-20, 33³ = 35,937 points at ~4 pts/s: ≈ 2.5 h.
  - ResNet-56, 33³: ≈ 6 h.
  - Per-epoch film for ResNet-20 at 21³: ≈ 40 min per checkpoint.
- **Batching:** `torch.func.functional_call` + `vmap` over parameter sets would evaluate several grid points per pass. That engineering could plausibly buy several ×, but it is unmeasured.

### Pitfalls
The asymmetry from `ml-art-directions.md` §4 grows with the extra dimension. Crumpled shells prove non-convexity. Smooth shells in a random 3-slice of 855,770 dimensions prove very little, and the caption says exactly that. The "thread" directions are not random, so the thread plate and the random-direction onion are captioned as different objects.

---

## 3. *Crystal* — cuBLAS's kernel choices over (m, n, k)
*(A hardware piece, cross-listed from `../hardware/`.)*

### The phenomenon
`hardware/lattice/` timed `torch.mm` with m and n swept and k = 4096 fixed. It found:
- parity stripes (odd n uses `cutlass_75 align1`, even n `cutlass_80 align2`);
- an 8-lattice;
- tiles at periods 128 and 256;
- kernel-switch cells where timing jumps a median 15%.

But the shape space of a GEMM is really three-dimensional, and the heuristic keys on all three sizes. The fixed-k plane is one slice of a crystal.

### Prior work
Chatterjee (arXiv:2605.29752, 2026) sweeps all 32,768 (M, N, K) shapes in {128…4096}³, but on an Intel Arc B580 with sycl-tla, not cuBLAS. No published (m, n, k) volume for cuBLAS was found.

### The piece
- **Categorical crystal:** every integer (m, n, k) ∈ [1, 256]³ is coloured by which kernel ran, and each kernel's region is rendered as a solid of crisp voxel faces (§0.5).
  - An exploded view pulls the regions apart along their centroids (declared).
  - A cutaway exposes the parity stripes as stacked lamellae.
- **Throughput fog:** a coarser timing volume shown as isosurfaces of GFLOP/s inside the crystal.
- **Slice check:** the plane k = 4096 at m, n ∈ [1, 256] is also measured, and must reproduce the g256 fingerprints bit for bit. The lattice retest was 100% identical.

### How to compute it
- **Kernel identity by output fingerprint:** hash C for fixed A and B, which the lattice piece found deterministic. Map fingerprint classes to kernel names with a profiler pass on a sample of shapes. The dense 16.7M-shape cube costs one call plus a hash per shape. Measure that rate on a 10⁴-shape toy before committing; at 1 ms per shape it would take ≈ 4.6 h.
- **Timing volume:** 64³ = 262,144 shapes on a step-4 grid, using lattice's hygiene (random order, median of 5 windows ≥ 1.5 ms, reference shape every 256). At the measured ~8 ms per shape, that is ≈ 35 min per dtype.

### Pitfalls
- **Timing needs an idle machine:** no art jobs and no `autonomous/` jobs (hw doc §0). Kernel identity is deterministic, but check on a sample that selection does not change under load before running it on a shared GPU.
- **Coincident rounding:** two kernels that happen to round identically would merge into one fingerprint class. The profiler sample bounds how often that happens.
- **Caption:** "GB10 + cuBLAS in torch 2.14.0+cu130", never "the GPU".

---

# Part II — Curves in space

## 4. *Ribbon* — the edge of stability in its own three directions

### The phenomenon
At the edge of stability, full-batch GD flips back and forth along the top Hessian eigenvector while drifting slowly underneath. `edge-of-stability/` drew this as a 2D braid. Cohen et al. (ICLR 2021, arXiv:2103.00065) and the central-flow paper (Cohen, Damian, Talwalkar, Kolter & Lee, ICLR 2025, arXiv:2410.24206) have no 3D trajectory figures. The latter says outright that "visualizing three-dimensional dynamics is difficult".

### Why it's beautiful
In three dimensions the two strands of even and odd steps become two sheets, and the path zig-zags between them while the sheets themselves drift and bend. It is a ribbon being stitched.

### The piece
- **Coordinates:**
  - axis 1 is ⟨θ_t − θ_ref, u₁⟩ for a fixed reference top eigenvector;
  - axes 2–3 are the top-2 PCs of the 21-step-smoothed drift, orthogonalised against u₁.
- **Marks:** even and odd steps are drawn as two tubes, joined by a translucent ribbon. Colour is the measured λ₁·η/2, which hovers at 1.
- **Canyon:** a loss volume on a 32³ grid over the same three directions around θ_ref. The ribbon is shown bouncing inside the valley it is stabilised by.
- **Film:** the camera follows the central flow.

### How to compute it
- **Stage A, from cache:** `main4.npz` holds a 1024-bin count-sketch of θ_t − θ₀ at every step, plus the sketch of u₁. A 3D version in sketch space costs nothing. It is declared as a sketch: a random linear projection that approximately preserves inner products.
- **Stage B, rerun:** one net at 2/η = 80 for 6000 steps, estimated at ≤ 2 h. The cache holds no full trajectory, which is why this is needed. Save:
  - projections onto a bank of fixed reference eigenvectors (top-3 every 250 steps, ≈ 190 MB);
  - θ every 10 steps (≈ 1.6 GB).
- **Stage C, canyon:** 32³ loss evaluations of the 3072-200-200-10 MLP on 5000 images, ≈ 20 min.

### Pitfalls
- **u₁ rotates during training.** The naive projection onto a moving u₁(t) is the one the existing README keeps as an "honesty" diptych. Use piecewise-fixed reference frames, aligned between windows by sign and Procrustes, and declare the alignment.
- **The threshold is 2/η for full-batch GD only.**

---

## 5. *Invariant Tori* — learning dynamics on the energy 3-sphere

### The phenomenon
Two players learning rock–paper–scissors by replicator dynamics, with tie payoffs ε_x = −ε_y = ε (Sato, Akiyama & Farmer, PNAS 99:4748, 2002), form a Hamiltonian system on a 4D state space. As `game-chaos/` measured:
- at ε = 0 the system is integrable and every orbit lies on a torus;
- at ε = 0.5 and H = 2.8, 29% of orbits are chaotic, a sea surrounding KAM islands.

SAF show Poincaré sections, not 3D orbits, and `game-chaos/` README item 8 lists a "3-D torus sculpture" as not pursued.

### The chart, which makes this piece exact rather than a projection
The energy H = −⅓Σ log x_i − ⅓Σ log y_i is conserved, so each orbit lives on a 3-dimensional level set.
- In logit coordinates, −⅓Σ log softmax(z)_i = logsumexp(z) − mean(z), which is strictly convex on the zero-mean subspace. H is therefore strictly convex on ℝ⁴, with its minimum 2 log 3 at the uniform strategy.
- Each level set H = h > 2 log 3 is the boundary of a convex body, so radial projection maps it homeomorphically onto S³.
- Stereographic projection then maps S³ minus one point onto ℝ³.

The result is a declared, conformal, one-to-one chart, and nothing is lost. The only distortion is the scale blow-up near the chosen pole. Choose the pole at a point no rendered orbit comes near, and check that numerically.

### Why it's beautiful
This is the classical picture of nested invariant tori, but made by two learners.
- **At ε = 0:** the tori should be smoothly nested doughnuts with orbits winding around them.
- **As ε grows:** tori should dissolve into fog while islands survive inside. (Measured in M1: the innermost torus goes chaotic first, at ε ≈ 0.04, not the outer ones.)

### The piece
- **Nested tori (ε = 0):** 8–12 orbits at increasing energy-surface radius, each a long thin copper tube whose winding draws its own torus.
- **Sea and islands (ε = 0.5, H = 2.8):**
  - a chaotic orbit's visit density binned into a 256³ volume and rendered as fog;
  - regular orbits drawn as tubes inside it;
  - the existing Poincaré section x_P − x_R + y_P − y_R = 0 drawn as a translucent membrane carrying its section dots, linking the new piece to the existing plates.
- **Film:** ε from 0 to 0.5 at fixed H, with the tori breaking up.
- **Object:** STL of one KAM torus, and a two-torus interlock if one exists in the data.

### How to compute it
CPU only. Use the existing C integrator (RK4 in logits, h = 0.01, H conserved to 3·10⁻⁹ over T = 10⁵), `compute_poincare.py` (about 1 min per 400 orbits) and the cached `traj.npz`. The one new step is the chart.

### Pitfalls
- **Stereographic scale varies by orders of magnitude.** Apparent thickness is not comparable across the image, so say so, and add a plate with a second pole choice.
- **The 29% figure is a finite-time Lyapunov classification** with threshold 5·10⁻³. The fog and tube split inherits that threshold.

---

## 6. *Number Knot* — the number line inside a language model

### The phenomenon
Kantamneni & Tegmark ("Language Models Use Trigonometry to Do Addition", arXiv:2502.00873, ICLR 2025 workshop) found that GPT-J, Pythia-6.9B and Llama 3.1-8B represent the integers 0–99 on a generalised helix: a linear term plus circles of periods T = 2, 5, 10, 100. They fit it by regressing the top-100 PCs of the residual stream onto [a, cos 2πa/T, sin 2πa/T]. Engels et al. (arXiv:2405.14860, ICLR 2025) found circles for days and months. No such study on Qwen or on any sub-1B model other than GPT-2-small was found.

### Model choice, checked locally
- **Qwen3-0.6B splits every number into single digits**, so it cannot host a per-number helix at all.
- **OLMo-2-0425-1B** (in the local HF cache) tokenises every integer 0–999 as a single token without a leading space (0 of 1000 multi-token, checked 2026-09-15). The number pieces use OLMo-2.
- Qwen3-0.6B does have single tokens for " Monday" and " January", so it hosts the days and months.
- Štefánik et al. (arXiv:2510.26285) was reported by our literature check to find sinusoidal number embeddings including OLMo 2; the arXiv abstract does not mention OLMo 2 (checked in number-knot M3), so treat that detail as unverified.

### The piece
- **Helix:** for the strongest period found, the 1000 numbers are measured activations projected onto that fitted (cos, sin) plane, plus the fitted linear direction. Beads are coloured by last digit. The fit chooses only the basis; every bead position is measured.
- **Knot:** the T = 100 plane is the large circle and the T = 10 plane the small circle, composed into a torus so the number line winds 10 times around it, i.e. a (1, 10) torus knot. The torus embedding is a *declared composition*; both angles and both radii are measured.
- **Depth tower:** the day-of-week and month circles in Qwen3-0.6B, and the number circle in OLMo-2, measured at every layer and stacked with layer as height (declared). The viewer watches the circle condense out of noise across depth.

### How to compute it
A few thousand forward passes and linear regressions: minutes of GPU time. Before claiming any period, Fourier-analyse the activations over a (Zhou et al., NeurIPS 2024, arXiv:2406.03445). Fit on 0–999 but report 0–99 separately, for comparability with K&T.

### Pitfalls
- **Null model:** run the same fit on shuffled number labels and on random single tokens, and report R² against those nulls. A regression onto sines will find *something* in any 100-dimensional cloud.
- **OLMo-2-1B may simply not have a clean helix.** That is a publishable negative for a small model, drawn as a messy knot next to its null.
- **Engels et al. found their circles by SAE clustering; ours are supervised.** Say which method was used.

---

# Part III — Surfaces and flows in space

## 7. *Rough Skin* — depth as the dial, one dimension up

### The phenomenon
Di Lillo, Marinucci, Salvi & Vigogna (arXiv:2504.06250) prove that infinite-width Gaussian networks on S^d with an activation of regularity index β ∈ (0, 1) have level sets of Hausdorff dimension d − β^L. Their proofs cover general d ≥ 2, though their experiments are on S² only. `depth-roughness/` measured 2 − 2^−L on S² for Heaviside; for example, L = 2 gave 1.618 against 1.619 for an exact-dimension field. That implies β = ½ for Heaviside. β = ½ is our reading, not stated in the paper. On S³ the prediction is **3 − 2^−L**: 2.5, 2.75 and 2.875 for L = 1, 2, 3. The zero set is a surface that grows toward filling space with every layer.

### The piece
- **The dial as a film:** L = 1 → 4 on one random draw, the zero set rendered as an isosurface with the measured 3D box dimension at each integer depth. ReLU (D = 2) runs in the other half of the frame.
- **Two casts:** STL prints of L = 1 Heaviside and its ReLU twin on the same noise: rough against smooth in the hand. L ≥ 2 is likely too space-filling to print or even read from outside, so it is shown as a slice film.
- **Cutaway atlas:** 12 parallel cuts through the patch, each a 2D coastline. This ties the piece back to the existing S² plates.

### How to compute it
- **Chart:** a cube patch of side ≈ 0.5 rad on S³, via the exponential map at a base point. Tangential stretch is at most ≈ 3% at the corners (declared).
- **Exact finite nets:** Heaviside networks at width n = 4096 on a 256³ grid, spacing 2·10⁻³ rad, above the ≈ 4/n = 10⁻³ rad scale below which finite nets stop being rough (`depth-roughness/` finding). Estimated ≈ 1 h in float32 at the roofline's measured ~100 TFLOP/s; the sign threshold makes float32 adequate, which should be checked on a slab.
- **Zoom sub-volumes:** Gaussian fields with the kernel's local power law, sampled by 3D FFT on a periodic box, the 3D analogue of the existing flat-sky octaves. This is a declared approximation, and it is checked against the exact nets where the two overlap.
- **Marching cubes:** for STL export, implement it or install `scikit-image` under the pip lock.

### Pitfalls
- **Calibrate the estimator.** The 2D box-counting estimator was biased by about 0.01–0.02 (`depth-roughness/` table), so calibrate the 3D estimator on synthetic fields of known dimension before quoting numbers.
- **Beauty risk:** a D = 2.75 surface may render as undifferentiated foam. If so, the dial is carried by slices and by L = 1, and the README says why.

---

## 8. *Combed* — flow-matching trajectories and memorisation

### The phenomenon
A flow-matching model transports Gaussian noise to data along an ODE. Two facts pull against each other:
- **The exactly optimal velocity field for a finite training set** is known in closed form (Bertrand, Gagneux, Massias & Emonet, arXiv:2506.03719) and only reproduces training points. The same holds for diffusion: Gu et al., "On Memorization in Diffusion Models", TMLR 2025, arXiv:2310.02664; Scarvelis, Borde & Solomon, "Closed-Form Diffusion Models", arXiv:2310.12395.
- **Trained networks generalise once the data set is large enough.** Kadkhodaie, Guth, Simoncelli & Mallat (ICLR 2024, arXiv:2310.02557) saw the switch at N ≈ 10³–10⁴ on 80×80 faces.

In ℝ³ the whole flow can be seen at once.

### The piece
- **Data:** N points sampled on a trefoil knot. The shape is a declared aesthetic choice; any curve works, and a knot makes generalisation along the curve visible.
- **Hair:** 20k ODE trajectories from fixed noise seeds, drawn as hairline tubes coloured by time, with endpoints glowing.
- **Memorising versus trained:** the closed-form field combs the hair into N tight tufts. A trained MLP at the same N either reproduces the tufts or spreads the ends along the knot.
- **The film is N itself:** N ∈ {16, 64, 256, 1024, 4096}, same seeds and same camera, with the tufts melting into a continuous knot. Plotted beside it, the measured fraction of memorised samples, using a nearest-neighbour distance-ratio criterion like Gu et al.'s; check their exact threshold.
- **Companion:** a basin volume that colours each starting-noise voxel by the training point it reaches under the closed-form flow. `diffusion-basins/` found the deterministic samplers' basins smooth, so no fractal claim is made; it is the 3D Voronoi-like shape of memorisation.

### How to compute it
Tiny MLPs and closed-form fields in ℝ³. Minutes of GPU time, and CPU would also do.

### Pitfalls
- **The ODE solver's step count changes the hair.** Use an adaptive solver or fixed small steps, and show a 2×-steps check.
- **Memorisation onset depends on network size as well as N.** Fix width, or sweep it as a second axis, but do not attribute to N what is due to capacity.

---

# 9. The renderer — one shared tool

The virtual environment contains matplotlib and nothing 3D (no pyvista, plotly, vispy, moderngl or scikit-image). Matplotlib's 3D mode has no real depth sorting or lighting. The proposal is a small torch renderer on the GB10 at `art/_shared/r3d/`, shared by all eight pieces, following the `_shared` palette precedent:

| Module | Does | Used by |
|---|---|---|
| `camera` | orthographic (plates) and perspective (films), turntable and follow paths, stereo pairs | all |
| `volume` | emission–absorption ray marching over a grid via 3D `grid_sample`, declared transfer functions, clip planes | 1, 2, 3, 5 |
| `iso` | isosurface ray marching with bisection refinement, gradient normals, shadow rays and ambient occlusion; nearest-neighbour **voxel mode** for categorical and fractal data | 1, 2, 3, 7 |
| `tubes` | curves as tubes and ribbons (ray–capsule or depth-buffered splats), additive glow for night styles; hidden-line SVG export for plotter styles, extending `neural-collapse/render_tetra.py`'s depth sort | 4, 5, 6, 8 |
| `mesh` | marching cubes, binary STL (reusing `render_tetra.export_stl` and `loss-landscape/render_extra.write_stl`), decimation | 1, 5, 7 |

**Renderer verification before any piece uses it:**
- analytic sphere and torus grids against closed-form silhouettes and normals;
- a smooth null volume (§11) through each transfer function;
- a voxel-mode render of a known checkerboard, to confirm nothing is interpolated.

**Styles**, extending the shared brief:
- **dark ground** with perceptual maps and glow;
- **plaster cast:** matte white with ambient occlusion, after the 19th-century mathematical model collections, which fits §1, §5 and §7;
- **Spectral split** for two-sided volumes, with the seam left transparent;
- **plotter hidden-line** for curves;
- **red–cyan anaglyph**, a stereo image that is also a two-ink riso (declared).

---

# 10. Media

| Medium | Fits | Notes |
|---|---|---|
| Turntable film | all | The native medium for 3D on a README page; GIF fallback as elsewhere in the gallery |
| Cutaway plate or slice atlas | 1, 2, 3, 7 | Required companion (§0.4); the bridge back to the existing 2D pieces |
| Resin or plaster 3D print | 1, 5, 7 | Rough against smooth (§7), a KAM torus (§5), the trainability body (§1). STL files in `gallery/` |
| Stereo pair or anaglyph | 4, 5, 8 | Cheap and delightful; curves read best in stereo |
| Interactive HTML viewer | any | Optional and last: GitHub READMEs cannot embed it, and vendoring three.js is extra weight |

---

# 11. Verification protocol for volumes

This extends fractals doc §11, all of which still applies.

1. **3D box counting.** Count boundary voxels, where a label or sign changes between 6-neighbours, in cubes of side ε, across at least two decades. The fitted slope is the surface dimension, in [2, 3].
2. **Slice consistency.** For a set of dimension D > 1 in ℝ³, the Marstrand–Mattila slicing theorem says that for almost every plane orientation, a positive-measure set of offsets gives sections of dimension D − 1, and none exceed it. Compare the 3D D with 1 + the D of **random oblique** slices. This is an expectation for typical slices, not a guarantee; axis-aligned slices may be atypical, so do not use them for this test.
3. **Plate reproduction** (§0.7): §1 at σ = 1, §3 at k = 4096, §7 against `depth-roughness/`.
4. **Resolution doubling** on a sub-volume. Real structure refines; interpolation artefacts change.
5. **Null volumes through the identical render path:** a three-learning-rate quadratic (§1), an analytic ellipsoidal onion (§2), a ReLU field (§7), the ε = 0 integrable tori (§5, which should show no fog), and shuffled labels (§6). If the null looks intricate, the renderer is the fractal.

---

# 12. Cost and scheduling

| # | Piece | GPU estimate | Needs idle GPU | Reuses |
|---|---|---|---|---|
| 1 | Solid Edge | free space-time solid; 64³ toy ≈ 25 min; 128³ ≈ 4–5 h; 256³ zoom ≈ 13 h (optional) | no | `tfractal.py`, `steps_zoomA2_384.npz` |
| 2 | Shells | ResNet-20 33³ ≈ 2.5 h; ResNet-56 33³ ≈ 6 h; film ≈ 40 min per checkpoint | no | `loss-landscape/cache/ckpt` |
| 3 | Crystal | fingerprint cube: measure (≈ 4.6 h at 1 ms/shape); timing 64³ ≈ 35 min per dtype | **yes**, for timing | `lattice/sweep.py`, g256 fingerprints |
| 4 | Ribbon | free from cache; rerun ≤ 2 h; canyon ≈ 20 min | no | `main4.npz` |
| 5 | Invariant Tori | CPU minutes | — | C integrator, `traj.npz` |
| 6 | Number Knot | minutes | no | HF cache |
| 7 | Rough Skin | ≈ 1 h exact nets; FFT fields minutes | no | `depth-roughness/` kernels |
| 8 | Combed | minutes | no | — |

- **Estimates use rates measured under ~8-way contention** in the existing READMEs, plus the roofline's idle ~100 TFLOP/s. Each toy run measures the real rate before scaling.
- **Rendering also uses the GPU.** A 1080² turntable of 480 frames through `volume` is expected to take tens of minutes, and runs in the same queue.
- **One GPU job at a time.** Two concurrent sweeps OOM'd the unified memory pool on 2026-09-14. Chain every stage in one detached script (`setsid nohup`, the `hardware/run_part2.sh` pattern) with per-chunk checkpoints. Use this chain, not the 8-slot `art/_shared/gpu_run.sh`.
- **The GPU is shared with `autonomous/` until 2026-09-29.** Its charter gives Fan Pu priority and pauses via `GPU_PAUSE`. §3's timing volume needs the autonomous work paused.

# 13. Sequencing

1. **Renderer plus free data (no new compute):**
   - build `r3d` and pass its analytic tests;
   - render §1's space-time solid, §4 Stage A from sketches, and §5's tori.
   These three exercise all four renderer modules on real measurements.
2. **Cheap new compute:** §8 *Combed*, §6 *Number Knot* and §7 at L = 1–2. All are minutes to an hour, and each is a complete piece.
3. **Flagship:** §1's 64³ toys for both axis sets, then the chosen 128³ volume.
4. **Heavy and conditional:** §2 at ResNet-20 first; §1 at 256³ only if 128³ shows structure; §4 Stage B if Stage A is too sketchy to believe.
5. **Idle-machine window:** §3 *Crystal*.

**The top three:** §1 *Solid Edge* (open ground, the strongest image), §5 *Invariant Tori* (an exact chart with no projection loss, and CPU only) and §7 *Rough Skin* (a theorem you can hold).

---

# 14. Blind spots

1. **A volume is still a slice.** Three hyperparameters out of dozens, or three weight directions out of 10⁵–10⁶. Everything in `ml-art-directions.md` blind spot 2 and fractals §14.2 applies unchanged.
2. **3D invites over-reading.** Viewers read a lit surface as a solid object with an inside and an outside. For §1 that reading is literal (converged set against diverged set). For §2 it is not, because the shells are level sets of a continuous function and the "surface" is our chosen level. Say which in each caption.
3. **Stereographic and exponential charts are honest but not isometric.** Size in §5 and §7 is not comparable across the frame.
4. **Fitted subspaces can manufacture circles.** §6 needs its nulls to be printed, not just run.
5. **On the art side:**
   - *Invariant tori and nested surfaces* have a visual history in mathematical visualisation, e.g. Banchoff's 4D projections, the Hopf fibration pictures and the Brill/Schilling plaster models.
   - *Glowing particle hair* has a large generative-art tradition.
   - *Escape route:* as elsewhere in this project, a real subject, a declared chart, and a slice plate that proves it.

---

## Core references

- Sohl-Dickstein, *The boundary of neural network trainability is fractal*, 2024 — arXiv:2402.06184
- Li, Xu, Taylor, Studer & Goldstein, *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018 — arXiv:1712.09913
- Chatterjee, *From Roofline to Ruggedness: Decomposing and Smoothing the GEMM Performance Landscape*, 2026 — arXiv:2605.29752 (Intel Arc, sycl-tla)
- Cohen, Kaur, Li, Kolter & Talwalkar, *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*, ICLR 2021 — arXiv:2103.00065
- Cohen, Damian, Talwalkar, Kolter & Lee, *Understanding Optimization in Deep Learning with Central Flows*, ICLR 2025 — arXiv:2410.24206
- Sato, Akiyama & Farmer, *Chaos in learning a simple two-person game*, PNAS 99(7):4748–4751, 2002
- Galla & Farmer, *Complex dynamics in learning complicated games*, PNAS 110(4):1232–1236, 2013 — arXiv:1109.4250
- Kantamneni & Tegmark, *Language Models Use Trigonometry to Do Addition*, 2025 — arXiv:2502.00873
- Engels, Michaud, Liao, Gurnee & Tegmark, *Not All Language Model Features Are One-Dimensionally Linear*, ICLR 2025 — arXiv:2405.14860
- Zhou, Fu, Sharan & Jia, *Pre-trained Large Language Models Use Fourier Features to Compute Addition*, NeurIPS 2024 — arXiv:2406.03445
- Levy & Geva, *Language Models Encode Numbers Using Digit Representations in Base 10*, 2024 — arXiv:2410.11781 (a template for digit-tokenised models like Qwen3)
- Štefánik et al., 2025 — arXiv:2510.26285 (sinusoidal number embeddings; OLMo 2 coverage unverified from the abstract)
- Di Lillo, Marinucci, Salvi & Vigogna, *Fractal and Regular Geometry of Deep Neural Networks*, 2025 — arXiv:2504.06250
- Bertrand, Gagneux, Massias & Emonet, *On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity*, 2025 — arXiv:2506.03719
- Kadkhodaie, Guth, Simoncelli & Mallat, *Generalization in diffusion models arises from geometry-adaptive harmonic representations*, ICLR 2024 — arXiv:2310.02557
- Gu, Du, Pang, Li, Lin & Wang, *On Memorization in Diffusion Models*, TMLR 2025 — arXiv:2310.02664
- Scarvelis, Borde & Solomon, *Closed-Form Diffusion Models*, TMLR 2025 — arXiv:2310.12395
- In-repo: `trainability-fractal/`, `loss-landscape/`, `../hardware/lattice/`, `edge-of-stability/`, `game-chaos/`, `depth-roughness/`, `neural-collapse/render_tetra.py`, `../hardware/roofline/`
