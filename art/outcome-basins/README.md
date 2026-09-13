# Not Whether, But Which

*Each pixel is a separate training run, coloured by **which** solution gradient descent reached, not just by whether it converged. When the step size is large, the borders between solutions fold into fractals.*

<img src="gallery/f3_s0_e0.3_wide_newton.png" width="49%"> <img src="gallery/f3_s0.5_e1.1_tassel_spectral.png" width="49%">

<img src="gallery/xor_s4_e1.2_spectral.png" width="49%"> <img src="gallery/zoom_zX_spectral.png" width="49%">

<video src="gallery/f3_zoomfilm_newton.mp4" autoplay loop muted playsinline width="49%"></video> <video src="gallery/f3_etafilm_newton.mp4" autoplay loop muted playsinline width="49%"></video>

<sub>Top left: *Four Roots*, 4096² runs of GD on ¼(xyz−1)², η = 0.3. Top right: the fringe of the same system at η = 1.1 (window 0.875 wide), in the Sohl-Dickstein Spectral split. Bottom left: a 1.3·10⁸× zoom into a boundary between solutions. Middle row: *Modulo Permutation*, the XOR 2-2-1 net at η = 1.2 in the Spectral split, and a 7776× zoom into its fan (D ≈ 1.79). Bottom right: the *Four Roots* slice as η rises from 0.02 to 1.32. GIF versions: [zoom](gallery/f3_zoomfilm_newton.gif), [η](gallery/f3_etafilm_newton.gif).</sub>

---

## 1. The phenomenon

Published trainability fractals, starting with Sohl-Dickstein (2024), colour hyperparameter space by one bit: did training converge or diverge? That is a Mandelbrot question. This project asks a Newton-fractal question instead: out of a small, countable set of solutions, **which one** did training land on? A 2D slice through initialisation space gets one tiny training run per pixel, and the pixel is coloured by the identity of the solution it reached.

**The iterated map** is plain full-batch gradient descent, θ ← θ − η∇L(θ). There are two systems, each with a discrete, enumerable solution set.

**A. *Four Roots*: depth-3 scalar factorisation.**
L(x,y,z) = ¼(xyz − 1)². The zero set xyz = 1 has four connected components, one for each sign pattern with product +1: **+++**, **+−−**, **−+−**, **−−+**. The slice is the plane x + y + z = √3·s, with in-plane axes u = (1,−1,0)/√2 and v = (1,1,−2)/√6. Permuting (x,y,z) acts on this plane as the dihedral group D₃, which is why the pictures have three-fold symmetry. The three "one positive" classes are rotated copies of one another, and +++ sits on the symmetry axis.

At a solution the Hessian is ½ggᵀ with g = (yz, xz, xy), so its sharpness is λ = ½(y²z² + x²z² + x²y²) ≥ 3/2. A minimum can be a stable fixed point of GD only when λ < 2/η. So no minimum is stable once η > 4/3, and for η near 1 only a narrow band of each solution component can capture trajectories (edge of stability).

**B. *Modulo Permutation*: XOR with a 2-2-1 tanh network.**
The network is ŷ = Σₖ aₖ tanh(wₖ·x + bₖ) + c, with 9 parameters. It is trained with MSE on the four XOR points (±1, ±1) with targets x₁x₂ ∈ {±1}. A run's *raw identity* is the ordered pair of boolean functions its two hidden units compute on the four inputs (4-bit sign codes of the pre-activations). Two symmetries leave the network function unchanged: permuting the hidden units, and flipping the sign of a tanh unit, (w, b, a) → (−w, −b, −a). Quotienting by both gives the *canonical identity*. This is what Git Re-Basin weight matching does; with two units, matching to a reference with signs reduces exactly to "sign-normalise each unit, then sort". Every converged run in our slices has one of **16 raw identities**, and these collapse to exactly **2 canonical solutions**. Canonical {1,7} uses units ≈ {¬x₁∧¬x₂, ¬(x₁∧x₂)}, i.e. AND/OR up to sign. Canonical {2,4} uses units ≈ {¬x₁∧x₂, x₁∧¬x₂}.

**Why η matters.** For small η, GD approximates gradient flow. Gradient flow is a smooth, invertible flow, so its basin boundaries are smooth: they are stable manifolds of saddles. (For the factorisation, gradient flow conserves x² − y², so the boundaries are literally straight lines in the slice.) Fractal filigree needs a map that **folds**: a discrete, non-invertible step taken near or beyond the edge of stability, where a single step can throw nearby initialisations to very different places. The η series and η film below show the change from straight borders to fractal fringes.

## 2. Hero images

<table>
<tr><td width="50%"><img src="gallery/f3_s0.5_e1.1_newton.png"></td><td width="50%"><img src="gallery/f3_s0.5_e1.1_tassel_newton.png"></td></tr>
<tr><td><sub><b>f3_s0.5_e1.1_newton</b>. <i>Four Roots</i>, s = 0.5, η = 1.1, window ±3.5, 4096². Hue = solution (gold +++, crimson +−−, green −+−, blue −−+). Brightness = log₁₀ steps to loss < 10⁻¹². Near-black = diverged, with a faint glow for slow escape. The dark inner lobes are real: they converge slowly because they land on minima with λ ≈ 2/η.</sub></td>
<td><sub><b>f3_s0.5_e1.1_tassel_newton</b>. The lower-left fringe (window 0.875 wide, 4096²). Tongues of all four solutions interleave along the divergence edge.</sub></td></tr>
<tr><td><img src="gallery/triptych_f3_s0.5_e1.1.png"></td><td><img src="gallery/zoom_zA_ink.png"></td></tr>
<tr><td><sub><b>triptych_f3_s0.5_e1.1</b>. The quotient story for the factorisation. Left: 4 classes. Middle: modulo permutations of (x,y,z), 2 classes. Right: modulo all symmetries (permutations and paired sign flips) there is one solution, so only converge/diverge is left, rendered in the Spectral split.</sub></td>
<td><sub><b>zoom_zA_ink</b>. Survey sheet of a nested zoom, 8× per level, 1.3·10⁸× in total. Each centre is picked automatically as the most boundary-dense point near the middle of the previous level. Heavy line = boundary between two solutions; light line = converge/diverge edge. Below ×500 the boundary is a statistically self-similar stack of stripes, locally a Cantor set × a line.</sub></td></tr>
</table>

## 3. Gallery

### A. *Four Roots* (¼(xyz − 1)²)

Every plate below comes from one cached 4096² map. The styles differ only in how the three measured fields (solution label, status, and smoothed convergence or escape time) are mapped to colour.

| | newton | ink | riso | dark time | Spectral |
|---|---|---|---|---|---|
| s = 0, η = 0.3, ±5 | <img src="gallery/f3_s0_e0.3_wide_newton.png" width="150"> | <img src="gallery/f3_s0_e0.3_wide_ink.png" width="150"> | <img src="gallery/f3_s0_e0.3_wide_riso.png" width="150"> | <img src="gallery/f3_s0_e0.3_wide_darktime.png" width="150"> | <img src="gallery/f3_s0_e0.3_wide_spectral.png" width="150"> |
| s = 0, η = 0.3, ±3.5 | <img src="gallery/f3_s0_e0.3_newton.png" width="150"> | <img src="gallery/f3_s0_e0.3_ink.png" width="150"> | <img src="gallery/f3_s0_e0.3_riso.png" width="150"> | <img src="gallery/f3_s0_e0.3_darktime.png" width="150"> | <img src="gallery/f3_s0_e0.3_spectral.png" width="150"> |
| s = 0.5, η = 1.1, ±3.5 | <img src="gallery/f3_s0.5_e1.1_newton.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_ink.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_riso.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_darktime.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_spectral.png" width="150"> |
| η = 1.1, fringe (0.875) | <img src="gallery/f3_s0.5_e1.1_tassel_newton.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_tassel_ink.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_tassel_riso.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_tassel_darktime.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_tassel_spectral.png" width="150"> |
| η = 1.1, fringe ×64 (0.109) | <img src="gallery/f3_s0.5_e1.1_tasselzoom_newton.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_tasselzoom_ink.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_tasselzoom_riso.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_tasselzoom_darktime.png" width="150"> | <img src="gallery/f3_s0.5_e1.1_tasselzoom_spectral.png" width="150"> |
| s = 0, η = 1.25, ±3.5 | <img src="gallery/f3_s0_e1.25_newton.png" width="150"> | <img src="gallery/f3_s0_e1.25_ink.png" width="150"> | <img src="gallery/f3_s0_e1.25_riso.png" width="150"> | <img src="gallery/f3_s0_e1.25_darktime.png" width="150"> | <img src="gallery/f3_s0_e1.25_spectral.png" width="150"> |
| s = 1.5, η = 0.8, ±3.5 | <img src="gallery/f3_s1.5_e0.8_newton.png" width="150"> | <img src="gallery/f3_s1.5_e0.8_ink.png" width="150"> | <img src="gallery/f3_s1.5_e0.8_riso.png" width="150"> | <img src="gallery/f3_s1.5_e0.8_darktime.png" width="150"> | <img src="gallery/f3_s1.5_e0.8_spectral.png" width="150"> |
| s = 1.5, η = 1.2, ±3.5 | <img src="gallery/f3_s1.5_e1.2_newton.png" width="150"> | <img src="gallery/f3_s1.5_e1.2_ink.png" width="150"> | <img src="gallery/f3_s1.5_e1.2_riso.png" width="150"> | <img src="gallery/f3_s1.5_e1.2_darktime.png" width="150"> | <img src="gallery/f3_s1.5_e1.2_spectral.png" width="150"> |

**Extra boundary-split palettes** (declared variants of the Spectral split from `art/color-research`, same ranks): `hubble_sho` (teal | ochre) and `aurora_ember` (green | ember).

| | hubble_sho | aurora_ember |
|---|---|---|
| η = 1.1, ±3.5 | <img src="gallery/f3_s0.5_e1.1_split_hubble_sho.png" width="200"> | <img src="gallery/f3_s0.5_e1.1_split_aurora_ember.png" width="200"> |
| fringe (0.875) | <img src="gallery/f3_s0.5_e1.1_tassel_split_hubble_sho.png" width="200"> | <img src="gallery/f3_s0.5_e1.1_tassel_split_aurora_ember.png" width="200"> |
| fringe ×64 (0.109) | <img src="gallery/f3_s0.5_e1.1_tasselzoom_split_hubble_sho.png" width="200"> | <img src="gallery/f3_s0.5_e1.1_tasselzoom_split_aurora_ember.png" width="200"> |

What each style measures, and what is a declared choice:

- **newton**. Measured: solution label and convergence time. Declared: a categorical palette, gold for +++ plus three Klimt jewel tones for the S₃-rotated classes (colour-research `klimt_categorical`); brightness = 1 − 0.7·q, with q the log₁₀ convergence time clipped to its 0.5–99.5 percentile range; diverged pixels get #141217 plus a 10% glow from their smoothed escape time. Convergence time is smoothed by log-linear interpolation of the loss across the step where it crosses 10⁻¹². Faint contour lines inside basins are level sets of the integer step count.
- **ink**. Measured: boundaries only. Declared: one ink (#1c1b1a) on paper (#f3eee2). A 3 px line marks a border between two different solutions; a 2 px line (at 55% ink) marks a converge/diverge edge.
- **riso**. Measured: label and convergence time. Declared: three stencil.wiki riso inks (fluo pink, yellow, medium blue). +++ = yellow, +−− = pink, −+− = blue, −−+ = pink over blue. A stochastic screen sets density from 95% (fast) to 35% (slow). The drums are misregistered by (0,0), (3,−2) and (−2,3) px. Diverged pixels are bare paper. Rendered from every other sample (2048²).
- **dark time**. Measured: convergence time only, so solution identity is invisible except where slow seams mark the borders. Declared: `cmc.lajolla`, histogram-equalised over converged pixels; diverged = black.
- **Spectral** (Sohl-Dickstein split, the user's favourite). Measured: status and time. Declared: converged pixels are ranked by convergence time and mapped purple (slowest, at the boundary) → pale yellow; diverged pixels are ranked by escape time and mapped deep red (slowest) → pale yellow; the palest 25% of each half is left unused, as in his colab (`palettes.render_split(..., 'sd_spectral', near_boundary='large')`). This style shows *whether*, not *which*. It is the right-hand panel of the triptych.

**η series and films**

<img src="gallery/eta_series_f3.png" width="100%">

<sub><b>eta_series_f3</b>. Same slice (s = 0.5, ±3.5, 2048² each) at η = 0.005 … 1.3. The label ηλ_min is η times the smallest sharpness any solution has (3/2); a stable minimum needs a value below 2. At η = 0.005 the borders between solutions are straight lines (gradient-flow regime). As η grows, a folded fringe appears wherever the local curvature along the trajectory makes ηλ ≳ 2; it moves inward, and at η ≥ 1.2 the dark slow zones of edge-of-stability convergence take over the petals.</sub>

| film | MP4 (1080², silent) | GIF |
|---|---|---|
| η from 0.02 to 1.32, 240 computed frames, newton | [f3_etafilm_newton.mp4](gallery/f3_etafilm_newton.mp4) | [gif](gallery/f3_etafilm_newton.gif) |
| η film, Spectral split (per-frame ranks) | [f3_etafilm_spectral.mp4](gallery/f3_etafilm_spectral.mp4) | [gif](gallery/f3_etafilm_spectral.gif) |
| zoom ×1.3·10⁸ into a solution boundary, η = 1.1, 420 computed frames, newton | [f3_zoomfilm_newton.mp4](gallery/f3_zoomfilm_newton.mp4) | [gif](gallery/f3_zoomfilm_newton.gif) |
| same zoom, Spectral split | [f3_zoomfilm_spectral.mp4](gallery/f3_zoomfilm_spectral.mp4) | [gif](gallery/f3_zoomfilm_spectral.gif) |

<sub>Every film frame is a fresh 1080² computation; nothing is interpolated between frames. Newton films use one fixed brightness range for all frames (log₁₀ steps 1.0–3.3), so the brightness does not flicker. In the zoom film the window centre drifts from the origin toward the target as (width/width₀)², so the zoom starts on the whole figure and ends centred on the boundary point. Zoom sheets in the other styles: [zoom_zA_newton](gallery/zoom_zA_newton.png), [zoom_zA_spectral](gallery/zoom_zA_spectral.png).</sub>

### B. *Modulo Permutation* (XOR 2-2-1)

<img src="gallery/diptych_xor_s4_e1.2.png" width="100%">

<sub><b>diptych_xor_s4_e1.2</b>. One random 2-plane through the 9-dimensional initialisation space of a 2-2-1 tanh net, ±3, 2048² runs at η = 1.2. Left: the 16 raw solution identities (which boolean function each hidden unit computes, in order, with sign), in a declared 16-colour palette: warm family = the AND/OR-type canonical solution, cool family = the x₁∧¬x₂-type one; legend shows each identity's share of converged runs. Right: the same runs modulo hidden-unit permutation and tanh sign flip (2 canonical solutions, vermilion and blue). Grey = still on a plateau at T = 20 000 (11%), near-black = diverged (15%). Brightness = log convergence time.</sub>

| newton (raw) | newton (canonical) | Spectral split | riso | ink | dark time |
|---|---|---|---|---|---|
| <img src="gallery/xor_s4_e1.2_raw_newton.png" width="130"> | <img src="gallery/xor_s4_e1.2_canon_newton.png" width="130"> | <img src="gallery/xor_s4_e1.2_spectral.png" width="130"> | <img src="gallery/xor_s4_e1.2_riso.png" width="130"> | <img src="gallery/xor_s4_e1.2_ink.png" width="130"> | <img src="gallery/xor_s4_e1.2_darktime.png" width="130"> |

- **newton raw / canonical**: measured label + convergence time; declared palettes as in the diptych.
- **Spectral**: measured status and time only (converged ranked purple→yellow, diverged ranked red→yellow, plateau runs = dark plum on the seam). The fan on the left, where diverging and slowly-converging runs interleave, is the most intricate texture in the whole project.
- **riso**: canonical {1,7} → fluo pink, canonical {2,4} → medium blue; a raw identity with an odd number of tanh sign flips relative to its canonical representative adds the yellow drum (pink+yellow reads orange, blue+yellow green); misregistered as in A.
- **ink**: heavy line = a boundary that survives quotienting by permutation and sign (between canonical solutions, or a converge/plateau/diverge edge); hairline at 60% ink = a boundary between raw identities that disappears modulo symmetry.
- **dark time**: convergence time only (`cmc.lajolla`), identity invisible.

**η series and films**

<img src="gallery/eta_series_xor_raw.png" width="100%">

<sub><b>eta_series_xor_raw</b> (also [canonical](gallery/eta_series_xor_canon.png)). Same plane at η = 0.3 … 1.25 (1024², hero at 2048² for η = 1.2). At η ≤ 0.8 the borders are smooth curves; the fan of fine interleaved diverge/plateau/solution filaments appears near η = 1.0 and thickens until at η = 1.25 only 56% of runs converge. The panel titles give the largest sharpness among converged runs vs 2/η.</sub>

| film | MP4 (1080², silent) | GIF |
|---|---|---|
| η from 0.6 to 1.26, 72 computed frames at 480² (shown 2× nearest-neighbour on a dark mat, declared), raw identities | [xor_etafilm_raw.mp4](gallery/xor_etafilm_raw.mp4) | [gif](gallery/xor_etafilm_raw.gif) |
| same, canonical solutions | [xor_etafilm_canon.mp4](gallery/xor_etafilm_canon.mp4) | [gif](gallery/xor_etafilm_canon.gif) |
| same, Spectral split (per-frame ranks) | [xor_etafilm_spectral.mp4](gallery/xor_etafilm_spectral.mp4) | [gif](gallery/xor_etafilm_spectral.gif) |

<video src="gallery/xor_etafilm_spectral.mp4" autoplay loop muted playsinline width="49%"></video> <video src="gallery/xor_etafilm_raw.mp4" autoplay loop muted playsinline width="49%"></video>

<sub>Films play at 8 frames per second; brightness range for the newton films is fixed across frames (log₁₀ steps 1.6–4.1, pooled 0.5–99.5 percentiles).</sub>

**Zoom into the fan**

<img src="gallery/zoom_zX_spectral.png" width="100%">

<sub><b>zoom_zX_spectral</b> (also [raw](gallery/zoom_zX_raw.png), [canonical](gallery/zoom_zX_canon.png)). Six nested levels, 6× each (width 1 → 1.3·10⁻⁴, 7776× in total), 1024² runs per level, η = 1.2. Level 0 is centred on the most boundary-dense point of the hero map; later centres are picked automatically by boundary density. Spectral split, rank-normalised per level (declared). At every level the window holds all 18 outcome labels (16 raw solutions + plateau + diverged) and 37–48% of pixels sit on a boundary: the fan is a self-similar stack of streaks, and it does not smooth out as we zoom. The share of diverged runs grows with depth (45% → 78%) because the zoom centre drifts deeper into the divergent side.</sub>

## 4. What was computed

| | A. Four Roots | B. XOR |
|---|---|---|
| model | L = ¼(xyz−1)², 3 parameters | 2-2-1 tanh MLP with linear output, 9 parameters, MSE on 4 XOR points |
| slice | plane x+y+z = √3·s (s ∈ {0, 0.5, 1.5}), axes u, v as above | θ₀ + αu + βv, θ₀ ~ N(0, I₉), u, v random orthonormal (numpy seed 4) |
| optimiser | full-batch GD, constant η | full-batch GD, constant η |
| steps | up to T = 20 000 | up to T = 20 000 |
| converged | loss < 10⁻¹² and still < 10⁻¹² 64 steps later | same |
| diverged | max\|θ\| > 10⁴ or non-finite | same |
| outcome | sign class of (x,y,z) | ordered pair of hidden-unit sign codes; canonical = sorted pair of min(c, 15−c) |
| grids | heroes 4096²; η series 2048²; zoom levels 2048²; resolution check 1025/2049/4097; films 1080² | hero 2048²; η series 1024²; fan zoom 1024² (6 levels ×6); film 72 frames at 480² (2× nearest-neighbour on a 1080² mat, declared) |
| precision | float64 end to end (raw CUDA kernels, one GPU thread per run) | float64 |

The engine, `cuda_gd.py`, compiles one CUDA kernel per problem with `torch.utils.cpp_extension.load_inline`. Each GPU thread runs one pixel's entire training run, stops early when that run finishes, and records a smoothed first-passage time. On a GPU shared with about 10 other jobs, this was 20–30× faster than a fused torch.compile step loop (`engine.py`, kept for reference and for the batched Jacobian/sharpness). The two engines agreed on 100% of labels in a check (`scratch/test_cuda.py`).

**Commands** (`G=/home/fzeng/ml/research/art/_shared/gpu_run.sh`, `PY=/home/fzeng/ml/research/art/.venv/bin/python`, run in this directory):

```bash
$G ./prod_fact3.sh      # heroes, η series, null, zoom levels, resolution check, uncertainty, both A films
$G ./prod_fact3b.sh     # extra A plates (wide, fringe, fringe zoom, s=1.5 η=1.2)
$G ./prod_fact3c.sh     # 10^6-sample uncertainty tail (riddling test)
$G ./prod_xor.sh        # B: hero + uncertainty raw/canon (first 3 lines were used; later lines superseded by:)
$G ./prod_xor2.sh; $G ./prod_xor_zoom.sh; for p in 0 1 2; do $G ./prod_xor_film.sh $p & done   # B null, η series, fan zoom, film
$PY analyze.py fact3; $PY analyze.py xor                        # verification JSON + plates
$PY render_fact3.py heroes heroes2 triptych eta zoom; $PY render_xor.py hero eta zoom
$PY render_films.py f3_etafilm newton 20; $PY render_films.py f3_etafilm spectral 20
$PY render_films.py f3_zoomfilm newton 24; $PY render_films.py f3_zoomfilm spectral 24
$PY render_fact3.py splits; $PY render_xor.py eta zoom
for s in raw canon spectral; do $PY render_films.py xor_etafilm $s 8; done
```

Wall time on the shared GB10 (8 GPU slots, about 95% utilised by other projects the whole time): A production 53 min in one slot (the two films took 36 min of that), A follow-ups 4 min, B hero 29 min, B uncertainty exponents 11 + 11 + 6.5 min, B η series 17 min, B η film 31 min (3 slots × ~11 min), B fan zoom 36 min, A riddling tail 2.4 min (total GPU slot time over the project ≈ 3.9 h). Rendering is CPU only (OMP_NUM_THREADS=4) and took about 15 min. Caches of raw arrays are in `cache/` (gitignored).

## 5. Verification and honesty (fractals doc §11)

<img src="gallery/verify_fact3.png" width="100%">

**A. Four Roots, η = 1.1, s = 0.5**

| test | result |
|---|---|
| box counting on the nested zoom (10 levels, 8× each; window width 7 → 5.2·10⁻⁸) | per-level fits over 1–128 px (2.1 decades each): D = 1.408, 1.409, 1.431, 1.384, 1.366, 1.376, 1.406, 1.394, 1.403, 1.404 (±0.01 each). **Mean 1.40 ± 0.02**, steady across 8 decades of zoom. The boxes span 2.5·10⁻¹¹ to 0.44 slice units overall. |
| split by boundary type (same levels) | borders **between two solutions**: D = 1.37 ± 0.02. Converge/diverge edge: D = 1.26 ± 0.02. So the "which" boundary is *more* intricate than the "whether" boundary. |
| uncertainty exponent (Grebogi et al.), M = 10⁵ random inits, ε from 10⁻¹ to 10⁻¹³ of window width | full window: f(ε) ∝ ε^0.57 over ε ∈ [3·10⁻⁷, 10⁻¹] → **D = 2 − α = 1.43**, agreeing with box counting. Fringe window (0.875): α = 0.28 → **D = 1.72** over 11 decades. |
| riddled / intermingled basins? | **No.** f(ε) keeps falling all the way to ε = 10⁻¹³ in both windows; riddling would make it flat (α ≈ 0). The fringe window does show a shallower tail (α ≈ 0.2 for ε < 10⁻⁵), so the boundary dimension there approaches 1.8: the basins are close to intermingled but not riddled. A 10⁶-sample follow-up (`prod_fact3c.sh`, ε = 10⁻⁴ … 10⁻¹⁴) confirms it: in the full window f falls from 6.1·10⁻³ to 1·10⁻⁵ (α = 0.28 over the whole tail, D = 1.72); in the fringe window f falls from 5.5·10⁻³ to 3·10⁻⁵ (α = 0.21, D = 1.79), a clean power law with no flattening. So below ε ≈ 10⁻⁶ even the full-window estimate is dominated by the fringe, which is why it drops from 0.57 to ≈ 0.25. The lowest points rest on only 10–30 uncertain samples (±30%). |
| resolution check (zoom level 3 window at 1025², 2049², 4097²; coarse samples land exactly on fine ones) | co-located samples agree bit for bit (100%). Box counts agree at matched physical ε, with D = 1.41 / 1.38 / 1.36 at the three resolutions. The fraction of new samples that disagree with their coarse neighbours falls from 2.19% to 1.30% per doubling. New detail keeps appearing and nothing aliases. |
| null model (same pipeline, η = 0.005, gradient-flow regime) | zoom levels D = 1.10, 1.04, 1.04, 1.04, 1.04. At depth, each 2048² level has exactly 4096 boundary pixels (a straight line). Uncertainty exponent α = 1.04 → **D = 0.96**. The pipeline does not make smooth boundaries look fractal; its bias is about +0.04 from 4-neighbour boundary thickness. |
| precision floor | float64. The deepest zoom has pixel spacing 2.5·10⁻¹¹ at coordinates of size ≈ 1.8, so rounding (≈ 4·10⁻¹⁶) sits five orders of magnitude below a pixel. The uncertainty curve stays a power law down to ε·W ≈ 10⁻¹³, which puts the floor near ε ≈ 10⁻¹⁴. We stopped the zoom at 5·10⁻⁸. |
| edge-of-stability selection | in every η-series map, the largest sharpness among converged runs is ≤ 2/η: 2.856 vs 2.857 (η = 0.7), 1.9995 vs 2 (η = 1), 1.666 vs 1.667 (η = 1.2), 1.538 vs 1.538 (η = 1.3). One pixel in 4·10⁶ at η = 1.1 exceeds it by 0.15%, a hold-window transient. Measured without being designed for: in this slice the share of +++ among converged runs rises from 16% (η = 0.005) to 54% (η = 1.3). |

**B. XOR**

<img src="gallery/verify_xor.png" width="100%">

| test | result |
|---|---|
| box counting on the nested fan zoom (6 levels, 6× each, width 1 → 1.3·10⁻⁴; 1024² per level) | D = 1.82, 1.83, 1.79, 1.76, 1.77, 1.77. **Mean 1.79 ± 0.03**, steady across 3.9 decades of zoom. |
| hero map (2048², ±3), quotient analysis | raw, all boundaries D = 1.67 ± 0.01; between raw solutions only D = 1.26 ± 0.03. Canonical, all boundaries D = 1.68; between canonical solutions only D = 1.14 ± 0.04. Canonicalisation removes **58%** of between-solution boundary pixels (16 → 2 classes). What survives is the fan: solve vs plateau vs diverge. |
| uncertainty exponent, M = 2·10⁴, ε = 10⁻¹ … 10⁻¹¹ of window width | raw labels α = 0.22 → **D = 1.78**; canonical α = 0.21 → **D = 1.79**. Agrees with the zoom box counts. f(ε) falls from 0.64 to 0.0033 without flattening, so **not riddled** down to 10⁻¹¹, but close to intermingled. |
| null model (same pipeline, η = 0.3) | uncertainty α = 0.92 → **D = 1.08**: smooth borders, as expected for a near-gradient-flow step size. The η series shows no fan at η ≤ 0.8. |
| resolution check | **not done for XOR** (XOR runs are ~10× costlier than A because plateau runs go to T = 20 000). The agreement of box counting (pixel-scale) and the uncertainty exponent (to 10⁻¹¹, far below a pixel) is our substitute. With ~40% boundary pixels, box counts at 1024² are near saturation; treat D ≈ 1.8 as ±0.05. |
| edge-of-stability selection | max sharpness among converged runs vs 2/η: 6.665 vs 6.667 (η = 0.3), 2.008 vs 2.000 (η = 1), 1.818 vs 1.818 (η = 1.1), 1.740 vs 1.739 (η = 1.15), 1.683 vs 1.667 (η = 1.2), 1.600 vs 1.600 (η = 1.25). The few runs above the line are ≤ 4 per 10⁶ (η = 0.8: 4 runs, max 2.568 vs 2.5), consistent with hold-window transients at loss < 10⁻¹². |
| precision | float64 throughout; the deepest XOR zoom pixel is 1.3·10⁻⁷, far above rounding. |

**What did not work / negative results**

- Colouring the 2-factor problem ¼(xy−1)² by branch gives only two colours, and inside the convergent region the branch boundary is the straight line x = −y. This is the published Zhu et al. object, and the sister project `gd-bifurcation` already maps it. We dropped it for the depth-3 version, which has four solutions.
- For XOR, η ≥ 1.3 leaves no converged runs at all in our slices, because every solution has λ > 2/η. The interesting band is narrow (η ≈ 1.0–1.25).
- A "stall detector" that marks plateau runs as stuck early saved no time and changed 1.8% of XOR labels, so it is off (`stall=0`).
- The first zoom-centre heuristic (most distinct labels) drifted onto an isolated single border by level 5. Picking by boundary density fixed it; the centres are listed in `prod_fact3.sh`.
- Uncertainty-exponent and box-counting estimates differ by window. The fringe window gives 1.72 by uncertainty exponent, while the stripe zoom gives 1.40 by box counting. The boundary is not uniformly self-similar, so there is no single dimension for "the" boundary, and we report per window.

## 6. Caveats

- **This is not specific to neural networks.** Liu (2024, arXiv:2406.13971) shows fractal trainability boundaries arising from trivially non-convex functions, and Part A is a 3-parameter polynomial. The honest caption is: *this is what iterating a folding map near an instability looks like, and large-step training is such a map*. Part B shows the same thing happening in an actual (tiny) network.
- **The 2D slice problem.** Fractal boundaries in a slice are strong evidence for fractal structure in the full space. Smooth boundaries in a slice would be weak evidence of anything. The XOR slice is one random 2-plane in ℝ⁹; the factorisation slice is one plane in ℝ³, chosen for its symmetry.
- **"Solution identity" is a declared coarse-graining.** For XOR, runs with the same hidden-unit sign codes still differ continuously (a 5-dimensional zero-loss manifold). The factorisation classes are exact topological components.
- **Canonicalisation.** For two units, sorting sign-normalised codes is exactly weight matching modulo permutation and sign. For wider networks, Git Re-Basin's matching is a heuristic and would add its own boundaries.
- **Not converged ≠ diverged.** XOR "not converged" runs sit on plateaus (loss 0.25 or 0.5) at T = 20 000 and are shown in two neutral greys. A small fraction converge later (see the stall test above), so those borders depend on T.
- The Spectral panels are rank-normalised per image (per frame in films), so colours are not comparable across images. This is declared.
- **Edge of stability** here means the full-batch GD threshold 2/η. SGD, momentum and Adam have different thresholds.

## 7. References

- J. Sohl-Dickstein, *The boundary of neural network trainability is fractal*, 2024. arXiv:2402.06184; colab and Spectral split at github.com/Sohl-Dickstein/fractal.
- Y. Liu, *Complex fractal trainability boundary can arise from trivial non-convexity*, 2024. arXiv:2406.13971 (checked: exists, and says what the fractals doc claims).
- X. Zhu, Z. Wang, X. Wang, M. Zhou, R. Ge, *Understanding Edge-of-Stability Training Dynamics with a Minimalist Example*, ICLR 2023. arXiv:2210.03294 (checked: exists).
- Liang & Montúfar, *Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region*, 2025. arXiv:2509.25351 (exists per the sister project; not reread here).
- S. Ainsworth, J. Hayase, S. Srinivasa, *Git Re-Basin*, ICLR 2023. arXiv:2209.04836.
- C. Grebogi, S. McDonald, E. Ott, J. Yorke, *Final state sensitivity: an obstruction to predictability*, Phys. Lett. A 99 (1983); S. McDonald et al., *Fractal basin boundaries*, Physica D 17 (1985). Uncertainty exponent.
- J. Alexander, J. Yorke, Z. You, I. Kan, *Riddled basins*, Int. J. Bifurcation & Chaos 2 (1992).
- S. Cohen et al., *Gradient descent on neural networks typically occurs at the edge of stability*, ICLR 2021. arXiv:2103.00065.
- H.-O. Peitgen, H. Jürgens, D. Saupe, *Chaos and Fractals*, Springer. Box counting.
- Palettes: ColorBrewer Spectral; MetBrewer Klimt; stencil.wiki riso inks; Crameri lajolla, via `art/color-research/palettes.py`.
