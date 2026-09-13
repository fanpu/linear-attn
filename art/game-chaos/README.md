# Two Players: chaos in learning against an opponent

*Two learners adapting to each other in rock–paper–scissors or a two-road traffic game can end up in chaos: their mixed strategies wander on a strange attractor instead of settling. I measured every "chaotic" label in this gallery with a Lyapunov exponent.*

<p align="center"><img src="gallery/lyap_full_spectral.png" width="49%"> <img src="gallery/poincare_ink_eps0.50.png" width="49%"></p>
<p align="center"><img src="gallery/lyap_zoom_sequence.png" width="49%"> <img src="gallery/simplex_riso_torus_k5.png" width="49%"></p>

<p align="center"><video src="gallery/lyap_zoom.mp4" autoplay loop muted playsinline width="32%"></video> <video src="gallery/poincare_accumulate_eps0.50.mp4" autoplay loop muted playsinline width="32%"></video> <video src="gallery/simplex_drawing.mp4" autoplay loop muted playsinline width="32%"></video></p>

---

## 1. The phenomenon

A **game** gives each player a payoff that depends on both players' choices. A **learning rule** is an iterated map: each player keeps a score per action and turns the scores into a mixed strategy with a softmax. The same machinery sits inside self-play RL, GANs and multi-agent training. The question here is whether that map converges, cycles, or goes chaotic.

Several easy claims are wrong, so I started from the corrections:

* **Simultaneous gradient descent–ascent on a bilinear zero-sum game is not chaotic.** It spirals outward (discrete time) or cycles (continuous time).
* **Replicator dynamics on standard RPS (ε = 0) is not chaotic either.** It is integrable: every orbit lies on a torus. My runs confirm this, with finite-time λ₁ ≤ 3·10⁻⁴ for all 409 orbits at ε = 0.
* Chaos appears when the game or the learning rule is perturbed. I built three of these cases:

**(a) Continuous learning, Hamiltonian chaos** (Sato, Akiyama & Farmer 2002, "SAF"). Two players play generalised RPS, where a tie pays ε_x to player 1 and ε_y to player 2:

$$A=\begin{pmatrix}\varepsilon_x&-1&1\\1&\varepsilon_x&-1\\-1&1&\varepsilon_x\end{pmatrix},\quad B=A(\varepsilon_y),\qquad \dot x_i=x_i[(Ay)_i-x^\top Ay],\ \ \dot y_i=y_i[(Bx)_i-y^\top Bx].$$

With ε_x = −ε_y = ε the game is zero-sum, so the flow is **conservative (Hamiltonian)**. The conserved energy is H = −⅓Σ log x_i − ⅓Σ log y_i. There is no attractor, and orbits return arbitrarily close to their start (Poincaré recurrence). Once ε ≠ 0, the invariant tori break up into **chaotic seas that surround KAM islands**. Conservative and chaotic can hold at the same time.

**(b) Discrete multiplicative weights with a large step, dissipative chaos** (Palaiopanos, Panageas & Piliouras 2017; Chotibut et al. 2020; Bielawski et al. 2021). Two agents choose between two links with linear costs c₁(ℓ) = aℓ and c₂(ℓ) = bℓ. Both use exponential MWU with step η. In logit coordinates u = log(x/(1−x)), and with both agents starting from the same mixed strategy, the map is

$$u_{t+1}=u_t-s\,(\sigma(u_t)-y^*),\qquad s=\eta(a+b),\quad y^*=\frac{2b-a}{a+b}.$$

It is a bimodal one-dimensional map. Palaiopanos et al. prove period-3 orbits (Li–Yorke chaos) for one instance (a = 1/4, b = 1.4/4, large η).

**(c) Experience-weighted attraction with memory loss** (Galla & Farmer 2013): Q′ = (1−α)Q + β·payoff, x = softmax(Q). I ran it on SAF's RPS payoffs.

The **largest Lyapunov exponent** λ is the average exponential growth rate of an infinitesimal perturbation. λ > 0 means chaos, λ < 0 means convergence to a periodic orbit or fixed point, and λ ≈ 0 means quasi-periodic motion. I propagated a tangent vector with the exact Jacobian and renormalised it at every step (Benettin's method).

---

## 2. Gallery

### 2.1 Lyapunov parameter maps: the congestion game (hero piece)

Each pixel is one learning run with its own (step size s, equilibrium load y*), coloured by the measured λ.

**Declared colour (Sohl-Dickstein Spectral split):**
* λ < 0 (periodic learning) runs purple → blue → green → pale yellow.
* λ > 0 (chaotic learning) runs deep red → orange → pale yellow.
* Each side is rank-normalised separately, and the dark ends meet at λ = 0, so the chaos/order boundary shows as a dark seam.

| | |
|---|---|
| <img src="gallery/plate_full.png" width="100%"> | <img src="gallery/lyap_zoom_sequence.png" width="100%"> |
| **Plate, full plane**, s ∈ [1, 100], y* ∈ (0, 1), 2400², 3000 + 5000 iterations. Measured: λ. Aesthetic: Spectral split, paper plate. | **Nested zoom** full → crossing → shrimp → inner shrimp. Boxes mark the next panel. Each panel is rank-normalised on its own pixels. |

**Shrimps.** The dark-seamed "shrimp" shapes are periodic windows: a stable periodic orbit exists inside each one, and each has the classic head with four trailing legs. Copies recur inside the chaotic sea at every zoom I computed (×7 → ×82 → ×1240 in s), and new ones keep appearing as I zoom in. This is the structure Gallas (1993) described for two-parameter maps. It appears here because this learning map has two critical points (1 − sσ′(u) = 0 has two roots once s > 4).

<table>
<tr><td><img src="gallery/lyap_z3_shrimp_spectral.png" width="100%"></td><td><img src="gallery/lyap_z4_shrimp_spectral.png" width="100%"></td><td><img src="gallery/lyap_z5_shrimp_spectral.png" width="100%"></td></tr>
<tr><td>z3 shrimp, s ∈ [29.6, 30.6]</td><td>z4 shrimp, s ∈ [23.2, 24.4]</td><td>z5 inner shrimp, s ∈ [24.07, 24.15]</td></tr>
<tr><td><img src="gallery/lyap_z1_crossing_spectral.png" width="100%"></td><td><img src="gallery/lyap_z2_hooks_spectral.png" width="100%"></td><td><img src="gallery/lyap_z6_chain_spectral.png" width="100%"></td></tr>
<tr><td>z1 crossing of two window families</td><td>z2 hooks, y* ≈ 0.7</td><td>z6 chain of small windows</td></tr>
</table>

Style variants of the same measured λ (declared palettes: `color-research/palettes.py` split pairings and cmcrameri berlin):

<table>
<tr><td><img src="gallery/lyap_full_verdigris.png" width="100%"></td><td><img src="gallery/lyap_full_aurora.png" width="100%"></td><td><img src="gallery/lyap_full_dark.png" width="100%"></td></tr>
<tr><td>verdigris / copper split</td><td>aurora / ember split</td><td>dark ground: berlin diverging centred at λ = 0, asinh stretch</td></tr>
<tr><td><img src="gallery/lyap_z3_shrimp_verdigris.png" width="100%"></td><td><img src="gallery/lyap_z1_crossing_aurora.png" width="100%"></td><td><img src="gallery/lyap_z1_crossing_dark.png" width="100%"></td></tr>
</table>

<table>
<tr><td><img src="gallery/plate_z4_shrimp.png" width="100%"></td><td><img src="gallery/lyap_z4_shrimp_verdigris.png" width="100%"></td><td><img src="gallery/lyap_z4_shrimp_dark.png" width="100%"></td></tr>
<tr><td>plate, z4 shrimp (also <a href="gallery/plate_z3_shrimp.png">z3 plate</a>)</td><td>z4 verdigris/copper (also <a href="gallery/lyap_z4_shrimp_aurora.png">aurora</a>)</td><td>z4 dark berlin (also <a href="gallery/lyap_z3_shrimp_dark.png">z3 dark</a>, <a href="gallery/lyap_z3_shrimp_aurora.png">z3 aurora</a>, <a href="gallery/lyap_z1_crossing_verdigris.png">z1 verdigris</a>)</td></tr>
</table>

**Motion:** [`gallery/lyap_zoom.mp4`](gallery/lyap_zoom.mp4) and [`gif`](gallery/lyap_zoom.gif). This is a 17-second geometric zoom (×2750 in s, ×7100 in y*) from the full plane into the z5 shrimp field, with every frame recomputed (360 frames × 900², float64).

**Bifurcation cascades** along one horizontal line of the plane. The ink density is the attractor, and the Spectral strip underneath is λ along the same line:

<img src="gallery/bifurcation_ystar0.4168.png" width="100%">
<img src="gallery/bifurcation_ystar0.7.png" width="100%">

### 2.2 Lyapunov maps: experience-weighted attraction on RPS

Discrete experience-weighted attraction (EWA) on SAF's rock–paper–scissors, largest λ per pixel. **Measured:** λ from tangent propagation, and the minimum strategy probability on the attractor. **Declared colour:** Spectral split with the seam at λ = 3·10⁻³, the finite-time noise floor for quasi-periodic orbits.

<table>
<tr><td width="50%"><img src="gallery/plate_rps_beta_eps_a0.30.png" width="100%"></td><td width="50%"><img src="gallery/rps_beta_eps_a0.30_zoom_spectral.png" width="100%"><br>Zoom β ∈ [2.5, 9], ε ∈ [−0.6, 0.6] (independent 1000² run; 8.2% chaotic here). The moth's wings are periodic "horns" (blue) cutting into chaotic lobes (orange).</td></tr>
</table>

In the (β, ε) plane at α = 0.3, chaos (red/orange) occupies only **2.9%** of the pixels. It sits in a mirror-symmetric "moth" around ε = 0, 3 < β < 9, inside a sea of quasi-periodic motion (blue) and near-pure best-response cycling (pale green, 67% of pixels, where strategies come within 10⁻⁶ of the simplex edge).
* **Coexisting attractors cause the straight vertical cuts.** Every pixel starts from the same (x₀, y₀). I checked with 8 random starts per β at ε = 0.15:
  * β = 5.4: 3 starts chaotic (λ = 0.046–0.087), 5 periodic (λ = −0.008).
  * β = 7.4 and 7.6: chaos (λ ≈ 0.015–0.05) coexists with a cycle at λ = −0.357.
* **The fine ripples at large β are not resolution-checked** (see §4) and may be partly aliased.

<table><tr><td><img src="gallery/rps_beta_eps_a0.30_indigo.png" width="100%"></td><td><img src="gallery/rps_beta_eps_a0.30_zoom_dark.png" width="100%"></td><td><img src="gallery/rps_beta_eps_a0.30_zoom_indigo.png" width="100%"></td></tr>
<tr><td>indigo / madder split (declared)</td><td>dark berlin diverging, seam at λ = 3·10⁻³</td><td>indigo / madder, zoom</td></tr></table>

Also: [bare Spectral print](gallery/rps_beta_eps_a0.30_spectral.png), [dark full plane](gallery/rps_beta_eps_a0.30_dark.png), [zoom plate with caption](gallery/plate_rps_beta_eps_a0.30_zoom.png).

### 2.3 Simplex trajectories and Poincaré sections (continuous learning, SAF)

| | |
|---|---|
| <img src="gallery/poincare_ink_eps0.50.png" width="100%"> | <img src="gallery/poincare_dark_eps0.50.png" width="100%"> |
| **Single ink**, ε = 0.5, energy H = 2.8. 395 orbits, ~4,400 returns each (1.8 M dots) through the section x_P − x_R + y_P − y_R = 0; axes x_R vs y_P. Nested closed curves are tori (KAM islands, genuinely repeating). The grey haze is a single chaotic sea. Aesthetic: ink coverage 1 − exp(−hits). | **Dark**, same data. Copper = orbits with finite-time λ ≤ 5·10⁻³; ice blue = chaotic orbits (λ > 5·10⁻³), each coloured by its own measured λ. |

<img src="gallery/poincare_eps_series.png" width="100%">

Also: [ε = 0 single-ink section](gallery/poincare_ink_eps0.00.png) (integrable: only tori) and [ε = 0.25 dark](gallery/poincare_dark_eps0.25.png).

*ε = 0 → 0.5 on one energy surface. The share of chaotic orbits grows 0% → 1% → 14% → 23% → 29%. Red marks orbits measured as chaotic.* At H = 3.0 (higher energy, closer to the simplex edge), 67% of orbits are chaotic.

| | |
|---|---|
| <img src="gallery/simplex_riso_players.png" width="100%"> | <img src="gallery/simplex_riso_torus_k5.png" width="100%"> |
| **Riso 2-ink, chaotic orbit** (SAF start k = 1, λ₁ = 0.050). Player 1 is printed in blue and player 2 in fluorescent pink on one simplex, t ≤ 1200. The pink plate is deliberately misregistered by 4 px (declared). | **Riso 2-ink, regular orbit** (k = 5, λ₁ → 0 as 1/T). A braided torus; the two players trace near-mirror images. |

<img src="gallery/simplex_plate_eps0.50.png" width="100%">

*Scientific plate: four starts in the same game, players in rows. Chaotic (k = 1, 2) fill the simplex; regular (k = 5, 20) stay on tori.*

| | |
|---|---|
| <img src="gallery/plotter_single_line.png" width="100%"> | <img src="gallery/butterfly_simplex_dark.png" width="100%"> |
| **Plotter drawing**: one continuous line, player 1's strategy on the chaotic orbit for t ≤ 1200 (1 px pen). A plotter-ready polyline is in [`plotter_single_line.svg`](gallery/plotter_single_line.svg), decimated to 0.15 mm. | **Butterfly effect**: 16 learners started 10⁻⁹ apart in logit space. Their spread is 2·10⁻⁶ at t = 200 and 0.56 at t = 600. |

**Motion:** [`poincare_accumulate_eps0.50.mp4`](gallery/poincare_accumulate_eps0.50.mp4) / [`gif`](gallery/poincare_accumulate_eps0.50.gif) shows the section filling in return by return, with the chaotic sea (red) arriving as haze around the islands. [`simplex_drawing.mp4`](gallery/simplex_drawing.mp4) / [`gif`](gallery/simplex_drawing.gif) shows both players' strategies being drawn.

### 2.4 Initialization maps

<img src="gallery/plate_icmap.png" width="100%">

*Which first move leads to chaos?* Player 1's initial mixed strategy sweeps the simplex (79,800 runs), player 2 starts at (½, ¼, ¼), and colour is the finite-time λ.
* The histogram of log λ is **cleanly bimodal**, so the λ = 5·10⁻³ seam sits in a real gap, not at an arbitrary cut.
* Chaos hugs the simplex edge, where energy is high. Filaments of chaos also reach into the regular core along separatrices.

<table><tr><td width="50%"><img src="gallery/icmap_spectral.png" width="100%"></td><td width="50%"><img src="gallery/icmap_zoom_spectral.png" width="100%"></td></tr>
<tr><td>Bare print, Spectral split, equilateral simplex (R bottom-left, P bottom-right, S top).</td><td><b>10× zoom</b> (x_P ∈ [0.33, 0.43], x_S ∈ [0.03, 0.13]; 160,000 fresh runs). Red chaotic tongues interleave with regular fans. The woven dot texture in the regular region is <b>moiré</b>: the finite-time λ of a regular orbit oscillates with its torus phase at t = T, and that phase varies faster than the pixel grid. Do not read the dots as structure; the claim is only the red/green boundary.</td></tr>
<tr><td><img src="gallery/icmap_cyanotype.png" width="100%"></td><td>Cyanotype / Van Dyke split variant (declared palette from <code>color-research</code>).</td></tr></table>

**Discrete MWU basin maps: a negative result.** See §4.

<img src="gallery/basin_atlas_negative.png" width="100%">

*Colour = which pure Nash profile the two learners reach from each pair of starts (declared categorical palette). Top row: large step; bottom row: the same game at a small step (null).*
* **2×2 coordination** (s = 20): the boundary bends, but it is smooth.
* **2×2 anti-coordination** (s = 30): the boundary is exactly the diagonal, identical at every step size, as the sign-invariance argument predicts.
* **Three-link congestion game** (costs (1, 1.2, 1.5), η = 60): smooth. With costs (1, 1.3, 1.1), η = 40, one detached island appears, the only non-trivial feature I found (D = 1.17, see §4).

<table><tr><td width="45%"><img src="gallery/basin_island_zoom.png" width="100%"></td><td>

**4× zoom of that island** (x_P ∈ [0.02, 0.27], x_S ∈ [0.52, 0.77], 1024², fresh runs). The large "bird" is smooth. Small lobes and thin slivers cluster along one straight line. That line is where player 2's start (the P↔R swap of player 1's start) *coincides* with player 1's. It is the symmetric invariant subspace that carries the Palaiopanos-type chaotic saddle. So the only intricacy in the discrete basin maps sits exactly where theory puts the chaos, but it stays thin.
</td></tr></table>

### 2.5 Self-play framing

<img src="gallery/selfplay_plate.png" width="100%">

Two softmax policies (three logits each) are trained against each other with exact expected-payoff updates, from SAF's chaotic start.
* **MWU at small step** reproduces the continuous chaos when λ is measured per unit of learning time: λ/η ≈ 0.06–0.09 for η ≲ 0.04, against 0.056 in continuous time.
* **MWU at larger step** throws the players onto the simplex edge instead: the minimum probability reaches 10⁻³⁰ at η ≈ 0.1 and underflows to 0 above η ≈ 1. This is Bailey & Piliouras's divergence. λ collapses to ≈ 0 there, and what remains is best-response cycling at the boundary, not chaos.
* **Policy gradient** multiplies the update by the softmax Jacobian, so it slows down near the edge and is only weakly chaotic (λ/η ≈ 0.01–0.02).

---

## 3. What was computed

No neural networks or datasets are involved: every "model" is a pair of mixed strategies, and every "optimizer" is a learning rule in logit coordinates. **Precision:** all float64. **Seeds:** tangent vectors use numpy/torch seed 0; nothing else is random except the documented 2×2 random-game search (seed 1).

| piece | system | grid / orbits | iterations | where |
|---|---|---|---|---|
| Congestion Lyapunov planes | u′ = u − s(σ(u) − y*), exact derivative | full 2400², six zooms 2000² | 3000 transient + 5000 averaged | `compute_lyap_congestion.py plates` (GPU, ~5–7 min per plate) |
| Resolution / iteration check | z4 window | 500², 1000², 2000²; 500² at 20k iterations | | `compute_lyap_congestion.py rescheck` |
| Zoom video | geometric zoom, 360 frames | 900² per frame | 1000 + 1500 | `compute_lyap_congestion.py video` (~100 min on the shared GPU) |
| Bifurcation plates | same map, y* ∈ {0.4168, 0.7} | 3000 step sizes | 3000 + 600 samples | `compute_bifurcation.py` |
| EWA on RPS | Q′ = (1−α)Q + βA(ε)y, tangent propagation | 1000² × 2 | 3000 + 5000 | `compute_lyap_rps.py rps_beta_eps_a0.30 rps_beta_eps_a0.30_zoom` (~25 min each) |
| SAF reproduction | replicator, RK4 h = 0.01 in logits, C/OpenMP | 25 starts × 3 ε | T = 10⁵ | `compute_poincare.py repro` |
| Poincaré sections | 26–40² seeds on one energy surface (secant-refined crossings) | ~400 orbits per ε, 5 values of ε + H = 3.0 | T = 4·10⁴ | `compute_poincare.py kam` (~1 min each) |
| Trajectories, butterfly, plotter line | SAF starts k = 1, 2, 5, 20 | | T = 4000 (h = 0.005); 20,000 for the plotter line | `compute_poincare.py traj` |
| Initial-condition λ map | player-1 start over the simplex | 79,800 orbits; zoom 160,000 | T = 2000 | `compute_icmap.py 400 2000 0.5` and `… 0.33 0.03 0.1` (11 / 22 min, 4 CPU threads) |
| Basin atlas | 2×2 MWU, 3-link MWU, + small-step nulls | 256², 512², 1024² | 3000–4000 | `compute_basins.py atlas` |
| Self-play | PG vs MWU, 1200 step sizes | | 5000 + 20,000 | `compute_selfplay.py` (CPU, 40 s) |

**Reproduce everything** (from this directory; long GPU jobs go through the slot limiter):

```bash
P=/home/fzeng/ml/research/art/.venv/bin/python; G=/home/fzeng/ml/research/art/_shared/gpu_run.sh
export OMP_NUM_THREADS=4
$P replicator_c.py                                 # compiles the C integrator, smoke-tests SAF Table I
$P compute_poincare.py all                         # repro + Poincaré sections + trajectories
$P compute_icmap.py 400 2000 0.5 && $P compute_icmap.py 400 2000 0.5 0.33 0.03 0.1
$G $P compute_lyap_congestion.py plates rescheck   # then: $G $P compute_lyap_congestion.py video
$G $P compute_lyap_rps.py rps_beta_eps_a0.30 rps_beta_eps_a0.30_zoom
$G $P compute_basins.py atlas && $G $P compute_basins.py island_zoom
$P compute_bifurcation.py && $P compute_selfplay.py
$G $P analyze_verify.py lyap && $P analyze_verify.py saf icmap basins
$P render_lyap.py all && $P render_rps.py && $P render_poincare.py all riso_torus
$P render_basins.py icmap atlas selfplay && $P render_bifurcation.py
```

**Wall time.** About 4.5 h of shared-GPU wall clock (the GPU ran at 90%+ utilisation from ~9 agents, so real compute was much less) and about 1.2 h of 4-thread CPU for the C integrator. **Exploration scripts** that led to these choices are kept in `cache/explore/` (gitignored); see §4.3.

**Files over 20 MB** (not committed): none in `gallery/`. `cache/` is gitignored.

---

## 4. Verification and honesty

### 4.1 Did chaos actually show up? Yes, in the three settings the literature predicts, and I measured it.

**SAF Table I reproduction.** Same initial conditions, T = 10⁵, RK4 h = 0.01 in logits. λ₁ is given ×10³:

| ε | k = 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 0.25, ours | **49.3** | **25.3** | **20.3** | 0.09 | 0.09 |
| 0.25, SAF | **49.0** | **35.3** | **16.6** | 0.4 | 0.4 |
| 0.50, ours | **55.6** | **39.2** | **21.9** | **8.4** | 0.10 |
| 0.50, SAF | **61.6** | **35.0** | **28.1** | **12.1** | 0.2 |
| 0.00, ours | 0.10 | 0.09 | 0.09 | 0.09 | 0.09 |

* **The same orbits are chaotic as in SAF** (bold): k ≤ 3 at ε = 0.25 and k ≤ 4 at ε = 0.5.
* Magnitudes agree to within the scatter expected for finite-time exponents of sticky Hamiltonian orbits.
* At ε = 0 every orbit is regular: λ decays as 1/T ([convergence plot](gallery/verify_lyap_convergence.png)).
* Energy H is conserved to 3·10⁻⁹ over T = 10⁵.

**Congestion-game map.** λ > 0 on **42.1%** of the full (s, y*) plane (max λ = 0.586 nats/iteration). On the row y* = 0.5 (fully symmetric costs), λ ≤ −0.006 for all s ≤ 100, which is Chotibut et al.'s "unless the game is fully symmetric".
* **Iteration check:** on the 500² shrimp window, the sign of λ agrees on **99.87%** of pixels between 5,000 and 20,000 averaging iterations (correlation 0.999).

**EWA on RPS.** Chaos is present but sparse: 2.9% of the (β, ε) plane at α = 0.3, max λ = 0.154.

**Self-play.** Discrete MWU at small η keeps λ/η ≈ 0.06–0.09, matching the continuous λ₁ = 0.056. At large η the dynamics are no longer chaotic; the players are thrown onto the simplex edge instead (§2.5).

### 4.2 Fractal / self-similarity claims (fractals doc §11)

**(i) The chaos/order seam in the congestion Lyapunov plane.**
* **Setup.** The boundary set is pixels whose 4-neighbourhood contains both λ > 0 and λ ≤ 0. The window is the z4 shrimp (s ∈ [23.2, 24.4], y* ∈ [0.4145, 0.4215]), computed independently at 500², 1000² and 2000².
* **Box counting** (box side 1–64 px, i.e. ε from 1/2000 to 1/31, 1.8 decades):

  | resolution | 500² | 1000² | 2000² |
  |---|---|---|---|
  | **D** | 1.38 | 1.40 | 1.33 |

* **Resolution scaling.** Boundary-pixel fraction is 9.7% → 7.4% → 5.5% per doubling. That implies D = 2 + slope ≈ **1.6**. A smooth curve would halve per doubling (D = 1).
* **Null model 1** (same pipeline on an analytic smooth curve, the period-doubling line s·y*(1−y*) = 2): D = 1.09 / 1.07 / 1.07, and the fraction halves per doubling.
* **Null model 2** (same pipeline on a smooth Lyapunov contour, λ = −0.05 in s ∈ [3, 12]): D = 1.22 / 1.17 / 1.11. It trends to 1 with resolution, which shows the pixel pipeline adds about 0.1–0.2 at low resolution.
* **Verdict.** The seam is clearly rougher than a smooth curve at every resolution, and new windows keep appearing on zoom (×1240, figure above). The two estimators disagree (1.33–1.40 vs 1.6), so I report "**D between about 1.3 and 1.6, non-integer**" rather than a single number.
  * Chaotic regions of such maps are expected to contain dense periodic windows, so the true boundary may be a fat fractal whose box dimension creeps toward 2 with resolution.
  * This makes shrimps self-similar in the statistical sense (the same morphology recurs at every scale) but not exactly self-similar.
* **Precision floor.** float64. At the deepest frame the pixel spacing is 4·10⁻⁵ in s and 1.6·10⁻⁷ in y*, ten decades above machine ε, so precision is not a limit here. Zooming further would need more iterations, because windows become narrower than the finite-time λ noise.

**(ii) The regular/chaotic boundary in the initial-condition map (§2.4).**
* **Setup.** The boundary is λ = 5·10⁻³, which sits in the gap of a bimodal histogram.
* **Box counting** (1–32 px, 1.5 decades):
  * full simplex, 400²: **D = 1.28**
  * independent 10× zoom, 400²: **D = 1.37**
* **Null** (same pipeline on a smooth level set of the energy H in the same pixels): D = 1.09 (full) and 1.18 (zoom).
* **Verdict.** The chaos boundary is rougher than a smooth curve by about 0.2, consistently at both scales. The fit range is short (§11 wants more than one decade; this is 1.5), and the null is not exactly 1, so I call it **suggestive, not proven, fractal**. That is the expected KAM picture of islands around islands.

**(iii) Discrete MWU basin boundaries.** Box counting (1–64 px on 1024²) and boundary-fraction scaling (256² → 512² → 1024²):

| game | large step: D (box) | large step: D (resolution) | small-step null: D (box) | small-step null: D (resolution) |
|---|---|---|---|---|
| 2×2 coordination | 1.06 | 1.05 | 1.03 | 1.00 |
| 2×2 anti-coordination | 1.01 | 1.00 | 1.01 | 1.00 |
| 3-link (1, 1.2, 1.5) | 1.03 | 0.99 | 1.03 | 0.98 |
| 3-link (1, 1.3, 1.1) | **1.17** | **1.12** | 1.03 | 0.98 |

All boundaries are smooth (D ≈ 1) except the island in the last game, which is mildly rough. The 4× island zoom gives D (box) = 1.17 at 512² and 1.16 at 1024². The boundary-fraction scaling between those resolutions gives **D = 1.07**, so the extra length does not keep growing as resolution increases. **Verdict: no fractal basin boundary in discrete MWU for these games.** The mild roughness comes from a few lobes accumulating near the symmetric chaotic saddle; a much deeper zoom along that line would be the place to look for riddling.

### 4.3 What didn't work (negative results)

* **Fractal basins for discrete MWU in small coordination / anti-coordination games: not found.**
  * 2×2 coordination games (MWU and EWA, s up to 60) give smooth basins.
  * A GPU random search over 300 random 2×2 bimatrix games, with and without memory loss (η up to 60), found no boundary that refined like a fractal between 256² and 1024². Every candidate was either D ≈ 1 or speckle from chaotic attractors that I had mislabelled as distinct.
  * **In 2×2 anti-coordination/congestion games the ordering sign(u − v) is invariant** (d′ = d + s(σ(u) − σ(v)) keeps its sign), so the basin boundary is exactly the diagonal. That diagonal is also where Palaiopanos et al.'s chaos lives. I show analytically that the diagonal attractor is always transversally unstable (|1 + sσ′| > |1 − sσ′|).
  * **The chaos they prove is a chaotic saddle of the full two-agent game.** Any asymmetry between the players' starts, however small, eventually sends them to a pure asymmetric equilibrium.
* **Three-link congestion game, symmetric start.** At η = 24–40 there is one chaotic attractor. At η ≥ 56 the "attractors" are float-saturated best-response cycles (softmax exactly 0/1). I did not trust those basin maps and did not render them.
* **Discrete RPS without memory loss (α = 0).** MWU drives strategies to the simplex boundary (Bailey–Piliouras). λ there is dominated by numerical saturation, so I did not show those planes.
* **EWA plane textures.** Fine ripple textures at large β in the EWA planes were not resolution-checked. Treat them as texture, not structure.

---

## 5. Caveats

* **"Learning is chaotic" needs conditions.** It holds for (a) a *perturbed* zero-sum RPS under continuous replicator learning (Hamiltonian chaos, which is conservative, not an attractor); (b) *large-step* MWU in a congestion game *with both agents starting identically*; (c) EWA with memory loss in narrow parameter regions. Plain RPS (ε = 0) is integrable. Plain GDA on bilinear games spirals or cycles.
* **Symmetric start (hero map).** Both agents begin from the same mixed strategy, which is the setting of the Palaiopanos et al. and Chotibut et al. proofs. In the full two-agent game that invariant line is transversally unstable, so real asymmetric learners leave the chaos and converge to a pure asymmetric equilibrium.
* **Cheung & Piliouras's "Lyapunov chaos"** is volume expansion in the cumulative-payoff space of MWU in zero-sum games, with Lyapunov time O(1/η²). It is not a positive Lyapunov exponent on a bounded attractor, and the primal strategies drift to the boundary. My self-play plate shows both faces.
* **Finite-time exponents.** Every λ here is a finite-time estimate.
  * Regular Hamiltonian orbits give λ ~ 1/T rather than 0.
  * Sticky chaotic orbits can masquerade as regular for a long time.
  * Thresholds (5·10⁻³ for continuous time, 3·10⁻³ for EWA) are declared and chosen from histogram gaps.
* **2-D slices** (fractals doc §14.2). Every map is a slice through a larger space (initial conditions of player 2, the third payoff parameter, α). Structure in a slice is evidence; smoothness in a slice is weak evidence.
* **Deflationary reading** (§14.1). Shrimps and period-doubling are what *any* two-parameter family of maps with two critical points does. The learning rule here is such a map, and nothing about it is special to games. The caption this deserves is "this is what iteration near an instability looks like, and a learning rule is iteration near an instability".
* **Rank normalisation** (Spectral split) equalises the histogram on each side. Colour differences inside a side show *rank*, not magnitude; only the seam position is absolute.
* **Not neural networks.** The "policies" are three logits each. The self-play framing is literal but minimal.

---

## 6. Ideas explored / not pursued

Brainstorm, ranked by what I built:

1. **Initial-condition λ map of continuous learning** ("which first move leads to chaos"). *Built* (§2.4). It is the strongest intricate-boundary piece, since discrete MWU basins turned out smooth.
2. **Discretisation plate for self-play** (policy gradient vs MWU, λ per unit learning time vs η). *Built* (§2.5).
3. **Wide bifurcation cascades** under the Lyapunov strip. *Built* (§2.1).
4. **Butterfly small multiples** (16 learners 10⁻⁹ apart). *Built*.
5. **ε-series of Poincaré sections** (tori breaking). *Built*.
6. **Galla–Farmer phase diagram** for random N = 50 games (Γ vs α/β, coloured by attractor dimension). *Not pursued*: it is a different object (high-dimensional chaos, not a picture), and a faithful version needs many payoff draws per pixel.
7. **Multistability map** (s, u₀) for the congestion map, to find Cantor-like basins in one dimension. *Explored at 512²*: coexisting attractors exist only in thin s-bands (e.g. s ≈ 20.8, 24.2), and almost everywhere the attractor does not depend on u₀. It was not a good image. The started 12-start multistability sweep was stopped to free the GPU.
8. **3-D torus sculpture / STL** of a KAM torus in (x_R, x_P, y_R). *Not pursued* (no fabrication pipeline here).
9. **Sonification** of the chaotic orbit (players as two voices). *Not pursued*.

**Critique rounds** (each image was viewed, then fixed):
* Round 1:
  * Poincaré dark variant: the chaotic sea was invisible, so it now uses a copper/ice two-colour scheme.
  * Plotter drawing: the full orbit was a grey mass, so it now shows t ≤ 1200, centred.
  * Bifurcation ink was too faint; label overlap fixed.
  * Self-play plate had overflowing text and a wrong axis range.
* Round 2:
  * Zoom-sequence boxes were invisible, so they are now thicker black+white.
  * Riso simplices had an empty top band, now cropped.
  * EWA plate caption had raw dict text and a large gap.

---

## 7. References

* Y. Sato, E. Akiyama, J. D. Farmer. *Chaos in learning a simple two-person game.* PNAS 99:4748 (2002). SFI WP 01-09-049. Checked: payoffs, section, and Table I (see §4).
* T. Galla, J. D. Farmer. *Complex dynamics in learning complicated games.* PNAS 110:1232 (2013), arXiv:1109.4250. EWA rule Q′ = (1−α)Q + Πx, x ∝ e^{βQ}. Their results are for random N = 50 games with payoff correlation Γ; I use their learning rule on SAF's 3×3 game instead.
* G. Palaiopanos, I. Panageas, G. Piliouras. *Multiplicative weights update with constant step-size in congestion games: convergence, limit cycles and chaos.* NeurIPS 2017, arXiv:1703.01138. Checked: the Li–Yorke chaos is proven for the **symmetric-start one-dimensional reduction** x₀ = y₀.
* Y. K. Cheung, G. Piliouras. *Vortices instead of equilibria in min-max optimization: chaos and butterfly effects of online learning in zero-sum games.* COLT 2019, arXiv:1905.08396. Their "Lyapunov chaos" is **volume expansion in the dual (cumulative-payoff) space**, with Lyapunov time O(1/η²). It is not a positive exponent on a bounded attractor.
* J. Bielawski, T. Chotibut, F. Falniowski, G. Kosiorowski, M. Misiurewicz, G. Piliouras. *Follow-the-regularized-leader routes to chaos in routing games.* ICML 2021, arXiv:2102.07974. **Correction to the task prompt:** this paper is about routing/congestion games (population size or cost scale as the bifurcation parameter), not coordination games.
* T. Chotibut, F. Falniowski, M. Misiurewicz, G. Piliouras. *The route to chaos in routing games: when is price of anarchy too optimistic?* NeurIPS 2020.
* J. P. Bailey, G. Piliouras. *Multiplicative weights update in zero-sum games.* EC 2018.
* J. A. C. Gallas. *Structure of the parameter space of the Hénon map.* PRL 70:2714 (1993). Shrimps.
* G. Benettin, L. Galgani, A. Giorgilli, J.-M. Strelcyn. Lyapunov characteristic exponents, Meccanica 15 (1980).
* J. Sohl-Dickstein. *The boundary of neural network trainability is fractal.* arXiv:2402.06184 (2024). Source of the Spectral split colouring.
