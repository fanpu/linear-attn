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

**Motion:** [`gallery/lyap_zoom.mp4`](gallery/lyap_zoom.mp4) and [`gif`](gallery/lyap_zoom.gif). This is a 15-second geometric zoom from the full plane into the z5 shrimp field, with every frame recomputed (360 frames × 900², float64).

**Bifurcation cascades** along one horizontal line of the plane. The ink density is the attractor, and the Spectral strip underneath is λ along the same line:

<img src="gallery/bifurcation_ystar0.4168.png" width="100%">
<img src="gallery/bifurcation_ystar0.7.png" width="100%">

### 2.2 Lyapunov maps: experience-weighted attraction on RPS

RPS_PLACEHOLDER

### 2.3 Simplex trajectories and Poincaré sections (continuous learning, SAF)

| | |
|---|---|
| <img src="gallery/poincare_ink_eps0.50.png" width="100%"> | <img src="gallery/poincare_dark_eps0.50.png" width="100%"> |
| **Single ink**, ε = 0.5, energy H = 2.8. 395 orbits, ~4,400 returns each (1.8 M dots) through the section x_P − x_R + y_P − y_R = 0; axes x_R vs y_P. Nested closed curves are tori (KAM islands, genuinely repeating). The grey haze is a single chaotic sea. Aesthetic: ink coverage 1 − exp(−hits). | **Dark**, same data. Copper = orbits with finite-time λ ≤ 5·10⁻³; ice blue = chaotic orbits (λ > 5·10⁻³), each coloured by its own measured λ. |

<img src="gallery/poincare_eps_series.png" width="100%">

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

ICMAP_PLACEHOLDER

**Discrete MWU basin maps: a negative result.** See §4.

BASIN_PLACEHOLDER

### 2.5 Self-play framing

<img src="gallery/selfplay_plate.png" width="100%">

Two softmax policies (three logits each) are trained against each other with exact expected-payoff updates, from SAF's chaotic start.
* **MWU at small step** reproduces the continuous chaos when λ is measured per unit of learning time: λ/η ≈ 0.06–0.09 for η ≲ 0.04, against 0.056 in continuous time.
* **MWU at larger step** throws the players onto the simplex edge instead: the minimum probability reaches 10⁻³⁰ at η ≈ 0.1 and underflows to 0 above η ≈ 1. This is Bailey & Piliouras's divergence. λ collapses to ≈ 0 there, and what remains is best-response cycling at the boundary, not chaos.
* **Policy gradient** multiplies the update by the softmax Jacobian, so it slows down near the edge and is only weakly chaotic (λ/η ≈ 0.01–0.02).

---

## 3. What was computed

COMPUTED_PLACEHOLDER

---

## 4. Verification and honesty

VERIFY_PLACEHOLDER

---

## 5. Caveats

CAVEATS_PLACEHOLDER

---

## 6. Ideas explored / not pursued

IDEAS_PLACEHOLDER

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
