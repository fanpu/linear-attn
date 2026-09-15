# Ribbon (§4) — M2 report

**Status: DONE.** All plates are rendered, every image was opened and checked, and the work is committed. No packages were installed. Directory: `/home/fzeng/ml/research/art/ribbon/`. Running log and decisions: `NOTES.md` ("M2" section).

## 1. What was done

| step | script | where it ran | time |
|---|---|---|---|
| chart coordinates (hero, honesty, context), canyon boxes | `prep_m2.py` | CPU | ~5 s |
| canyon loss volumes: `hero32`, `context32`, `hero64` | `canyon_box.py` | CPU, 4 threads | 144 s / 149 s / 915 s |
| hero, honesty, stereo, context plates | `render_ribbon.py` via `run_m2_renders.sh` | GPU, one `gpu1.sh` job | 12 / 10.6 / 8.0 / 8.5 s |
| plotter SVG + proof; 32³ vs 64³ check | `render_ribbon.py plotter`, `check64` | CPU | < 1 s / 48 s |

- **Canyon grids ran on CPU.** The gpu1 queue was four art jobs deep, so I cancelled my queued GPU job (it was still waiting for the lock and never touched the GPU) and ran the grid on CPU (`logs/canyon_box.log`, `logs/canyon_box_cpu.log`). The fused first layer matches the plain forward to ≤ 1.4e-8.
- **The render job waited 1 h 56 min** in the queue (04:52 to 06:48), then ran in 50 s (`logs/m2_renders.log`).

## 2. Window choice and chart numbers (`cache/m2/prep_info.json`)

**Hero window: Stage B steps 3050–3449.** It is centred on t_ref = 3250 and contains three clear bursts.
- Among 400-step windows starting 2950–3250, the four near-centred ones (starts 3050–3200) show the most u_ref capture, 0.32–0.37; 3050 was chosen for centring.
- In this window, u_ref captures **36 %** of the detrended oscillation energy in the top-3 bank subspace at step 3250. The flip shares the near-degenerate top-3 subspace, so the hero shows about a third of it.
- λ₁η/2 ranges from 1.011 to 1.116, so every step in the window is above the edge. The naive u₁(t) swaps (overlap < 0.95) on 34 % of its steps.

**Context plate: piecewise frames.**
- In each bank frame (every 250 steps), axis 1 is the principal oscillation direction inside that frame's top-3 subspace. It captures a median 41 % of the top-3 energy.
- Frames are sign-aligned by the exact inner product and blended with triangular weights.
- Adjacent frames overlap only |⟨v_i, v_{i−1}⟩| median **0.39** (min 0.006). Axis 1 is a local oscillation coordinate, not one continuous direction.

**Canyon boxes** (chart offsets from θ_ref along u_ref, pc1, pc2):
- hero: the window's trajectory range ± 30 % on each side;
- context: pc1 and pc2 range ± 10 %, and axis 1 ±1.3 × max |axis 1|.

Loss ranges: hero 0.162–0.279, context 0.102–0.434.

**Interpolation check:** trilinear 32³ vs the 64³ grid differ by 6.1e-4 at most (median 3.6e-4); the two renders look identical.

## 3. Scaling (exaggeration relative to pc1, all orthographic)

| plate | u_ref / axis 1 | pc1 | pc2 |
|---|---|---|---|
| hero, honesty, stereo, plotter | ×4 | ×1 | ×4 |
| context | ×20 (piecewise axis 1) | ×1 | ×2 |

In the hero, pc2 ×2 looked planar; ×4 lets the slow pc2 drift read as depth. Declared.

## 4. Images: measured vs declared

Shared declarations for all plates:
- **Colour:** measured λ₁η/2 of each step, shown with Spectral_r and a two-slope norm 0.90 | 1.00 | 1.12 (declared diverging map centred at the edge).
- **Strands:** even and odd steps as `splat_spheres` tubes, with Lambert shading for form only.
- **Chords:** every GD step θ_t → θ_{t+1} as a straight `splat_additive` hairline. The ribbon is the set of real steps, not a fitted surface.
- **Canyon:** measured full-batch loss on the (u_ref, pc1, pc2) slice through θ(3250).
- **Transfer function (declared):** upper 65 % of `cmc.oslo` over the grid's loss range; opacity is 6 Gaussian shells (σ 0.018 in normalised loss) at levels 0.08…0.88; density 6/box extent (hero) and 1.2 (context); trilinear.
- **Half cutaway:** removes pc2 < the window median. Chords are split at the cut plane, so the ones behind it are fogged by the canyon and the ones in front are not. The volume stops at the strand depth buffer (`render_volume(depth=...)`).
- **Source:** Stage B's own trajectory only; no main4 burst times are cited.

Plates:
1. **`gallery/hero_glow.png`** (2400², dark glow).
   - Measured: steps 3050–3449 in the fixed frame, λ₁η/2, the local 32³ canyon.
   - Declared: u_ref ×4 and pc2 ×4, camera az −120° / el 38°, the transfer function, the half cutaway.
   - What it shows: wave-packet bursts along u_ref inside flat valley walls, with peaks rising through the lowest shells. Strands swap at the amplitude nodes. Strand colours alternate between even and odd steps because λ₁ itself oscillates with period 2 (measured).
2. **`gallery/honesty_fixed_vs_moving.png`** (3200×1600).
   - Left: the flip along the fixed u_ref.
   - Right: the same render with only the oscillation part of axis 1 replaced by the stored x_t = ⟨θ − θ̄, u₁(t)⟩ along the moving eigenvector. The slow part, pc axes, canyon and camera are identical (declared).
   - What it shows: the moving frame turns smooth packets into sawtooth jumps.
3. **`gallery/stereo_crosseye.png`** (2 × 1400²). A cross-eye pair (right-eye view on the left). Declared: rotation stereo at az ± 2.5°, because parallel-shift stereo gives no parallax with orthographic cameras.
4. **`gallery/plotter_hero.svg`** plus `gallery/plotter_hero_proof.png` (2048²). Three Inkscape layers:
   - the GD path, hidden-line against a thin tube depth buffer;
   - loss contours on the far face pc2 = hi, at the same six transfer-function levels;
   - the box wireframe.

   Declared: ink colours and line weights.
5. **`gallery/context_eos_glow.png`** (3000×1860).
   - Measured: steps 405–5989 (the EoS phase) in piecewise frames, and the global canyon rerun on the tightened box.
   - Declared: axis 1 ×20, pc2 ×2.
   - The caption states that the canyon is the fixed slice through θ(3250), that away from θ_ref the trajectory is not on that slice, and that a fixed axis 1 only holds locally (top-3 subspace overlap 0.30 after 250 steps).
   - The cyan spike at the right end is the catapult (λ₁η/2 < 1, around step 560).
6. **`gallery/check_canyon_32_vs_64.png`** (2048×1024). The same box, transfer function and camera at 32³ and 64³ (interpolation check).

## 5. Decisions (logged in NOTES.md)

- Hero window 3050–3449.
- Box margins: ± 30 % (hero); ± 10 % and 1.3 × max |axis 1| (context).
- Exaggeration factors as in §3.
- Colour norm 0.90 | 1.00 | 1.12, the same on every plate.
- Transfer function as in §4.
- Half cutaway at the window median of pc2, with chords split at the cut plane.
- Honesty panel swaps only the oscillation component of axis 1.
- Rotation stereo.
- Plotter layers as in §4.
- Canyon grids on CPU to avoid a 2 h queue.
- Context axis 1 = per-frame principal oscillation direction (not bank u₁, whose labels swap).

## 6. Risks and open items for M3

- **Context plate reads as a seismograph of spikes, not a ribbon.** The tubes (R 0.0035) hide the chords over 5600 steps. A thinner-tube variant would help; it needs a GPU rerun (8 s, plus the queue wait).
- **Hero framing:** the bottom ~20 % of the image is empty. Crop, or shift the camera target.
- **The hero's u_ref carries only 36 % of the top-3 oscillation energy,** and the piecewise frames are only weakly aligned (median 0.39). Captions have to carry this in M3.
- **The canyon shells look like flat parallel sheets** because loss is nearly quadratic along u_ref and changes slowly along the PCs. That is faithful to the data, but it is not a dramatic canyon.
- **The M3 film** (camera follows the 21-step mean path with a 400-step trailing window) needs many GPU frames; checkpoint per frame and expect long queue waits.

## 7. Commits

- `dc0f44d` art/ribbon: M2 prep, canyon boxes, renderer, plotter SVG, 32 vs 64 check
- `2da52c2` art/ribbon: M2 renders — hero glow, honesty panel, stereo pair, context plate, plotter SVG
- No further commits: the tree is clean after 2da52c2.
