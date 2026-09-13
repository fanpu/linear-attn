# PAUSED — game-chaos checkpoint

All background jobs were stopped. Nothing is running.

## Done
- Papers read (in `cache/papers/`, gitignored). Bielawski et al. 2021 (arXiv:2102.07974) is about **routing/congestion games**, not coordination games. Note this in the README.
- `gamelib.py`: torch float64 dynamics (1-D congestion MWU map, RPS EWA with tangent-vector Lyapunov exponents) and `spectral_split` (the Sohl-Dickstein cdf restretch, seam at λ=0).
- `replicator_c.py`: C/OpenMP RK4 integrator for the SAF replicator equations (logits, variational equation, secant-refined Poincaré crossings). It **reproduces SAF Table I**: at T=1e5, ε=0.25 gives λ1·1e3 = 49.3, 25.3, 20.3, 0.09, … (SAF: 49.0, 35.3, 16.6, 0.4); ε=0.5 gives 55.6, 39.2, 21.9, 8.4, 0.1 (SAF: 61.6, 35.0, 28.1, 12.1, 0.2). H drift is below 3e-9.
- Cached: `cache/saf_repro.npz` and `cache/kam_eps{0.00,0.10,0.25,0.40}.npz` (single energy H0=2.8, ~400 orbits, T=4e4). Chaotic fraction (λ>5e-3) is 0, 0.01, 0.14, 0.23.
- Exploration (in `cache/explore/`, with PNGs):
  - The congestion Lyapunov plane (s, y*) is rich: mirror-symmetric hooks, and **genuine shrimps** at (30.18, 0.3914) and (24.10, 0.4172), nested. Straight-edged "tears" mark coexisting attractors, where the fixed u0 changes basin.
  - RPS EWA planes show Arnold tongues and sparse chaos. The (β up to 12) exploration did not finish.
  - Basin maps: 2×2 MWU/EWA coordination and anti-coordination basins are smooth. A GPU random search over 300 2×2 games found no fractal boundary between regular attractors (D≈1, or noise from chaotic labels). In 2×2 anti-coordination the ordering sign(u−v) is invariant, so the basin boundary is exactly the diagonal. 3-link congestion with symmetric start (a 2-D map on the simplex) has many coexisting cycles at η=24–40. **This is the best basin candidate** (`cache/explore/sym3.py`).

## Next (exact commands, from this directory)
```
P=/home/fzeng/ml/research/art/.venv/bin/python; G=/home/fzeng/ml/research/art/_shared/gpu_run.sh
OMP_NUM_THREADS=4 $P compute_poincare.py kam    # re-run: adds eps 0.5 + H0=3.0 (skip done eps if desired)
OMP_NUM_THREADS=4 $P compute_poincare.py traj
nohup $G $P compute_lyap_congestion.py plates rescheck multistab > logs/cong_plates.log 2>&1 &
nohup $G $P compute_lyap_congestion.py video > logs/cong_video.log 2>&1 &   # ~80 min, 360x900^2 frames
nohup $G $P cache/explore/rps_explore.py > logs/rps_explore.log 2>&1 &
```
Then:
1. Build `compute_basins.py` from sym3.py: classify attractors (period and orbit signature, LE), box counting / uncertainty exponent, 2×/4× resolution check, null model at small η.
2. Optional self-play piece (softmax policy gradient on RPS).
3. Renders: Spectral split (primary), dark diverging, single-ink Poincaré, riso 2-ink trajectories, scientific plate. Motion: zoom, section accumulating, trajectory drawing.
4. Brainstorm ≥5 ideas and build 2 (for example the coexisting-attractor map, ε-sweep of KAM sections).
5. Critique loop, README with an ideas section, link check, commit via commit.sh.
