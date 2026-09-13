# ouroboros NOTES (living handoff)

## Environment / conventions
- CPU only (GPU saturated). Always `OMP_NUM_THREADS=2..4 OUROBOROS_DT=float32` (common.py reads OUROBOROS_DEV / OUROBOROS_DT; default cpu/float64). All caches so far are float32.
- Python: `P=/home/fzeng/ml/research/art/.venv/bin/python`, run from `/home/fzeng/ml/research/art/ouroboros`.
- **Edit files with the Edit tool only** — sed/python rewrites echo the whole file back into context (cost me ~15k tokens).
- Machine load ~35 on 20 cores (other agents); keep <= 3 of my processes.

## Code
- common.py: targets (ring 8-mode GMM sigma=0.08 r=1, spiral, Barnsley fern scaled into [-1.45,1.45]), CRN pools, batched GMM-EM (suff-stat covariance, REG_COVAR 1e-6), KDE LOO-CV bandwidth (per-chain golden-section over log10 h in [-3,0.3], checked vs 400-pt brute grid: <1%), sliced W2 (64 dirs, 2048 eval vs 2048 ref), hist overlap, ring mode mass.
- chains.py: run_batch(target, model, regime, lam[], n, seeds[], G, K, em_iters0=400, em_iters=10 warm-start, KDE_Q=256 LOO queries). Records per-gen metrics; with n_display records display samples and (GMM) disp_pi/mu/cov.
- compute_film.py `<target> <model> <n> <G> [--K] [--nd] [--every] [--regimes replace,anchored,accumulate] [--tag]` -> cache/film_*.npz (regimes: replace lam=0, anchored lam=0.25 replace, accumulate lam=0).
- compute_phase.py (lambda x n grid, multi-worker via part files + .lock; `--rev` for 2nd worker; assembles cache/phase_<t>_<m>_<regime>.npz when all parts exist; rerun same command to assemble).
- style.py (fonts P052/C059 URW, inks, paper texture, splat, LUTs via color-research/palettes.py, video() mp4+gif).
- render_film.py `<cache> dark|paper [--sub 3] [--gmax G] [--cmap klimt_gold] [--still g,...]`
- render_spiral.py `<cache> dark|paper|riso [--arms replace,accumulate] --G 200 --stride 4 --r 0.94 --dtheta 0.45 --tile 0.2`
- render_ridgeline.py `<cache ring gmm> <regime> night|paper --G 160 --step 2 --lo -1.6 --hi 0.7` (needs disp_mu)
- render_phase.py `ring gmm [--tag] [--G] [--tau 0.25] [--only seq,split]` (seq: log ratio sw2(G)/sw2(0) in batlow/lajolla/sepia; split: escape-time signed field with sd_spectral/aurora_ember/hubble_sho). Accumulate grid coarser -> nearest upsampled (declare).
- render_fern.py `<cache fern gmm> sheet|hero --gens 0,10,40,100,200` (stipple samples + engraved component ellipses; "Filix ouroborum" plate).
- peek.py: diagnostic contact sheet -> logs/.

## State (agent 2, 14:45)
- DONE: phase_ring_gmm_replace.npz assembled (97 lam x 65 n in [8,512], G80). Rendered replace-only maps: `render_phase.py ring gmm --regimes replace` -> gallery/phase_ring_gmm_replace_{sw2_*,split_*}.png (single panel uses 2:1 cells).
- DONE: film_phase_ring_gmm_replace_sd_spectral.mp4/.gif (`render_phase_film.py ring gmm`), fern_sheet_{sepia_ink,iron_gall}.png + fern_hero_replace_sepia_ink.png (cache/film_fern_gmm_n4096_all.npz = merged), spirals rerendered with new tone (`--wexp 1`, per-tile blur, guides dimmer; dark/paper --tile 0.18, riso two-arm --tile 0.09).
- RUNNING (nohup): accumulate phase workers logs/phase_gmm_acc_w{1,2}.log (w1 reruns to assemble cache/phase_ring_gmm_accumulate.npz; if both exit w/o assembling, rerun w1 command after deleting stale locks). Then: `render_phase.py ring gmm` (diptych) and `render_phase_film.py ring gmm --regime accumulate` optional.
- RUNNING: `verify_compute.py all` (logs/verify_compute.log): floor + seeds done; native window n=32..96 x2 seeds (~1.5 h, parts in cache/parts/verify_native_n*.npz, resumable). Then `verify.py` -> verify_results.txt + gallery/verify_boundary.png.
- verify so far: CRN exact (rows with same n_r identical, diff 0). Seeds 0-4 n128 G200: replace sW2 0.93+-0.24, modes 1,1,1,2,0; anchored 0.23, 8 modes; accumulate 0.15, 8 modes; true-sample floor 0.046+-0.013. Box counting replace map: D(1-8 cells)=1.53, smooth-only 1.11, phase-random null 1.46+-0.03, AAFT null 1.35+-0.04 (real rougher than both nulls at 100% quantile, but boundary-cell count 701 vs null 727/735), local slopes 1.3-1.8 (no power law, <1 decade). Native raster D(4-32px) ~1.15 -> noise-roughened, not fractal.
- TODO: README.md (full), final verify run, NOTES final.

## Caches (cache/, gitignored)
- film_ring_gmm_n128.npz (G200, with params), film_ring_kde_n128.npz (G120), film_spiral_kde_n512 (G100), film_spiral_gmm_n1024 K24 (G200)
- film_fern_gmm_n4096_p.npz: replace+anchored WITH params (K64, G200, nd 20000). film_fern_gmm_n4096.npz (all 3 regimes, NO params; accumulate slow, was at g175 at 14:05) — merge its accumulate/* keys into the _p file for the sheet (see merge one-liner used for toy: load both, d.update(accumulate keys), np.savez). Accumulate row then has no ellipses unless recomputed with params (~20 min CPU: `compute_film.py fern gmm 4096 200 --K 64 --nd 20000 --regimes accumulate --tag _pacc`).
- then fern kde n2048 G80 runs in the same job (logs/film_B.log).
- phase parts: cache/parts/phase_ring_gmm_replace_L97_0-1_G80_e10_s0_n*.npz (97 lambda x 65 n in [8,512], G80); 2 workers running at 14:05 (logs/phase_gmm_replace_w*.log), ~13/55 unique columns done.
- accumulate phase NOT yet run (killed for CPU): `$P compute_phase.py ring gmm accumulate --L 33 --N 33 --nmin 8 --nmax 512 --G 80 --bs 33` (+ a `--rev` worker). n=8,9 parts exist. Expensive (pool grows to 80n): expect 1-2 h with 2 workers. Delete stale `*.lock` without matching npz before resuming.

## Gallery so far
- ridgeline_ring_gmm_{replace_night,replace_paper,anchored_night,accumulate_night}.png — strong. Ridge slides left from 1 sigma to ~0.15 sigma by gen 150; share beyond 2 sigma 10.8% (gen0) -> 0.8% (gen10) -> 0.0% (gen50+). anchored stays ~5-9%, accumulate ~7.5%.
- spiral_ring_gmm_replace_{dark,paper}.png, spiral_ring_gmm_replace-accumulate_riso.png (rendering at 14:07). Critique: gen-0 tiles too faint (area-normalised weight (60/half)^2 -> try ^1), guide circles slightly too prominent; two-arm version needs tile <= 0.09.
- film_ring_gmm_{dark,paper}.mp4/.gif, film_ring_kde_dark.mp4 (rendering via logs/render_films.log).

## Findings so far
- GMM replace lambda=0 n=128: ring collapses to one needle-like component by g~120 (sw2 0.18 -> 1.0); lambda=0.25 keeps 8 modes but squashes them into needles (sw2 ~0.2 flat); accumulate stable (sw2 ~0.2).
- KDE (LOO bandwidth) EXPLODES instead of collapsing: each gen adds h^2 variance and h grows with spread -> runaway (sw2 16 at g120, n=128); lambda=0.25 also diffuses by g40; accumulate stable. Opposite failure mode to GMM — good README point.
- Fern GMM K32 n512 toy: replace -> a few blobs/needles by g40; anchored -> needles; accumulate holds the silhouette.
- Toy phase map (32x24, G40, n 16-1024): collapse region = small n & small lambda; at small n even lambda=1 has large sw2 (gen-0 fit error) -> render uses ratio / excess over generation 0 (CRN makes gen 0 identical across lambda at fixed n).

## TODO (next agent)
1. When phase replace assembled: run accumulate phase (above), then `render_phase.py ring gmm` (check the split seam, tau choice; maybe G=40/80 comparison). Consider a phase-map *film* over generations (map at g=1..80; histories are stored) — cheap, strong motion piece.
2. verify.py: (a) collapse metrics table over seeds 0-4 for film settings (sw2, modes, var_ratio at G) replace vs anchored vs accumulate; (b) sw2 noise floor (true samples vs reference); (c) boundary box-counting of the escape set in the phase map + resolution check (refine a window with `--lam0/--lam1/--nmin/--nmax` at 2x/4x) + null model (seed-averaged map, or thresholded iid noise with same autocorrelation). Expect honest negative: boundary is noise-roughened, not fractal; note lambda resolution floor = 1/n (n_r = floor(lambda n)).
3. Finish fern sheet + hero (render_fern.py) with full cache; spiral tweaks; KDE film; spiral_kde / spiral_gmm films optional.
4. README.md (BRIEF structure), captions measured vs aesthetic; references Shumailov 2024 Nature, Alemohammad 2023 (MAD), Bertrand 2023 (stability of iterative retraining), Gerstgrasser 2024 (accumulate avoids collapse). Precision: float32 CPU.
5. Commit via `../_shared/commit.sh ouroboros "..."`.
