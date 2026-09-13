# outcome-basins: PAUSED (coordinator stop, user token budget)

## Done
- Engines: `cuda_gd.py` (float64 per-pixel CUDA kernels, 20-30x faster than torch loop), `engine.py`, `problems.py`.
- **Piece A, Four Roots** (1/4(xyz-1)^2, 4 sign-class solutions), complete: 8 hero maps x 5 styles (newton, ink, riso, darktime, Spectral), triptych (raw / mod permutation / mod all = Spectral), eta series, 10-level zoom sheets (3 styles), eta film and 1.3e8x zoom film (newton + Spectral, MP4 + GIF).
- Verification A (`gallery/verify_fact3.png`, `cache/verify_fact3.json`): box-counting D = 1.40 +- 0.02, steady over 10 zoom levels (8 decades); boundaries between solutions D = 1.37, converge/diverge edge D = 1.26; uncertainty exponent D = 1.43 (full window), 1.72 (fringe); null (eta = 0.005) D = 1.04 / 0.96; resolution check 1025/2049/4097 consistent; not riddled; EoS: max sharpness <= 2/eta at every eta.
- **Piece B, XOR 2-2-1**: hero 2048^2 at eta = 1.2 (16 raw classes -> 2 canonical), diptych + 5 styles rendered. Quotient analysis (`cache/quotient_hero_xor_s4_e1.2.json`): 58% of between-solution boundary removed by canonicalisation (D 1.26 -> 1.14), but the overall boundary (with diverged / plateau runs) stays D ~ 1.68. Uncertainty exponent: raw alpha = 0.22 (D = 1.78), canon alpha = 0.21 (D = 1.79); `cache/uncert/xor_e1.2*.npz` saved. Hero max sharpness 1.683 vs 2/eta = 1.667 (2 px of 3M above).
- README.md drafted: A sections complete; B sections are placeholders (`XOR_SECTION_PLACEHOLDER`, `XOR_VERIFY_PLACEHOLDER`, `WALL_XOR`, `RIDDLE_PLACEHOLDER`).

## Next (not done)
1. XOR null uncertainty (eta = 0.3), eta series (768^2), zoom (zX), eta film (360^2 x 60): remaining lines of `prod_xor.sh` (killed during the null uncertainty). Remove the already-finished first 3 lines before re-running.
2. `prod_fact3c.sh` (10^6-sample uncertainty tail, riddling test): never got a GPU slot.
3. `python analyze.py xor`; `python render_xor.py eta zoom`; `python render_films.py xor_etafilm raw 12` (and `canon`).
4. Fill README placeholders (XOR gallery table, verification table with the numbers above, wall time: XOR hero 29 min, uncertainty 11 min each), then run a link check on the README.

## Resume
`cd art/outcome-basins; G=../_shared/gpu_run.sh`; edit `prod_xor.sh` to drop its finished lines; `$G ./prod_xor.sh`; `$G ./prod_fact3c.sh`; then steps 3-4.
GPU used so far: about 2.3 h of slot time (A 57 min, B 51 min, toys ~20 min).
