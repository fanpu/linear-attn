# outcome-basins: NOTES (living handoff)

## State (session 2)
- Piece A (*Four Roots*, ¼(xyz−1)²) complete and verified: D = 1.40±0.02 box counting over 10 zoom levels (1.3e8×), uncertainty D = 1.43 (full) / 1.72 (fringe), null η=0.005 D ≈ 1.0, res check OK, not riddled to ε=1e-13. Numbers in `cache/verify_fact3.json`, README §5.
- Piece B (XOR 2-2-1, η=1.2, seed-4 slice): hero 2048², 16 raw → 2 canonical; quotient removes 58% of between-solution boundary (D 1.26→1.14), total boundary D ≈ 1.68; uncertainty D = 1.78 raw / 1.79 canon (`cache/quotient_hero_xor_s4_e1.2.json`, `cache/uncert/xor_e1.2*.npz`).
- Session 2 added: `render_fact3.py splits` (hubble_sho, aurora_ember split variants, done); `compute_frames.py --part/--parts`; film renderer mats 480² XOR frames at 2× nearest on 1080².
- Deleted `cache/render_f3_*` (film PNG intermediates, regenerable via render_films.py) to keep cache < 5 GB.

## Running GPU jobs (launched session 2; logs in logs/)
- `prod_xor2.sh` (XOR null uncertainty η=0.3; η series 1024²) → logs/prod_xor2.log
- `prod_xor_zoom.sh` (6-level ×6 zoom into divergence fan, 1024², tag zX) → logs/prod_xor_zoom.log
- `prod_xor_film.sh 0|1|2` (η 0.6→1.26, 72 frames 480², tag xor_etafilm) → logs/prod_xor_film{0,1,2}.log
- `prod_fact3c.sh` (1e6-sample uncertainty tail, riddling) → logs/prod_fact3c.log
All are resumable (maps skip? no: maps recompute; frames skip existing files). Check with `tail -n 3 logs/prod_*.log`.

## Next
1. When done: `python analyze.py xor`; `python render_xor.py eta zoom`; `python render_films.py xor_etafilm raw 8` (+ canon, spectral).
2. Riddling tail from `cache/uncert/f3_e1.1_*_bigM.npz` → README RIDDLE_PLACEHOLDER.
3. Fill README placeholders (XOR_SECTION_PLACEHOLDER, XOR_VERIFY_PLACEHOLDER, WALL_XOR, RIDDLE_PLACEHOLDER), add split variants to gallery, link check.
GPU used: ~2.3 h slot time before session 2.
