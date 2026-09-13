# outcome-basins: NOTES (living handoff)

## State (session 2)
- Piece A (*Four Roots*, ¼(xyz−1)²) complete and verified: D = 1.40±0.02 box counting over 10 zoom levels (1.3e8×), uncertainty D = 1.43 (full) / 1.72 (fringe), null η=0.005 D ≈ 1.0, res check OK, not riddled to ε=1e-13. Numbers in `cache/verify_fact3.json`, README §5.
- Piece B (XOR 2-2-1, η=1.2, seed-4 slice): hero 2048², 16 raw → 2 canonical; quotient removes 58% of between-solution boundary (D 1.26→1.14), total boundary D ≈ 1.68; uncertainty D = 1.78 raw / 1.79 canon (`cache/quotient_hero_xor_s4_e1.2.json`, `cache/uncert/xor_e1.2*.npz`).
- Session 2 added: `render_fact3.py splits` (hubble_sho, aurora_ember split variants, done); `compute_frames.py --part/--parts`; film renderer mats 480² XOR frames at 2× nearest on 1080².
- Deleted `cache/render_f3_*` (film PNG intermediates, regenerable via render_films.py) to keep cache < 5 GB.

## Done in session 2
- fact3c riddling tail (1e6 samples): full α=0.28 (D 1.72), fringe α=0.21 (D 1.79), no flattening to ε=1e-14 → README filled.
- XOR null uncertainty η=0.3: α=0.923, D=1.08. XOR η series 1024² (maps eta_xor_*), sheets `eta_series_xor_{raw,canon}.png`. EoS: ≤4 px/1e6 exceed 2/η (η=0.8: 4, max 2.568 vs 2.5; η=1: 1; η=1.15: 4).
- XOR η film 72×480² (3 parts, ~11 min each), rendered raw/canon/spectral at 8 fps (TQ 1.6–4.1 for XOR).
- README: XOR gallery section written; split variants table added. Remaining placeholders: XOR_ZOOM_PLACEHOLDER, XOR_VERIFY_PLACEHOLDER, WALL_XOR.

## Session 2 finish
- XOR fan zoom zX done (36 min): box D = 1.82,1.83,1.79,1.76,1.77,1.77 (mean 1.79±0.03), all levels 18 labels, ~40% boundary px. `analyze.py xor` → cache/verify_xor.json, gallery/verify_xor.png. `render_xor.py zoom` → zoom_zX_{raw,canon,spectral}.png.
- README complete: all placeholders filled, link check clean. GPU total ≈ 3.9 h slot time.
- No background jobs running.

## Possible next steps (optional polish)
- XOR resolution check (fan L1 window at 513/1025/2049) — not done, stated in README.
- XOR fan zoom film (e.g. 540², 120 frames, 3 parts via compute_frames.py zoom --part) — would be the strongest B motion piece; ~2 h slot time.
- Large print crop of zoom_zX L2 in Spectral / hubble_sho at native 1024 → needs recompute at 2048+ for print.
