# 04-lazy-rich-mup: PAUSED (token budget wind-down)

## Done (measured, cached, figures committed)
- NTK core + tests (`test_ntk_core.py`: torch.func NTK == explicit Jacobian; wide NTK -> closed form; linearized GD; toy grads).
- Width sweep m=64..16384 x3 seeds (`compute_ntk.py widths`): slopes dW -0.47, da -0.53, dK_init -0.53, net-vs-linearization -0.53,
  kernel change during training -0.92 (~1/m; sign-cancellation of first-order terms). Figures: `ntk_width_scaling.png`, `ntk_linearization.png`.
- Chizat alpha sweep m=1024, 9 alphas x2 seeds, CPU (`compute_ntk.py alpha 0|1`): dW ~ 1/alpha; kernel change flattens to ~alpha^-1/2 at
  alpha>=10 (hypothesis: ReLU activation flips; `check_flips.py` written, NOT run). Test error ~flat 14.5-15.3% except alpha=0.01. Figure `alpha_sweep.png` (needs label polish).
- Hero MP4 (`render_hero.py`, toy data `compute_toy.py hero`), toy widget (`widgets/toy.js`, selftest passes).
- Coordinate check (`coord_check.py`, SP blows up ~370x at width 2048, muP flat). Figure `coord_check.png`.
- muP explorer widget `widgets/mup.js` + exporter (selftest passes on partial data).
- post.md: hero, intro, sections 1 and 3 written; section 2 partially; 4-6, reproduce, references TODO.

## Interrupted (rerun from theory/)
- LM width sweep, 15/56 runs done (resumable, skips finished runs; one ckpt for mup_d512 lr3.1e-2):
  `for s in 0 1; do _shared/gpu_run.sh .venv/bin/python 04-lazy-rich-mup/train_lm.py sweep width $s 2; done` (run in background)
- Depth sweep (never started): `_shared/gpu_run.sh .venv/bin/python 04-lazy-rich-mup/train_lm.py sweep depth`
- Kernel-change decomposition: `.venv/bin/python 04-lazy-rich-mup/check_flips.py`
- Positive-readout kernel test: `DEV=cpu .venv/bin/python 04-lazy-rich-mup/compute_ntk.py posa`

## Next steps
1. Finish sweeps; `analyze_lm.py`, `render_mup.py` (+`--anim`), `export_widgets.py mup`.
2. `render_alpha.py --anim` (kernel alignment MP4); fix alpha_sweep label clipping.
3. Per-layer feature movement figure from `move` field in cache/lm/*.json.
4. Write sections 2 (results), 4, 5 (depth, feature movement), 6 (where it breaks: 1/m kernel, alpha^-1/2 flips, lazy not worse on CIFAR), reproduce, references (verified list in agent report: Lee et al. 3rd author Schoenholz; TP5 is NeurIPS 2021).
5. Render with `_shared/render_post.py --shot`, check console selftests, README.md.
