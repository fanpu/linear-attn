# §8 Combed — M1 report (flows and memorisation numbers)

**Status:** DONE_WITH_CONCERNS. Every acceptance item is met. The concerns are about how to read the numbers (see Risks 1–3), not about missing work.

## What was done
- `art/combed/combed_common.py`: the trefoil x(s) = (sin s + 2 sin 2s, cos s − 2 cos 2s, −sin 3s)/3 with nested N ∈ {16, 64, 256, 1024, 4096}. s ~ U[0, 2π) from `default_rng(0)`, and each smaller set is a prefix of the 4096 draws. Also here: the closed-form field, `VelocityMLP`, RK4 (on a uniform grid and on an arbitrary grid), and the Gu et al. criterion.
- **Closed-form field**, checked against Bertrand et al. arXiv:2506.03719, Prop. 1 / eq. 6: û*(x,t) = Σᵢ λᵢ(x,t)(x⁽ⁱ⁾ − x)/(1 − t), with λ = softmax_j(−‖x − t x⁽ʲ⁾‖²/(2(1 − t)²)). Time runs from t = 0 (noise N(0, I)) to t = 1 (data). This is exactly the plan's formula. Computed in float64.
- **Unit tests** (`art/combed/test_combed.py`, 7 pass):
  - The continuity equation ∂ₜp + ∇·(p v*) = 0, checked by central finite differences (h = 1e-5, float64) at 20 random (x, t). Points were drawn where p_t is non-negligible, with t ∈ [0.05, 0.95]. Worst relative residual 9.5e-8, against a 1e-4 threshold.
  - eq. 6 against a direct sum.
  - The sets are nested prefixes.
  - RK4 is exact on a one-point (straight-line) flow, on both grids.
  - The criterion on hand-made cases, and the MLP shape.
- **Memorisation criterion:** Gu et al. arXiv:2310.02664, credited there to Yoon et al. 2023. A sample counts as memorised if its ℓ2 distance to the nearest training point is below **1/3** of the distance to the second nearest. The memorised fraction is the share of the 20k samples that pass.
- **Trained fields:** one MLP per N.
  - Architecture: input [x, sin/cos(f·t) at 16 frequencies geometric in [1, 1000]], then 4 hidden layers × 256 with SiLU, then R³. 207,363 parameters; the width is the same at every N.
  - Loss: conditional flow matching, ‖v_θ(x_t, t) − (x₁ − x₀)‖², with t ~ U[0,1].
  - **Budget:** 20,000 Adam steps, batch 1024, lr 1e-3 with cosine decay to 0, float32, torch seed 0, identical for every N.
- **Sampling:**
  - 20,000 noise seeds (`default_rng(1)`), shared by all 10 fields.
  - RK4 with 256 steps on t ∈ [0, 1 − 10⁻³], then repeated with 512 steps. Both fields were integrated in float64 (the MLP was cast to double).
  - Trajectories are cached at 64 time points, uniform in step index.
- **Extra diagnostics** (the reasons are under Decisions):
  - `converge`: a 1024/2048-step order check on the 256 worst and 256 random seeds.
  - `limit`: continues integration from 1 − 10⁻³ to 1 − 10⁻⁶ with 96 RK4 steps geometric in (1 − t), repeated with 192.
  - A null: 20k fresh points on the continuous knot, passed through the same criterion.
  - The distance from each x̂ to the continuous knot.
- **Previews** (matplotlib, all viewed with the Read tool), in `art/combed/cache/preview/`:
  - `hair_N{16,64,256,1024,4096}.png`: closed-form vs trained, 1500 hairs coloured by t, plus a panel showing only t ≥ 0.6.
  - `endpoints_all.png`
  - `memorised_vs_N.png`

## Device
**CPU, declared.** At launch the GPU queue was held by `autonomous/`, and the models are tiny. I used at most 4 threads in total, mostly two 2-thread processes. Wall clock was about 80 min of mostly unattended compute, longer than the ~1 h estimate: the closed-form N = 4096 run took 35 min at 2 threads. `gpu1.sh` was not used, and no packages were installed.

## Commands (from `art/combed/`)
```
../.venv/bin/python -m pytest -q test_combed.py
OMP_NUM_THREADS=2 ../.venv/bin/python compute.py sample closed --threads 2   # logs/sample_closed.log
OMP_NUM_THREADS=2 ../.venv/bin/python compute.py train --threads 2           # logs/train.log
OMP_NUM_THREADS=2 ../.venv/bin/python compute.py sample mlp --threads 2      # logs/sample_mlp*.log
OMP_NUM_THREADS=2 ../.venv/bin/python compute.py converge closed|mlp --threads 2
OMP_NUM_THREADS=2 ../.venv/bin/python compute.py limit closed|mlp --threads 2
../.venv/bin/python compute.py summary     # cache/summary.json, logs/summary.json
../.venv/bin/python preview.py
```

## Memorised fraction vs N (criterion d1 < d2/3, 20k samples)
The headline columns (**bold**) measure x̂ = x + (1 − t) v(x, t) at the declared stop t = 1 − 10⁻³, which amounts to one Euler step to t = 1. For v*, x̂ is the posterior mean E[x₁ | x_t].

| N | **closed-form** | closed raw endpoint | closed, t = 1−10⁻⁶ | **trained MLP** | MLP raw | MLP, t = 1−10⁻⁶ | null: fresh knot points | MLP x̂ median dist. to knot | MLP final excess loss |
|---|---|---|---|---|---|---|---|---|---|
| 16 | **0.954** | 0.881 | 1.000 | **0.704** | 0.700 | 0.704 | 0.263 | 3.9e-3 | 0.009 |
| 64 | **0.988** | 0.962 | 1.000 | **0.434** | 0.430 | 0.434 | 0.367 | 5.3e-3 | 0.026 |
| 256 | **0.909** | 0.762 | 1.000 | **0.183** | 0.174 | 0.183 | 0.313 | 5.4e-3 | 0.015 |
| 1024 | **0.719** | 0.379 | 0.999 | **0.049** | 0.044 | 0.049 | 0.333 | 5.6e-3 | 0.006 |
| 4096 | **0.471** | 0.061 | 1.000 | **0.005** | 0.005 | 0.005 | 0.345 | 5.5e-3 | 0.005 |

- **Closed-form x̂** lies about 6e-6 from the knot at every N.
- **Why closed-form falls short of 100% at N ≤ 64:** every miss sits between training points 7 and 14. The seed-0 draw put them only 1.5e-3 apart, which the stop cannot resolve.
- **Why closed-form drops at N ≥ 1024:** the declared stop leaves a residual width of 10⁻³, larger than the median point spacing (2.9e-3 at N = 1024, 8.3e-4 at N = 4096). x̂ therefore averages neighbours. Only 7% of samples have a largest softmax weight above 0.99 at N = 4096. In the t → 1 limit the field memorises 99.9–100%.
- **Trained MLP:** the memorised fraction falls from 70% to 0.5%.

## Step-doubling error (256 vs 512 RK4 steps, raw endpoint, all 20k seeds)
| N | closed max | closed p99.9 | MLP max | MLP p99.9 |
|---|---|---|---|---|
| 16 | 3.1e-3 | 1.2e-3 | 2.8e-1 (one seed changes basin; 9/20k change nearest point) | 3.9e-3 |
| 64 | 2.9e-3 | 1.1e-3 | 1.2e-2 | 2.9e-3 |
| 256 | 9.0e-3 | 1.8e-3 | 1.3e-2 | 2.6e-3 |
| 1024 | 1.4e-2 | 2.4e-3 | 8.6e-3 | 3.8e-3 |
| 4096 | 4.2e-3 | 1.8e-3 | 4.6e-3 | 1.9e-3 |

- **Median gap:** closed-form 3e-10 to 1e-4; MLP about 2.5e-5.
- **Order check, MLP (worst 256 seeds):** the gap shrinks about 300× from 256→512 to 512→1024, then about 40–90× per further doubling. The MLP is converged at 512 steps.
- **Order check, closed-form (worst seeds):** the gap shrinks only about 2–10× per doubling; the worst seeds still differ by 3e-4 to 5e-4 between 1024 and 2048 steps. On random seeds the gap shrinks about 5–10× per doubling, reaching ≤ 7e-5.
- **Cause of the slow closed-form convergence:** seeds that commit to a basin late, near t → 1, where the softmax sharpens like 1/(1 − t)². The errors are about 1e-3 on a picture whose scale is about 2, so they do not show in the hair.
- **Tail integration (1−10⁻³ → 1−10⁻⁶), 96 vs 192 steps:** max gap 7e-9 (closed-form), 4e-13 (MLP).

## Decisions (all logged in `art/combed/NOTES.md`)
1. **CPU instead of GPU**, for the reasons under Device.
2. **Headline measure is x̂, not the raw endpoint.** At the stop, the raw endpoint sits about 2e-3 from the training point it reached, which is comparable to the spacing at large N. Scored on raw endpoints, the criterion would report "not memorised" because of the stop, not because of the field. The raw fraction is reported alongside.
3. **`limit` stage (t → 1 − 10⁻⁶)**, to separate the field's behaviour from the declared stop.
4. **Null of fresh on-knot points, plus distance to the knot.** On a 1D curve the 1/3 criterion calls a perfect generaliser "memorised" 26–37% of the time. Gu et al. designed it for high-dimensional images.
5. **Order check on worst and random seeds** in addition to the plain step-doubling maximum.
6. **Seeds:** data 0, noise 1, null 7, probe set 99, torch 0. Trajectories are stored uniform in step index.

## Risks and open issues
1. **The trained fraction reads below the null from N = 256.** The MLP's samples form a knot thickened by about 5e-3 (p90 about 1.2e-2), roughly the same at every N and set by approximation error (excess loss 0.005–0.026). Once that blur exceeds the point spacing, "not memorised" means "blurred along or off the knot", not "new points exactly on the knot". The M3 film caption should print the null beside the fraction, or M2/M3 should add an along-curve coverage measure. The spec's "tufts melting into a continuous knot" does hold visually (`hair_N4096.png` and `endpoints_all.png`).
2. **The declared stop 1 − 10⁻³ smooths the closed-form flow at N ≥ 1024.** The hair diptych at N = 4096 will show a nearly continuous closed-form knot too. It needs either the limit endpoints or a caption. I recommend rendering endpoints from `cache/limit_closed_N*.npz` (`xhat_tail96`) for the closed-form field.
3. **Capacity and budget are fixed, not swept** (the spec's pitfall). The MLP's memorised fraction at N = 16 (70%) depends on the 20k-step budget and width 256, so it should not be attributed to N alone.
4. **The seed-0 data contain a near-duplicate pair at 1.5e-3.** It is visible as one merged tuft at N = 16; the pair was kept rather than re-seeded.
5. **Cache is 219 MB, gitignored.** The N = 16 basin volume (M2) is not computed yet; with closed-form RK4 at 128³ it should take about 10 min on CPU.

## Files
- Code: `art/combed/combed_common.py`, `art/combed/compute.py`, `art/combed/preview.py`, `art/combed/test_combed.py`
- Log and handoff: `art/combed/NOTES.md` (committed); `art/combed/logs/` (gitignored, local only, including `summary.json`)
- Cache: `art/combed/cache/flow_{closed,mlp}_N*.npz`, `limit_*`, `converge_*`, `mlp_N*.pt`, `summary.json`, `preview/`
