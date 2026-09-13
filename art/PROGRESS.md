# ML art projects: progress log

**Status as of 2026-09-13 ~00:55 (paused).** All 18 project agents were launched at ~00:35 and hit the API session limit together at ~00:50, about 15 minutes in. Their leftover GPU/CPU processes were killed by hand, and the GPU is idle.

**No project has a `gallery/` or `README.md` yet.** What exists is compute code (often tested), paper reads, and a few toy runs.

**Update:** resumed with at most 5 agents at a time; the other agents keep their context and continue where they stopped.
- Running: attention-textile, trainability-fractal, depth-roughness, gd-bifurcation, grokking. Each has an added brainstorm + self-critique step (≥5 ideas, iterate 2 rounds on the top pieces, "Ideas explored" README section).
- Stopped by user interrupt (cancelled; needs a fresh launch if wanted): loss-landscape, mode-connectivity.
- Queue: game-chaos → decode-map → signal-propagation → outcome-basins → diffusion-basins → edge-of-stability → neural-collapse → hessian-spectrum → weight-spectrum → scaling-dimension → ouroboros.

## Shared setup (done, reusable)

| What | Where |
|---|---|
| Proposal docs | `ml-art-directions.md`, `ml-art-fractals.md` |
| Agent brief (deliverables, style rules, GPU etiquette, README spec) | `_shared/BRIEF.md` |
| GPU slot limiter (8 slots) | `_shared/gpu_run.sh` |
| Python env (torch 2.14+cu130, torchvision, scipy, sklearn, imageio, colorcet, cmcrameri, transformers 5.15) | `.venv/` (separate from the repo-root `.venv` so its pins stay untouched) |
| Datasets: CIFAR-10, MNIST, FashionMNIST | `data/` |
| Ignore rules | `.gitignore` (`.venv/`, `*/cache/`, slot locks) |

The per-project task prompts, including each project's corrections to the proposal docs, live in the conversation that launched them. The key corrections are summarized per project below, so a relaunch can reuse them.

## Per-project state

Legend: ⬜ nothing · 🟨 code written · 🟧 code + partial/toy results · 🟥 killed mid-run

| # | Directory | Source | State | What exists | Notes / partial findings |
|---|---|---|---|---|---|
| 1 | `edge-of-stability/` | main §1 | 🟨 | `eos_train.py` (Cohen et al. fc-tanh 3072-200-200-10, full-batch GD, dense sharpness logging) | Not yet run. |
| 2 | `grokking/` | main §2 | 🟧 | `train.py` (ensemble of Nanda-style 1-layer transformers, p=113, float64 log-softmax) | Toy run started (`cache/toy/`), no result recorded. |
| 3 | `neural-collapse/` | main §3 | 🟥 | `train_nc.py` (ResNet18, Papyan recipe, NC1–NC4 logging) | Toy 10-class run (width 64) got 3/350 epochs: NC1 1.78, train acc 0.72. Checkpoint in `cache/toy10/`. |
| 4 | `loss-landscape/` | main §4 | 🟥 | `common.py` (Li et al. ResNets ± shortcuts, filter-norm dirs), `train.py`, `landscape.py` (resumable 2D/1D surfaces) | 4 runs killed at 60-epoch schedule: ResNet-20 ep17 (79.1% test), ResNet-20-noshort ep18 (63.4%), ResNet-56 ep8 (71.5%), ResNet-56-noshort ep9 (39.4%). Checkpoints in `cache/ckpt/`. `train.py` has **no resume**, so restart from scratch (~35 min each when 4 run in parallel) or add resume. |
| 5 | `mode-connectivity/` | main §5 | 🟧 | `common.py` (weight matching, REPAIR, Bézier curves), `compute_hero.py` | **MNIST toy (width-512 MLP, 3 epochs) gave real results:** naive-lerp train barrier 1.44 → weight-matched 0.038 → matched+REPAIR 0.006; Bézier ≈0. Barrier vs epoch: ~0 at init, grows, then matched barrier falls as training proceeds (the LMC-is-emergent story). Killed during the 2-D planes step. Results are only in `cache/hero_mnist_toy.log` (npz not written). |
| 6 | `signal-propagation/` | main §6 + fractals §3 | 🟨 | `sp_core.py` (CRN deep random nets over the (σ_w, σ_b) grid, mean-field quadrature, box counting) | Found the finite-width paper: D'Inverno et al. 2025, code at github.com/jon-dong/fractal-deep-info-prop, whose conventions it follows. Not yet run. |
| 7 | `hessian-spectrum/` | main §7 | 🟧 | `common.py` (HVP/GN-VP, Lanczos, SLQ, Papyan decomposition, float64), `train.py`, `analyze.py` | Toy MNIST MLP trained (3 epochs, 7 checkpoints); one spectrum computed (`cache/spectra/toy_mnist_mlp/step_001404.npz`), not yet inspected. |
| 8 | `weight-spectrum/` | main §8a | 🟧 | `train.py` (MLP/MiniAlexNet, ESD of WᵀW/N at log-spaced checkpoints) | Toy run started; only step-0 matrices saved. |
| 9 | `attention-textile/` | main §8b | 🟧 | `toy_common.py`, `train_toy.py` (2-layer attn-only), `compute_qwen.py` | **Qwen3-0.6B part finished** (`cache/qwen_main.npz`, `qwen_book.npz`, 295 MB). Induction heads found: top L20H14 (0.68), L21H8 (0.66), L6H11 (0.61); null max 0.008. Loss on repeats 14.4 → 0.92 → 0.41. Pattern book of 18 variants includes Thue–Morse, Fibonacci-word, and nested repetition sequences. Toy training not yet run. **Ready to render.** |
| 10 | `trainability-fractal/` | fractals §1 | 🟧 | `tfractal.py` (batched float64 reimplementation of Sohl-Dickstein's colab, manual grads), `test_core.py` | 128² tanh toy grid saved (`cache/toy_128_tanh.npy`); gradient-check results weren't recorded. |
| 11 | `gd-bifurcation/` | fractals §2 | 🟨 | `common.py` (vectorized float64 GD maps + logistic-map null) | The 3 cited papers (2210.03294, 2502.20531, 2509.25351) were downloaded to `cache/`, so those arXiv IDs exist. |
| 12 | `diffusion-basins/` | fractals §4 | 🟧 | `common.py` (VP diffusion, PF-ODE), `toy.py` (2D GMM + analytic score + MLP ε-net) | Toy ring-8 model trained (`cache/toy_ring8.pt`). |
| 13 | `outcome-basins/` | fractals §5 | 🟨 | `engine.py` (torch.compile'd float64 batched GD, manual grads), `scratch/` toys (XOR 2-2-1, xy=1) | No saved results. |
| 14 | `decode-map/` | fractals §6 | 🟥 | `qwen.py` (hand-written fp32 Qwen3 with static KV cache, no padding), `test_model.py`, `sample.py` (inverse-CDF sampling with shared uniforms, HF-style repetition penalty) | 64², L=32 toy sweep killed before writing output. |
| 15 | `ouroboros/` | fractals §7 | 🟥 | `common.py` (targets, CRN pools, batched GMM/KDE, metrics), `chains.py` | `cache/targets_check.png` shows target distributions. GMM/KDE toy killed mid-run. |
| 16 | `depth-roughness/` | fractals §8 | 🟨 | `common.py` (Di Lillo et al. model on S², kernel iteration) | Paper source pulled to `cache/paper/`. The agent noted the paper writes the weight scale as n^{-1/2} where eq. (2.3) needs variance 1/n. |
| 17 | `game-chaos/` | fractals §9 | ⬜ | only empty `cache/ gallery/ papers/` dirs | Nothing to resume. |
| 18 | `scaling-dimension/` | fractals §10 | 🟧 | `idlib.py` (TwoNN, MLE, correlation integral; GPU brute-force kNN), `test_idlib.py`, `ts_train.py` (batched teacher/student sweep) | One toy sweep saved (`cache/toy/toy1.npz`), not yet inspected. |

## Resuming cheaply

What made the first attempt expensive was 18 Opus agents running concurrently, each reading long docs and papers. Options, roughly from cheapest:

1. **Render what's already computed**: attention-textile's Qwen tapestry (ready now) and mode-connectivity's MNIST toy (rerun `compute_hero.py`, ~10 min). Output reaches the gallery with little agent time.
2. **Relaunch in waves of 3–5**, prioritizing the projects the user cares most about (fractal/repeating structure): trainability-fractal, gd-bifurcation, depth-roughness, game-chaos, grokking (star polygons), decode-map. Tell each agent to **reuse the existing code in its directory** rather than start over.
3. Use a cheaper model (Sonnet) for render-heavy or straightforward projects (weight-spectrum, hessian-spectrum, neural-collapse, loss-landscape), and keep Opus for the subtle fractal-verification ones.
4. Long training (loss-landscape ResNets, neural-collapse 350 epochs) can run as plain background shell jobs through `_shared/gpu_run.sh` with no agent attached; agents only need to come back to render.
