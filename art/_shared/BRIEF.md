# Shared brief for every art project agent

You are one of ~18 agents, each building one project from the two proposal docs:

- `/home/fzeng/ml/research/art/ml-art-directions.md` (main doc)
- `/home/fzeng/ml/research/art/ml-art-fractals.md` (fractals companion)

Read your project's section **and** §0, the media section, the verification protocol (fractals doc §11), and the blind-spots section of the doc it comes from. The corrections in your task prompt override the docs where they disagree.

## Goal

Your deliverable is a **gallery**: visualizations plus GIFs/videos of a real, measured ML phenomenon, rendered in **several styles** so the user can decide which one they like best. It is art first and educational second. Every visual choice is either a faithful rendering of a measured quantity or a declared aesthetic choice.

The user especially loves **fractal, repeating, self-similar, or intricate patterns**. Wherever your phenomenon genuinely has such structure (zooms, periodic windows, star polygons, nested cells, interference-like textures), lean into it. Never fake it. If you claim something is fractal, back it up with box counting across scales, a resolution check, and a null model (fractals doc §11).

## Where things go

Put everything in your own directory, `/home/fzeng/ml/research/art/<your-dir>/`. Do not touch any file outside it, except for reading the docs and shared data.

```
<your-dir>/
  README.md        # the accompanying writeup (see below)
  *.py             # code: compute scripts separate from render scripts
  cache/           # raw measured arrays (.npz/.pt), gitignored by pattern; keep < ~5 GB
  gallery/         # final PNG / GIF / MP4 (lossless PNG, never JPEG for data images)
```

Computing and rendering must be separate steps: save the raw measurements to `cache/`, then render every style from the cache. That way restyling never needs recompute.

## Environment

- Python: `/home/fzeng/ml/research/art/.venv/bin/python` (3.12, torch 2.14+cu130, torchvision, numpy, scipy, scikit-learn, matplotlib, pillow, imageio, imageio-ffmpeg, colorcet, cmcrameri, pandas, tqdm, transformers 5.15). **Do not** use or modify the repo-root `.venv`, and do not edit any `pyproject.toml`.
- If you truly need another package, install it under a lock: `flock /home/fzeng/ml/research/art/_shared/.pip.lock uv pip install --python /home/fzeng/ml/research/art/.venv/bin/python <pkg>`, then mention it in your report. Prefer implementing small things yourself (Lanczos, TwoNN, weight matching, and so on).
- Datasets are already downloaded in `/home/fzeng/ml/research/art/data` (torchvision layout: `CIFAR10`, `MNIST`, `FashionMNIST`). Use `root='/home/fzeng/ml/research/art/data', download=False`.
- HF models are cached in `~/.cache/huggingface/hub` (for example `Qwen/Qwen3-0.6B`). Load them with `local_files_only=True`.
- `ffmpeg` and ImageMagick `convert` are at `/usr/bin`.
- You can read papers with the WebFetch/WebSearch tools (load them via ToolSearch). Check the key paper's actual setup before reproducing it. Some citations in the docs are 2025 papers that have not been verified; if one doesn't exist or says something different, say so in your README.

## The GPU is shared (NVIDIA GB10, ~120 GB unified CPU+GPU memory, 20 CPU cores)

- Any GPU job longer than about 2 minutes, or using more than 4 GB, must run through the slot limiter: `/home/fzeng/ml/research/art/_shared/gpu_run.sh <cmd...>`. It blocks until one of 8 slots is free, so run it with Bash `run_in_background` and watch progress through a log file rather than a foreground call that may time out.
- In GPU scripts, call `torch.cuda.set_per_process_memory_fraction(0.10)` (≈12 GB) and chunk big sweeps to fit.
- Set `OMP_NUM_THREADS=4` for CPU-heavy rendering, and don't spawn more than 4 worker processes.
- Build a toy version first (for example a 128² grid or a few hundred steps), look at it, and only then scale up what shows structure. Aim for a total GPU budget of a few hours, not days. Use float64 wherever deep zooms or chaos make float32 precision matter, and state the precision floor in the README.
- Use `nvidia-smi` and `free -g` if things seem slow. Other agents are running.

## Style variants (required)

For each main piece, render **at least 3–4 distinct styles**, for example:

- dark ground with a perceptually uniform map (magma/inferno/viridis/cividis, cmcrameri batlow/oslo/lajolla, colorcet fire/bmy)
- single-ink / line-only on paper white, in the style of plotter art (Molnár/Mohr) or survey sheets
- two- or three-spot-colour risograph-style, with optional deliberate misregistration
- a scientific-plate idiom (spectral plate, geological cross-section, observatory archive)
- cyclic maps (twilight, colorcet cyclic) only for genuinely cyclic data such as phase or residues

**User favourite, required wherever it fits:** the **Sohl-Dickstein "Spectral" style** (matplotlib `Spectral`, ColorBrewer), as in his trainability-fractal figures (github.com/Sohl-Dickstein/fractal). A two-sided quantity (converge vs diverge, λ<0 vs λ>0, ordered vs chaotic, f<c vs f>c) is split at its boundary. Each side is rank/CDF-normalized separately and mapped onto one half of Spectral so the two dark ends (deep red #9e0142 and purple #5e4fa2) meet at the boundary: one side runs purple → blue → green → pale yellow, the other deep red → orange → pale yellow. The boundary becomes a sharp dark seam, and the smooth interiors glow in pastel gradients. Check his colab for the exact normalization. Declare it in captions. It's a declared aesthetic mapping of a signed quantity. For purely sequential data it can create false bands, so use it there only as a labelled variant.

Never use jet/rainbow on sequential data, because false banding looks like structure. Categorical basin maps need a thought-through categorical palette, and that palette is a declared choice. Vary more than colour: composition, line weight, small multiples versus a single large plate, zoom sequences.

**Motion:** wherever the phenomenon unfolds in time (training steps, depth, zoom, sequence length), make an MP4 (H.264, `-pix_fmt yuv420p`, 1080p or square 1080², slow and silent) and a smaller GIF (under ~15 MB; palettegen/paletteuse or gifski-quality dithering). Final stills should be high-resolution PNG (≥2000 px on the long side for hero images).

## README.md (required)

This is the accompanying document. Markdown with embedded HTML is fine: `<img>` grids, `<video src="gallery/x.mp4" autoplay loop muted playsinline width="...">`, `<table>`/`<figure>` layouts. Use relative paths. Contents:

1. **Title and a one-sentence hook.**
2. **The phenomenon**, explained plainly with the key equations.
3. **Hero image(s)** near the top.
4. **Gallery** of every style variant, grouped, each with a one-line caption saying what is measured and what is aesthetic.
5. **What was computed**: model, data, optimizer, steps, grid sizes, precision, seeds, wall time. Anyone should be able to reproduce it (give the exact commands).
6. **Verification / honesty**: did the phenomenon actually show up in *your* runs? Include the metrics. For fractal claims, give the box-counting fit with ε range, the resolution check, and the null model. List what didn't work and any negative results; a clean negative is a legitimate result.
7. **Caveats** a viewer should know, adapted from the docs' blind spots and your task prompt.
8. **References**.

## Don'ts

- **Commit regularly** with `/home/fzeng/ml/research/art/_shared/commit.sh <your-dir> "<message>"`: after each working script, each finished piece or style batch, and at least every ~30 minutes of progress. Use messages like `art/<your-dir>: add hachure plates`. The script commits only your directory, takes a lock (other agents commit concurrently), and skips files over 20 MB; list those in your README. Never run `git add`/`git commit`/`git stash`/`git checkout` yourself, never create branches, and never push.
- Don't touch other agents' directories.
- **Checkpoint for cheap resumption:** keep a short `NOTES.md` in your directory, a living handoff: state, key numbers, decisions, next steps, exact resume commands. Update it at every commit, so a fresh agent can continue from `NOTES.md` + `_shared/TASKS.md` without your history. Avoid dumping huge logs or images into your own context: tail logs, downscale images before viewing them.
- **Palettes:** `/home/fzeng/ml/research/art/color-research/palettes.py` (`import palettes as P; P.render_split(M, 'aurora_ember')`; see its README) provides many artistic palettes and boundary-split pairings. Use 1–2 as extra variants.
- Don't leave long-running background processes when you finish.
- Don't claim success you didn't verify: look at your own images (use the Read tool on the PNGs) before calling a style done.

## Final report (your last message, ≤ 250 words)

Include: whether the phenomenon appeared (with the key number), the 3 strongest image/video paths, anything fractal/self-similar found (with dimension estimates), negative results or surprises, doc claims that turned out wrong or unverifiable, and GPU time used.
