# Shared brief for every theory/ project agent

You are one of 8 agents. Each one turns a section of `/home/fzeng/ml/research/theory/PROJECT_IDEAS.md` into a finished piece. Up to 3 agents run at once. Read your section of that doc and its "Ground rules" and "Sizing" sections. Where your task prompt disagrees with the doc, the task prompt wins.

## The deliverable: a blog post

Write **one pedagogically excellent blog post**: `post.md`, markdown with embedded HTML for figures, videos, and interactive widgets. It reproduces the classical result and then builds on it. The results in it are measured by you, on this machine.

**Audience:** someone with a general CS/ML background. They know gradient descent, linear algebra, and what a neural net is. They do *not* know this corner of theory. Take them from "why should I care" to "I understand the result, I've seen it happen, and I've seen where it breaks."

### Writing

- **Open with a hook.** Lead with a striking visual and the question it raises, not with definitions.
- **Build intuition before formalism.** Introduce each equation after the reader already wants it, then explain what every symbol means in words. Use derivation sketches, not full proofs; link out for rigor.
- **Show every claim.** Each key claim gets a figure or widget right next to it. Theory curves are overlaid on measured data wherever the theory makes a prediction.
- **Have a "where it breaks" section.** Finite size, violated assumptions, discrepancies, negative results. This is often the most interesting part, so be honest.
- **Make the "build on" part real.** It is a genuine experiment with a stated question, method, result, and caveats. Label clearly what is established literature and what is new measurement.
- **Stay tight and concrete.** Prefer short paragraphs. No filler, no hype. Use a friendly, curious, precise voice (think Distill, or 3Blue1Brown in prose).
- **End with** a short "Reproduce it" section (exact commands) and a references list.
- **Check every citation** with WebSearch/WebFetch (load them via ToolSearch). The ideas doc was written from memory. If a paper's setup, numbers, or title differs from the doc, follow the paper and note the correction in your final report.

### Visuals: the bar is *beautiful and inspiring*

The user especially cares about this. Every figure should look like it belongs in a great Distill article or a science-museum exhibit, not a default matplotlib dump.

- **Invoke the `dataviz` skill** (Skill tool) before writing your first plotting code, and follow it.
- **Design one visual system per post:** a cohesive palette, typography, and line weights shared by static plots, GIFs, and widgets. Use a dark cinematic look for hero pieces if it suits; clean light plates for explanatory plots.
- **No jet/rainbow on sequential data.** Use perceptually uniform maps (`cmcrameri`, `colorcet`, or viridis/magma-family) or a hand-built palette. Label directly on the plot instead of using legends where possible. Remove chartjunk and use generous whitespace.
- **Quantity:** aim for roughly **8–15 figures**, including **at least 3 animations** (GIF, and/or MP4 via `<video autoplay loop muted playsinline>`), and **at least 2 interactive widgets**.
  - The **hero animation** goes at the very top and should make someone want to read on.
  - Animations should show the *process* (training dynamics, a sweep unfolding, a trajectory), be smooth (≥20 fps where motion matters), and loop cleanly.
  - GIFs under ~12 MB each (ffmpeg palettegen/paletteuse). MP4 as H.264 `-pix_fmt yuv420p`. Hero stills ≥1600 px wide.
- **Widgets** let the reader *play with the theory*: drag a slider for init scale, width, or γ and watch a curve or trajectory respond. Precompute data where the math isn't cheap in JS; implement it live in JS where it is (closed forms, small simulations). Hand-written JS + `<canvas>`/SVG is fine.
- **Look at everything you make.** Use the Read tool on the PNGs, on GIF frames extracted with ffmpeg, and on the page screenshots. Iterate at least twice on the hero and on the widgets: critique composition, colour, legibility, and whether the image teaches the point, then improve.

### Technical rules for the post (it must work offline, from `file://`)

- **Render it** with `/home/fzeng/ml/research/theory/.venv/bin/python /home/fzeng/ml/research/theory/_shared/render_post.py <dir>/post.md --shot`. This writes `post.html` (base CSS from `_shared/post.css` + vendored KaTeX) and page screenshots in `<dir>/_preview/`. Read the docstring at the top of `render_post.py`.
- **Math:** `$inline$` and `$$display$$`.
- **Styling:** add post-specific CSS in a `<style>` block. Use the classes `.widget`, `.wide`, `.full`, `figure`/`figcaption`, `.callout`.
- **Widget JS goes in `widgets/*.js`,** loaded with `<script src="widgets/x.js"></script>`. Don't write long inline scripts in markdown.
- **Load data through script tags, not `fetch()`.** `fetch()` of local JSON is blocked under `file://`. Put precomputed data in `widgets/data_x.js` as `window.X_DATA = {...}` and load it with a script tag. Keep each data file under ~3 MB; downsample or quantize.
- **No CDNs or network at view time.** If you really need a JS library, vendor it under `_shared/vendor/` and say so in your report.
- **Check for JS errors** with the headless-Chrome command in the `render_post.py` docstring. Zero console errors before you finish. Check interactive widgets both in screenshots and by exercising their code paths (e.g. a small self-test when the URL has `#selftest`).
- **Use relative paths only.**

## Where things go

Put everything in `/home/fzeng/ml/research/theory/<your-dir>/`. Do not modify files outside it; the only exceptions are adding a vendored JS lib under `_shared/vendor/` and installing a package (below).

```
<your-dir>/
  post.md            # the blog post (the deliverable)
  README.md          # 10–20 lines: what this is, how to reproduce, where outputs live
  *.py               # compute scripts separate from render scripts; small tests for core math
  cache/             # raw measurements (.npz/.pt), gitignored; keep < ~5 GB
  figures/           # final PNG / GIF / MP4 referenced by post.md (committed)
  widgets/           # widget JS + data JS (committed)
  _preview/          # page screenshots (gitignored)
```

Compute and render separately. Save raw measurements to `cache/`, then render from the cache, so restyling never needs recompute.

## Environment

- **Python:** `/home/fzeng/ml/research/theory/.venv/bin/python` (3.12, torch 2.14+cu130, torchvision, flash-linear-attention 0.5.2, transformers 5.15, numpy, scipy, scikit-learn, cvxpy, pandas, matplotlib, pillow, imageio, imageio-ffmpeg, colorcet, cmcrameri, tqdm, markdown). Do **not** use the repo-root `.venv` or `art/.venv`, and don't edit any `pyproject.toml`.
- **Extra packages:** `flock /home/fzeng/ml/research/theory/_shared/.pip.lock uv pip install --python /home/fzeng/ml/research/theory/.venv/bin/python <pkg>`. Mention any install in your report.
- **Datasets** (torchvision layout: CIFAR10, MNIST, FashionMNIST) are in `/home/fzeng/ml/research/art/data`. Use `download=False` and read only. HF models/datasets are cached in `~/.cache/huggingface`; use `local_files_only=True` where possible.
- **Tools:** `ffmpeg`, ImageMagick `convert`, `google-chrome` (headless), and `node` are available.
- **Reusable code:** `art/` contains code worth reading (and copying into your dir, not importing): HVP/Lanczos in `art/hessian-spectrum/common.py`, training loops, and render helpers.

## The machine is shared (NVIDIA GB10, ~120 GB unified CPU+GPU memory, 20 CPU cores)

A separate set of `art/` agents is running heavy GPU jobs at the same time.

- **Slot limiter:** any GPU job longer than ~2 minutes, or using more than ~4 GB, must run through `/home/fzeng/ml/research/theory/_shared/gpu_run.sh <cmd...>`, a 3-slot limiter shared by the theory agents. Launch it with Bash `run_in_background` and watch a log file.
- **Memory cap:** in GPU scripts, call `torch.cuda.set_per_process_memory_fraction(0.08)` (≈10 GB).
- **CPU:** set `OMP_NUM_THREADS=4`, and run at most 4 worker processes. Check `free -g` and `nvidia-smi` if things are slow, and back off if memory is under ~15 GB free.
- **Toy first, then scale.** Prefer many tiny models vectorized with `torch.func.vmap` over big models. Use float64 for comparisons against closed forms.
- **Budget:** a few GPU-hours total, not days. If something needs more, run a reduced version and say what the full version would take.

## Git: commit to `main` at natural checkpoints

Commit your directory whenever a meaningful chunk is done, e.g. after "core math + tests", "reproduction figures", "first full draft of post", "widgets working", "build-on results", "final polish". Expect roughly 4–8 commits over the project. **Always** use the helper:

```
/home/fzeng/ml/research/theory/_shared/commit.sh <your-dir> "<your-dir>: short message"
```

- The helper commits **only** your directory, serializes with the other agents via a lock, and refuses files over 25 MB.
- Never run `git add`/`git commit` directly, and never commit other paths.
- Never push, rebase, reset, stash, checkout, or create branches.
- Make sure `cache/` artifacts are never committed. Keep the total committed size of your dir reasonable (target < ~80 MB; shrink GIFs/MP4s if needed).

## Don'ts

- Don't touch other agents' directories, `art/`, or anything else in the repo.
- Don't leave background processes running when you finish.
- Don't claim a result you didn't verify, and don't fake a visual. Every plotted curve comes from a real computation or a stated closed form.

## Final report (your last message, ≤ 250 words)

- whether the reproduction matched the theory (with the key numbers)
- the "build on" finding
- the 3 strongest figures/widgets (paths)
- citation corrections
- anything that didn't work
- GPU time used
- the list of commits
