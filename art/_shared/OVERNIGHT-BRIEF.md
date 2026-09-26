# Overnight brief — 2026-09-26, shared by every project agent

Fan Pu is asleep. He gave the GPU to the art work for 10 hours (until **21:30 EDT, 2026-09-26**)
and asked for *new* ML-phenomena art: ideas and approaches that have not been done, with a
preference for findings about **small neural networks** (they are cheap and probably generalise).
At most two project agents run at once; you are one of them. The other one is using the GPU too.

## Read first (10 minutes, no more)

- `art/ml-art-directions.md` §0 — the governing rule: every visual decision is either a faithful
  rendering of a measured quantity or an explicitly declared aesthetic choice.
- `art/CRITIQUE.md` — what makes these images work or fail. Load-bearing findings:
  - commit to a **key** (clearly dark or clearly light ground); mid-grey fields die.
  - do **not** use `common_render.cdf_img`-style per-phase histogram equalisation; it flattens value.
  - the thing the work is *about* must be the most visible thing in the frame.
  - the ink-on-cream / specimen-sheet register is the collection's most distinctive work; the
    spectral-colormap fractal plane is its most generic. Don't default to Spectral-on-dark.
- `art/display-directions.md` §3.2 — print size rule: ~8,600 px per metre of print width.
- Skim one finished project README for format, e.g. `art/grokking/README.md`.

## GPU: pasar only

Every CUDA process goes through pasar (`pasar guide` has everything). Never start a GPU process
directly from the shell. CPU-only prototyping at tiny sizes is fine.

- `pasar submit --time <est> --name <name> --tag art-<project> --note "<why>" --by <agent-name> --cwd /home/fzeng/ml/research/art -- .venv/bin/python <project>/<script>.py ...`
- **Small-network jobs are good sharing candidates**: pass `--mem <honest estimate, e.g. 6G>` so
  your jobs can run beside the other agent's. Use whole-GPU only for jobs that really fill it
  (e.g. LLM batched decoding).
- One job per configuration (granular). Keep bid 1000. **Never** `--preempt`. **Never** `--on`
  (no cloud).
- Inside scripts: `import pasar_job`; call `pasar_job.progress(step, total)` at least every few
  minutes; for anything over ~20 min, checkpoint under `pasar_job.persist_dir()` and resume on
  `pasar_job.resuming()`. `pasar_job` is installed in `art/.venv`.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in shared jobs.
- Use `--json` when parsing; `pasar wait <id> --timeout 540` (loop it; Bash calls time out at
  10 min) and `pasar logs <id>` when something fails. Cancel jobs you don't need.
- Stay inside your GPU-hour budget (given in your task). All your GPU work must finish by
  **21:30 EDT**. Write big arrays to `<project>/cache/` (gitignored).

## Standards

- **Honesty over beauty.** Every claimed structure needs a null control (shuffled labels, random
  init, untrained network, a trivially-explained baseline...) rendered with the *same* treatment.
  If the phenomenon turns out trivial or absent, say so plainly — a well-documented null is a
  good result; a pretty artefact is not.
- **Prior art:** spend a few minutes with WebSearch checking whether this image/approach already
  exists. Say what you found in the README.
- **Look at your images.** Read your PNGs, critique them against CRITIQUE.md, and iterate at least
  once. Produce `gallery/contact_sheet.png` of your best 4–8 images.
- Keep `gallery/` under ~60 MB; big stuff in `cache/`.

## Deliverables in `art/<project>/`

Code, `gallery/`, and a `README.md` in the style of the other projects: the phenomenon, what was
measured, declared aesthetic choices, the null control and its result, prior art, the best images,
compute used (job ids, GPU time), and honest next moves.

## Do not

- commit or push (the coordinator commits), or edit anything outside your project directory
  (including `art/README.md`);
- touch other projects' directories or other people's pasar jobs;
- add any Claude/Anthropic attribution anywhere.

## Final report (your last message, ≤300 words)

What you found, with numbers; whether it is real or null; the 2–4 best image paths (absolute);
GPU time used and job ids; the single most promising next move.
