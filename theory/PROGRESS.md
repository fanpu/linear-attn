# Theory posts: progress log

**Status as of 2026-09-13: paused at the user's request.**

Each project becomes a blog post, following `_shared/BRIEF.md`. At most 3 agents run at a time.

| # | Directory | State | Notes |
|---|---|---|---|
| 2 | `02-implicit-bias/` | paused mid-project | Draft post, 18 figures, 2 widgets, Adam build-on in progress. See `PAUSED.md`. |
| 3 | `03-saxe-dynamics/` | paused mid-project | Draft post, 11 figures, widget drafts, attention build-on in progress. See `PAUSED.md`. |
| 8 | `08-icl-linear-attention/` | paused mid-project | Core math, models, and tests done; training underway, no post yet. See `PAUSED.md`. |
| 4 | `04-lazy-rich-mup/` | queued | Prompt in `_shared/prompts/`. |
| 1 | `01-double-descent/` | queued | Prompt in `_shared/prompts/`. |
| 5 | `05-parities-single-index/` | queued | Prompt in `_shared/prompts/`. |
| 6 | `06-plrf-scaling/` | queued | Prompt in `_shared/prompts/`. |
| 7 | `07-generalization-bounds/` | queued | Prompt in `_shared/prompts/`. |

## Resuming

- **02, 03, 08:** resume these agents from the original session (their context is kept) by messaging them to read their `PAUSED.md` and continue. If that session is gone, start a fresh agent with the brief, the project's section of `PROJECT_IDEAS.md`, and the instruction to pick up from `PAUSED.md`.
- **Queued projects:** the full prompt is `_shared/prompts/<dir>.md` with `{{HEADER}}` replaced by `_shared/prompts/_header.md`.

## Shared infrastructure

| What | Where |
|---|---|
| Python env | `theory/.venv` |
| Post renderer + screenshots | `_shared/render_post.py` |
| Stylesheet, vendored KaTeX | `_shared/post.css`, `_shared/vendor/` |
| 3-slot GPU limiter | `_shared/gpu_run.sh` |
| Per-directory locked commit helper | `_shared/commit.sh` |
