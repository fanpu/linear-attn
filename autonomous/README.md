# autonomous/: Claude's independent research on linear attention

Claude works here as an independent research colleague, alongside Fan Pu's own linear-attention work in the rest of this repo.
**The goal is one publishable result, drafted by 2026-09-29 at the latest** (token budget, not the calendar, sets the pace), written up clearly enough that the core idea fits in one figure and three sentences.

## Status

| | |
|---|---|
| **Updated** | 2026-09-15 |
| **Phase** | Deepen (Gate A passed 2026-09-15) |
| **Current direction** | S1′: how much can Kalman-style memories beat a well-configured gated delta rule? Tracking theory + trained models |
| **Headline result so far** | Under drift, a Kalman memory beats the best gated delta rule by ≤ 1.47× (isotropic), and a square-root key metric Σ^(−½) keeps it there at any anisotropy (R-002, R-003) |
| **GPU** | **PAUSED** (2026-09-15 08:35, at Fan Pu's request). Resume: `autonomous/tools/resume.sh` |

## Where to look

| If you want to know… | Read |
|---|---|
| The rules I work by (mission, budget, pausing, standards) | [`CHARTER.md`](CHARTER.md) |
| The two-week timeline and decision gates | [`PLAN.md`](PLAN.md) |
| What I did on a given day, and why | [`journal/`](journal/) |
| What we actually know so far (including dead ends) | [`RESULTS.md`](RESULTS.md) |
| Which ideas are alive, dead, or promoted | [`IDEAS.md`](IDEAS.md) |
| Paper notes and the field map | [`literature/`](literature/) |
| Individual experiments | `experiments/NNN-slug/README.md` |
| Messages between us | [`SYNC.md`](SYNC.md) |

## Layout

```
autonomous/
  CHARTER.md  PLAN.md  IDEAS.md  RESULTS.md  SYNC.md
  journal/YYYY-MM-DD.md
  literature/            paper and topic notes, novelty log
  common/                tested shared code: reference mixers, synthetic tasks, training loop (built in the Map phase)
  experiments/NNN-slug/  one directory per experiment (template: experiments/TEMPLATE.md)
  paper/                 the draft (once the key figure exists)
  tools/                 gpu_run.sh (polite GPU job runner), pause.py, pause.sh, resume.sh
  .venv/                 pytest only; sees the repo-root .venv packages via a .pth file (not committed)

Tests: `.venv/bin/python -m pytest -q --rootdir common common/` (from autonomous/)
```
