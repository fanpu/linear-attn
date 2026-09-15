# autonomous/: Claude's independent research on linear attention

Claude works here as an independent research colleague, alongside Fan Pu's own linear-attention work in the rest of this repo.
**The goal is one publishable result by 2026-09-29**, written up clearly enough that the core idea fits in one figure and three sentences.

## Status

| | |
|---|---|
| **Day** | 1 of 14 (2026-09-15) |
| **Phase** | Map: reading, building a tested toolkit, shortlisting ideas |
| **Current direction** | Not chosen yet. Gate A (pick one) is at the end of 2026-09-19 |
| **Headline result so far** | None yet |
| **GPU** | Idle. Pause anytime: `autonomous/tools/pause.sh` (or `touch autonomous/GPU_PAUSE`) |

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
  common/                tested shared code: reference mixers, synthetic tasks, training loop (from day 1–2)
  experiments/NNN-slug/  one directory per experiment (template: experiments/TEMPLATE.md)
  paper/                 the draft (from ~day 8)
  tools/                 gpu_run.sh (polite GPU job runner), pause.py, pause.sh, resume.sh
```
