# dl-alchemy: course charter

A private, self-study replica of Stanford **CS 312: Deep Learning Alchemy** (Fall 2026) for Fan Pu, run on his
own GPU, with Claude acting as course staff. Snapshot of the source course: `source/cs312-site-2026-09-19.md`.

**New session? Read this file, then `STATE.md`. Do nothing else until Fan Pu asks for something.**

## 1. Standing instructions from Fan Pu

These are his words, condensed. They override anything else in this file.

1. **Purpose** (2026-09-19): "make a course like this for me, using my own gpu to actually perform the
   experiments." For his own education; "it's ok to directly copy or use ideas from them."
2. **Follow their curriculum** (2026-09-19): "don't be biased by the things that i'm doing right now. follow
   their curriculum actually." Do not steer content toward the rest of this repo (linear attention, MQAR,
   DeltaNet, GB10 low-precision, the theory posts). Repo context is for practical constraints only.
3. **Pacing** (2026-09-19): "do one unit at a time, when i ask you to (so we don't exhaust tokens or require jobs
   that take too many gpu hours)." Never start the next unit, or the next phase that costs GPU, on your own
   initiative. Their calendar is reference only.
4. **GPU** (2026-09-19): "i'll ask you to do something when i know my gpu will be idle for a while." Never launch
   a GPU job unless he has said in the current session that the GPU is free. When he does, check `nvidia-smi`
   first anyway. Tell him the estimated GPU-hours of a batch before launching it.
5. **State** (2026-09-19): "make sure you track all these instructions and state somewhere in the project
   directory because i will be asking from a new session." Everything needed to resume lives in this directory.
   Update `STATE.md` before ending any session that changed anything.
6. **He thinks; Claude does the engineering** (2026-09-19): "i'm already doing another research sprint so i want
   to mostly be thinking and not writing code or babysitting more runs." He never has to write code, write
   configs, launch jobs, or watch them. See §3 for the split.

Machine rules that apply to every GPU job here (from earlier incidents on this box):
one GPU job at a time, chained sequentially in a detached script (`setsid nohup ... < /dev/null &`); every
training script checkpoints and can resume; if a job dies without a traceback check `journalctl -k | grep NVRM`.
Other projects (`autonomous/`, `art/`) share this GPU.

## 2. What the source course is

Seven units. Each unit: an open-ended assignment that scopes out empirical phenomena, lectures on the common
intuitions, weekly 30-minute tutorials, an ungraded report, then a 40-minute closed-book **prediction quiz**:
each question is a held-out experiment the staff actually ran, shown as a description plus code diff, and the
student predicts the outcome. Answers and mental models are discussed right after. Grade = best 5 of 7 quizzes.
Units 1-4 use small-scale language-model pre-training; units 5-7 use DNA.

| # | Unit | Domain |
|---|---|---|
| 1 | Basics: Hyperparameter tuning and scaling | LM |
| 2 | Optimization 1: Hyperparameter invariants | LM |
| 3 | Optimization 2: Sharp and flat basins | LM |
| 4 | Architecture 1: Extending model capacity | LM |
| 5 | Architecture 2: Stability across depth/time | DNA |
| 6 | Generalization: Data-efficient algorithms | DNA |
| 7 | Exotic: ??? (they have not said) | DNA |

## 3. Roles

CS 312 requires experiments to be "run by the student" because the learning is in *experiment design*, committing
to predictions, and reading results; their AI policy allows AI for "implementation, visualization, mock quizzes".
So the split is: **Fan Pu does the thinking, Claude does everything else.**

| Fan Pu (student) | Claude (staff + lab tech) |
|---|---|
| Reads `notes.pdf`, then `handout.pdf`. | Writes them. |
| Picks which directions to pursue and says what to run in plain English ("sweep LR at 3 widths; I expect the optimum to shift down with width"). | Turns that into configs and code diffs; flags confounders and unclean comparisons *before* running; states GPU-hours. |
| Gives a one-line prediction before each batch (Claude must ask for it if missing, and record it). | Queues and launches jobs when he says the GPU is free, monitors them, restarts failures, makes the plots, reports back with a short results summary + figures. |
| Looks at the plots, says what he now believes, picks the next experiment. | Pushes back in tutorial: alternative explanations, what would distinguish them. |
| Writes the report in his own words (short is fine; bullet points + Claude's figures). | Designs and runs the sealed held-out experiments; writes, administers and grades the quiz; writes the debrief. |

Claude must **not** choose his experiments for him or interpret results before he has (no "this shows that..." in
the results summary; give numbers and plots, let him draw the conclusion, then discuss). That is the part that
cannot be delegated without turning the unit into reading a paper.

**Menu mode** (fallback, only for a unit/week where he asks for it): the handout carries a fixed menu of
experiments; he writes a prediction for each; Claude runs them all; he reviews predictions vs results. Less of his
time, less experiment-design practice.

His own experiments and records live in `unitN-*/experiments/`: `log.md` (one entry per batch: his request in his
words, his prediction, configs, run paths, result figures, his takeaway), plus `configs/`, `runs/`, `plots/`.

## 4. The unit loop

One unit at a time. Each phase starts only when Fan Pu asks. `STATE.md` records the current phase.

| Phase | What happens | GPU? |
|---|---|---|
| P0 sync | Re-fetch the course site and GitHub org. If they have published the real handout / lecture / quiz for this unit, **use theirs** and adapt only the compute scale. Save a dated snapshot in `source/`. | no |
| P1 handout | Write `handout.typ` → `handout.pdf` (open-ended, like theirs: phenomena to understand + directions to explore, not a checklist; includes a per-run time estimate and a suggested total GPU budget) and `notes.typ` → `notes.pdf` (lecture-notes stand-in: the common intuitions + short reading list). If the real lecture recording exists, link it instead. | no |
| P2 scaffold | Add what the unit needs to the shared `alchemy/` package; smoke test. Unit 1 also does calibration (§6). | small |
| P3 experiments | Fan Pu designs, Claude executes (§3): he says what to run + a prediction; Claude builds configs, flags confounders, launches when the GPU is free, monitors, plots, logs the batch in `experiments/log.md`, reports numbers and figures without interpreting. Repeat. Tutorials on request: Claude asks why he chose these experiments, pokes at confounders, gives practice problems. No spoilers from sealed runs. | yes |
| P4 report | Fan Pu writes his report in his own words (any format; short is fine; Claude supplies figures/tables on request but not the conclusions). Ungraded. | no |
| P5 held-out runs | Claude designs and runs the quiz experiments (may happen any time the GPU is free, before or after P4; the report "may inspire questions"). Results go to `quiz/sealed/`. | yes |
| P6 quiz | 40 min, closed book. Claude sends `quiz/quiz.pdf`; Fan Pu answers in `quiz/answers.md` or in chat. | no |
| P7 debrief | Reveal results (his answer vs Claude's staff prediction vs truth), grade, discuss mental models, record the score in `STATE.md`. | no |

### Quiz format (copy theirs)

- Each question = one held-out experiment actually run on this machine. Give a description and a code diff
  against the unit's baseline.
- Mix of easy / medium / hard by how far the question extrapolates from the handout's directions. About 5-8
  questions for 40 minutes.
- Answer formats as in their examples: rankings with `<`, `>`, `=` (`=` means within the noise threshold; theirs
  is 0.01, ours is set from measured seed noise), and numeric predictions such as a loss difference
  `L_variant − L_default`. When a question compares methods, follow their protocol of best loss over an LR grid
  if budget allows; otherwise state the fixed LR in the question.
- Closed book. He may cite his report from memory; sound reasoning earns partial credit.
- Grading: ranking questions by pairwise agreement; numeric questions full credit inside a band set from seed
  noise, partial credit for right sign / right order of magnitude. Record the rubric in the quiz file before grading.

### Staff-side GPU work (confirmed 2026-09-19)

Ground truth is always a real run on this machine, never Claude's belief. All of it obeys §1.4 (only when he says
the GPU is free, hours quoted first, one job at a time).

- **Calibration** (Unit 1 P2; again at Unit 5 for DNA): throughput at a few widths → baseline of ~15-20 min/run;
  the six-point LR grid on the baseline as a sanity check; 3-5 seeds for the noise floor (defines `=` in rankings
  and the full-credit band for numeric answers). ~2-4 h.
- **Pilots** (each unit, before finalizing the handout): check that the unit's phenomena actually appear at this
  scale; if one does not reproduce, change the direction or say so in the handout. ~1-3 h. Pilot results are
  **sealed** like quiz results (`unitN-*/quiz/sealed/pilots/`).
- **Held-out quiz runs** (P5): one real run per question (LR grid where the question compares methods, extra
  seeds for close calls) plus a fresh **baseline control** in every batch; if the control moves outside the noise
  floor, the batch is suspect: find the bug before writing questions. ~1 night.
- **Staff predictions**: before launching each held-out run, Claude writes its own prediction to
  `quiz/sealed/staff_predictions.md`. Used to check difficulty labels, and shown in the debrief next to his
  answer and the truth.

### Sealing rules

- Held-out results live only in `unitN-*/quiz/sealed/`. Fan Pu does not open that directory before the debrief.
- While running held-out jobs, Claude must not print results, losses, or plots of them in chat, and must not
  mention them in tutorials. Report only job status (queued / running / done / failed).
- Held-out configs also stay in `sealed/` until the quiz, since the list of experiments leaks the questions.

### Lecture notes (`notes.pdf`): the teaching document

Asked 2026-09-19: "before each assignment will there also be some intro or lectures on how to think about the
questions? like referencing some other previous papers, results, other assignments ... if i just do them blind
i'll be guessing." Yes: CS 312 has 1-2 lectures per unit that "explain common intuitions"; `notes.pdf` is our
stand-in and is read **before** the handout. About 6-10 pages, in this order:

1. **The unit's questions and why they matter at scale.**
2. **Mental models**: 2-4 simple pictures to reason with, each derived briefly (blue `model` boxes).
3. **What is already known**: key results from prior papers with their numbers, the scale they were shown at,
   and caveats. Tiered reading list: 2-3 to read, a few to skim, the rest optional.
4. **Worked predictions**: 2-3 walkthroughs in quiz format (code diff → reasoning from the mental models →
   prediction → what a published result found). This teaches the skill the quiz tests.
5. **Where the literature disagrees or is thin**: the handout's directions point here.
6. **Links to earlier units**: which earlier results and models carry over, including his own reports and debriefs.
7. **Experiment design for this unit**: the controls that matter (LR tuned per arm, compute matching, seeds vs
   the noise floor) and the classic ways to fool yourself.
8. **Practice questions with answers**: an unsealed mock quiz; reused in tutorials.

Rules: (a) no spoilers: the notes give models and published results, never the sealed held-out results, and
quiz questions must extrapolate beyond the notes; (b) verify every cited paper and number by looking it up
(WebSearch/WebFetch) when writing, never from memory; mark anything unverified as such; (c) when CS 312's real
lecture for the unit is public, link it, watch for what it covers, and shrink the notes to what it leaves out.

### Document format (asked 2026-09-19: "give me a pdf instead of md")

Everything Fan Pu reads is a **PDF compiled from Typst**, in the visual style of `../day4/handout/template.typ`
(style only; that template's content, day/version numbering, adapters and log-entry conventions do not apply
here). That covers `handout`, `notes`, `quiz`, `debrief`. `COURSE.md`, `STATE.md` and `source/*.md` stay
markdown (they are for session state); his own `report` is in any format he likes.

- Template: `template/template.typ` (helpers listed in its header: `doc`, `problem`, `question`, `example`,
  `result`, `tip`/`lowres`/`debugtip`/`model`, `parts`, `deliverable`, `resources`, `code`, `prompt`,
  `answerline`, `algorithm`). Import as `#import "/template/template.typ": *`. Do not change anything visual
  between units. `template/style_sample.pdf` shows every element.
- Build: `template/build.sh unitN-*/handout.typ` (no typst binary on this box; the script runs the PyPI `typst`
  package through `uv run --no-project --with typst`; CPU only). Writes the PDF next to the source.
- Code diffs go in `#code[```diff ... ```]`. Citations, when needed: `template/handout.csl` plus a `refs.bib`.
- After compiling, look at the PDF pages (Read tool) before handing it over, then send it with SendUserFile.
  No `#ph[...]` placeholder may survive into a delivered PDF.

## 5. Token and GPU budget discipline

- Work only on the unit and phase he asked for. No look-ahead drafting of later units.
- Handouts ~3-5 PDF pages, notes ~6-10 PDF pages (the notes carry the teaching; see §4). No multi-agent fan-outs unless he asks.
- Before any GPU batch: state number of runs × minutes per run = total hours, and get a yes.
- Target run length about 15-20 minutes on the GB10 so that a night covers a meaningful sweep. Staff-side
  held-out batch per quiz: aim for one night or less.

## 6. Compute scaling (the one deliberate departure from the source)

They assume a B200 and a depth-8 baseline trained on 614M tokens. This box is an NVIDIA GB10 (measured ~95 TFLOP/s
bf16 matmul, realistically ~35 TFLOP/s in training), so width and token count get scaled down; everything else
about the baseline is copied from their snippet (depth 8, pre-norm RMSNorm, learned absolute position
embeddings, `update()` returning `attn + mlp`, LR grid {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}).

Calibration happens in Unit 1 P2, when the GPU is free: measure tokens/s for 2-3 candidate widths, pick
width/tokens/vocab so one run is ~15-20 min, then run 3 seeds of the baseline to get the seed-noise floor that
defines `=` in quizzes. Record the chosen numbers in `STATE.md`. Rough planning estimate until then: ~15-25M
non-embedding parameters, a few hundred M tokens, small BPE vocab so the LM head does not dominate.

**Result (2026-09-19):** baseline = depth 8, width 256, BPE-8192 vocabulary, T 512, 120M tokens, 32,768-token
batch (`unit1-basics/baseline.toml`); ~28 min/run with the GPU shared. The GPT-2 50k vocabulary was dropped because
its output layer was 55% of the FLOPs; FineWeb-Edu was retokenized into `data/fineweb_edu_bpe8k/`
(`alchemy/retokenize.py`). Measurements and reasoning are in `STATE.md`.

Datasets: LM: they have not named theirs; we use FineWeb-Edu (source shards in `../day3/data/fineweb_edu`). DNA: wait for their Assignment 5 to see what
they use; decide at Unit 5 P0.

## 7. Unit content: what is known vs guessed

Only the unit titles and one example problem are public as of 2026-09-19. Until their handouts appear, handouts
here are **reconstructions** and must say so at the top. Guesses below come from the titles and the staff's
published research; treat them as starting points for P1, not commitments, and drop them the moment real
material is available.

1. **Basics: tuning and scaling.** Guess: LR / batch size / warmup / schedule / weight-decay sensitivity, the
   best-of-LR-grid protocol, small scaling laws in parameters and data and extrapolating them.
2. **Hyperparameter invariants.** Guess: which combinations of hyperparameters leave training unchanged: LR
   transfer across width/depth (muP-style), batch size vs LR scaling rules, AdamW's lr×wd timescale, β2/ε vs batch size.
3. **Sharp and flat basins.** Guess: loss-landscape geometry vs LR schedule (river-valley view of WSD, from
   Kaiyue Wen's work), sharpness, weight averaging, LR decay, optimizer choice.
4. **Extending model capacity.** Known: their example problem (component ablations, residual schemes; see
   snapshot) plausibly belongs here. Guess: depth vs width, MLP ratio, heads, other ways to add capacity.
5. **Stability across depth/time** (DNA). Ambiguous: "time" may mean training time (loss spikes) or sequence
   length. Guess: norm placement, init and residual scaling as depth grows; long-sequence behaviour.
6. **Data-efficient algorithms** (DNA). Guess: fixed data, many epochs: regularization, ensembling, distillation
   (Suhas Kotha's data-constrained pre-training work).
7. **Exotic.** Unknown. Wait for them.

## 8. Directory layout

```
dl-alchemy/
  COURSE.md            this charter (instructions + design)
  STATE.md             live state: current unit/phase, decisions, scores, log
  source/              dated snapshots of the CS 312 site and any released materials
  alchemy/             shared Python package: baseline model, train loop, configs (created in Unit 1 P2)
  template/            template.typ, build.sh, handout.csl, style_sample.{typ,pdf}
  unit1-basics/
    handout.typ/.pdf  notes.typ/.pdf
    experiments/       log.md (request, prediction, runs, takeaway per batch), configs/, runs/, plots/
    report.*           his, any format
    quiz/  quiz.typ/.pdf  answers.md  sealed/  debrief.typ/.pdf
  unit2-invariants/ ...
```

Create directories only when their unit starts.

## 9. Session protocol

Start of session: read `COURSE.md` §1 and `STATE.md`. Wait for the request. Typical requests and what they mean:

- "start unit N" → P0 then P1 for unit N (no GPU). Stop and report.
- "gpu is free for X hours" → propose the next GPU batch for the current unit that fits in X (P2 scaffold/calibration,
  or P5 held-out runs), with its hour estimate; launch after a yes.
- "run X" / "try X" (an experiment idea in plain English) → P3: ask for his prediction if he gave none, flag
  confounders, write configs, quote GPU-hours, launch if the GPU is free (else queue it in `STATE.md`), log it.
- "tutorial" → P3 discussion on the current unit.
- "quiz me" → P6, only if the P5 results exist; otherwise say what is missing.
- "debrief" / after answers are in → P7.

End of session: update `STATE.md` (phase, decisions, running jobs with log paths and how to check them, next
action) and append a dated line to its log.
