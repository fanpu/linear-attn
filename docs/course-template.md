# Study courses: the format for "teach me research area X"

Instructions for Claude. Applies whenever Fan Pu asks for an assignment-style curriculum on a
research area he does not know yet, or pastes a brief like `training-dynamics/BRIEF.md`.
Each course lives in its own top-level directory (`training-dynamics/`, ...). This file is the
format; the per-course `COURSE.md` records that course's decisions and `STATE.md` its progress.

**New session, course request or course mention? Read this file, then `<course>/COURSE.md`, then
`<course>/STATE.md`.**

## 1. What this is, and what it is not

This is the daily-sprint assignment format (`day3/`, `day4/`: a Typst handout, starter code with
`[you]` stubs wired through `tests/adapters.py`, tests that are never edited) pointed at
*learning a field* instead of running a research day. It is **not** dl-alchemy: there is no sealed
ground truth, no quiz Claude administers, and Fan Pu writes the core code himself. The point is
that he implements the crux and derives the key results by hand, and the boilerplate is done for him.

## 2. Standing instructions from Fan Pu (2026-10-29)

His words, condensed. They override anything else here.

1. **Format.** "assignment style, with starter code and core parts and tests for me to fill in my
   implementation, so i implement the crux of things and understand how things happen ... most of
   the time is spent on learning etc, less boilerplate, but i always do the parts that are necessary
   for learning."
2. **Curriculum.** "structure it into a curriculum, build things up from basics (assume basic ugrad
   ml knowledge), similar to the level of the research sprint knowledge assumptions."
3. **Layout.** One directory per course, "one folder for each assignment in this course".
4. **Solutions.** "it should also include the worked solutions in case i cant figure it out / only
   have offline access." Plain sight is fine: `solutions/` inside each assignment folder, no sealing.
5. **Build everything at once.** "i want to build out all assignments at once so i can work through
   it on the flight." No one-unit-at-a-time pacing for courses. Build the whole course, verify it,
   hand it over.
6. **Light on compute, offline-capable.** "i might be doing this on a flight so may not have a gpu at
   all, so if possible try to make this light on compute." Core assignments run on a laptop CPU.
   GPU work is an optional, clearly marked extension, submitted through pasar when he is home
   (see `~/.claude/CLAUDE.md` for the pasar rules).
7. **Reading.** "for concepts that are borrowed from a paper, say so, and say to look at that paper
   for more information. but ultimately the assignment itself should be self contained."

Also inherited from his other instructions: no tailoring the content toward his current projects
(follow the brief, not the rest of this repo), commit straight to `main`, send him the PDF when
something is ready to look at, not just the path.

## 3. Inputs: the brief

A course starts from a brief: areas, papers per area, what to take from each, a learner profile,
an anchor model or running example, exercise hooks, and ordering notes. Save it verbatim as
`<course>/BRIEF.md` before anything else. If he asks for a course without a brief, draft one in that
shape (areas, 2-5 papers each, hooks, dependency order), get his yes, then proceed. The brief is
the authority on *content*; this file is the authority on *format*.

## 4. Directory layout

```
<course>/
  BRIEF.md              his brief, verbatim
  COURSE.md             curriculum map (assignment list, dependency graph, what each covers), the
                        anchor model, course-specific decisions, and compute tiers per assignment
  STATE.md              build status per assignment, then his progress: date, problems done, time,
                        his notes, open questions for a tutorial
  README.md             setup (uv sync, offline check), how to run tests, how to build PDFs
  pyproject.toml, uv.lock   one locked env for the whole course (numpy, scipy, torch CPU wheels,
                        matplotlib, pytest; nothing that needs a GPU to import)
  template/             Typst template copied from day4/handout/template.typ (visual style only),
                        build.sh as in dl-alchemy/template/build.sh, handout.csl, refs.bib
  papers/               PDFs of every paper in the brief, named `<firstauthor><year>-<slug>.pdf`,
                        plus the framing posts saved as .md or .html, for offline reading
  <pkg>/                shared boilerplate package (data loaders, plotting, a tiny trainer,
                        Hessian power iteration, ODE integrators); all [AI]; stable across assignments
  a0-<slug>/            one folder per assignment, numbered in study order
    handout.typ / .pdf  the assignment (self-contained)
    <pkg_a0>/ or *.py   the assignment's own code: [AI] scaffolding plus [you] stubs
    tests/              adapters.py (he fills in), test_<slug>.py (never edited), conftest.py
    solutions/
      solutions.typ / .pdf   worked derivations for every pen-and-paper part, and the expected
                             numbers/plots for every numerical check
      <stub>_solution.py     reference implementations that make the tests pass
      figures/               the plots the solutions reference
    notes.md            his scratch space (empty at build time)
  a1-<slug>/ ...
  capstone/             if the brief asks for one
  m0-<primer>/          optional math primer: handout.typ / .pdf only (no tests, no stubs), a
                        reference for the math the assignments assume, read on demand; see §7
```

Assignments are numbered in the brief's dependency order. `a0` is always the prerequisites warm-up
when the brief lists prerequisites.

## 5. Building a course

Build phases. Phases B1 and B2 get his yes before B3 starts; after that, build everything.

| Phase | What happens |
|---|---|
| B0 brief | Save `BRIEF.md`. Read every paper's abstract and the sections the brief points at (WebFetch; download the PDFs into `papers/`). Never work from memory of a paper. |
| B1 map | Write `COURSE.md`: the assignment list (title, one-sentence goal, which areas and papers it draws on, the state of the anchor model at the end of it, estimated hours for him, compute tier), the dependency graph, and the prerequisites assignment. Show him the map and get a yes. Typically 6-10 assignments of 3-6 hours each. |
| B2 scaffold | `pyproject.toml`, `template/`, shared package, `README.md`, smoke test on CPU. Decide and record how each paper's experiment is shrunk to CPU scale (§8). |
| B3 assignments | For each assignment in order: write the handout, the starter code and stubs, the tests, the solutions; run the reference solutions through the tests (all green) and run every numerical check end to end on CPU, recording wall time; compile both PDFs and look at every page (Read tool) before moving on. Reuse the previous assignment's code; the anchor model grows in the brief's fixed order. |
| B4 offline bundle | `uv sync` and verify imports with networking off (`unshare -n` or equivalent); predownload datasets into `data/`; every PDF built; `papers/` complete; `README.md` has the offline checklist. |
| B5 handover | `STATE.md` with build status and per-assignment time estimates. Commit to main. Send him `a0`'s handout PDF and the course map; tell him the total estimated hours and which extensions want the GPU. |

Multi-agent fan-out across assignments is fine in B3 if the map is fixed and the shared package is
done, since assignments only depend on each other through code that already exists. Each agent gets
this file, `COURSE.md`, `BRIEF.md`, and the previous assignment's folder.

## 6. Anatomy of an assignment

The handout follows the day-sprint handout shape, about 4-8 PDF pages, in this order:

1. **Overview.** The question this assignment answers, in one paragraph, and where it sits in the
   course (what the last one established, what the next one needs). Then the three fixed lists:
   *What you will derive*, *What you will implement*, *What you will run*. Each item links to its
   problem.
2. **Background.** Only what the problems need, built up from undergrad ML. Every non-ML term
   defined on first use (e.g. "a *saddle* is a fixed point whose linearization has eigenvalues of
   both signs"). Every borrowed idea attributed at the point of use: "This argument is from Du, Hu
   and Lee (2018), Section 3; read it for the general homogeneous case." Papers are *for more*, never
   *required to continue*.
3. **Problems**, in orange `problem` boxes, each tagged with its type, estimated time, and `[you]`:
   - **Derive.** Pen and paper. States exactly what to show, gives the starting equations, and ends
     with a *deliverable* that is a concrete formula or number he can type into a stub (a closed form,
     a critical value, a rate), so the numerical check below can verify the derivation. Every Derive
     part has a hint ladder, `#hints(...)` after the deliverable (or a blue `tip` box for long ones),
     ordered from gentle to strong; see §7.
   - **Implement.** A function in a named file with a stated interface (`defline`, `arglist`,
     `iolist`), tested through `tests/adapters.py`. Ends with the fixed test sentence
     ("To test your implementation, complete `run_<slug>` in `tests/adapters.py` and run
     `pytest -k test_<slug>`"). Scope: the crux only. Integrators, plotting, data loading, and
     training loops are `[AI]` in the shared package unless writing them *is* the lesson.
   - **Check.** A short numerical experiment (a script that is mostly written) whose output is
     compared against the Derive deliverable. States the expected shape of the result and the
     tolerance, and what disagreement would mean.
   - **Predict.** Before running a Check, he writes down what he expects (one line in `notes.md`).
     Used where the brief's hooks say "predict, before running".
   - **Extension (GPU).** Optional. Marked `[GPU]` in the title with a GB10 time estimate and a ready
     `pasar submit` line; never on the critical path of the assignment or of later ones.
4. **Worked example.** At least one black `example` box per assignment: a hand-traceable instance of
   the main derivation or computation with its answer, so he can calibrate before the real problem.
5. **What to write down.** Three to five sentences for `notes.md`: what he now believes, and one
   question to bring to a tutorial.

Every Derive problem is paired with a Check. Every Implement problem has a test. Every problem has
a worked solution in `solutions/`.

### Code conventions

- Stubs raise `NotImplementedError` and carry a docstring stating the contract and the equation
  number in the handout they implement.
- `tests/adapters.py` is the only file he edits to connect code to tests; test files have explicit
  tolerances and a comment naming the handout equation they check. Tests run on CPU in seconds.
- `solutions/<stub>_solution.py` drops into the same interface; `pytest --solutions` (a conftest
  flag) swaps them in so the build can verify the tests are green.
- Scripts take `--out` and write figures and a `results.json`; the solutions folder holds the
  reference `results.json` and figures so he can diff offline.
- Seeds fixed everywhere. No `nohup`, no background jobs in the core track.

## 7. Writing rules

- All math in Typst math (reads as LaTeX in the PDF). Notation is fixed in `COURSE.md` and reused
  across assignments; the anchor model keeps the brief's symbols.
- Level: strong CS undergrad math (calculus, linear algebra, probability, basic real analysis),
  comfortable with PyTorch; not fluent in dynamical systems, SDEs or kernel methods. Build those three
  up from scratch where they first appear.
- Verify every citation, venue and quoted number by looking it up while writing; mark anything
  unverified. Paper links in the handout point at the version in `papers/` and the arXiv URL.
- No spoilers of the derivation's answer in the handout body; answers live in `solutions/`.
- Each handout stands alone offline: no step depends on a web page, an API, or a download not
  already in the repo.
- Same visual template for every assignment; no `#ph[...]` placeholder survives into a delivered PDF.

Added 2026-10-29 after he found A0-A2's derivations hard without more background ("I think I need a
math primer ... I'm not really good with ODEs") and an audit found hint ladders on only 12 of 47
Derive parts. These four apply to every Derive part from A3 on:

- **Primer pointers.** If the course has an `m0-<primer>/` (here `m0-math-primer/`, sections named
  `ODE.4`, `LA.5`, ...), point into it inline wherever a part assumes a technique:
  `#primer("ODE.4", "logistic")` renders "(primer §ODE.4, logistic)". A technique no primer section
  covers is either added to the primer (a short subsection: definition, one worked example in the
  course notation, a "Where you use it" line, an exercise with its answer) or built up in the handout's
  Background.
- **A hint ladder on every Derive part.** `#hints([...], [...], [...])`, two or three hints, gentle to
  strong. The first names the technique and the primer section; the last is the first line of the
  worked solution, quoted. A part without a ladder does not ship.
- **Concrete deliverables.** Each Derive deliverable is a named formula, a number, or a one-sentence
  statement with a specified form ("the identity ∇²L·X = M∇L with M a constant matrix"; "the class
  of losses, stated as a condition on L"; "the per-neuron statement as an equation between two
  norms"). Never "conclude something about", "one sentence", "state for which".
- **Time box.** Every handout's Setup paragraph carries: "If a Derive part takes more than 15 minutes
  with all hints, read that part in solutions.pdf, close it, and redo the part from a blank page before
  moving on. The solutions are part of the method, not a failure."

## 8. Compute tiers

Classify every experiment in the brief and record the tier in `COURSE.md`.

| Tier | Definition | Treatment |
|---|---|---|
| Light | Seconds to a few minutes on a laptop CPU: closed-form models, ODE integration, toy regressions, small MLPs on small datasets, kernel computations on a handful of examples | Core track, as written |
| Medium | A published experiment that is CPU-feasible once shrunk (smaller net, data subset, fewer steps), 10-40 minutes | Core track, shrunk; the handout states the shrink, the target wall time, and what the full-scale version would show; a `--full` flag or GPU extension restores the original scale |
| Heavy | Learning-rate sweeps across widths, anything on a Transformer, finetuning a language model, many seeds | Optional `[GPU]` extension with a pasar line and a GB10 estimate; the core track reaches the same concept on a Light or Medium stand-in |

Rule of thumb for the shrink: keep the protocol (sweep, early stopping, controls) and cut the scale,
never the other way round. Say in the handout which published number the shrunk version is and is
not expected to reproduce.

## 9. Offline checklist (B4)

- `uv sync` completes; `python -c "import <pkg>"` and the full `pytest --solutions` pass with
  networking disabled.
- All datasets under `data/`, small (MNIST, CIFAR subsets, synthetic). Loader functions never
  download at run time.
- Every handout and solutions PDF built and checked page by page.
- `papers/` has every PDF from the brief and the framing posts.
- `README.md` lists the above as a checklist he can run before leaving.

## 10. Working through it (student sessions)

When he is on an assignment (he says which), Claude is a TA, not a solver:

- Hints before answers. Point at the handout's tip boxes first, then the relevant background
  paragraph, then the solution's first step. Give the full solution only when he asks for it.
- Check his derivations by comparison with `solutions/`, and his numerical results against the
  reference `results.json`; say where they differ and why that might be, without rewriting his code.
- When he is stuck on a test, read his adapter and his stub and ask what he expects each line to
  do before pointing at the bug.
- Record in `STATE.md`: date, assignment, problems completed, hours, his notes-to-self, and any
  question he wants a tutorial on. Tutorials on request: the explaining-ml-concepts skill applies.
- Do not launch GPU extensions unless he has said the GPU is idle in the current session; then go
  through pasar with the handout's line.

## 11. Quality gates before handover

- Reference solutions make every test pass; every Check script reproduces the solutions' numbers
  on CPU within the stated tolerance and within the stated wall time.
- Every Derive problem has a Check, every Implement a test, every problem a solution.
- Every Derive part has a `#hints(...)` ladder and a primer pointer where it assumes a technique:
  `grep -c '#tag("derive")'` equals `grep -c '#hints('` in each handout.
- Every paper in the brief is used somewhere and attributed; every quoted number verified.
- A reader with the stated background can reach the end of each handout using only the repo.
- PDFs inspected page by page; the total estimated hours and GPU extensions are listed in
  `COURSE.md` and quoted to him at handover.
