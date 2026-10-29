# training-dynamics: state

## Build status

| Phase | Status | Date |
|---|---|---|
| B0 brief and papers | done: `BRIEF.md`, 21 PDFs + post in `papers/` | 2026-10-29 |
| B1 curriculum map | done: `COURSE.md` | 2026-10-29 |
| B2 scaffold | done: `pyproject.toml` + `uv.lock` (CPU torch), `template/` (template.typ, build.sh, handout.csl, refs.bib), `tdyn/` (integrate, optim, hessian, models, plotting, results, data, swap), `data/` (mnist.npz, cifar10_5k.npz via `scripts/fetch_data.py`), root `conftest.py` (`--solutions`), `README.md` | 2026-10-29 |
| B3 assignments | in progress (see below) | |
| B4 offline bundle | partly: `uv sync --frozen` into a clean venv and `pytest --offline --solutions` verified; datasets in `data/`; A0 PDFs built | 2026-10-29 |
| B5 handover | not started | |

### Assignments

| # | Folder | Status | Date | Tests (stubs / `--solutions`) | Check CPU (measured) | PDFs |
|---|---|---|---|---|---|---|
| A0 | `a0-prereqs` | done | 2026-10-29 | 48 fail with NotImplementedError / 48 pass | 6 checks, 0.2-1.5 s each (about 10 s total including start-up) | handout 9 pp. (8 + references), solutions 9 pp. |
| A1 | `a1-symmetries` | done | 2026-10-29 | 48 fail with NotImplementedError / 48 pass | 6 checks, 1.2-4.2 s each (about 13 s total including start-up) | handout 9 pp. (8 + references), solutions 8 pp. |
| A2 | `a2-weight-decay` | not started | | | | |
| A3 | `a3-edge-of-stability` | not started | | | | |
| A4 | `a4-sgd-noise` | not started | | | | |
| A5 | `a5-lazy-rich` | not started | | | | |
| A6 | `a6-function-space` | not started | | | | |
| A7 | `a7-parameterization` | not started | | | | |
| C | `capstone` | not started | | | | |

Notes for whoever builds A1 onward: tests run one assignment directory per pytest call
(`uv run pytest a1-symmetries`); each assignment's `tests/conftest.py` puts its own root and tests
dir first on `sys.path` and drops a cached `adapters` module, and pytest runs with
`--import-mode=importlib`, so test file names may repeat across assignments. Stubs and scripts load
code through `tdyn.swap.impl(ROOT, "<pkg>.<module>")`; reference solutions are
`solutions/<module>_solution.py`. Check scripts write one section of `out/results.json` each through
`tdyn.results.update_results`. Shared test-only helpers can live in `tests/_<assignment>_helpers.py` (A1: `_a1_helpers.py`,
reference simulations built from `tdyn` only); give them an assignment-specific name, since `tests/` is on
`sys.path`. A1 added the `du2018balanced` and `kunin2021mechanics` entries to `template/refs.bib`.

## Fan Pu's progress

(empty until he starts)

| Date | Assignment | Problems done | Hours | Notes / tutorial questions |
|---|---|---|---|---|
