# training-dynamics

A self-study course on neural-network training dynamics: nine assignments and a capstone, each a
Typst handout with starter code, `[you]` stubs, tests and worked solutions. The curriculum is in
`COURSE.md`, the brief it was built from in `BRIEF.md`, build status and progress in `STATE.md`.
Everything in the core track runs on a laptop CPU and works offline once set up.

## Setup

```bash
cd training-dynamics
uv sync                               # one locked env: numpy, scipy, matplotlib, pytest, CPU torch
uv run python -c "import tdyn"        # smoke test
uv run python scripts/fetch_data.py   # only if data/*.npz are missing (needs network, ~170 MB download)
```

On Linux torch comes from the PyTorch CPU wheel index, so no CUDA packages are installed; on macOS
it comes from PyPI. `uv.lock` is universal (Linux, macOS, Windows; Python 3.11+).

## Working through an assignment

Start with `a0-prereqs/handout.pdf`. Each assignment folder looks like

```
a0-prereqs/
  handout.typ / handout.pdf   the assignment
  prereqs/                    your code: stubs raising NotImplementedError   [you]
  scripts/check_*.py          the numerical checks                           [AI]
  tests/                      adapters.py (pre-wired; edit only if your signatures differ),
                              test_*.py (never edited), conftest.py
  solutions/                  solutions.pdf, *_solution.py, figures/, results.json
  notes.md                    your notes
  out/                        written by the check scripts (gitignored)
```

### Tests

```bash
uv run pytest a0-prereqs                      # your code; fails with NotImplementedError until filled in
uv run pytest a0-prereqs -k test_gd           # one function
uv run pytest a0-prereqs --solutions          # the same tests against solutions/*_solution.py
TDYN_SOLUTIONS=1 uv run pytest a0-prereqs     # equivalent to --solutions
```

How the swap works: every `tests/adapters.py` loads your module through `tdyn.swap.impl(ROOT,
"prereqs.gd")`. That returns `prereqs/gd.py`, or `solutions/gd_solution.py` when the environment
variable `TDYN_SOLUTIONS=1` is set. The course-root `conftest.py` defines `--solutions`, which just
sets that variable. Run one assignment directory per pytest call.

### Checks

```bash
uv run python a0-prereqs/scripts/check_flow_limit.py            # writes a0-prereqs/out/*.png, results.json
uv run python a0-prereqs/scripts/check_gd_stability.py --solutions   # reference implementations
uv run python a0-prereqs/scripts/check_gd_stability.py --out /tmp/x  # another output directory
uv run python -m tdyn.results a0-prereqs/out/results.json a0-prereqs/solutions/results.json
```

Each check script owns one section of `results.json`. The last command lists every number that differs
from the reference by more than 5% (`--rtol` to change). Sections driven by random sampling (the OU
check) agree only within sampling error if your code draws its random numbers in a different order.

## Building the PDFs

```bash
template/build.sh a0-prereqs/handout.typ a0-prereqs/solutions/solutions.typ
```

`build.sh` uses the `typst` binary if one is installed (here `~/.local/bin/typst`), otherwise the PyPI
`typst` package through `uv run --no-project --with typst` (that needs network the first time). The
root is this directory, so documents import `/template/template.typ` and cite `/template/refs.bib`.
It refuses to finish if a `#ph[...]` placeholder survived into a PDF.

## Offline checklist (run before leaving)

- [ ] `uv sync` completes, then with Wi-Fi off `uv run --offline python -c "import tdyn, torch"` works (on Linux `unshare -rn` before the command simulates no network, where user namespaces are allowed)
- [ ] `uv run --offline pytest a0-prereqs --solutions` (and every other assignment) is green
- [ ] `data/mnist.npz` and `data/cifar10_5k.npz` exist; `uv run python -c "from tdyn.data import load_mnist, load_cifar10_5k; load_mnist(); load_cifar10_5k()"` works (loaders never download)
- [ ] every `handout.pdf` and `solutions/solutions.pdf` is built and opens
- [ ] `papers/` holds every PDF from the brief and the post (`papers/list.txt` lists the arXiv ids)
- [ ] optional: the PyPI `typst` package is cached (`uv run --no-project --with typst python -c "import typst"`) if you want to rebuild PDFs without the binary

## Layout

```
BRIEF.md  COURSE.md  STATE.md  README.md  pyproject.toml  uv.lock  conftest.py
template/   template.typ (visual style of day4's handout), build.sh, handout.csl, refs.bib
papers/     the brief's papers as PDFs, plus the source post (.md, .html)
data/       mnist.npz, cifar10_5k.npz (predownloaded; scripts/fetch_data.py recreates them)
scripts/    fetch_data.py
tdyn/       shared [AI] package: integrate, optim, hessian, models, plotting, results, data, swap
a0-prereqs/ ... one folder per assignment, in study order (see COURSE.md)
```

`tdyn` holds the staff versions of tools that an early assignment asks you to write (GD, the Hessian
tools); later assignments import those, so a bug in your A0 code never blocks A3.
