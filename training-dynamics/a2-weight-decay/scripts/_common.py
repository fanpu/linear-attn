"""Shared plumbing for the A2 check scripts.  [AI]

Every check script takes
    --out DIR      where to write its PNGs and its section of results.json (default a2-weight-decay/out/)
    --solutions    use solutions/*_solution.py instead of your wdecay/ code (sets TDYN_SOLUTIONS=1)
and prints its wall time. Compare your results with the reference afterwards:
    uv run python -m tdyn.results a2-weight-decay/out/results.json a2-weight-decay/solutions/results.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # a2-weight-decay/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", type=Path, default=ROOT / "out", help="output directory")
    p.add_argument("--solutions", action="store_true", help="run against solutions/*_solution.py")
    args = p.parse_args()
    if args.solutions:
        os.environ["TDYN_SOLUTIONS"] = "1"
    args.out.mkdir(parents=True, exist_ok=True)
    from tdyn.plotting import set_style
    set_style()
    args.t0 = time.perf_counter()
    return args


def mine(module: str):
    """Your module wdecay.<module>, or its reference solution under --solutions."""
    from tdyn.swap import impl
    return impl(ROOT, f"wdecay.{module}")


def finish(args, section: str, results: dict) -> None:
    from tdyn.results import update_results
    results["wall_seconds"] = round(time.perf_counter() - args.t0, 2)
    path = update_results(args.out / "results.json", section, results)
    print(f"[{section}] wrote {path} (section '{section}'), wall time {results['wall_seconds']} s")
