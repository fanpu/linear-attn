"""results.json helpers: every check script writes one, solutions/ holds the reference.  [AI]"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _plain(x: Any) -> Any:
    """numpy scalars / arrays -> Python floats / lists, recursively; NaN/inf kept as strings."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else str(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def save_results(path: str | Path, results: dict) -> Path:
    """Write `results` as indented JSON (numpy-safe). Creates the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(results), indent=2, sort_keys=True) + "\n")
    return path


def load_results(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: d}


def compare_results(reference: dict, got: dict, *, rtol: float = 1e-2, atol: float = 1e-8,
                    tolerances: dict[str, tuple[float, float]] | None = None) -> list[str]:
    """Compare two results dicts leaf by leaf. Returns a list of human-readable mismatches
    (empty means they agree). Numbers use |got-ref| <= atol + rtol*|ref|; `tolerances` maps a
    dotted key (e.g. "slope.fit") to its own (rtol, atol). Non-numbers must be equal."""
    tolerances = tolerances or {}
    ref, new = _flatten(reference), _flatten(got)
    problems = []
    for key, r in ref.items():
        if key.endswith("wall_seconds"):       # timing differs from machine to machine
            continue
        if key not in new:
            problems.append(f"{key}: missing")
            continue
        g = new[key]
        rt, at = tolerances.get(key, (rtol, atol))
        if isinstance(r, list) and isinstance(g, list):
            if len(r) != len(g):
                problems.append(f"{key}: length {len(g)} != {len(r)}")
            elif not all(_close(x, y, rt, at) for x, y in zip(g, r)):
                problems.append(f"{key}: lists differ beyond rtol={rt}, atol={at}")
        elif not _close(g, r, rt, at):
            problems.append(f"{key}: got {g}, reference {r}")
    return problems


def _close(g, r, rtol, atol) -> bool:
    if isinstance(r, bool) or isinstance(g, bool) or not isinstance(r, (int, float)) \
            or not isinstance(g, (int, float)):
        if isinstance(r, list) and isinstance(g, list):
            return len(r) == len(g) and all(_close(x, y, rtol, atol) for x, y in zip(g, r))
        return g == r
    return abs(g - r) <= atol + rtol * abs(r)


def update_results(path: str | Path, section: str, values: dict) -> Path:
    """Merge `values` into results.json under key `section` (each check script owns one section)."""
    path = Path(path)
    current = load_results(path) if path.exists() else {}
    current[section] = _plain(values)
    return save_results(path, current)


def main(argv=None) -> int:
    """python -m tdyn.results OURS.json REFERENCE.json [--rtol R]: print mismatches, exit 1 if any."""
    import argparse

    p = argparse.ArgumentParser(description="Compare a results.json against the reference.")
    p.add_argument("got")
    p.add_argument("reference")
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--atol", type=float, default=1e-6)
    a = p.parse_args(argv)
    ref, got = load_results(a.reference), load_results(a.got)
    sections = [s for s in ref if s in got]
    missing = [s for s in ref if s not in got]
    problems = compare_results({s: ref[s] for s in sections}, {s: got[s] for s in sections},
                               rtol=a.rtol, atol=a.atol)
    for s in missing:
        print(f"[not run yet] {s}")
    for line in problems:
        print(f"[differs] {line}")
    print(f"{len(sections)} section(s) compared, {len(problems)} difference(s) "
          f"(rtol={a.rtol}, atol={a.atol}).")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
