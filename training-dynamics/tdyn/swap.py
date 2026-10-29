"""Import a stub module or its reference solution.  [AI]

Each assignment's tests/adapters.py and scripts/*.py load your code through `impl`:

    gd = impl(ROOT, "prereqs.gd")      # -> <ROOT>/prereqs/gd.py              (default)
                                       # -> <ROOT>/solutions/gd_solution.py   (TDYN_SOLUTIONS=1)

`pytest --solutions` (course-root conftest.py) and every script's `--solutions` flag set
TDYN_SOLUTIONS=1, so the same tests and checks run against the reference implementations.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def use_solutions() -> bool:
    return os.environ.get("TDYN_SOLUTIONS", "") == "1"


def impl(root: str | Path, dotted: str) -> ModuleType:
    """Return module `dotted` (e.g. "prereqs.gd") from assignment directory `root`, or
    `root/solutions/<last part>_solution.py` when TDYN_SOLUTIONS=1."""
    root = Path(root).resolve()
    if use_solutions():
        name = dotted.rsplit(".", 1)[-1]
        path = root / "solutions" / f"{name}_solution.py"
        modname = f"_tdyn_solution_{root.name.replace('-', '_')}_{name}"
        if modname in sys.modules:
            return sys.modules[modname]
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(dotted)
