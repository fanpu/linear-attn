"""Per-assignment test setup.  [AI; do not edit]

Puts this assignment's root (for `import prereqs`) and this tests/ directory (for
`import adapters`) at the front of sys.path, and forgets any `adapters` module another
assignment's tests imported earlier in the same pytest run.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
sys.modules.pop("adapters", None)
