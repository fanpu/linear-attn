"""Cooperative pausing for Claude's GPU jobs (see CHARTER.md §3).

Usage in a training loop:

    from pause import should_pause, PAUSE_EXIT
    for step in range(start_step, n_steps):
        ...
        if should_pause():
            save_checkpoint(step)
            sys.exit(PAUSE_EXIT)   # gpu_run.sh reruns the script after resume; it must load the checkpoint

should_pause() only touches the filesystem at most once every `every_s` seconds, so it is cheap to call every step.
"""

import os
import time
from pathlib import Path

PAUSE_EXIT = 75
PAUSE_FILE = Path(os.environ.get("AUTONOMOUS_PAUSE_FILE", Path(__file__).resolve().parent.parent / "GPU_PAUSE"))

_last_check = 0.0
_last_value = False


def should_pause(every_s: float = 10.0) -> bool:
    global _last_check, _last_value
    now = time.monotonic()
    if now - _last_check >= every_s:
        _last_check = now
        _last_value = PAUSE_FILE.exists()
    return _last_value
