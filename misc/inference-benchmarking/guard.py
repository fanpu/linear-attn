"""Memory watchdog for benchmark subprocesses.

GB10's memory is unified, so a runaway benchmark does not hit a clean CUDA OOM --
it eats the host's memory and can wedge the machine. This module runs each
benchmark in its own process group and kills the whole group if system
availability falls below a floor, or if the run overruns its time budget.

Two details matter and are covered by tests:

* Availability is read from /proc/meminfo MemAvailable, not NVML. GB10 reports
  `memory.used = N/A` -- there is no framebuffer counter to read.
* The whole process *group* is killed. vLLM runs its EngineCore in a separate
  process, and killing only the direct child would leave the engine alive and
  still holding tens of GiB.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

GIB = 1 << 30

# Leave the host this much memory. Below it, the child dies.
DEFAULT_FLOOR_BYTES = 12 * GIB

# Grace period between SIGKILL and giving up on reaping the group.
_REAP_TIMEOUT_S = 10.0


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


@dataclass
class GuardResult:
    returncode: int
    output: str
    elapsed_s: float
    floor_tripped: bool
    timed_out: bool
    min_available_bytes: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.floor_tripped and not self.timed_out

    def status(self) -> str:
        if self.floor_tripped:
            return "killed:memory-floor"
        if self.timed_out:
            return "killed:timeout"
        return "ok" if self.returncode == 0 else f"failed:rc={self.returncode}"


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's entire process group, then reap it."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pass


def run_guarded(
    argv: list[str],
    timeout_s: float,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    poll_s: float = 1.0,
    env: dict | None = None,
    cwd: str | None = None,
    echo: bool = False,
) -> GuardResult:
    """Run `argv` under the watchdog, capturing its output.

    Returns a GuardResult rather than raising: a tripped watchdog is data the
    survey should report, not an error that aborts the sweep.
    """
    t0 = time.time()
    min_avail = available_memory_bytes()

    # start_new_session puts the child in its own process group, which is what
    # makes killpg able to take down vLLM's EngineCore along with it.
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
        cwd=cwd,
        bufsize=1,
    )

    chunks: list[str] = []
    floor_tripped = timed_out = False

    # Drain stdout on a thread so a chatty child cannot fill the pipe and
    # deadlock while the watchdog is sleeping between polls.
    import threading

    def _drain():
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            if echo:
                print(line, end="")

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    while proc.poll() is None:
        time.sleep(poll_s)
        avail = available_memory_bytes()
        min_avail = min(min_avail, avail)
        if avail < floor_bytes:
            floor_tripped = True
            _kill_group(proc)
            break
        if time.time() - t0 > timeout_s:
            timed_out = True
            _kill_group(proc)
            break

    proc.wait()
    drainer.join(timeout=5.0)

    return GuardResult(
        returncode=proc.returncode,
        output="".join(chunks),
        elapsed_s=time.time() - t0,
        floor_tripped=floor_tripped,
        timed_out=timed_out,
        min_available_bytes=min_avail,
    )


if __name__ == "__main__":
    print(f"available now {available_memory_bytes()/GIB:.1f} GiB, "
          f"floor {DEFAULT_FLOOR_BYTES/GIB:.0f} GiB")
