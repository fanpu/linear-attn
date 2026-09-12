"""Tests for guard.py — the memory watchdog.

Uses small, bounded allocations (a few GiB against ~110 GiB available), so these
are safe to run on the benchmark machine itself.
"""

import sys
import time

import budget
import guard

GIB = 1 << 30
PY = sys.executable


def test_clean_command_runs_untouched():
    r = guard.run_guarded([PY, "-c", "print('hello')"], timeout_s=60, floor_bytes=GIB)
    assert r.returncode == 0
    assert "hello" in r.output
    assert not r.floor_tripped and not r.timed_out


def test_nonzero_exit_is_reported_not_masked():
    r = guard.run_guarded([PY, "-c", "raise SystemExit(3)"], timeout_s=60, floor_bytes=GIB)
    assert r.returncode == 3
    assert not r.floor_tripped


def test_timeout_kills_child():
    t0 = time.time()
    r = guard.run_guarded([PY, "-c", "import time; time.sleep(120)"],
                          timeout_s=3, floor_bytes=GIB)
    assert r.timed_out
    assert time.time() - t0 < 30
    assert r.returncode != 0


def test_floor_breach_kills_child():
    """An impossible floor must trip immediately, even on a benign child."""
    floor = budget.total_memory_bytes() * 2
    r = guard.run_guarded([PY, "-c", "import time; time.sleep(60)"],
                          timeout_s=45, floor_bytes=floor, poll_s=0.2)
    assert r.floor_tripped
    assert r.returncode != 0


def test_floor_breach_kills_a_genuinely_allocating_child():
    """The real scenario: a child eating memory is killed as availability falls.

    The floor is set 1 GiB below current availability and the child allocates
    ~4 GiB in 256 MiB steps, so the watchdog must fire partway through.
    """
    floor = budget.available_memory_bytes() - GIB
    child = (
        "import time\n"
        "buf = []\n"
        "for _ in range(16):\n"
        "    buf.append(bytearray(256 * 1024 * 1024))\n"
        "    time.sleep(0.3)\n"
        "print('ALLOCATED_ALL')\n"
    )
    r = guard.run_guarded([PY, "-c", child], timeout_s=60, floor_bytes=floor, poll_s=0.2)
    assert r.floor_tripped, "watchdog did not fire while memory was being consumed"
    assert "ALLOCATED_ALL" not in r.output, "child finished despite the floor breach"


def test_grandchildren_are_killed_too():
    """vLLM spawns a separate EngineCore process, so killing only the direct
    child would leave the engine holding memory. The guard must kill the group."""
    child = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print('GRANDCHILD_PID', p.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    r = guard.run_guarded([PY, "-c", child], timeout_s=4, floor_bytes=GIB, poll_s=0.2)
    assert r.timed_out
    pid = int([l for l in r.output.splitlines() if "GRANDCHILD_PID" in l][0].split()[1])
    time.sleep(1.0)
    alive = True
    try:
        import os
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        alive = False
    assert not alive, f"grandchild {pid} survived the guard"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:
                fails += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'FAILED' if fails else 'OK'} ({fails} failures)")
    raise SystemExit(1 if fails else 0)
