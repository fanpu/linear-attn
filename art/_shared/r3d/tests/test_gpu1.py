import os, signal, subprocess, time, pathlib, stat

SCRIPT = os.environ.get("GPU1_SCRIPT", "/home/fzeng/ml/research/art/_shared/gpu1.sh")   # override to test a candidate


def _env(tmp, others_flag=None, ignore=False, exclusive=False, meminfo=None, min_free_gb=None):
    fake = tmp / "bin"; fake.mkdir(exist_ok=True)
    smi = fake / "nvidia-smi"
    flag = others_flag or (tmp / "no_such_flag")
    smi.write_text(f"#!/usr/bin/env bash\n[ -e {flag} ] && echo '123, python'\nexit 0\n")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    if meminfo is None:
        meminfo = tmp / "meminfo"
        meminfo.write_text("MemTotal:       200000000 kB\nMemAvailable:   200000000 kB\n")
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}",
               GPU1_AUTONOMOUS_LOCK=str(tmp / "autonomous" / "tools" / ".gpu.lock"),
               GPU1_ART_LOCK=str(tmp / "gpu1.lock"),
               GPU1_MEMINFO=str(meminfo),
               GPU1_POLL_S="0.2")
    env.pop("GPU1_HELD", None)          # the suite may itself run inside a gpu1 job
    (tmp / "autonomous" / "tools").mkdir(parents=True, exist_ok=True)
    if ignore:
        env["GPU1_IGNORE_OTHERS"] = "1"
    if exclusive:
        env["GPU1_EXCLUSIVE"] = "1"
    if min_free_gb is not None:
        env["GPU1_MIN_FREE_GB"] = str(min_free_gb)
    return env


def test_exit_status_passthrough(tmp_path):
    # Default share mode. _env's default meminfo has ample MemAvailable so this
    # never depends on the real machine's memory state.
    env = _env(tmp_path)
    r = subprocess.run([SCRIPT, "bash", "-c", "exit 7"], env=env, capture_output=True, text=True)
    assert r.returncode == 7


def test_exit_status_passthrough_exclusive_mode(tmp_path):
    env = _env(tmp_path, ignore=True, exclusive=True)
    r = subprocess.run([SCRIPT, "bash", "-c", "exit 7"], env=env, capture_output=True, text=True)
    assert r.returncode == 7


def test_two_jobs_are_serialised(tmp_path):
    # Default share mode. _env's default meminfo has ample MemAvailable so this
    # never depends on the real machine's memory state.
    env = _env(tmp_path)
    log = tmp_path / "log"
    cmd = f"echo start $(date +%s.%N) >> {log}; sleep 1; echo end $(date +%s.%N) >> {log}"
    a = subprocess.Popen([SCRIPT, "bash", "-c", cmd], env=env)
    time.sleep(0.2)
    b = subprocess.Popen([SCRIPT, "bash", "-c", cmd], env=env)
    assert a.wait(20) == 0 and b.wait(20) == 0
    ev = [l.split() for l in log.read_text().splitlines()]
    assert [e[0] for e in ev] == ["start", "end", "start", "end"]


def test_two_jobs_are_serialised_exclusive_mode(tmp_path):
    env = _env(tmp_path, ignore=True, exclusive=True)
    log = tmp_path / "log"
    cmd = f"echo start $(date +%s.%N) >> {log}; sleep 1; echo end $(date +%s.%N) >> {log}"
    a = subprocess.Popen([SCRIPT, "bash", "-c", cmd], env=env)
    time.sleep(0.2)
    b = subprocess.Popen([SCRIPT, "bash", "-c", cmd], env=env)
    assert a.wait(20) == 0 and b.wait(20) == 0
    ev = [l.split() for l in log.read_text().splitlines()]
    assert [e[0] for e in ev] == ["start", "end", "start", "end"]


def test_waits_for_other_gpu_process(tmp_path):
    flag = tmp_path / "busy"; flag.write_text("1")
    env = _env(tmp_path, others_flag=flag, exclusive=True)
    out = tmp_path / "ran"
    p = subprocess.Popen([SCRIPT, "bash", "-c", f"touch {out}"], env=env)
    time.sleep(1.0)
    assert not out.exists()
    flag.unlink()
    assert p.wait(20) == 0 and out.exists()


def test_waits_for_autonomous_lock_but_not_when_paused(tmp_path):
    env = _env(tmp_path, ignore=True, exclusive=True)
    lock = tmp_path / "autonomous" / "tools" / ".gpu.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "3"])
    time.sleep(0.3)
    out = tmp_path / "ran"
    p = subprocess.Popen([SCRIPT, "bash", "-c", f"touch {out}"], env=env)
    time.sleep(1.0)
    assert not out.exists()            # blocked behind the autonomous job
    (tmp_path / "autonomous" / "GPU_PAUSE").write_text("")
    assert p.wait(20) == 0 and out.exists()   # pause means their jobs cannot start: stop waiting
    holder.wait()


def test_share_mode_ignores_other_gpu_processes_and_autonomous_lock(tmp_path):
    flag = tmp_path / "busy"; flag.write_text("1")
    env = _env(tmp_path, others_flag=flag)   # default: share mode, no GPU1_EXCLUSIVE
    lock = tmp_path / "autonomous" / "tools" / ".gpu.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "3"])
    time.sleep(0.3)
    out = tmp_path / "ran"
    p = subprocess.Popen([SCRIPT, "bash", "-c", f"touch {out}"], env=env)
    assert p.wait(10) == 0 and out.exists()   # not blocked by the busy nvidia-smi or the autonomous lock
    holder.wait()


def test_waits_for_available_memory(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       200000000 kB\nMemAvailable:        1000 kB\n")
    env = _env(tmp_path, meminfo=meminfo)
    out = tmp_path / "ran"
    p = subprocess.Popen([SCRIPT, "bash", "-c", f"touch {out}"], env=env)
    time.sleep(1.0)
    assert not out.exists()
    meminfo.write_text("MemTotal:       200000000 kB\nMemAvailable:   67108864 kB\n")  # 64 GB
    assert p.wait(20) == 0 and out.exists()


def test_sets_allocator_conf(tmp_path):
    env = _env(tmp_path)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    r = subprocess.run([SCRIPT, "bash", "-c", "echo $PYTORCH_CUDA_ALLOC_CONF"],
                        env=env, capture_output=True, text=True)
    assert "expandable_segments:True" in r.stdout


def _run_killable(argv, env, timeout):
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        raise AssertionError(f"timed out after {timeout} s (deadlock?): {argv}")
    return p.returncode, out, err


def test_nested_call_inside_a_gpu1_job_completes(tmp_path):
    env = _env(tmp_path)
    out = tmp_path / "ran"
    inner = f"{SCRIPT} bash -c 'touch {out}; exit 5'"
    rc, _, err = _run_killable([SCRIPT, "bash", "-c", inner], env, timeout=10)
    assert rc == 5 and out.exists(), err
    # the outer job still holds the art lock while it runs: a sibling job queues behind it
    log = tmp_path / "log"
    a = subprocess.Popen([SCRIPT, "bash", "-c", f"{SCRIPT} bash -c 'echo start >> {log}; sleep 1; echo end >> {log}'"],
                         env=env)
    time.sleep(0.3)
    b = subprocess.Popen([SCRIPT, "bash", "-c", f"echo start >> {log}; echo end >> {log}"], env=env)
    assert a.wait(20) == 0 and b.wait(20) == 0
    assert log.read_text().split() == ["start", "end", "start", "end"]


def test_failed_blocking_flock_does_not_run_the_job(tmp_path):
    env = _env(tmp_path)
    badbin = tmp_path / "badbin"; badbin.mkdir()
    fl = badbin / "flock"
    fl.write_text("#!/usr/bin/env bash\nexit 1\n")                 # both the -n try and the blocking wait fail
    fl.chmod(fl.stat().st_mode | stat.S_IEXEC)
    env["PATH"] = f"{badbin}:{env['PATH']}"
    out = tmp_path / "ran"
    rc, _, err = _run_killable([SCRIPT, "bash", "-c", f"touch {out}"], env, timeout=10)
    assert rc != 0 and not out.exists()
    assert "lock" in err
