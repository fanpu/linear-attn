import os, subprocess, time, pathlib, stat

SCRIPT = "/home/fzeng/ml/research/art/_shared/gpu1.sh"


def _env(tmp, others_flag=None, ignore=False):
    fake = tmp / "bin"; fake.mkdir(exist_ok=True)
    smi = fake / "nvidia-smi"
    flag = others_flag or (tmp / "no_such_flag")
    smi.write_text(f"#!/usr/bin/env bash\n[ -e {flag} ] && echo '123, python'\nexit 0\n")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}",
               GPU1_AUTONOMOUS_LOCK=str(tmp / "autonomous" / "tools" / ".gpu.lock"),
               GPU1_POLL_S="0.2")
    (tmp / "autonomous" / "tools").mkdir(parents=True, exist_ok=True)
    if ignore:
        env["GPU1_IGNORE_OTHERS"] = "1"
    return env


def test_exit_status_passthrough(tmp_path):
    env = _env(tmp_path, ignore=True)
    r = subprocess.run([SCRIPT, "bash", "-c", "exit 7"], env=env, capture_output=True, text=True)
    assert r.returncode == 7


def test_two_jobs_are_serialised(tmp_path):
    env = _env(tmp_path, ignore=True)
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
    env = _env(tmp_path, others_flag=flag)
    out = tmp_path / "ran"
    p = subprocess.Popen([SCRIPT, "bash", "-c", f"touch {out}"], env=env)
    time.sleep(1.0)
    assert not out.exists()
    flag.unlink()
    assert p.wait(20) == 0 and out.exists()


def test_waits_for_autonomous_lock_but_not_when_paused(tmp_path):
    env = _env(tmp_path, ignore=True)
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
