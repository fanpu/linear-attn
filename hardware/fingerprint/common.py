"""Shared helpers for Fingerprint: stack record, nvidia-smi snapshots, kernel-name capture, bit diffs."""
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
GALLERY = os.path.join(ROOT, "gallery")
LOGS = os.path.join(ROOT, "logs")
DECODE_MAP = "/home/fzeng/ml/research/art/decode-map"   # reuse its hand-written Qwen3 (bf16 body, fp32 head)
PY = sys.executable


def smi(tag):
    """Log a full nvidia-smi dump + compute apps + loadavg to logs/smi_<tag>.txt; return a short dict."""
    full = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    load = open("/proc/loadavg").read().strip()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(LOGS, f"smi_{tag}.txt"), "w") as f:
        f.write(f"{now}\nloadavg {load}\n{full}\ncompute apps:\n{apps}\n")
    return dict(time=now, util_temp_power=q, apps=apps.strip(), loadavg=load)


def stack():
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    try:
        from importlib.metadata import version
        cublas = version("nvidia-cublas")
    except Exception:
        cublas = "?"
    return dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                driver_name=drv, cuda=torch.version.cuda, torch=torch.__version__,
                cublas=cublas, cublasLt=torch._C._cuda_getCompiledVersion() if hasattr(torch._C, "_cuda_getCompiledVersion") else None,
                cudnn=torch.backends.cudnn.version(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                date=datetime.date.today().isoformat())


def cuda_kernels(fn):
    """Run fn() once under torch.profiler; return sorted unique CUDA kernel names (a dispatch signature)."""
    act = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=act) as prof:
        fn()
        torch.cuda.synchronize()
    names = set()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            names.add(e.name)
    return sorted(names)


def int_view(t):
    """Bit pattern of a float tensor as a numpy int array (same width)."""
    it = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32,
          torch.float64: torch.int64}[t.dtype]
    return t.contiguous().view(it).cpu().numpy()


def popcount(a):
    """Per-element number of set bits of an unsigned/signed int array."""
    u = a.view(np.dtype(f"u{a.dtype.itemsize}"))
    b = np.unpackbits(u.view(np.uint8).reshape(*u.shape, u.itemsize), axis=-1)
    return b.sum(-1).astype(np.uint8)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
