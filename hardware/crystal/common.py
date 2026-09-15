"""common.py - crystal: output fingerprints and profiler kernel names of torch.mm over (m, n, k).

    C = A @ B,   A in R^{m x k},  B in R^{k x n},   C in R^{m x n}

The fingerprint method is lattice's (hardware/lattice/sweep.py), reused exactly:
  * masters(): cancellation probes, copied verbatim for k >= 64 (so the k = 4096 plane reproduces
    lattice's cached g256 fingerprints bit for bit). For k < 64 the original construction cannot place
    64 big terms, so it uses 2*(k//4) big terms instead (declared adaptation, see NOTES.md).
  * mats(): all rows of A identical, all columns of B identical, fresh contiguous tensors, out=C.
  * fingerprint = raw bits of C[0,0], C[m//2,n//2], C[-1,-1] for two probes (E from PROBE_E).
Kernel names: lattice's --mode kernels profiler pass (correlation id -> launch ts -> record_function).
Labels: hardware/lattice/common.py:kernel_label (imported).
"""
import bisect, ctypes, datetime, json, os, subprocess, sys, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, PREV, LOGS = f"{HERE}/cache", f"{HERE}/cache/preview", f"{HERE}/logs"
LATTICE = os.path.normpath(f"{HERE}/../lattice")
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("lattice_common", f"{LATTICE}/common.py")
lattice_common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lattice_common)
kernel_label, short_kernel = lattice_common.kernel_label, lattice_common.short_kernel  # hardware/lattice/common.py

DT = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
IT = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}
PROBE_E = dict(fp32=(20, 12), fp16=(10, 8), bf16=(20, 12))   # hardware/lattice/sweep.py


def masters(k, dtype, E, seed=0):
    """hardware/lattice/sweep.py:masters, verbatim for k >= 64. For k < 64: nbig = 2*(k//4)."""
    g = torch.Generator().manual_seed(seed)
    a = 2.0 ** (torch.rand(k, generator=g, dtype=torch.float64) * 4 - 6)
    b = 2.0 ** (torch.rand(k, generator=g, dtype=torch.float64) * 4 - 6)
    a *= torch.sign(torch.rand(k, generator=g, dtype=torch.float64) - .5)
    nbig = 64 if k >= 64 else 2 * (k // 4)
    idx = torch.randperm(k, generator=g)[:nbig]; sg = torch.ones(nbig, dtype=torch.float64); sg[nbig // 2:] = -1
    a[idx] = sg * 2.0 ** (E // 2); b[idx] = 2.0 ** (E - E // 2)
    return a.to(dtype).cuda(), b.to(dtype).cuda()


class Probes:
    """Per-(k, dtype) master vectors, cached."""
    def __init__(self, dtype_name):
        self.name, self.dtype, self.cache = dtype_name, DT[dtype_name], {}

    def get(self, k):
        if k not in self.cache:
            if len(self.cache) > 512:
                self.cache.clear()
            E1, E2 = PROBE_E[self.name]
            self.cache[k] = (masters(k, self.dtype, E1, 0), masters(k, self.dtype, E2, 1))
        return self.cache[k]


def mats(m, k, n, a, b, dtype):
    """hardware/lattice/sweep.py:main.mats (fresh contiguous A, B; empty C)."""
    A = a[None, :].expand(m, k).contiguous()
    B = b[:, None].expand(k, n).contiguous()
    return A, B, torch.empty(m, n, device="cuda", dtype=dtype)


def fingerprints(shapes, probes, sync_every=256):
    """shapes: (S,3) int array of (m, k, n). Returns (S,6) int64 fingerprints (lattice order)."""
    dtype, it = probes.dtype, IT[probes.dtype]
    S = len(shapes); out = np.zeros((S, 6), np.int64); pend = []
    for j, (m, k, n) in enumerate(shapes):
        m, k, n = int(m), int(k), int(n)
        (a, b), (a2, b2) = probes.get(k)
        A, B, C = mats(m, k, n, a, b, dtype); torch.mm(A, B, out=C)
        A2, B2, C2 = mats(m, k, n, a2, b2, dtype); torch.mm(A2, B2, out=C2)
        ci, ci2 = C.view(it), C2.view(it)
        pend.append(torch.stack([ci[0, 0], ci[m // 2, n // 2], ci[-1, -1],
                                 ci2[0, 0], ci2[m // 2, n // 2], ci2[-1, -1]]))
        if len(pend) == sync_every or j == S - 1:
            out[j + 1 - len(pend):j + 1] = torch.stack(pend).cpu().numpy(); pend = []
    return out


def kernel_names(shapes, probes, chunk=256, log=None):
    """lattice --mode kernels, generalised to (m, k, n). Returns list of ' | '-joined kernel names."""
    dtype = probes.dtype
    names = [""] * len(shapes)
    for c0 in range(0, len(shapes), chunk):
        ids = range(c0, min(c0 + chunk, len(shapes)))
        ms = []
        for i in ids:
            m, k, n = map(int, shapes[i]); (a, b), _ = probes.get(k)
            ms.append(mats(m, k, n, a, b, dtype))
        for A, B, C in ms:
            torch.mm(A, B, out=C)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as p:
            for i, (A, B, C) in zip(ids, ms):
                with torch.profiler.record_function(f"S{i}"):
                    torch.mm(A, B, out=C)
                torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json"); p.export_chrome_trace(path)
            ev = json.load(open(path))["traceEvents"]
        ann = sorted((e["ts"], e["ts"] + e["dur"], int(e["name"][1:])) for e in ev
                     if e.get("cat") == "user_annotation" and e["name"].startswith("S"))
        starts = [x[0] for x in ann]
        launch = {e["args"]["correlation"]: e["ts"] for e in ev
                  if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
        per = {}
        for e in sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"]):
            ts = launch.get(e["args"].get("correlation"))
            if ts is None:
                continue
            q = bisect.bisect_right(starts, ts) - 1
            if q >= 0 and ts <= ann[q][1]:
                per.setdefault(ann[q][2], []).append(e["name"])
        for i in ids:
            names[i] = " | ".join(per.get(int(i), []))
        del ms
    return names


def smi():
    """hardware/lattice/sweep.py:smi, plus the other processes' pids and memory."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    apps = [a for a in apps.splitlines() if a and not a.startswith(str(os.getpid()))]
    u, t, p = [x.strip() for x in q.split(",")]
    return dict(time=datetime.datetime.now().isoformat(timespec="seconds"), util=u, temp=t, power=p,
                other_apps=apps, loadavg=os.getloadavg())


def stack(dtype):
    """hardware/lattice/sweep.py:stack (without k)."""
    lt = ctypes.CDLL("libcublasLt.so.13"); lt.cublasLtGetVersion.restype = ctypes.c_size_t
    cb = ctypes.CDLL("libcublas.so.13"); v = ctypes.c_int(); cbv = []
    for i in range(3):
        cb.cublasGetProperty(i, ctypes.byref(v)); cbv.append(v.value)
    drv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return dict(gpu=drv, capability=list(torch.cuda.get_device_capability()), arch="sm_121a",
                torch=torch.__version__, cuda=torch.version.cuda, cublas=".".join(map(str, cbv)),
                cublasLt=int(lt.cublasLtGetVersion()), dtype=dtype, allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                blas=str(torch.backends.cuda.preferred_blas_library()),
                CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                date=datetime.date.today().isoformat())


def slice_subgrid():
    """64 values in [1, 256] covering every residue mod 4: v_j = 1 + 4j + (j mod 4)."""
    j = np.arange(64)
    return 1 + 4 * j + j % 4
