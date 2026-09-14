"""sweep.py - measure torch.mm (cuBLAS) over output shapes (m, n) with the inner dimension k fixed.

    C = A @ B,   A in R^{m x k},  B in R^{k x n},   C in R^{m x n}

Every entry C_ij is a dot product of the same k numbers (row 0 of a master A with column 0 of a
master B: all rows of A are identical, all columns of B are identical), so its bits depend only on
the order in which the kernel accumulates those k products. That makes the output bits a
shape-independent fingerprint of the algorithm (cf. misc/strassen/strassen_forensics.py).

Per shape (randomized order):
  * warmup 3 calls; estimate one call; choose N calls per window so a window lasts >= --window s;
  * --reps synchronized windows (perf_counter around N calls + torch.cuda.synchronize());
    per-call time = median over windows; also min and IQR/median (noise);
  * fingerprint: raw bits of C[0,0], C[m//2,n//2], C[-1,-1] and #distinct values of C, for two
    cancellation probes (big exponent E from PROBE_E) - see masters().
Every --ref-every shapes a fixed reference shape (and a 1x1 overhead probe) is re-timed (drift).
nvidia-smi (util, temp, power) + compute-apps + loadavg are logged before and after; a sweep is
flagged contaminated if another compute process appears or util before start > 5 %.

Modes: --mode time (default) or --mode kernels (torch.profiler kernel names per shape, chunked).

    python sweep.py --dtype bf16 --grid g256            # every integer in [1,256]^2
    python sweep.py --dtype fp32 --grid g256 --tf32
    python sweep.py --dtype fp16 --grid s2048           # stride-13 grid to 2041
    python sweep.py --dtype bf16 --grid g256 --mode kernels
"""
import argparse, ctypes, datetime, gc, json, os, subprocess, tempfile, time, bisect
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DT = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
IT = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}


def grid(name):
    if name == "g256":
        v = np.arange(1, 257)
    elif name == "g128":
        v = np.arange(1, 129)
    elif name.startswith("p"):                      # pilot: p3 = stride 3 in [1,256]
        v = np.arange(1, 257, int(name[1:]))
    elif name == "s2048":
        v = np.arange(13, 2049, 13)
    else:
        raise ValueError(name)
    return v


def smi():
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    apps = [a for a in apps.splitlines() if a and not a.startswith(str(os.getpid()))]
    u, t, p = [x.strip() for x in q.split(",")]
    return dict(time=datetime.datetime.now().isoformat(timespec="seconds"), util=u, temp=t, power=p,
                other_apps=apps, loadavg=os.getloadavg())


def stack(dtype, k, tf32):
    lt = ctypes.CDLL("libcublasLt.so.13"); lt.cublasLtGetVersion.restype = ctypes.c_size_t
    cb = ctypes.CDLL("libcublas.so.13"); v = ctypes.c_int(); cbv = []
    for i in range(3):
        cb.cublasGetProperty(i, ctypes.byref(v)); cbv.append(v.value)
    drv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return dict(gpu=drv, capability=list(torch.cuda.get_device_capability()), arch="sm_121a",
                torch=torch.__version__, cuda=torch.version.cuda, cublas=".".join(map(str, cbv)),
                cublasLt=int(lt.cublasLtGetVersion()), dtype=dtype, k=k, allow_tf32=tf32,
                blas=str(torch.backends.cuda.preferred_blas_library()),
                CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                date=datetime.date.today().isoformat())


PROBE_E = dict(fp32=(20, 12), fp16=(10, 8), bf16=(20, 12))


def masters(k, dtype, E, seed=0):
    """Cancellation probe: 64 'big' products of exactly +-2^E (half +, half -, cancel exactly) hidden
    among k-64 small products in [2^-12, 2^-4] with random signs. The exact dot product is ~0.05, so
    rounding in the kernel's accumulator (absorbing small terms next to big partial sums) dominates
    the low bits of the result, and different accumulation orders / precisions give different bits."""
    g = torch.Generator().manual_seed(seed)
    a = 2.0 ** (torch.rand(k, generator=g, dtype=torch.float64) * 4 - 6)
    b = 2.0 ** (torch.rand(k, generator=g, dtype=torch.float64) * 4 - 6)
    a *= torch.sign(torch.rand(k, generator=g, dtype=torch.float64) - .5)
    idx = torch.randperm(k, generator=g)[:64]; sg = torch.ones(64, dtype=torch.float64); sg[32:] = -1
    a[idx] = sg * 2.0 ** (E // 2); b[idx] = 2.0 ** (E - E // 2)
    return a.to(dtype).cuda(), b.to(dtype).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=DT)
    ap.add_argument("--grid", default="g256")
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--mode", default="time", choices=["time", "kernels"])
    ap.add_argument("--window", type=float, default=1.5e-3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ref-every", type=int, default=256)
    ap.add_argument("--ref", type=int, nargs=2, default=[200, 200])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    dtype, k = DT[args.dtype], args.k
    v = grid(args.grid)
    M, Nn = np.meshgrid(v, v, indexing="ij")
    shapes = np.stack([M.ravel(), Nn.ravel()], 1)
    order = np.random.default_rng(args.seed).permutation(len(shapes))
    name = f"{args.mode}_{args.dtype}{'_tf32' if args.tf32 else ''}_{args.grid}_k{k}{args.tag}"
    E1, E2 = PROBE_E[args.dtype]
    a, b = masters(k, dtype, E1, 0)
    a2, b2 = masters(k, dtype, E2, 1)
    info = dict(stack=stack(args.dtype, k, args.tf32), args=vars(args), smi_before=smi())
    print(json.dumps(info), flush=True)
    if info["smi_before"]["other_apps"]:
        print("WARNING other compute apps present", flush=True)

    def mats(m, n, a=a, b=b):
        A = a[None, :].expand(m, k).contiguous()
        B = b[:, None].expand(k, n).contiguous()
        return A, B, torch.empty(m, n, device="cuda", dtype=dtype)

    t0 = time.time()
    gc.disable()
    if args.mode == "time":
        S = len(shapes)
        t_med, t_min, iqr = np.zeros(S), np.zeros(S), np.zeros(S)
        nwin = np.zeros(S, np.int32); fp = np.zeros((S, 6), np.int64); nuniq = np.zeros((S, 2), np.int32)
        pos = np.zeros(S, np.int32); ref = []

        def timeit(A, B, C):
            for _ in range(3):
                torch.mm(A, B, out=C)
            torch.cuda.synchronize(); s = time.perf_counter(); torch.mm(A, B, out=C); torch.cuda.synchronize()
            est = time.perf_counter() - s
            N = int(min(5000, max(1, args.window / max(est, 1e-7))))
            ts = []
            for _ in range(args.reps):
                torch.cuda.synchronize(); s = time.perf_counter()
                for _ in range(N):
                    torch.mm(A, B, out=C)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - s) / N)
            ts = np.array(ts)
            q1, q3 = np.percentile(ts, [25, 75])
            return np.median(ts), ts.min(), (q3 - q1) / np.median(ts), N

        refA = mats(*args.ref); oneA = mats(1, 1)
        for j, idx in enumerate(order):
            if j % args.ref_every == 0:
                r = timeit(*refA)[0]; o = timeit(*oneA)[0]
                ref.append((j, time.time() - t0, r, o))
                if j % (args.ref_every * 16) == 0:
                    print(f"{j}/{S} {time.time()-t0:.0f}s ref {r*1e6:.2f}us one {o*1e6:.2f}us", flush=True)
            m, n = map(int, shapes[idx])
            A, B, C = mats(m, n)
            t_med[idx], t_min[idx], iqr[idx], nwin[idx] = timeit(A, B, C)
            A2, B2, C2 = mats(m, n, a2, b2); torch.mm(A2, B2, out=C2)
            ci, ci2 = C.view(IT[dtype]), C2.view(IT[dtype])
            fp[idx] = torch.stack([ci[0, 0], ci[m // 2, n // 2], ci[-1, -1],
                                   ci2[0, 0], ci2[m // 2, n // 2], ci2[-1, -1]]).cpu().numpy()
            nuniq[idx] = (torch.unique(C).numel(), torch.unique(C2).numel())
            del A2, B2, C2
            pos[idx] = j
            if not torch.isfinite(C[0, 0]):
                print("nonfinite", m, n)
        r = timeit(*refA)[0]; o = timeit(*oneA)[0]; ref.append((S, time.time() - t0, r, o))
        info["smi_after"] = smi(); info["wall_s"] = time.time() - t0
        np.savez_compressed(f"{HERE}/cache/{name}.npz", grid=v, m=shapes[:, 0], n=shapes[:, 1], t_med=t_med,
                            t_min=t_min, iqr=iqr, nwin=nwin, fp=fp, nuniq=nuniq, pos=pos, ref=np.array(ref),
                            info=json.dumps(info))
    else:
        prof_names = [""] * len(shapes)
        chunk = 1024 if v.max() <= 256 else 48
        for c0 in range(0, len(shapes), chunk):
            ids = order[c0:c0 + chunk]
            ms = [mats(*map(int, shapes[i])) for i in ids]
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
                prof_names[i] = " | ".join(per.get(int(i), []))
            print(f"{c0+len(ids)}/{len(shapes)} {time.time()-t0:.0f}s", flush=True)
        info["smi_after"] = smi(); info["wall_s"] = time.time() - t0
        np.savez_compressed(f"{HERE}/cache/{name}.npz", grid=v, m=shapes[:, 0], n=shapes[:, 1],
                            names=np.array(prof_names), info=json.dumps(info))
    gc.enable()
    print(json.dumps(dict(smi_after=info["smi_after"], wall_s=info["wall_s"])), flush=True)


if __name__ == "__main__":
    main()
