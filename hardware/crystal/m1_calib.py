"""m1_calib.py - crystal M1 (GPU): fingerprint rate, determinism, slice check, kernel names, load check.

Phases (each writes cache/m1_<phase>.npz and is skipped if that file exists, so a rerun resumes):
  load0   2000 sampled shapes, fingerprints + nvidia-smi state (start of job)
  rate1   10^4 random shapes in [1,256]^3, fingerprints, wall time (under contention: share mode)
  rate2   the same 10^4 shapes in an independent random order (determinism)
  slice   k = 4096, 64x64 sub-grid of g256 (compare to hardware/lattice/cache/time_bf16_g256_k4096.npz)
  kern    profiler kernel names for the 2000 sampled shapes (+ their fingerprints)
  slab<k> dense (m, n) in [1,256]^2 at fixed k: fingerprints + kernel names (class purity, per-slab wall time)
  plane<n> dense (m, k) in [1,256]^2 at fixed n: fingerprints + kernel names (structure along k)
  loadS   the 2000 shapes again while a thread in this process runs 2048^2 bf16 matmuls (self-made load)
  load1   2000 sampled shapes again + nvidia-smi state (end of job)

    python m1_calib.py [--phases load0,rate1,...] [--tag X]
"""
import argparse, json, os, threading, time
import numpy as np
import torch
from common import CACHE, Probes, fingerprints, kernel_names, smi, stack, slice_subgrid

SLABS = [3, 16, 64, 127, 128, 256]
PLANES = [127, 128]


def rand_shapes(N, seed):
    return np.random.default_rng(seed).integers(1, 257, size=(N, 3))  # columns (m, k, n)


def save(name, **kw):
    np.savez_compressed(f"{CACHE}/m1_{name}.npz", **kw)
    print(f"saved m1_{name}", flush=True)


def timed_fp(shapes, probes):
    torch.cuda.synchronize(); s0 = smi(); t = time.perf_counter()
    fp = fingerprints(shapes, probes)
    torch.cuda.synchronize(); wall = time.perf_counter() - t
    torch.cuda.empty_cache()
    return fp, wall, s0, smi()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--phases", default="load0,rate1,rate2,slice,kern," + ",".join(f"slab{k}" for k in SLABS)
                    + "," + ",".join(f"plane{n}" for n in PLANES) + ",loadS,load1")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    probes = Probes(args.dtype)
    info = dict(stack=stack(args.dtype))
    print(json.dumps(info), flush=True)
    samp = rand_shapes(2000, 7)
    toy = rand_shapes(10000, 0)
    # warm the CUDA context / cuBLAS handles so the first phase's rate is not a start-up cost
    fingerprints(rand_shapes(64, 99), probes)

    for ph in args.phases.split(","):
        name = ph + args.tag
        if os.path.exists(f"{CACHE}/m1_{name}.npz"):
            print("skip", name, flush=True); continue
        print(f"=== {ph} {time.strftime('%H:%M:%S')}", flush=True)
        if ph.startswith("load") and ph != "loadS":
            fp, wall, s0, s1 = timed_fp(samp, probes)
            save(name, shapes=samp, fp=fp, wall=wall, info=json.dumps(dict(info, smi_before=s0, smi_after=s1)))
        elif ph == "loadS":
            stop = threading.Event(); count = [0]

            def burn():
                g = torch.cuda.Stream()
                with torch.cuda.stream(g):
                    X = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16); Y = torch.empty_like(X)
                    while not stop.is_set():
                        torch.mm(X, X, out=Y); g.synchronize(); count[0] += 1
            th = threading.Thread(target=burn); th.start(); time.sleep(2.0)
            fp, wall, s0, s1 = timed_fp(samp, probes)
            stop.set(); th.join()
            save(name, shapes=samp, fp=fp, wall=wall, burn_calls=count[0],
                 info=json.dumps(dict(info, smi_before=s0, smi_after=s1)))
        elif ph in ("rate1", "rate2"):
            order = np.random.default_rng(0 if ph == "rate1" else 1).permutation(len(toy))
            fp_o, wall, s0, s1 = timed_fp(toy[order], probes)
            fp = np.zeros_like(fp_o); fp[order] = fp_o
            print(f"{ph}: {len(toy)} shapes {wall:.1f}s = {wall/len(toy)*1e6:.0f} us/shape", flush=True)
            save(name, shapes=toy, fp=fp, order=order, wall=wall, info=json.dumps(dict(info, smi_before=s0, smi_after=s1)))
        elif ph == "slice":
            v = slice_subgrid(); M, N = np.meshgrid(v, v, indexing="ij")
            sh = np.stack([M.ravel(), np.full(M.size, 4096), N.ravel()], 1)
            order = np.random.default_rng(2).permutation(len(sh))
            fp_o, wall, s0, s1 = timed_fp(sh[order], probes)
            fp = np.zeros_like(fp_o); fp[order] = fp_o
            save(name, shapes=sh, fp=fp, wall=wall, info=json.dumps(dict(info, smi_before=s0, smi_after=s1)))
        elif ph == "kern":
            fp, wall_fp, s0, _ = timed_fp(samp, probes)
            t = time.perf_counter(); names = kernel_names(samp, probes); torch.cuda.synchronize()
            wall_k = time.perf_counter() - t; torch.cuda.empty_cache()
            print(f"kern: fp {wall_fp:.1f}s names {wall_k:.1f}s", flush=True)
            save(name, shapes=samp, fp=fp, names=np.array(names), wall_fp=wall_fp, wall_names=wall_k,
                 info=json.dumps(dict(info, smi_before=s0, smi_after=smi())))
        elif ph.startswith("slab") or ph.startswith("plane"):
            x = int(ph[4:] if ph.startswith("slab") else ph[5:])
            v = np.arange(1, 257); P, Q = np.meshgrid(v, v, indexing="ij")
            if ph.startswith("slab"):     # rows m, cols n, fixed k
                sh = np.stack([P.ravel(), np.full(P.size, x), Q.ravel()], 1)
            else:                          # rows m, cols k, fixed n
                sh = np.stack([P.ravel(), Q.ravel(), np.full(P.size, x)], 1)
            if ph.startswith("plane"):     # group by k so the per-k master cache is hit
                order = np.lexsort((np.random.default_rng(3).random(len(sh)), sh[:, 1]))
            else:
                order = np.random.default_rng(3).permutation(len(sh))
            fp_o, wall_fp, s0, _ = timed_fp(sh[order], probes)
            t = time.perf_counter(); names_o = kernel_names(sh[order], probes); torch.cuda.synchronize()
            wall_k = time.perf_counter() - t; torch.cuda.empty_cache()
            fp = np.zeros_like(fp_o); fp[order] = fp_o
            names = np.empty(len(sh), dtype=object); names[order] = names_o
            print(f"{ph}: fp {wall_fp:.1f}s ({wall_fp/len(sh)*1e6:.0f} us/shape) names {wall_k:.1f}s "
                  f"({wall_k/len(sh)*1e6:.0f} us/shape)", flush=True)
            save(name, shapes=sh, fp=fp, names=names.astype(str), wall_fp=wall_fp, wall_names=wall_k,
                 info=json.dumps(dict(info, smi_before=s0, smi_after=smi())))
        else:
            raise ValueError(ph)
    print("done", flush=True)


if __name__ == "__main__":
    main()
