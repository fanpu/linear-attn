"""timing.py - crystal M2 (GPU, needs an idle machine): time torch.mm over a 64^3 (m, k, n) grid, bf16.

Grid: each axis v_j = 1 + 4j + (j mod 4), j = 0..63 (step ~4 in [1, 256], every residue mod 4 present, so the
parity lamellae are sampled; the same values as the M1 k = 4096 slice check).
Hygiene copied from hardware/lattice/sweep.py (--mode time) and run_all.sh:
  * wait (<= 5 min) for GPU temp <= 50 C and util <= 5 % before starting;
  * shapes in random order (seed 0); per shape 3 warmup calls, 1 estimate, N calls per window so a window lasts
    >= 1.5 ms, median of 5 synchronized windows (also min and IQR/median);
  * every 256 shapes: reference shape (200, 200, 200) and the 1x1x1 overhead probe re-timed, with GPU temp/util;
  * nvidia-smi + compute apps + loadavg before and after; contaminated flag if another compute app appears.
Checkpoint every 16 384 shapes to cache/timing_<dtype>_g64.partial.npz; a rerun resumes (resumption is logged).

    GPU1_EXCLUSIVE=1 gpu1.sh python timing.py
"""
import argparse, json, os, subprocess, time, gc
import numpy as np
import torch
from common import CACHE, Probes, mats, smi, stack, slice_subgrid


def gpu_ut():
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    u, t, p = [x.strip() for x in q.split(",")]
    return float(u), float(t), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--window", type=float, default=1.5e-3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ref-every", type=int, default=256)
    ap.add_argument("--ckpt-every", type=int, default=16384)
    args = ap.parse_args()
    name = f"timing_{args.dtype}_g64"
    final, part = f"{CACHE}/{name}.npz", f"{CACHE}/{name}.partial.npz"
    if os.path.exists(final):
        print("exists", final); return
    v = slice_subgrid()
    Mg, Kg, Ng = np.meshgrid(v, v, v, indexing="ij")
    shapes = np.stack([Mg.ravel(), Kg.ravel(), Ng.ravel()], 1)
    S = len(shapes)
    order = np.random.default_rng(0).permutation(S)
    probes = Probes(args.dtype); dtype = probes.dtype

    for i in range(60):                                   # run_all.sh: wait for a cool, idle GPU
        u, t, _ = gpu_ut()
        if t <= 50 and u <= 5:
            break
        time.sleep(5)
    info = dict(stack=stack(args.dtype), args=vars(args), grid=v.tolist(), smi_before=smi(),
                start_temp=t, start_util=u, waited_s=5 * i)
    info["contaminated_before"] = bool(info["smi_before"]["other_apps"])
    print(json.dumps(info), flush=True)

    if os.path.exists(part):
        z = np.load(part)
        t_med, t_min, iqr, nwin, done = z["t_med"], z["t_min"], z["iqr"], z["nwin"], int(z["done"])
        ref = [tuple(r) for r in z["ref"]]; resumes = json.loads(str(z["resumes"])) + [dict(at=done, smi=info["smi_before"])]
        print(f"resume at {done}", flush=True)
    else:
        t_med, t_min, iqr = np.zeros(S), np.zeros(S), np.zeros(S); nwin = np.zeros(S, np.int32)
        done, ref, resumes = 0, [], []

    def timeit(A, B, C):                                  # hardware/lattice/sweep.py:timeit
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

    def mk(m, k, n):
        (a, b), _ = probes.get(k)
        return mats(m, k, n, a, b, dtype)

    refA, oneA = mk(200, 200, 200), mk(1, 1, 1)
    t0 = time.time(); gc.disable()
    other_seen = []

    def save(path, final_=False):
        np.savez_compressed(path, grid=v, shapes=shapes, order=order, t_med=t_med, t_min=t_min, iqr=iqr, nwin=nwin,
                            done=done, ref=np.array(ref), resumes=json.dumps(resumes, default=str),
                            other_seen=json.dumps(other_seen), info=json.dumps(info, default=str))

    for j in range(done, S):
        if j % args.ref_every == 0:
            r = timeit(*refA)[0]; o = timeit(*oneA)[0]; u, t, p = gpu_ut()
            ref.append((j, time.time() - t0, r, o, t, u))
            if j % (args.ref_every * 16) == 0:
                s = smi()
                if s["other_apps"]:
                    other_seen.append((j, s))
                print(f"{j}/{S} {time.time()-t0:.0f}s ref {r*1e6:.1f}us one {o*1e6:.2f}us temp {t} util {u} "
                      f"other {len(s['other_apps'])}", flush=True)
        if j and j % args.ckpt_every == 0:
            done = j; save(part)
        idx = order[j]
        A, B, C = mk(*map(int, shapes[idx]))
        t_med[idx], t_min[idx], iqr[idx], nwin[idx] = timeit(A, B, C)
        del A, B, C
    done = S
    r = timeit(*refA)[0]; o = timeit(*oneA)[0]; u, t, p = gpu_ut(); ref.append((S, time.time() - t0, r, o, t, u))
    gc.enable()
    info["smi_after"] = smi(); info["wall_s"] = time.time() - t0
    info["contaminated"] = bool(info["contaminated_before"] or info["smi_after"]["other_apps"] or other_seen)
    save(final, True)
    if os.path.exists(part):
        os.remove(part)
    print(json.dumps(dict(smi_after=info["smi_after"], wall_s=info["wall_s"], contaminated=info["contaminated"])),
          flush=True)


if __name__ == "__main__":
    main()
