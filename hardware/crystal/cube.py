"""cube.py - crystal M2 (GPU): kernel name + fingerprint of torch.mm at every (m, k, n) in [1, 256]^3.

One k-slab (65 536 shapes, (m, n) in [1,256]^2) per checkpoint: cache/cube_<dtype>/k<kkk>.npz with
  names_vocab  raw ' | '-joined profiler kernel names seen in the slab
  name_id      (256, 256) uint16 index into names_vocab, [m-1, n-1]
  fp           (256, 256, 6) int32 fingerprint (lattice method, common.py)
  smi          nvidia-smi state at slab start (other compute apps, util, temp)
  check        load check: labels of 200 fixed shapes (first 200 of the M1 sample) re-profiled at this slab,
               compared with M1's cache/m1_kern.npz (kernel names do not depend on probe values)
A rerun skips finished slabs. Order: the four spread slabs first (so --max-hours can decide early), then the rest.

    python cube.py --dtype bf16
    python cube.py --dtype fp32 --max-hours 1      # skips (writes SKIPPED.json) if the projection exceeds 1 h
"""
import argparse, json, os, time
import numpy as np
import torch
from common import CACHE, Probes, fingerprints, kernel_names, kernel_label, smi, stack

G = 256
FIRST = [32, 96, 160, 256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--nref", type=int, default=200)
    args = ap.parse_args()
    out = f"{CACHE}/cube_{args.dtype}"; os.makedirs(out, exist_ok=True)
    probes = Probes(args.dtype)
    info = dict(stack=stack(args.dtype))
    json.dump(info, open(f"{out}/stack.json", "w"), indent=1)
    print(json.dumps(info), flush=True)

    ref = np.load(f"{CACHE}/m1_kern.npz")
    ref_shapes = ref["shapes"][:args.nref]
    ref_labels = [kernel_label(x) for x in ref["names"][:args.nref]]
    check_ref = args.dtype == "bf16"          # M1 reference is bf16

    v = np.arange(1, G + 1); M, N = np.meshgrid(v, v, indexing="ij")
    order_k = FIRST + [k for k in range(1, G + 1) if k not in FIRST]
    fingerprints(np.array([[8, 8, 8]] * 16), probes)   # warm up
    walls = []
    t_job = time.time()
    for i, k in enumerate(order_k):
        path = f"{out}/k{k:03d}.npz"
        if os.path.exists(path):
            continue
        s0 = smi(); t0 = time.perf_counter()
        sh = np.stack([M.ravel(), np.full(M.size, k), N.ravel()], 1)
        order = np.random.default_rng(k).permutation(len(sh))
        fp_o = fingerprints(sh[order], probes)
        t_fp = time.perf_counter() - t0
        names_o = kernel_names(sh[order], probes)
        t_nm = time.perf_counter() - t0 - t_fp
        fp = np.zeros_like(fp_o); fp[order] = fp_o
        vocab = sorted(set(names_o)); vid = {x: j for j, x in enumerate(vocab)}
        name_id = np.zeros(len(sh), np.uint16); name_id[order] = [vid[x] for x in names_o]
        # load check at every slab (flagged by whether another compute process was present)
        chk = {}
        if check_ref:
            s_chk = smi()
            lab = [kernel_label(x) for x in kernel_names(ref_shapes, probes)]
            chk = dict(n=len(lab), agree=int(sum(a == b for a, b in zip(lab, ref_labels))),
                       other_apps=s_chk["other_apps"], util=s_chk["util"],
                       differ=[(ref_shapes[j].tolist(), lab[j], ref_labels[j]) for j in range(len(lab))
                               if lab[j] != ref_labels[j]][:10])
        s1 = smi()
        wall = time.perf_counter() - t0
        torch.cuda.empty_cache()
        np.savez_compressed(path, names_vocab=np.array(vocab), name_id=name_id.reshape(G, G),
                            fp=fp.reshape(G, G, 6).astype(np.int32), wall=wall, wall_fp=t_fp, wall_names=t_nm,
                            info=json.dumps(dict(smi_before=s0, smi_after=s1, check=chk)))
        walls.append(wall)
        busy = bool(s0["other_apps"] or s1["other_apps"] or chk.get("other_apps"))
        print(f"k={k} ({i+1}/{G}) {wall:.1f}s (fp {t_fp:.1f} names {t_nm:.1f}) vocab {len(vocab)} "
              f"busy={busy} check {chk.get('agree')}/{chk.get('n')} temp {s0['temp']} util {s0['util']} "
              f"elapsed {time.time()-t_job:.0f}s", flush=True)
        if args.max_hours and len(walls) == len(FIRST):
            proj = np.mean(walls) * G / 3600
            print(f"projection {proj:.2f} h (limit {args.max_hours} h)", flush=True)
            if proj > args.max_hours:
                json.dump(dict(projected_h=proj, limit_h=args.max_hours, slab_walls=walls,
                               time=time.strftime("%F %T")), open(f"{out}/SKIPPED.json", "w"), indent=1)
                print("SKIPPED: projection over limit", flush=True)
                return
    print(f"done {time.time()-t_job:.0f}s", flush=True)


if __name__ == "__main__":
    main()
