"""Boiling-film measurement: many independent Ideami-style measurements (kind B: a FRESH random
minibatch of 16 CIFAR-10 train images per grid point, BN in train mode) of the SAME plane as
measure.py, one full res x res grid per film frame.

    python boil.py --res 256 --frames 120 --sub1024 [--P 1024]
writes cache/boil/B_r{res}_f{k:03d}_{loss,acc}.npy; frames already on disk are skipped (resume).

Evaluator: dl_common.BatchedNet (grouped conv, per-net images), the same code that measured B
and B'.  bench_boil.py measured it at ~4,100 points/s (16 s per 256^2 frame); a stacked-direction
alternative was 2.3x slower.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from dl_common import CACHE, grid_axis


def main():
    import pasar_job
    from dl_common import BatchedNet, load_cifar, load_trained, random_direction
    p = argparse.ArgumentParser()
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--P", type=int, default=1024)
    p.add_argument("--sub1024", action="store_true",
                   help="use every 4th point of the 1024^2 axis (the exact 256^2 subgrid of B, B', C)")
    p.add_argument("--seed0", type=int, default=101, help="frame k uses sampling seed seed0 + k")
    args = p.parse_args()
    pasar_job.apply_memory_limit()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    net = load_trained(dev)
    bnet = BatchedNet(net, random_direction(net, 1), random_direction(net, 2), dev)
    xtr, ytr = load_cifar(True, dev)
    R = args.res
    axn = grid_axis(1024)[::4][:R] if args.sub1024 else grid_axis(R)
    ax = torch.tensor(axn, device=dev, dtype=torch.float32)
    Bg, Ag = torch.meshgrid(ax, ax, indexing="ij")          # row = beta, col = alpha (as measure.py)
    A_, B_ = Ag.flatten().contiguous(), Bg.flatten().contiguous()
    out = os.path.join(CACHE, "boil"); os.makedirs(out, exist_ok=True)
    done = [k for k in range(args.frames) if os.path.exists(os.path.join(out, f"B_r{R}{'s' if args.sub1024 else ''}_f{k:03d}_acc.npy"))]
    if done and pasar_job.resuming():
        pasar_job.resumed(len(done))
    t0 = time.time()
    for k in range(args.frames):
        fn = os.path.join(out, f"B_r{R}{'s' if args.sub1024 else ''}_f{k:03d}")
        if os.path.exists(fn + "_acc.npy"):
            continue
        g = torch.Generator(device=dev).manual_seed(args.seed0 + k)
        loss = torch.empty(R * R, device=dev, dtype=torch.float64); acc = torch.empty(R * R, device=dev)
        with torch.no_grad():
            for c0 in range(0, R * R, args.P):
                a, b = A_[c0:c0 + args.P], B_[c0:c0 + args.P]
                idx = torch.randint(0, 50000, (a.numel(), 16), device=dev, generator=g)
                lg = bnet.forward(xtr[idx], a, b, bn_train=True, shared_input=False)                                       # (16, P, 10)
                y = ytr[idx].t()
                loss[c0:c0 + args.P] = F.cross_entropy(lg.permute(0, 2, 1), y, reduction="none").double().mean(0)
                acc[c0:c0 + args.P] = (lg.argmax(-1) == y).float().mean(0)
        np.save(fn + "_loss.npy", loss.view(R, R).float().cpu().numpy())
        np.save(fn + "_acc.npy", acc.view(R, R).cpu().numpy())         # acc last = frame complete
        pasar_job.checkpoint(k + 1)
        pasar_job.progress(k + 1, args.frames, loss_mean=float(loss.mean()))
        print(json.dumps(dict(frame=k, secs=round(time.time() - t0, 1), loss_mean=float(loss.mean()))), flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
