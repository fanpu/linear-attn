"""Combed M2 compute: dense trajectories for rendering, and the N = 16 closed-form basin volume.

  setsid nohup art/_shared/gpu1.sh art/.venv/bin/python art/combed/compute_m2.py dense --device cuda > logs/dense.log 2>&1 < /dev/null &
  setsid nohup art/_shared/gpu1.sh art/.venv/bin/python art/combed/compute_m2.py basin --device cuda > logs/basin.log 2>&1 < /dev/null &
  art/.venv/bin/python art/combed/compute_m2.py boxcount

Stop time for every M2 render (controller ruling on M1): t = 1 - 1e-6. The trajectory is the M1 256-step RK4 run on
t in [0, 1 - 1e-3] (all 257 states kept) followed by 96 RK4 steps geometric in (1 - t) to 1 - 1e-6 (every 8th kept).
Outputs are one file per (kind, N) and are skipped if present (resumable).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import combed_common as C
import compute as K

TAIL_STEPS, TAIL_KEEP = 96, 8
T_END = 1.0 - 1e-6


def fields(kind: str, n: int, device: str):
    data = torch.tensor(C.training_set(n), dtype=torch.float64, device=device)
    if kind == "closed":
        return lambda x, t: C.closed_form_velocity(x, t, data), data
    model = K.load_mlp(n).to(device)
    return (lambda x, t: C.mlp_velocity(model, x, t)), data


def integrate_to_end(field, x0: torch.Tensor, keep_all: bool):
    """RK4 256 uniform steps to 1-1e-3, then 96 geometric steps to 1-1e-6. Returns (states, t) or final state."""
    ts = np.concatenate([C.time_grid(256), C.tail_grid(TAIL_STEPS)[1:]])
    keep = np.concatenate([np.arange(257), 256 + np.arange(TAIL_KEEP, TAIL_STEPS + 1, TAIL_KEEP)])
    x = x0.clone()
    states = [x.float().cpu()] if keep_all else None
    keep_set = set(keep.tolist())
    for k in range(len(ts) - 1):
        t, h = float(ts[k]), float(ts[k + 1] - ts[k])
        k1 = field(x, t)
        k2 = field(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = field(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = field(x + h * k3, t + h)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if keep_all and k + 1 in keep_set:
            states.append(x.float().cpu())
    if keep_all:
        return torch.stack(states).numpy(), ts[keep]
    return x


def dense(kind: str, n: int, device: str):
    out = C.CACHE / f"dense_{kind}_N{n}.npz"
    if out.exists():
        print(f"[dense] {kind} N={n} exists", flush=True)
        return
    t0 = time.time()
    field, data = fields(kind, n, device)
    z = np.load(C.CACHE / f"flow_{kind}_N{n}.npz")
    x0 = torch.tensor(z["x0"], device=device)
    states, t = integrate_to_end(field, x0, keep_all=True)
    # consistency with the M1 CPU run (same arithmetic up to device rounding)
    gap256 = float(np.abs(states[256].astype(np.float64) - z["end256"]).max())
    end = states[-1].astype(np.float64)
    i, d1, d2 = C.nn_two(end, z["data"])
    np.savez(out, states=states, t=t, data=z["data"], end_nn=i, end_d1=d1, end_d2=d2, device=device,
             gap_vs_m1_end256=gap256, wall_s=time.time() - t0)
    print(f"[dense] {kind} N={n} {time.time() - t0:.0f}s  |state@1-1e-3 - M1 end256|max={gap256:.2e}  "
          f"mem(1-1e-6)={(d1 < C.MEM_RATIO * d2).mean():.4f}", flush=True)
    if device == "cuda":
        torch.cuda.empty_cache()


def basin(res: int, device: str, n: int = 16, chunk: int = 1 << 21):
    """Label each noise voxel of a res^3 grid on [-2.5, 2.5]^3 (grid-point centres, r3d convention) by the training
    point that the closed-form flow reaches at t = 1 - 1e-6. Null: label by the Voronoi cell of x0 itself."""
    out = C.CACHE / f"basin_N{n}_R{res}.npz"
    if out.exists():
        print(f"[basin] R={res} exists", flush=True)
        return
    t0 = time.time()
    field, data = fields("closed", n, device)
    g = torch.linspace(-2.5, 2.5, res, dtype=torch.float64)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    pts = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    labels = np.empty(len(pts), np.int16)
    ratio = np.empty(len(pts), np.float32)
    null = np.empty(len(pts), np.int16)
    data_np = C.training_set(n)
    for a in range(0, len(pts), chunk):
        x0 = pts[a:a + chunk].to(device)
        xe = integrate_to_end(field, x0, keep_all=False).cpu().numpy()
        i, d1, d2 = C.nn_two(xe, data_np)
        labels[a:a + chunk], ratio[a:a + chunk] = i, d1 / d2
        null[a:a + chunk] = C.nn_two(pts[a:a + chunk].numpy(), data_np)[0]
        print(f"[basin] R={res} {a + len(x0)}/{len(pts)} ({time.time() - t0:.0f}s)", flush=True)
    shape = (res, res, res)
    np.savez_compressed(out, labels=labels.reshape(shape), ratio=ratio.reshape(shape), null_voronoi_x0=null.reshape(shape),
                        lo=-2.5, hi=2.5, data=data_np, device=device, wall_s=time.time() - t0)
    if device == "cuda":
        torch.cuda.empty_cache()


def boundary_mask(lab: np.ndarray) -> np.ndarray:
    """Voxels with at least one 6-neighbour of a different label."""
    m = np.zeros(lab.shape, bool)
    for ax in range(3):
        d = np.diff(lab, axis=ax) != 0
        sl_lo = [slice(None)] * 3; sl_lo[ax] = slice(0, -1)
        sl_hi = [slice(None)] * 3; sl_hi[ax] = slice(1, None)
        m[tuple(sl_lo)] |= d
        m[tuple(sl_hi)] |= d
    return m


def box_count(mask: np.ndarray):
    """Occupied boxes of side eps = 1, 2, 4, ... voxels (grid truncated to a multiple of eps)."""
    res = mask.shape[0]
    eps, counts = [], []
    e = 1
    while e <= res // 2:
        m = res // e
        b = mask[:m * e, :m * e, :m * e].reshape(m, e, m, e, m, e).any(axis=(1, 3, 5))
        eps.append(e); counts.append(int(b.sum()))
        e *= 2
    return np.array(eps), np.array(counts)


def fit_dimension(eps, counts, lo_e, hi_e):
    sel = (eps >= lo_e) & (eps <= hi_e)
    slope, _ = np.polyfit(np.log(eps[sel]), np.log(counts[sel]), 1)
    local = -np.diff(np.log(counts)) / np.diff(np.log(eps))
    return float(-slope), local.tolist()


def boxcount():
    res_out = {}
    for res in (128, 256):
        f = C.CACHE / f"basin_N16_R{res}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        row = {}
        for name in ("labels", "null_voronoi_x0"):
            mask = boundary_mask(z[name])
            eps, counts = box_count(mask)
            # large boxes saturate (a box of side res/4 always meets some boundary), so fit small-to-mid eps only
            D, local = fit_dimension(eps, counts, 1, res // 16)
            D_wide, _ = fit_dimension(eps, counts, 1, res // 8)
            row[name] = dict(eps_vox=eps.tolist(), eps_world=(eps * 5.0 / (res - 1)).tolist(), counts=counts.tolist(),
                             D_fit=D, D_fit_range_vox=[1, res // 16], D_fit_wide=D_wide, D_fit_wide_range_vox=[1, res // 8],
                             local_slopes=local,
                             boundary_voxels=int(mask.sum()), n_labels=int(len(np.unique(z[name]))))
        row["label_counts"] = np.bincount(z["labels"].ravel(), minlength=16).tolist()
        row["frac_ratio_lt_third"] = float((z["ratio"] < C.MEM_RATIO).mean())
        row["null_agreement"] = float((z["labels"] == z["null_voronoi_x0"]).mean())
        res_out[res] = row
        print(res, {k: (round(v["D_fit"], 3), round(v["D_fit_wide"], 3), [round(x, 3) for x in v["local_slopes"]], v["boundary_voxels"]) for k, v in row.items()
                    if isinstance(v, dict)}, "mem", row["frac_ratio_lt_third"], "agree_null", row["null_agreement"])
    if 128 in res_out and 256 in res_out:
        res_out["boundary_voxel_ratio_256_over_128"] = res_out[256]["labels"]["boundary_voxels"] / res_out[128]["labels"]["boundary_voxels"]
        print("boundary voxels 256/128:", res_out["boundary_voxel_ratio_256_over_128"], "(surface: ~4, volume-filling: ~8)")
    (C.CACHE / "basin_boxcount.json").write_text(json.dumps(res_out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dense", "basin", "boxcount"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ns", type=int, nargs="*", default=list(C.NS))
    ap.add_argument("--kinds", nargs="*", default=["closed", "mlp"])
    ap.add_argument("--res", type=int, nargs="*", default=[128, 256])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.device == "cuda":
        print("device", torch.cuda.get_device_name(0), torch.__version__, flush=True)
    if a.stage == "dense":
        for n in a.ns:
            for kind in a.kinds:
                dense(kind, n, a.device)
    elif a.stage == "basin":
        for r in a.res:
            basin(r, a.device)
    else:
        boxcount()
