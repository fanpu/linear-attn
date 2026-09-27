"""Films (frames -> cache/film/<name>/f%04d.png; encode with ffmpeg afterwards, see README).

  boil     fixed camera and light; frame k = boiling measurement k (cache/boil, a FRESH bs-16
           minibatch at every point of the 256^2 subgrid, BN train).  Also renders the fixed-
           minibatch surface C on the same subgrid once (C.png) for the split version.
  sweep    fixed camera over the dense fur (B1, 256^2 subgrid); the key light's azimuth circles once.
  settle   fixed camera; frame k = the MEAN of the first k+1 boiling measurements: the fur averages
           away (as 1/sqrt(k)) into the smooth mean surface.
  flyover  a slow camera move over a smooth surface (C: one fixed minibatch, or A), raking light.

    python film.py --mode boil --frames 120 --width 2560 --device cuda
Frames on disk are skipped, so a preempted job resumes where it stopped.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d
import hf, solid
from dense import field, colour, Z0, make_cam
from dl_common import CACHE


def save(arr, path):
    from PIL import Image
    Image.fromarray(arr).save(path, compress_level=1)


def smooth(t):
    return t * t * (3 - 2 * t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["boil", "sweep", "flyover", "settle"])
    p.add_argument("--name", default=None)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--width", type=int, default=2560)
    p.add_argument("--ss", type=int, default=2)
    p.add_argument("--up", type=int, default=4)
    p.add_argument("--K", type=float, default=0.4)
    p.add_argument("--colour", default="acc_tam")
    p.add_argument("--t_lo", type=float, default=0.0)
    p.add_argument("--orbit", default="0,0.05,0.1,1.1,-90,50,50")
    p.add_argument("--orbit2", default=None, help="flyover: end camera (orbit params interpolated)")
    p.add_argument("--path", default=None, help="flyover: eye/target keyframes 'ex,ey,ez,tx,ty,tz;...'")
    p.add_argument("--fov", type=float, default=55)
    p.add_argument("--key", default="200,22")
    p.add_argument("--fill", default="20,35")
    p.add_argument("--key_I", type=float, default=3.2)
    p.add_argument("--fill_I", type=float, default=0.55)
    p.add_argument("--ka", type=float, default=0.55)
    p.add_argument("--soft", type=float, default=18.0)
    p.add_argument("--fog", type=float, default=0.25)
    p.add_argument("--ks", type=float, default=0.35)
    p.add_argument("--shin", type=float, default=40.0)
    p.add_argument("--exposure", type=float, default=1.4)
    p.add_argument("--field", default="C")
    p.add_argument("--sub", type=int, default=1)
    p.add_argument("--step", type=float, default=0.6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="render at most this many new frames (preview)")
    a = p.parse_args()
    dev = a.device
    import pasar_job
    if dev == "cuda":
        pasar_job.apply_memory_limit()
    torch.set_num_threads(int(os.environ.get("NT", "20")))
    name = a.name or a.mode
    out = os.path.join(CACHE, "film", name); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
    W = a.width; H = W * 9 // 16
    z0 = Z0()
    key = solid.light_dir(*(float(v) for v in a.key.split(",")))
    fill = solid.light_dir(*(float(v) for v in a.fill.split(","))) if a.fill else None
    kw = dict(fill=fill, key_I=a.key_I, fill_I=a.fill_I, ka=a.ka, fog=a.fog, ks=a.ks, shininess=a.shin)
    todo = [k for k in range(a.frames) if not os.path.exists(os.path.join(out, f"f{k:04d}.png"))]
    if len(todo) < a.frames and pasar_job.resuming():
        pasar_job.resumed(a.frames - len(todo))
    if a.limit:
        todo = todo[:a.limit]
    t0 = time.time()
    alb_cache = {}

    def albedo(n):
        if n not in alb_cache:
            alb_cache[n] = torch.tensor(colour(a.colour, n, a.t_lo), dtype=torch.float32, device=dev)
        return alb_cache[n]

    if a.mode == "settle":       # frame k = mean of the first k+1 fresh measurements (same camera/light)
        cam = make_cam(a.orbit, W, H)
        for i, k in enumerate(todo):
            Hm, GX, GY = solid.prep(field(f"boilmean:{k + 1}", 1), a.up, a.K, z0, dev)
            sh = solid.shadow_fast(Hm, key, softness=a.soft); ao = solid.ao_coarse(Hm, a.up)
            G = solid.gbuffer(Hm, cam, ss=a.ss, step_texels=a.step, device=dev)
            save(solid.tonemap(solid.shade(G, Hm, GX, GY, albedo(Hm.shape[0]), key, shadow=sh, ao=ao, **kw), a.exposure),
                 os.path.join(out, f"f{k:04d}.png"))
            pasar_job.progress(a.frames - len(todo) + i + 1, a.frames)
            print(json.dumps(dict(frame=k, secs=round(time.time() - t0, 1))), flush=True)
    elif a.mode == "boil":
        cam = make_cam(a.orbit, W, H)
        cpath = os.path.join(out, "C.png")
        if not os.path.exists(cpath):          # the fixed-minibatch surface on the same subgrid
            Hm, GX, GY = solid.prep(field("C", 4), a.up, a.K, z0, dev)
            sh = solid.shadow_fast(Hm, key, softness=a.soft); ao = solid.ao_coarse(Hm, a.up)
            G = solid.gbuffer(Hm, cam, ss=a.ss, step_texels=a.step, device=dev)
            save(solid.tonemap(solid.shade(G, Hm, GX, GY, albedo(Hm.shape[0]), key, shadow=sh, ao=ao, **kw), a.exposure), cpath)
        for i, k in enumerate(todo):
            if not os.path.exists(os.path.join(CACHE, "boil", f"B_r256s_f{k:03d}_acc.npy")):
                print("measurement", k, "not on disk yet; stopping"); break
            Hm, GX, GY = solid.prep(field(f"boil:{k}", 1), a.up, a.K, z0, dev)
            sh = solid.shadow_fast(Hm, key, softness=a.soft); ao = solid.ao_coarse(Hm, a.up)
            G = solid.gbuffer(Hm, cam, ss=a.ss, step_texels=a.step, device=dev)
            save(solid.tonemap(solid.shade(G, Hm, GX, GY, albedo(Hm.shape[0]), key, shadow=sh, ao=ao, **kw), a.exposure),
                 os.path.join(out, f"f{k:04d}.png"))
            pasar_job.progress(a.frames - len(todo) + i + 1, a.frames)
            print(json.dumps(dict(frame=k, secs=round(time.time() - t0, 1))), flush=True)
    elif a.mode == "sweep":
        cam = make_cam(a.orbit, W, H)
        Hm, GX, GY = solid.prep(field(a.field, a.sub), a.up, a.K, z0, dev)
        ao = solid.ao_coarse(Hm, a.up)
        G = solid.gbuffer(Hm, cam, ss=a.ss, step_texels=a.step, device=dev)
        az0, el = (float(v) for v in a.key.split(","))
        for i, k in enumerate(todo):
            az = az0 + 360.0 * k / a.frames
            L = solid.light_dir(az, el)
            Lf = solid.light_dir(az + 180, 35) if fill is not None else None
            sh = solid.shadow_fast(Hm, L, softness=a.soft)
            img = solid.shade(G, Hm, GX, GY, albedo(Hm.shape[0]), L, shadow=sh, ao=ao,
                              **dict(kw, fill=Lf))
            save(solid.tonemap(img, a.exposure), os.path.join(out, f"f{k:04d}.png"))
            pasar_job.progress(a.frames - len(todo) + i + 1, a.frames)
            print(json.dumps(dict(frame=k, az=az, secs=round(time.time() - t0, 1))), flush=True)
    else:   # flyover
        Hm, GX, GY = solid.prep(field(a.field, a.sub), a.up, a.K, z0, dev)
        sh = solid.shadow_fast(Hm, key, softness=a.soft); ao = solid.ao_coarse(Hm, a.up * a.sub)
        if a.orbit2:          # orbit parameters (tx,ty,tz,r,az,el,fov) interpolated with ease-in-out
            o1 = np.array([float(v) for v in a.orbit.split(",")]); o2 = np.array([float(v) for v in a.orbit2.split(",")])
        else:
            keys = [[float(v) for v in s.split(",")] for s in a.path.split(";")]
            kf = np.array(keys)                                                     # (n_key, 6)
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(np.linspace(0, 1, len(kf)), kf, axis=0, bc_type="clamped")
        for i, k in enumerate(todo):
            s = smooth(k / max(1, a.frames - 1))
            if a.orbit2:
                tx, ty, tz, rad, az, el, fov = (1 - s) * o1 + s * o2
                cam = r3d.orbit((tx, ty, tz), rad, az_deg=az, el_deg=el, width=W, height=H, fov_deg=fov)
            else:
                e = cs(s)
                cam = r3d.Camera(tuple(e[:3]), tuple(e[3:6]), (0, 0, 1), W, H, a.fov)
            G = solid.gbuffer(Hm, cam, ss=a.ss, step_texels=a.step, device=dev)
            img = solid.shade(G, Hm, GX, GY, albedo(Hm.shape[0]), key, shadow=sh, ao=ao, **kw)
            save(solid.tonemap(img, a.exposure), os.path.join(out, f"f{k:04d}.png"))
            pasar_job.progress(a.frames - len(todo) + i + 1, a.frames)
            print(json.dumps(dict(frame=k, secs=round(time.time() - t0, 1))), flush=True)
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
