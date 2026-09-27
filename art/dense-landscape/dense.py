"""Dense-solid stills (renderer v2, solid.py).  usage:
    python dense.py --field B1 --sub 4 --up 6 --colour acc_tam --orbit tx,ty,tz,r,az,el,fov \
        --key 200,22 --width 4096 --ss 2 --device cuda --out cache/renders/G1.png

Fields (log10 loss; the same plane as everything else):
  B1, B2   Ideami protocol, fresh bs16 per point (1024^2 measured); --sub 4 -> its 256^2 subgrid
  C        one fixed minibatch of 16 (1024^2)
  EB       E[B], 128^2 measured          A   the BN-eval 1024-image surface, 256^2 measured
  boil:k   frame k of the boiling-film measurement (256^2 subgrid, boil.py)
  boilmean:m  log10 of the mean loss over boil frames 0..m-1 (Monte-Carlo estimate of E[B])
Colours (data; the palette is declared):
  acc_tam  accuracy of the fixed 1024 images in BN-train minibatches of 16 (E[B] protocol, 128^2),
           mapped linearly from chance (0.10) to the plane's max through the 'tam' palette
           (art/color-research/palettes.py; CIELAB-interpolated purple -> red -> orange -> gold)
  fid      function identity: PCA3 of predictions on 256 probes -> OKLab (a, b, L), saturated
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import r3d
import hf, solid
from scene import ld, up, oklab_to_srgb_linear, srgb2lin
from dl_common import CACHE, grid_axis

K_DEFAULT = 0.4


def Z0():
    return float(np.log10(ld("A_n1024_r1024_rows508-516_loss")).min())


def field(name, sub):
    if name.startswith("boilmean:"):     # log10 of the Monte-Carlo mean loss over the first m boil frames
        m = int(name.split(":")[1])
        L = sum(np.load(os.path.join(CACHE, "boil", f"B_r256s_f{k:03d}_loss.npy")).astype(np.float64) for k in range(m))
        return np.log10(L / m)
    if name.startswith("boil:"):
        k = int(name.split(":")[1])
        return np.log10(np.load(os.path.join(CACHE, "boil", f"B_r256s_f{k:03d}_loss.npy")).astype(np.float64))
    F = {"B1": lambda: np.log10(ld("B_s1_r1024_loss")), "B2": lambda: np.log10(ld("B_s2_r1024_loss")),
         "C": lambda: np.log10(ld("C_r1024_loss")), "EB": lambda: np.log10(ld("A16_n1024_r128_loss")),
         "A": lambda: np.log10(ld("A_n1024_r256_loss"))}[name]()
    if sub > 1:
        n = (F.shape[0] - 1) // sub + 1
        F = F[::sub, ::sub][:n, :n]
    return F


def palette_lut(name, n=512):
    import palettes as P
    cm = P.get(name)
    return np.asarray(cm(np.linspace(0, 1, n)))[:, :3]


def colour(kind, n, t_lo=0.0, chroma=0.26):
    """(n, n, 3) linear-RGB albedo texture."""
    if kind.startswith("acc_"):
        acc = up(ld("A16_n1024_r128_acc"), n)
        t = np.clip((acc - 0.10) / (float(ld("A16_n1024_r128_acc").max()) - 0.10), 0, 1)
        t = t_lo + (1 - t_lo) * t
        lut = palette_lut(kind[4:])
        srgb = np.stack([np.interp(t, np.linspace(0, 1, len(lut)), lut[:, c]) for c in range(3)], -1)
        return srgb2lin(srgb)
    if kind == "fid":
        z = [ld(f"A_n1024_r256_pc{k}") for k in (1, 2, 3)]
        z = [np.clip(zz / (2.5 * zz.std()), -1, 1) for zz in z]
        rgb = oklab_to_srgb_linear(0.62 + 0.12 * z[2], chroma * z[0], chroma * z[1])
        return solid.up_colour(rgb, n)
    raise KeyError(kind)


def make_cam(orbit, W, H):
    tx, ty, tz, rad, az, el, fov = (float(v) for v in orbit.split(","))
    return r3d.orbit((tx, ty, tz), rad, az_deg=az, el_deg=el, width=W, height=H, fov_deg=fov)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="B1")
    p.add_argument("--sub", type=int, default=4)
    p.add_argument("--up", type=int, default=6)
    p.add_argument("--K", type=float, default=K_DEFAULT)
    p.add_argument("--colour", default="acc_tam")
    p.add_argument("--t_lo", type=float, default=0.0)
    p.add_argument("--orbit", default="0,0.1,0.1,1.2,-95,40,50")
    p.add_argument("--key", default="200,22", help="key light azimuth, elevation (deg)")
    p.add_argument("--fill", default="20,35")
    p.add_argument("--key_I", type=float, default=3.2)
    p.add_argument("--fill_I", type=float, default=0.55)
    p.add_argument("--ka", type=float, default=0.55)
    p.add_argument("--soft", type=float, default=18.0, help="shadow penumbra sharpness")
    p.add_argument("--fog", type=float, default=0.25)
    p.add_argument("--ks", type=float, default=0.35)
    p.add_argument("--shin", type=float, default=40.0)
    p.add_argument("--chroma", type=float, default=0.26)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--aspect", type=float, default=16 / 9)
    p.add_argument("--ss", type=int, default=1)
    p.add_argument("--step", type=float, default=0.6)
    p.add_argument("--window", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = a.device
    if dev == "cuda":
        import pasar_job; pasar_job.apply_memory_limit(); pasar_job.progress(0, 4)
    torch.set_num_threads(int(os.environ.get("NT", "20")))
    t0 = time.time()
    T = {}
    def tick(k):
        if dev == "cuda": torch.cuda.synchronize()
        T[k] = round(time.time() - t0 - sum(T.values()), 2)
    F = field(a.field, a.sub)
    Hm, GX, GY = solid.prep(F, a.up, a.K, Z0(), dev)
    n = Hm.shape[0]; tick("prep")
    alb = torch.tensor(colour(a.colour, n, a.t_lo, a.chroma), dtype=torch.float32, device=dev); tick("colour")
    key = solid.light_dir(*(float(v) for v in a.key.split(",")))
    fill = solid.light_dir(*(float(v) for v in a.fill.split(","))) if a.fill else None
    sh = solid.shadow_fast(Hm, key, softness=a.soft, step_texels=0.75); tick("shadow")
    if dev == "cuda": pasar_job.progress(1, 4)
    ao = solid.ao_coarse(Hm, a.up); tick("ao")
    if dev == "cuda": pasar_job.progress(2, 4)
    W = a.width; H = int(round(W / a.aspect))
    cam = make_cam(a.orbit, W, H)
    win = tuple(int(v) for v in a.window.split(",")) if a.window else None
    G = solid.gbuffer(Hm, cam, ss=a.ss, step_texels=a.step, device=dev, window=win); tick("gbuf")
    if dev == "cuda": pasar_job.progress(3, 4)
    img = solid.shade(G, Hm, GX, GY, alb, key, shadow=sh, ao=ao, fill=fill, key_I=a.key_I,
                      fill_I=a.fill_I, ka=a.ka, fog=a.fog, ks=a.ks, shininess=a.shin)
    arr = solid.tonemap(img, a.exposure); tick("shade")
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(arr).save(a.out)
    hitfrac = float(G["m"].float().mean())
    print(json.dumps(dict(out=a.out, field=a.field, sub=a.sub, up=a.up, grid=F.shape[0], fine=n, K=a.K,
                          colour=a.colour, orbit=a.orbit, key=a.key, fill=a.fill, W=W, H=H, ss=a.ss,
                          hit_frac=round(hitfrac, 3), mean_lum=round(float(arr.mean()) / 255, 3),
                          secs=round(time.time() - t0, 1), T=T)), flush=True)
    if dev == "cuda": pasar_job.progress(4, 4)


if __name__ == "__main__":
    main()
