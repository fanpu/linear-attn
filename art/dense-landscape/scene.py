"""Render one scene of the dense landscape.  usage:
    python scene.py --scene truth_fid --width 5120 --ss 2 [--device cuda] [--out gallery/...png]

Heights (all scenes): z = K * (log10 L - Z0), K = 0.25 world units per decade on a plane of width 2
(declared exaggeration; the same K for every scene, so smooth and noisy surfaces share one scale).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d
import hf
from scipy.interpolate import RectBivariateSpline
from dl_common import CACHE, GALLERY, grid_axis

K = 0.25          # world units per decade of loss (plane is 2 wide)  -- declared exaggeration
G = os.path.join(CACHE, "grid")


def ld(n):
    return np.load(os.path.join(G, n + ".npy")).astype(np.float64)


def up(F, n_dst=1024):
    ax = grid_axis(F.shape[0])
    return RectBivariateSpline(ax, ax, F, kx=3, ky=3)(grid_axis(n_dst), grid_axis(n_dst))


# ------------------------------------------------------------------ colour (declared maps)
def oklab_to_srgb_linear(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077085312 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return np.clip(np.stack([r, g, bb], -1), 0, 1)


def fid_colour(n=1024, chroma=0.19, Lc=0.63, Lspan=0.14):
    """Function identity: PCA3 of the net's softmax outputs on 256 fixed test probes (basis: 64^2
    coarse grid of the same plane, cache/probe_pca.npz).  PC1 -> OKLab a, PC2 -> OKLab b,
    PC3 -> OKLab L; each PC scaled by 2.5 sd and clipped to [-1, 1]."""
    z = [ld(f"A_n1024_r256_pc{k}") for k in (1, 2, 3)]
    z = [np.clip(zz / (2.5 * zz.std()), -1, 1) for zz in z]
    rgb = oklab_to_srgb_linear(Lc + Lspan * z[2], chroma * z[0], chroma * z[1])
    return np.stack([up(rgb[..., c], n) for c in range(3)], -1).clip(0, 1)


def ramp(x, stops):
    """piecewise-linear colour ramp; stops: [(pos, (r,g,b)), ...] in linear RGB."""
    pos = np.array([s[0] for s in stops]); col = np.array([s[1] for s in stops])
    return np.stack([np.interp(x, pos, col[:, c]) for c in range(3)], -1)


def srgb2lin(c):
    c = np.asarray(c, float); return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


ACC_STOPS = [(0.10, (0.07, 0.07, 0.22)), (0.25, (0.20, 0.25, 0.60)), (0.45, (0.22, 0.60, 0.64)),
             (0.65, (0.82, 0.78, 0.50)), (0.85, (1.00, 0.94, 0.80))]


def acc_colour(n=1024, src="A"):
    """accuracy on the fixed 1024-image eval set (src A: BN eval, 256^2; src A16: BN train in
    minibatches of 16, 128^2 -- the protocol of B and C).  Declared sRGB ramp: chance 0.10 indigo,
    0.25 blue, 0.45 teal, 0.65 pale gold, 0.85 warm white.  A MEASURED quantity, not height."""
    acc = up(ld("A_n1024_r256_acc" if src == "A" else "A16_n1024_r128_acc"), n)
    return ramp(acc, [(p_, srgb2lin(c_)) for p_, c_ in ACC_STOPS])


def resid_colour(B, EB, sd):
    """minibatch residual z = (log10 B - log10 E[B] - mean) / sd:  z<0 (batch easier than average)
    -> cold blue-white, z>0 (harder) -> ember.  |z| drives saturation; neutral = dim bone."""
    z = np.clip(((B - EB) - (B - EB).mean()) / (2 * sd), -1, 1)
    neutral = srgb2lin((0.62, 0.60, 0.56))
    cold = srgb2lin((0.35, 0.62, 1.00)); hot = srgb2lin((1.00, 0.42, 0.12))
    w = np.abs(z)[..., None] ** 0.8
    return np.where(z[..., None] < 0, neutral * (1 - w) + cold * w, neutral * (1 - w) + hot * w)


# ------------------------------------------------------------------ fields
def fields(which):
    if which == "A":      # 256^2 measured, bicubic to 1024^2 (checked against exact 1024 rows)
        return up(np.log10(ld("A_n1024_r256_loss")))
    if which == "B1":
        return np.log10(ld("B_s1_r1024_loss"))
    if which == "B2":
        return np.log10(ld("B_s2_r1024_loss"))
    if which == "B1_256":  # every 4th measured point: the same measurement at 256^2
        return np.log10(ld("B_s1_r1024_loss"))[::4, ::4][:256, :256]
    if which == "EB":
        return up(np.log10(ld("A16_n1024_r128_loss")))
    if which == "C":
        return np.log10(ld("C_r1024_loss"))
    raise KeyError(which)


CAMS = {
    # eye, target, fov
    "low":   dict(eye=(0.10, -2.05, 0.62), target=(0.0, 0.05, 0.02), fov_deg=38),
    "rim":   dict(eye=(0.0, -1.35, 0.55), target=(0.0, 0.25, -0.05), fov_deg=55),
    "high":  dict(eye=(0.0, -1.55, 1.25), target=(0.0, 0.0, 0.0), fov_deg=45),
    "graze": dict(eye=(-0.35, -1.6, 0.30), target=(0.05, 0.2, 0.0), fov_deg=48),
    "funnel": dict(eye=(0.0, -1.25, 0.95), target=(0.0, 0.1, 0.05), fov_deg=52),
    "icarus": dict(eye=(0.05, -1.05, 0.62), target=(0.0, 0.35, 0.10), fov_deg=60),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--aspect", type=float, default=16 / 9)
    p.add_argument("--ss", type=int, default=1)
    p.add_argument("--cam", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--light", default="-0.75,0.55,0.26")
    p.add_argument("--K", type=float, default=K, help="world units per decade (declared)")
    p.add_argument("--fog", type=float, default=0.35)
    p.add_argument("--orbit", default=None, help="tx,ty,tz,radius,az_deg,el_deg,fov_deg (overrides --cam)")
    p.add_argument("--ka", type=float, default=0.30)
    p.add_argument("--ks", type=float, default=0.6)
    p.add_argument("--shin", type=float, default=80.0)
    p.add_argument("--up", type=int, default=1, help="cubic-spline upsampling factor (declared; passes through every measured point)")
    p.add_argument("--rows_only", action="store_true", help="x-ray: only lines of constant beta")
    p.add_argument("--every", type=int, default=2, help="x-ray: draw every n-th grid line")
    p.add_argument("--sub", type=int, default=1, help="use every sub-th measured point of 1024^2 fields")
    p.add_argument("--window", default=None, help="x0,y0,w,h crop of the output (preview)")
    p.add_argument("--rim", default=None, help="second light x,y,z,intensity (no shadow)")
    args = p.parse_args()
    dev = args.device
    if dev == "cuda":
        import pasar_job; pasar_job.apply_memory_limit(); pasar_job.progress(0, 3)
    torch.set_num_threads(int(os.environ.get("NT", "16")))
    t0 = time.time()
    L = tuple(float(v) for v in args.light.split(","))
    sc = args.scene
    stats = json.load(open(os.path.join(CACHE, "noise_stats.json"))) if os.path.exists(os.path.join(CACHE, "noise_stats.json")) else {}
    cam_name = args.cam or ("low" if sc.startswith("truth") else "rim")
    Z0 = float(np.log10(ld("A_n1024_r1024_rows508-516_loss")).min())   # common datum: min of the exact central rows of A
    xray = sc.startswith("xray")
    if sc in ("truth_fid", "truth_acc", "xray_A"):
        F = fields("A"); alb = fid_colour() if sc != "truth_acc" else acc_colour()
    elif sc in ("noise_acc", "noise_acc_256", "noise_fid_256", "xray_B_acc"):
        key = "B1_256" if sc.endswith("256") else "B1"
        F = fields(key); n = F.shape[0]
        alb = fid_colour(n) if sc.startswith("noise_fid") else acc_colour(n, "A16")
    elif sc in ("noise_resid", "xray_B", "noise_fid", "noise_resid_256"):
        key = "B1_256" if sc.endswith("256") else "B1"
        F = fields(key)
        EB = fields("EB")
        if key == "B1_256":
            EB = EB[::4, ::4][:256, :256]
        sd = stats.get("B_noise_std_log10", 0.1)
        alb = resid_colour(F, EB, sd) if sc != "noise_fid" else fid_colour(F.shape[0])
    elif sc == "noise_B2":
        F = fields("B2"); alb = resid_colour(F, fields("EB"), stats.get("B_noise_std_log10", 0.1))
    elif sc in ("fixed_C", "xray_C"):
        F = fields("C"); alb = acc_colour(1024, "A16")
    elif sc == "mean_EB":
        F = fields("EB"); alb = acc_colour(1024, "A16")
    else:
        raise KeyError(sc)
    if args.sub > 1:                     # a coarser grid of the SAME measurement (declared)
        n_ = (F.shape[0] - 1) // args.sub + 1
        F = F[::args.sub, ::args.sub][:n_, :n_]; alb = alb[::args.sub, ::args.sub][:n_, :n_]
    if args.up > 1:
        n_ = (F.shape[0] - 1) * args.up + 1
        F = up(F, n_); alb = np.stack([up(alb[..., c], n_) for c in range(3)], -1).clip(0, 1)
    Hm = torch.tensor(args.K * (F - Z0), dtype=torch.float32, device=dev)
    albedo = torch.tensor(alb, dtype=torch.float32, device=dev)
    W = args.width; Hh = int(round(W / args.aspect))
    if args.orbit:
        tx, ty, tz, rad, az, el, fov = (float(v) for v in args.orbit.split(","))
        cam = r3d.orbit((tx, ty, tz), rad, az_deg=az, el_deg=el, width=W, height=Hh, fov_deg=fov)
        cam_name = "orbit"
    else:
        cam = r3d.Camera(width=W, height=Hh, **CAMS[cam_name])
    if xray:
        acc = hf.xray(Hm, cam, colour=albedo if sc in ("xray_A", "xray_B_acc") else None,
                      densify=2, every=args.every, device=dev,
                      orients=(0,) if args.rows_only else (0, 1))
        # density normalisation: mean line coverage per lit pixel
        m = acc.sum(-1) > 0
        ref = float(acc.sum(-1)[m].median())
        img = 1 - torch.exp(-args.exposure * 0.6 * acc / ref)
        img = img ** (1 / 1.0)
    else:
        sh = hf.shadow_map(Hm, L)
        if dev == "cuda":
            pasar_job.progress(1, 3)
        ao = hf.ao_map(Hm, n_dirs=12)
        img = hf.render(Hm, albedo, cam, L, ss=args.ss, shadow=sh, ao=ao, device=dev, fog=args.fog,
                        ks=args.ks, shininess=args.shin, ka=args.ka,
                        rim=tuple(float(v) for v in args.rim.split(",")) if args.rim else None,
                        window=tuple(int(v) for v in args.window.split(",")) if args.window else None)
        if dev == "cuda":
            pasar_job.progress(2, 3)
        img = img * args.exposure
        img = img / (1 + img * 0.25)                           # soft shoulder (declared tonemap)
        img = torch.where(img <= 0.0031308, 12.92 * img, 1.055 * img.clamp_min(1e-8) ** (1 / 2.4) - 0.055)
    out = args.out or os.path.join(GALLERY, f"{sc}_{cam_name}_{W}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    arr = (img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    from PIL import Image
    Image.fromarray(arr).save(out, optimize=True)
    lum = arr.mean() / 255
    print(json.dumps(dict(scene=sc, cam=cam_name, out=out, W=W, H=Hh, ss=args.ss, K=args.K, Z0=Z0, L=L,
                          secs=round(time.time() - t0, 1), mean_lum=round(float(lum), 3))), flush=True)
    if dev == "cuda":
        pasar_job.progress(3, 3)


if __name__ == "__main__":
    main()
