"""Stills for wild-landscapes: every scene is (height field, colour field) from cache/, rendered by
wl_render.Scene (dense-landscape's lit-solid renderer).  All choices per scene are declared in SCENES.

    python stills.py --scene rnn_T30 --width 4096 --ss 2 --device cuda --out cache/renders/rnn_T30.png
    python stills.py --scene rnn_T30 --width 800 --ss 1 --device cpu --out pilot/p.png      # preview
"""
import argparse, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wl_render as R

C = os.path.join(HERE, "cache")


def oklab_srgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    lin = np.stack([4.0767416621 * l - 3.3077085312 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1).clip(0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def hex_rgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# ------------------------------------------------------------------------------------------ scenes
def rnn(path, floor=-1.5, lam_scale=0.8, pal="crameri_berlin", sub=1):
    """height: log10 loss clipped below at `floor` (the crater floor is flat below it: declared);
    colour: largest Lyapunov exponent lambda of the driven dynamics, 0.5 + lambda/lam_scale through a
    dark-centred diverging palette, so the edge of chaos (lambda = 0) is the dark seam."""
    d = np.load(path)
    F = np.maximum(np.log10(d["loss"].astype(np.float64)), floor)[::sub, ::sub]
    t = np.clip(0.5 + d["lyap"][::sub, ::sub] / lam_scale, 0, 1)
    return F, R.cmap(t, pal), floor


def tiny(path, chroma=0.16):
    """height: the loss itself (linear: its terraces are the number of inputs the net gets wrong);
    colour: function identity, PC1/PC2 of the net's output vector over all inputs -> OKLab a/b,
    lightness from correctness (L = 0.9 at loss 0 down to 0.35 at the maximum)."""
    d = np.load(path)
    F = d["loss"].astype(np.float64)
    pc = d["pc"]
    z = [np.clip(p / (2.2 * p.std()), -1, 1) for p in pc]
    Lk = 0.9 - 0.55 * np.sqrt(np.clip(F / F.max(), 0, 1))
    return F, oklab_srgb(Lk + 0.04 * z[2], chroma * z[0], chroma * z[1]), 0.0


def three(path, gamma=6.0, chroma=0.22, hues=(25, 145, 265), use="dg", lo=None, hi=None, Lhi=0.78, Llo=0.08, Lgam=0.7):
    """height: log10 training loss (2048 fixed images); colour = function identity: w_k = ag_k^gamma
    normalised, where ag_k is the agreement of the point's argmax predictions with net k on the
    'disputed' test probes (where the three nets do not all agree); OKLab hue = sum_k w_k u_k with u_k
    unit vectors at `hues` degrees (net 0 red, net 1 green, net 2 blue), chroma = `chroma` x the length
    of that sum; OKLab lightness from the loss (0.88 at the plane's lowest loss, 0.22 at its highest,
    linear in log10 loss between the given lo/hi or the plane's own range)."""
    d = np.load(path)
    F = np.log10(d["loss"].astype(np.float64))
    A = np.stack([d[f"{use}{k}"] for k in range(3)], -1).astype(np.float64)
    W = A ** gamma; W = W / W.sum(-1, keepdims=True).clip(1e-12)
    U = np.stack([[np.cos(np.radians(h)), np.sin(np.radians(h))] for h in hues])
    ab = chroma * (W @ U)
    lo = F.min() if lo is None else lo; hi = F.max() if hi is None else hi
    Lk = Lhi - (Lhi - Llo) * np.clip((F - lo) / (hi - lo), 0, 1) ** Lgam
    return F, oklab_srgb(Lk, ab[..., 0], ab[..., 1]), float(F.min())


def tiny_sign(path, chroma=0.15, Llo=0.16, Lhi=0.86, pw=2.5):
    """height: the loss (linear).  colour = which Boolean function the net computes (the sign pattern
    of f over all 2^k inputs): lightness from how many inputs it gets right, L = Llo + (Lhi - Llo) *
    (right/2^k)^pw, so only near-solutions are bright; hue = the function's index times the golden
    angle (an arbitrary but fixed label, declared), chroma fixed."""
    d = np.load(path)
    F = d["loss"].astype(np.float64)
    fn = d["fn"].astype(np.int64); yt = int(d["ytrue"])
    n = int(round(np.log2(int(fn.max()) + 1))) if fn.max() > 0 else 1
    nin = max(n, int(np.ceil(np.log2(yt + 1))))
    nin = 1 << int(np.ceil(np.log2(nin))) if nin & (nin - 1) else nin
    x = np.bitwise_xor(fn, yt)
    wrong = np.zeros_like(x)
    for b in range(nin):
        wrong += (x >> b) & 1
    right = nin - wrong
    Lk = Llo + (Lhi - Llo) * (right / nin) ** pw
    ids, inv = np.unique(fn, return_inverse=True)
    h = (ids * 137.50776 % 360.0)[inv.reshape(fn.shape)]
    return F, oklab_srgb(Lk, chroma * np.cos(np.radians(h)), chroma * np.sin(np.radians(h))), 0.0


def field(path, lo=None, hi=None, key="agree", pal="aurora", clo=0.0, chi=1.0, gam=1.0):
    """generic slice: height = log10 loss clipped to [lo, hi] (declared per image, shared within a
    pair); colour = a measured per-point quantity (default: agreement of the point's argmax
    predictions with the unperturbed net, i.e. function identity) through a palettes.py ramp."""
    d = np.load(path)
    F = np.log10(d["loss"].astype(np.float64))
    lo = F.min() if lo is None else lo; hi = F.max() if hi is None else hi
    F = np.clip(F, lo, hi)
    c = d[key].astype(np.float64) if key in d else 1 - (F - lo) / (hi - lo)
    t = np.clip((c - clo) / (chi - clo), 0, 1) ** gam
    return F, R.cmap(t, pal), lo


SCENES = {
    "rnn_T30": dict(fn=lambda: rnn(os.path.join(C, "rnn", "plane_T30_r2049_s2.5.npz")), K=0.25,
                    orbit="0,0.05,0.05,2.5,-100,42,45", key=(200, 30), exposure=1.4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--path", default=None)
    ap.add_argument("--K", type=float, default=None)
    ap.add_argument("--orbit", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--aspect", type=float, default=16 / 9)
    ap.add_argument("--ss", type=int, default=1)
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--up", type=int, default=1)
    ap.add_argument("--fog", type=float, default=0.2)
    ap.add_argument("--ks", type=float, default=0.35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kw", default="{}", help="json kwargs for the scene function")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device == "cuda":
        import pasar_job; pasar_job.apply_memory_limit(); pasar_job.progress(0, 2)
    torch.set_num_threads(int(os.environ.get("NT", "16")))
    t0 = time.time()
    sc = SCENES.get(a.scene, {})
    kind = a.scene.split("_")[0]
    if a.path:
        F, col, Z0 = {"rnn": rnn, "tiny": tiny, "tinysign": tiny_sign, "three": three, "field": field}[kind](a.path, **json.loads(a.kw))
    else:
        F, col, Z0 = sc["fn"]()
    if a.sub > 1:
        F, col = F[::a.sub, ::a.sub], col[::a.sub, ::a.sub]
    K = a.K or sc.get("K", 0.3)
    S = R.Scene(F, col, K=K, Z0=Z0, up=a.up, device=a.device)
    key = tuple(float(v) for v in a.key.split(",")) if a.key else sc.get("key", (200, 30))
    img, G = S.render(a.orbit or sc.get("orbit", "0,0,0.1,2.6,-100,40,45"), width=a.width, aspect=a.aspect, ss=a.ss,
                      key=key, exposure=a.exposure or sc.get("exposure", 1.3), fog=a.fog, ks=a.ks)
    R.save(img, a.out)
    print(json.dumps(dict(out=a.out, scene=a.scene, grid=F.shape[0], K=K, hit=round(float(G["m"].float().mean()), 3), bad_rays=getattr(S, "bad_rays", 0),
                          mean_lum=round(float(img.mean()) / 255, 3), secs=round(time.time() - t0, 1))), flush=True)
    if a.device == "cuda":
        pasar_job.progress(2, 2)


if __name__ == "__main__":
    main()
