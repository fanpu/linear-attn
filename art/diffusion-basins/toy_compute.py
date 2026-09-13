"""Compute every 2-D toy measurement into cache/.  Rendering is separate (render_toy.py).

Maps are evaluated on a square grid of *starting points*:
  - samplers (DDIM / ODE / DDPM-frozen): the plane is the 2-D noise z = x_T (VP units).  In 2-D the
    noise plane IS the whole noise space, so no slice (and no shell problem) is involved.
  - iterated over-relaxed denoiser: the plane is the starting point x in data coordinates.

  python toy_compute.py maps        # sampler x score x layout basin maps
  python toy_compute.py steps       # DDIM step-count sweep (animation)
  python toy_compute.py iter        # over-relaxed denoiser maps + gamma sweep
  python toy_compute.py zoom <tag>  # nested zoom stacks (see ZOOMS)
  python toy_compute.py verify      # box counting, uncertainty exponent, resolution, null + Newton control
"""
import json
import math
import sys
import time

import numpy as np
import torch

from common import CACHE, T_MIN, box_count, boundary_mask, ddim, ddpm_frozen, fit_dimension, gpu_setup, ode_rk4, sigma_of_t
from toy import GMM, analytic_eps_fn, learned_eps_fn, load_net, mixture

DEV = gpu_setup()
F64 = torch.float64
LAYOUTS = ["ring8", "grid25", "scatter12", "tri3"]
HALF = 3.0  # noise-plane half width
ODE_STEPS = 100
SIG_MIN_ANALYTIC = 1e-3


def grid(cx, cy, hw, R, dtype=F64):
    a = (torch.arange(R, device=DEV, dtype=dtype) + 0.5) / R * 2 - 1
    xs, ys = cx + hw * a, cy + hw * a
    Y, X = torch.meshgrid(ys, xs, indexing="ij")  # row = y (bottom row first when origin='lower')
    return torch.stack([X.flatten(), Y.flatten()], 1)


def run_chunked(fn, P, chunk=1 << 20):
    outs = [fn(P[i:i + chunk]) for i in range(0, len(P), chunk)]
    if isinstance(outs[0], tuple):
        return tuple(torch.cat([o[j] for o in outs]) for j in range(len(outs[0])))
    return torch.cat(outs)


def sampler_fn(kind, eps_fn, analytic):
    """kind: ddim<N> | ode | ddpm<N>s<seed>"""
    if kind.startswith("ddim"):
        n = int(kind[4:])
        return lambda z: ddim(eps_fn, z, n)
    if kind == "ode":
        smin = SIG_MIN_ANALYTIC if analytic else None
        return lambda z: ode_rk4(eps_fn, z, ODE_STEPS, sigma_min=smin)
    if kind.startswith("ddpm"):
        n, seed = kind[4:].split("s")
        g = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(int(n), 2, generator=g, dtype=F64).to(DEV)
        return lambda z: ddpm_frozen(eps_fn, z, int(n), noise[:, None, :])
    raise ValueError(kind)


def eps_for(layout, score):
    if score == "analytic":
        return analytic_eps_fn(GMM(layout, DEV))
    net = load_net(layout, DEV, torch.float32)
    return learned_eps_fn(net)


def basin_map(layout, score, kind, cx=0.0, cy=0.0, hw=HALF, R=1024, chunk=1 << 20):
    g = GMM(layout, DEV)
    fn = sampler_fn(kind, eps_for(layout, score), score == "analytic")
    P = grid(cx, cy, hw, R)
    x0 = run_chunked(fn, P, chunk)
    lab = g.label(x0)
    return lab.reshape(R, R).cpu().numpy().astype(np.uint8), x0.reshape(R, R, 2).cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------- iterated map
def fixed_points(g, sigma):
    """zeros of D(x)-x near each mean (independent of gamma); found with gamma=1 iteration"""
    x = g.mu.clone()
    for _ in range(5000):
        x = g.denoise(x, sigma)
    return x


FP_CACHE = {}
TRACK_EVERY = 5  # residual checked every 5 iterations; nu log-interpolated inside the interval


def iterate_map(g, sigma, gamma, P, n_iter, delta=1e-3, track=True):
    """x <- x + gamma (D(x, sigma) - x).  Returns label (nearest stable fixed point, 255 if the end
    point is > 0.1 from every fixed point), smooth convergence count nu (first n with residual < delta,
    fractional part by log interpolation; inf if never), log10 final residual."""
    fp = FP_CACHE.get((g.name, sigma))
    if fp is None:
        fp = FP_CACHE[(g.name, sigma)] = fixed_points(g, sigma)
    x = P.clone()
    nu = torch.full((len(P),), float("inf"), device=DEV, dtype=F64)
    prev = torch.cdist(x, fp).min(1).values
    for n in range(1, n_iter + 1):
        x = x + gamma * (g.denoise(x, sigma) - x)
        if not track:
            continue
        if n % TRACK_EVERY and n != n_iter:
            continue
        r = torch.cdist(x, fp).min(1).values
        hit = torch.isinf(nu) & (r < delta)
        if hit.any():
            frac = torch.log(prev[hit] / delta) / torch.log(prev[hit] / r[hit]).clamp(min=1e-12)
            nu[hit] = n - TRACK_EVERY + TRACK_EVERY * frac.clamp(0, 1)
        prev = r
    d = torch.cdist(x, fp)
    rmin, lab = d.min(1)
    lab = torch.where(rmin < 0.1, lab, torch.full_like(lab, 255))
    return lab, nu, torch.log10(rmin.clamp(min=1e-300))


def iter_map(layout, sigma, gamma, cx, cy, hw, R, n_iter, chunk=1 << 20):
    g = GMM(layout, DEV)
    P = grid(cx, cy, hw, R)
    lab, nu, lr = run_chunked(lambda p: iterate_map(g, sigma, gamma, p, n_iter), P, chunk)
    sh = (R, R)
    return (lab.reshape(sh).cpu().numpy().astype(np.uint8), nu.reshape(sh).cpu().numpy().astype(np.float32),
            lr.reshape(sh).cpu().numpy().astype(np.float32))


def iterate_map_learned(net, sigma, gamma, P, n_iter, fp):
    eps = learned_eps_fn(net)
    x = P.clone()
    for n in range(n_iter):
        D = x - sigma * eps(x, sigma)
        x = x + gamma * (D - x)
    d = torch.cdist(x.to(F64), fp)
    rmin, lab = d.min(1)
    return torch.where(rmin < 0.1, lab, torch.full_like(lab, 255)), torch.log10(rmin.clamp(min=1e-300))


ITER_CFG = {  # layout: (sigma, hero gamma, plane half width)
    "ring8": (0.4, 2.12, 4.0),
    "ring6": (0.4, 2.10, 4.0),
    "scatter12": (0.35, 2.10, 3.2),
}
N_ITER = 2000


# ----------------------------------------------------------------------------- tasks
def task_maps(score):
    t0 = time.time()
    R = 1024 if score == "analytic" else 512
    for layout in ["scatter12", "ring8", "grid25"]:
        kinds = ["ddim10", "ddim50", "ddim1000", "ode", "ddpm1000s0"]
        if layout == "scatter12":
            kinds += ["ddpm1000s1", "ddpm1000s2"]
        if True:
            out = {}
            for kind in kinds:
                t1 = time.time()
                lab, x0 = basin_map(layout, score, kind, R=R, chunk=1 << 18 if score == "learned" else 1 << 20)
                out[f"{kind}_lab"], out[f"{kind}_x0"] = lab, x0
                print(f"maps {layout} {score} {kind} R={R} {time.time()-t1:.0f}s", flush=True)
            np.savez_compressed(f"{CACHE}/maps_{layout}_{score}.npz", hw=HALF, R=R, **out)
    if score != "analytic":
        return
    # ODE step convergence (analytic, scatter12): label agreement between RK4 step counts
    global ODE_STEPS
    res = {}
    labs = {}
    for n in [50, 100, 200, 400]:
        ODE_STEPS = n
        labs[n], x0 = basin_map("scatter12", "analytic", "ode", R=512)
        res[n] = x0
    ODE_STEPS = 100
    conv = {f"{n}_vs_400": float((labs[n] != labs[400]).mean()) for n in [50, 100, 200]}
    conv.update({f"maxdx_{n}_vs_400": float(np.abs(res[n] - res[400]).max()) for n in [50, 100, 200]})
    json.dump(conv, open(f"{CACHE}/ode_convergence.json", "w"), indent=1)
    print("ode convergence", conv, f"total {time.time()-t0:.0f}s", flush=True)


STEP_LIST = sorted(set([int(round(v)) for v in np.geomspace(1, 300, 56)]))


def task_steps(score):
    t0 = time.time()
    todo = [("scatter12", "analytic", 768)] if score == "analytic" else [("scatter12", "learned", 384)]
    for layout, score, R in todo:
        eps = eps_for(layout, score)
        g = GMM(layout, DEV)
        P = grid(0, 0, HALF, R)
        labs, x0s = [], []
        for n in STEP_LIST:
            x0 = run_chunked(lambda z: ddim(eps, z, n), P, 1 << 18)
            labs.append(g.label(x0).reshape(R, R).cpu().numpy().astype(np.uint8))
            x0s.append(x0.reshape(R, R, 2).cpu().numpy().astype(np.float32))
        np.savez_compressed(f"{CACHE}/steps_{layout}_{score}.npz", steps=np.array(STEP_LIST), lab=np.stack(labs),
                            x0=np.stack(x0s), hw=HALF)
        print(f"steps {layout} {score} {time.time()-t0:.0f}s", flush=True)


GAMMAS = np.round(np.concatenate([np.linspace(1.0, 1.9, 19)[:-1], np.linspace(1.9, 2.12, 45)]), 5)  # 2.125 = mode stability limit


def task_iter(part):
    t0 = time.time()
    for layout, (sigma, gam, hw) in ITER_CFG.items():
        if part != "hero":
            break
        lab, nu, lr = iter_map(layout, sigma, gam, 0, 0, hw, 2048 if layout == "ring8" else 1024, N_ITER)
        np.savez_compressed(f"{CACHE}/iter_{layout}_hero.npz", lab=lab, nu=nu, logres=lr, sigma=sigma, gamma=gam, hw=hw,
                            n_iter=N_ITER, fp=fixed_points(GMM(layout, DEV), sigma).cpu().numpy())
        print(f"iter hero {layout} {time.time()-t0:.0f}s", flush=True)
    # gamma sweep for the animation (tri3 and ring8)
    for layout in ["ring8"]:
        if part != "gamma":
            break
        sigma, _, hw = ITER_CFG[layout]
        labs, nus = [], []
        for gam in GAMMAS:
            lab, nu, lr = iter_map(layout, sigma, float(gam), 0, 0, hw, 540, 1000)
            labs.append(lab); nus.append(nu)
        np.savez_compressed(f"{CACHE}/iter_{layout}_gamma.npz", gammas=GAMMAS, lab=np.stack(labs), nu=np.stack(nus),
                            sigma=sigma, hw=hw, n_iter=1000)
        print(f"iter gamma sweep {layout} {time.time()-t0:.0f}s", flush=True)
    # learned-denoiser version of the same map (float32 net, 512^2)
    for layout in ["ring8", "scatter12"]:
        if part != "learned":
            break
        sigma, gam, hw = ITER_CFG[layout]
        net = load_net(layout, DEV, torch.float32)
        g = GMM(layout, DEV)
        fp_exact = fixed_points(g, sigma)
        # fixed points of the learned denoiser: iterate gamma=1 from the exact ones
        eps = learned_eps_fn(net)
        fp = fp_exact.clone()
        for _ in range(3000):
            fp = fp - sigma * eps(fp, sigma)
        P = grid(0, 0, hw, 512)
        out = {}
        for gg in [1.0, gam]:
            lab, lr = run_chunked(lambda p: iterate_map_learned(net, sigma, gg, p, N_ITER // 2, fp), P, 1 << 18)
            lab_a, nu_a, lr_a = iter_map(layout, sigma, gg, 0, 0, hw, 512, N_ITER // 2)
            out[f"learned_g{gg}_lab"] = lab.reshape(512, 512).cpu().numpy().astype(np.uint8)
            out[f"analytic_g{gg}_lab"] = lab_a
        np.savez_compressed(f"{CACHE}/iter_{layout}_learned.npz", sigma=sigma, gamma=gam, hw=hw, fp_learned=fp.cpu().numpy(),
                            fp_exact=fp_exact.cpu().numpy(), **out)
        print(f"iter learned {layout} {time.time()-t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- zooms
def labeller(spec):
    """spec -> function(cx, cy, hw, R) -> uint8 label image"""
    if spec["map"] == "sampler":
        def f(cx, cy, hw, R):
            return basin_map(spec["layout"], spec["score"], spec["kind"], cx, cy, hw, R,
                             chunk=1 << 18 if spec["score"] == "learned" else 1 << 20)[0]
    elif spec["map"] == "iter":
        def f(cx, cy, hw, R):
            g = GMM(spec["layout"], DEV)
            P = grid(cx, cy, hw, R)
            lab = run_chunked(lambda p: iterate_map(g, spec["sigma"], spec["gamma"], p, spec["n_iter"], track=False)[0], P)
            return lab.reshape(R, R).cpu().numpy().astype(np.uint8)
    return f


def bisect_boundary(spec, p, q, n=60):
    """refine a boundary point on the segment p-q whose end labels differ"""
    f = labeller(spec)

    def lab(pt):
        return f(pt[0], pt[1], 0.0, 1)[0, 0]
    lp = lab(p)
    for _ in range(n):
        m = 0.5 * (p + q)
        if lab(m) == lp:
            p = m
        else:
            q = m
    return 0.5 * (p + q)


def find_centre(spec, cx, cy, hw, levels, R=192, factor=2.0, prefer="junction"):
    """descend: at each level pick a boundary pixel near the centre (junction-rich if requested),
    re-centre on it. Returns the deep centre."""
    f = labeller(spec)
    for k in range(levels):
        lab = f(cx, cy, hw, R)
        b = boundary_mask(lab)
        if not b.any():
            print(f"  level {k}: no boundary in window", flush=True)
            break
        yy, xx = np.nonzero(b)
        dc = np.hypot(yy - R / 2, xx - R / 2) / R
        score = dc.copy()
        if prefer == "junction":
            from scipy.ndimage import generic_filter
            nlab = generic_filter(lab.astype(float), lambda v: len(np.unique(v)), size=5)
            score = dc - 0.08 * nlab[yy, xx]
        i = int(np.argmin(score))
        px = cx + hw * ((xx[i] + 0.5) / R * 2 - 1)
        py = cy + hw * ((yy[i] + 0.5) / R * 2 - 1)
        cx, cy = px, py
        hw = hw / factor
    return cx, cy


ZOOMS = {
    # smooth-flow hypothesis: analytic probability-flow ODE, and the learned DDIM-50 map
    "ode_scatter12": dict(map="sampler", layout="scatter12", score="analytic", kind="ode", hw0=3.0, levels=32, R=768, start=(0.137, -0.071)),
    "ddim50_learned_scatter12": dict(map="sampler", layout="scatter12", score="learned", kind="ddim50", hw0=3.0, levels=16, R=512, start=(0.137, -0.071)),
    # non-invertible over-relaxed denoiser
    "iter_ring8": dict(map="iter", layout="ring8", sigma=0.4, gamma=2.12, n_iter=2000, hw0=4.0, levels=36, R=768, start=(2.9, 2.9)),
    "iter_ring6": dict(map="iter", layout="ring6", sigma=0.4, gamma=2.10, n_iter=2000, hw0=4.0, levels=30, R=768, start=(1.1, 0.6)),
}


def task_zoom(tag):
    spec = ZOOMS[tag]
    t0 = time.time()
    cx, cy = find_centre(spec, *spec["start"], spec["hw0"], spec["levels"] - 4, R=128)
    print(f"zoom {tag}: centre ({cx!r}, {cy!r}) found in {time.time()-t0:.0f}s", flush=True)
    f = labeller(spec)
    labs = []
    for k in range(spec["levels"]):
        hw = spec["hw0"] * 2.0 ** (-k)
        labs.append(f(cx, cy, hw, spec["R"]))
        b = boundary_mask(labs[-1])
        print(f"  level {k} hw={hw:.3e} boundary px={b.sum()} labels={len(np.unique(labs[-1]))} {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(f"{CACHE}/zoom_{tag}.npz", lab=np.stack(labs), cx=cx, cy=cy, hw0=spec["hw0"], factor=2.0,
                            spec=json.dumps(spec))


# ----------------------------------------------------------------------------- verification
def newton_z3(cx, cy, hw, R, n_iter=200):
    P = grid(cx, cy, hw, R)
    z = torch.complex(P[:, 0], P[:, 1])
    for _ in range(n_iter):
        z = z - (z ** 3 - 1) / (3 * z ** 2 + 1e-300)
    roots = torch.tensor([1, complex(-0.5, math.sqrt(3) / 2), complex(-0.5, -math.sqrt(3) / 2)], device=DEV)
    return (z[:, None] - roots[None]).abs().argmin(1).reshape(R, R).cpu().numpy().astype(np.uint8)


def uncertainty_exponent(labfn_points, sample_box, eps_list, M=1 << 20, seed=0):
    """fraction f(eps) of random points whose label changes under a random displacement of size eps.
    f ~ eps^alpha, boundary dimension D = 2 - alpha (McDonald, Grebogi, Ott, Yorke 1985)"""
    g = torch.Generator(device=DEV).manual_seed(seed)
    (x0, x1), (y0, y1) = sample_box
    P = torch.stack([x0 + (x1 - x0) * torch.rand(M, device=DEV, generator=g, dtype=F64),
                     y0 + (y1 - y0) * torch.rand(M, device=DEV, generator=g, dtype=F64)], 1)
    th = 2 * math.pi * torch.rand(M, device=DEV, generator=g, dtype=F64)
    d = torch.stack([th.cos(), th.sin()], 1)
    l0 = labfn_points(P)
    fr = []
    for e in eps_list:
        fr.append(float((labfn_points(P + e * d) != l0).double().mean()))
    return np.array(fr)


def task_verify():
    t0 = time.time()
    out = {}
    # 1. global box counting at 2048^2 (iterated-map heroes are reused from cache/iter_*_hero.npz)
    R = 2048
    cases = {}
    cases["ode_scatter12"] = basin_map("scatter12", "analytic", "ode", R=R)[0]
    print(f"verify ode {R} {time.time()-t0:.0f}s", flush=True)
    cases["ddim50_scatter12"] = basin_map("scatter12", "analytic", "ddim50", R=R)[0]
    cases["ddim10_scatter12"] = basin_map("scatter12", "analytic", "ddim10", R=R)[0]
    for lay in ["ring8", "ring6"]:
        cases[f"iter_{lay}"] = np.load(f"{CACHE}/iter_{lay}_hero.npz")["lab"]
    cases["iter_ring8_gamma1"] = iter_map("ring8", 0.4, 1.0, 0, 0, 4.0, R, 300)[0]
    cases["newton_z3"] = newton_z3(0, 0, 2.0, R)
    # null models: a disc edge, and the exactly-straight rays of the symmetric ring8 analytic DDIM map
    yy, xx = np.mgrid[0:R, 0:R]
    cases["null_circle"] = ((xx - R / 2) ** 2 + (yy - R / 2) ** 2 < (0.37 * R) ** 2).astype(np.uint8)
    cases["null_ring8_rays"] = basin_map("ring8", "analytic", "ddim50", R=R)[0]
    box = {}
    for k, lab in cases.items():
        sizes, counts = box_count(boundary_mask(lab))
        D, se = fit_dimension(sizes, counts, lo=2, hi=256)
        box[k] = dict(sizes=sizes.tolist(), counts=counts.tolist(), D_2_256=D, se=se,
                      D_1_16=fit_dimension(sizes, counts, lo=1, hi=16)[0], D_32_512=fit_dimension(sizes, counts, lo=32, hi=512)[0])
        print(f"box {k}: D[2,256 px]={D:.3f}+-{se:.3f}  D[1,16]={box[k]['D_1_16']:.3f} D[32,512]={box[k]['D_32_512']:.3f}", flush=True)
    out["box2048"] = box
    np.savez_compressed(f"{CACHE}/verify_2048.npz", **cases)

    # 2. resolution check: same window at 256, 512, 1024, 2048
    reso = {}
    windows = {"ode_scatter12": dict(map="sampler", layout="scatter12", score="analytic", kind="ode"),
               "iter_ring8": dict(map="iter", layout="ring8", sigma=0.4, gamma=2.12, n_iter=N_ITER),
               "newton_z3": None}
    for k, spec in windows.items():
        f = labeller(spec) if spec else (lambda cx, cy, hw, R: newton_z3(cx, cy, hw, R))
        cx, cy, hw = {"ode_scatter12": (0.6, -0.4, 0.25), "iter_ring8": (2.2, 1.1, 0.3), "newton_z3": (-0.5, 0.3, 0.25)}[k]
        imgs = {r: f(cx, cy, hw, r) for r in [256, 512, 1024, 2048]}
        rr = {}
        for r in [512, 1024, 2048]:
            fct = r // 256
            sub = imgs[r][fct // 2::fct, fct // 2::fct]  # nearest fine pixel to each coarse centre
            rr[f"mismatch_256_vs_{r}"] = float((sub != imgs[256]).mean())
        for r in [256, 512, 1024, 2048]:
            rr[f"boundary_px_{r}"] = int(boundary_mask(imgs[r]).sum())
        reso[k] = rr
        np.savez_compressed(f"{CACHE}/resolution_{k}.npz", **{str(r): v for r, v in imgs.items()})
        print("resolution", k, rr, f"{time.time()-t0:.0f}s", flush=True)
    out["resolution"] = reso

    # 3. uncertainty exponent
    eps_list = [10.0 ** (-e) for e in np.arange(1, 5.25, 0.5)]
    M = 1 << 18
    ue = {}
    gs = GMM("scatter12", DEV)
    fns = {
        "ddim50_scatter12": (HALF, lambda P: gs.label(ddim(analytic_eps_fn(gs), P, 50))),
        "ode_scatter12": (HALF, lambda P: gs.label(ode_rk4(analytic_eps_fn(gs), P, ODE_STEPS, sigma_min=SIG_MIN_ANALYTIC))),
        "newton_z3": (2.0, None),
    }
    for lay in ["ring8", "ring6"]:
        sg, gm, hw = ITER_CFG[lay]
        gl = GMM(lay, DEV)
        fns[f"iter_{lay}"] = (hw, (lambda gl, sg, gm: lambda P: iterate_map(gl, sg, gm, P, N_ITER, track=False)[0])(gl, sg, gm))
    def newton_pts(P, n_iter=200):
        z = torch.complex(P[:, 0], P[:, 1])
        for _ in range(n_iter):
            z = z - (z ** 3 - 1) / (3 * z ** 2 + 1e-300)
        roots = torch.tensor([1, complex(-0.5, math.sqrt(3) / 2), complex(-0.5, -math.sqrt(3) / 2)], device=DEV)
        return (z[:, None] - roots[None]).abs().argmin(1)
    fns["newton_z3"] = (2.0, newton_pts)
    for k, (hw, fn) in fns.items():
        fr = uncertainty_exponent(lambda P, fn=fn: run_chunked(fn, P, 1 << 20), ((-hw, hw), (-hw, hw)), eps_list, M=M)
        sel = fr * M >= 25
        slope = np.polyfit(np.log(np.array(eps_list)[sel]), np.log(fr[sel]), 1)[0] if sel.sum() >= 3 else float("nan")
        ue[k] = dict(eps=eps_list, f=fr.tolist(), alpha=float(slope), D=float(2 - slope), n_fit=int(sel.sum()), M=M)
        print(f"uncertainty {k}: alpha={slope:.3f} D={2-slope:.3f} f={fr} {time.time()-t0:.0f}s", flush=True)
    out["uncertainty"] = ue
    out["wall_s"] = time.time() - t0
    json.dump(out, open(f"{CACHE}/verify_toy.json", "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "maps":
        task_maps(sys.argv[2])
    elif what == "steps":
        task_steps(sys.argv[2])
    elif what == "iter":
        for part in sys.argv[2:]:
            task_iter(part)
    elif what == "zoom":
        for tag in sys.argv[2:]:
            task_zoom(tag)
    elif what == "verify":
        task_verify()
