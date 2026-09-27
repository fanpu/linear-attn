"""Numbers for the README (CPU, from cache/).  -> cache/stats.json"""
import json, os, sys
import numpy as np
from scipy.ndimage import map_coordinates
HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "cache")
out = {}


def rough(L):
    """RMS second difference of log10 loss (both axes), relative to the plane's 2-98% range."""
    d2 = np.sqrt(0.5 * (np.mean(np.diff(L, 2, 0) ** 2) + np.mean(np.diff(L, 2, 1) ** 2)))
    return float(d2), float(d2 / (np.percentile(L, 98) - np.percentile(L, 2)))


def boxdim(mask, sizes=(2, 4, 8, 16, 32, 64, 128)):
    """box-counting dimension of the boundary of a boolean mask."""
    b = (mask[1:, 1:] != mask[:-1, 1:]) | (mask[1:, 1:] != mask[1:, :-1])
    n = []
    for s in sizes:
        m = b[: b.shape[0] // s * s, : b.shape[1] // s * s]
        n.append(m.reshape(m.shape[0] // s, s, m.shape[1] // s, s).any((1, 3)).sum())
    k = np.polyfit(np.log(sizes), np.log(n), 1)[0]
    return float(-k), [int(v) for v in n]


# ---- RNN
f = os.path.join(C, "rnn", "rough.json")
if os.path.exists(f):
    r = json.load(open(f))
    out["rnn_structure_function"] = {k: v["H"] for k, v in r.items() if k.startswith("T")}
    out["rnn_fp32_vs_fp64"] = {k: v for k, v in r.items() if k.startswith("fp32")}
for T in (10, 30, 100):
    f = os.path.join(C, "rnn", f"plane_T{T}_r2049_s2.5.npz")
    if os.path.exists(f):
        d = np.load(f); L = np.log10(d["loss"]); ly = d["lyap"]
        D, n = boxdim(ly > 0)
        chaotic = ly > 0.05; ordered = ly < -0.05
        g = lambda m: float(np.sqrt(np.mean((np.diff(L, 2, 1)[:, :][m[:, 1:-1]]) ** 2)))
        out[f"rnn_T{T}"] = dict(chaotic_frac=float((ly > 0).mean()), edge_of_chaos_boxdim=D, box_counts=n,
                                rms_d2_chaotic=g(chaotic), rms_d2_ordered=g(ordered), log10_range=[float(L.min()), float(L.max())])

# ---- tiny nets
for name in ("p4_h4_rand_s5_r4096", "p4_h4_rand_s0.8_r4096"):
    f = os.path.join(C, "tiny", name + ".npz")
    if os.path.exists(f):
        d = np.load(f); F = d["loss"]; fn = d["fn"]
        u, c = np.unique(fn, return_counts=True)
        levels = np.round(F * 4) / 4
        flat = float((np.abs(F - levels) < 0.02).mean())
        out[f"tiny_{name}"] = dict(n_functions=int(len(u)), solved_frac=float((fn == int(d["ytrue"])).mean()),
                                   top_function_share=float(c.max() / c.sum()), on_terrace_frac=flat)

# ---- three minima: barriers along the three edges of the triangle
for al in (0, 1):
    f = os.path.join(C, "three", f"plane_a{al}_r512.npz")
    if os.path.exists(f):
        d = np.load(f); info = json.loads(str(d["info"])); L = d["loss"].astype(np.float64); span = float(d["span"])
        R = L.shape[0]
        def at(u, v):
            return map_coordinates(L, [[(v + span) / (2 * span) * (R - 1)], [(u + span) / (2 * span) * (R - 1)]], order=1)[0]
        mk = info["marks"]; bars = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            ts = np.linspace(0, 1, 201)
            vals = [at(mk[i][0] + t * (mk[j][0] - mk[i][0]), mk[i][1] + t * (mk[j][1] - mk[i][1])) for t in ts]
            bars.append(float(max(vals) - 0.5 * (vals[0] + vals[-1])))
        out[f"three_a{al}"] = dict(loss_at_nets=[float(at(*m)) for m in mk], loss_at_centroid=float(at(0, 0)),
                                   edge_barriers=bars, plane_max=float(L.max()), dist_between_nets=info["dist"],
                                   disputed_frac=info["disputed_frac"], align=info.get("align1"))

# ---- ResNet / GPT slices
for name in ("rand_resnet56_s1_r128", "hess_resnet56_s0.05_r128", "rand_resnet56_noshort_s3_r256", "rand_resnet56_s3_r128",
             "pilot_resnet56_noshort_s10_r64", "pilot_resnet56_s10_r64", "pilot_resnet56_noshort_s4_r64", "pilot_resnet56_s4_r64"):
    f = os.path.join(C, "resnet", name + ".npz")
    if os.path.exists(f):
        d = np.load(f); L = np.log10(d["loss"])
        r_abs, r_rel = rough(L)
        out[name] = dict(log10_range=[float(L.min()), float(L.max())], rms_d2=r_abs, rms_d2_rel=r_rel,
                         agree_min=float(d["agree"].min()), span=float(d["span"]), nonfinite=int((~np.isfinite(d["loss"])).sum()))
f = os.path.join(C, "gpt", "plane_adamw_s1_r128.npz")
if os.path.exists(f):
    d = np.load(f); L = np.log10(d["loss"]); r_abs, r_rel = rough(L)
    # isotropy: ratio of principal curvatures of a quadratic fit to the central 33x33
    c = L.shape[0] // 2; w = 16; ax = np.linspace(-1, 1, L.shape[0])[c - w:c + w + 1]
    X, Y = np.meshgrid(ax, ax); Z = L[c - w:c + w + 1, c - w:c + w + 1]
    A = np.stack([X.ravel() ** 2, Y.ravel() ** 2, X.ravel() * Y.ravel(), X.ravel(), Y.ravel(), np.ones(X.size)], 1)
    co = np.linalg.lstsq(A, Z.ravel(), rcond=None)[0]
    ev = np.linalg.eigvalsh([[2 * co[0], co[2]], [co[2], 2 * co[1]]])
    out["gpt_adamw_s1_r128"] = dict(log10_range=[float(L.min()), float(L.max())], rms_d2=r_abs, rms_d2_rel=r_rel,
                                     curvature_ratio=float(ev.max() / ev.min()), centre_loss=float(d["centre_loss"]))
    f2 = os.path.join(C, "resnet", "rand_resnet56_s1_r128.npz")
    if os.path.exists(f2):
        L2 = np.log10(np.load(f2)["loss"]); Z = L2[c - w:c + w + 1, c - w:c + w + 1]
        co = np.linalg.lstsq(A, Z.ravel(), rcond=None)[0]; ev = np.linalg.eigvalsh([[2 * co[0], co[2]], [co[2], 2 * co[1]]])
        out["rand_resnet56_s1_r128"]["curvature_ratio"] = float(ev.max() / ev.min())
f = os.path.join(C, "resnet", "hess_resnet56.pt")
if os.path.exists(f):
    import torch
    h = torch.load(f); out["hessian_resnet56"] = dict(top=h["vals"], most_negative=h["neg"], n=h["n"])
    f = os.path.join(C, "resnet", "hess_resnet56_noshort.pt")
    if os.path.exists(f):
        h = torch.load(f); out["hessian_resnet56_noshort"] = dict(top=h["vals"], most_negative=h["neg"], n=h["n"])
json.dump(out, open(os.path.join(C, "stats.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

# ---- zoom-out funnels: log10 L vs log10 r
fun = {}
for name in ("pilot_resnet56_noshort_s10_r64", "pilot_resnet56_s10_r64", "rand_resnet56_noshort_s3_r256", "rand_resnet56_s3_r128"):
    f = os.path.join(C, "resnet", name + ".npz")
    if os.path.exists(f):
        d = np.load(f); L = np.log10(d["loss"]); s = float(d["span"]); ax = np.linspace(-s, s, L.shape[0])
        X, Y = np.meshgrid(ax, ax); r = np.hypot(X, Y); m = (r > 1.5) & (r < s)
        p = np.polyfit(np.log10(r[m]), L[m], 1)
        fun[name] = dict(exponent=float(p[0]), resid_sd=float((L[m] - np.polyval(p, np.log10(r[m]))).std()),
                         basin_frac_below_chance=float((d["loss"] < np.log(10)).mean()))
out["zoomout_power_law"] = fun
json.dump(out, open(os.path.join(C, "stats.json"), "w"), indent=1)
print(json.dumps(fun, indent=1))
