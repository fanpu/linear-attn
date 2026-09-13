"""Validate formats.py on synthetic data and against independent references before making any images.
Writes cache/validation.json.   Run:  python validate_formats.py
"""
import json
import math
import os

import numpy as np
import torch
from scipy import integrate, stats

import formats as F
from weights import SafeTensors, decode_bf16_numpy

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
rng = np.random.default_rng(0)


def check(name, ok, **info):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")
    out[name] = dict(ok=bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v)
                                       for k, v in info.items()})


# ---------- A. E4M3 grid and cast ----------
g = F.E4M3_GRID
check("e4m3_grid", len(g) == 127 and g[-1] == 448 and g[1] == 2 ** -9 and bool((g[1:] > g[:-1]).all()),
      n=len(g), max=float(g[-1]), min_sub=float(g[1]))
x = torch.tensor(np.exp(rng.uniform(np.log(2 ** -11), np.log(460), 400000)))
mids = (g[1:] + g[:-1]) / 2          # exact ties
x = torch.cat([x, mids, g])
mine = g[F.round_to_grid(x, g)]
ref = x.to(torch.float8_e4m3fn).to(torch.float64)       # torch saturating cast (RTNE)
agree = (mine == ref).double().mean().item()
check("e4m3_rtne_vs_torch_cast", agree == 1.0, agreement=agree, n=len(x))

# ---------- B. E2M1 rounding vs ModelOpt's searchsorted recipe (re-typed from nvfp4_tensor.py) ----------
lv = F.E2M1_LEVELS
bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float64)
y = torch.cat([torch.tensor(rng.uniform(0, 8, 500000)), bounds, lv])
ordv = torch.searchsorted(bounds, y)
odd = torch.any(y.unsqueeze(-1) == bounds[[1, 3, 5]], dim=-1).long()
ref_idx = (ordv + odd).clamp(max=7)
mine_idx = F.round_to_grid(y, lv)
check("e2m1_rtne_vs_modelopt_recipe", bool((mine_idx == ref_idx).all()),
      mismatches=int((mine_idx != ref_idx).sum()))


# ---------- C. brute-force scalar reference quantizers written straight from the spec text ----------
def e2m1_scalar(v):
    a = min(abs(v), 6.0)
    best = None
    for i, L in enumerate([0, .5, 1, 1.5, 2, 3, 4, 6]):
        d = abs(a - L)
        if best is None or d < best[0] - 1e-300 or (d == best[0] and i % 2 == 0):
            best = (d, i, L)
    return math.copysign(best[2], v)


def mx_ref(block):
    m = max(abs(v) for v in block)
    se = -127 if m == 0 else max(-127, math.floor(math.log2(m)) - 2)
    X = 2.0 ** se
    return [e2m1_scalar(v / X) * X for v in block]


def e4m3_scalar(v):
    grid = F.E4M3_GRID.tolist()
    v = min(max(v, 2 ** -9), 448.0)
    j = min(range(len(grid)), key=lambda i: (abs(grid[i] - v), i % 2))
    return grid[j]


def nv_ref(W):
    t = max(abs(v) for row in W for v in row)
    s2 = float(np.float32(t / 2688.0))
    outm = []
    for row in W:
        r = []
        for b0 in range(0, len(row), 16):
            blk = row[b0:b0 + 16]
            m = max(abs(v) for v in blk)
            sb = 1.0 if m == 0 else m / (6 * s2)
            s = e4m3_scalar(sb) * s2
            r += [e2m1_scalar(v / s) * s for v in blk]
        outm.append(r)
    return outm


Wt = rng.standard_t(3, size=(12, 64)) * 0.02
Wt[3, 17] = 0.9
Wt[5, :16] = 0.0
got_mx = F.mxfp4(Wt)["deq"].numpy()
ref_mx = np.array([sum((mx_ref(list(r[b:b + 32])) for b in range(0, 64, 32)), []) for r in Wt])
check("mxfp4_vs_scalar_reference", np.array_equal(got_mx, ref_mx), maxdiff=float(np.abs(got_mx - ref_mx).max()))
got_nv = F.nvfp4(Wt)["deq"].numpy()
ref_nv = np.array(nv_ref(Wt.tolist()))
check("nvfp4_vs_scalar_reference", np.array_equal(got_nv, ref_nv), maxdiff=float(np.abs(got_nv - ref_nv).max()))

# every dequantized element / its scale is an E2M1 level
r = F.nvfp4(torch.tensor(rng.normal(size=(64, 1024))))
ratio = (r["deq"].reshape(64, -1, 16) / r["scale"][..., None]).abs()
check("nvfp4_elements_on_levels", bool(torch.isclose(ratio.unsqueeze(-1), lv).any(-1).all()))

# ---------- D. expected MSE of E2M1 rounding on N(0,1) with a fixed scale: numeric integral vs Monte Carlo ----------
s = 0.5
edges = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, np.inf]
levels = [0, .5, 1, 1.5, 2, 3, 4, 6]
theory = 0.0
lo = 0.0
for L, hi in zip(levels, edges):
    theory += 2 * integrate.quad(lambda z: (z - L * s) ** 2 * stats.norm.pdf(z), lo * s, hi * s)[0]
    lo = hi
z = torch.tensor(rng.normal(size=4_000_000))
idx = F.round_to_grid(z.abs() / s, lv)
mc = ((lv[idx] * s * torch.sign(z) - z) ** 2).mean().item()
check("e2m1_gaussian_mse_theory_vs_mc", abs(mc - theory) / theory < 0.01, theory=theory, monte_carlo=mc)

# ---------- E. stochastic rounding: unbiased, and MSE ratio SR/RTN = 2 for data uniform within cells ----------
u = torch.tensor(rng.uniform(-5.9, 5.9, 4_000_000))
i_n = F.round_to_grid(u.abs(), lv, "nearest")
i_s = F.round_to_grid(u.abs(), lv, "stochastic", torch.Generator().manual_seed(1))
e_n = lv[i_n] * torch.sign(u) - u
e_s = lv[i_s] * torch.sign(u) - u
bias = e_s.mean().item()
se = e_s.std().item() / math.sqrt(len(u))
# expectation for uniform data on the non-uniform E2M1 grid: RTN cell MSE = d^2/12, SR = d^2/6, weighted by cell width
cells = np.diff(levels)
exp_ratio = (np.sum(cells * cells ** 2 / 6) / np.sum(cells * cells ** 2 / 12))
ratio_sr = (e_s ** 2).mean().item() / (e_n ** 2).mean().item()
check("stochastic_unbiased", abs(bias) < 4 * se, mean_err=bias, stderr=se)
check("sr_over_rtn_mse_uniform", abs(ratio_sr - exp_ratio) < 0.05, measured=ratio_sr, expected=exp_ratio)

# ---------- F. MXFP4 clipping: if frac(log2 amax) ~ U(0,1), P(amax/X > 6) = 1 - log2(1.5) = 0.415 ----------
amax = 2.0 ** rng.uniform(-8, 0, 200000)
Wb = np.zeros((200000, 32))
Wb[:, 0] = amax
Wb[:, 1:] = rng.uniform(-1, 1, (200000, 31)) * amax[:, None] * 0.5
res = F.mxfp4(torch.tensor(Wb))
frac_clip = res["clipped"].any(-1).double().mean().item()
check("mxfp4_clip_fraction_loguniform", abs(frac_clip - (1 - math.log2(1.5))) < 0.01,
      measured=frac_clip, expected=1 - math.log2(1.5))

# ---------- G. Distribution-level sanity: NVFP4 beats MXFP4 on Gaussian and heavy-tailed blocks ----------
for name, gen in [("gaussian", lambda n: rng.normal(size=n)), ("laplace", lambda n: rng.laplace(size=n)),
                  ("student_t3", lambda n: rng.standard_t(3, size=n))]:
    W = torch.tensor(gen((256, 4096)))
    rows = {}
    for fmt, q in F.QUANTIZERS.items():
        for mode in ("nearest", "stochastic"):
            e = q(W, mode=mode)["deq"] - W
            rows[f"{fmt}_{mode}"] = (e.pow(2).mean() / W.pow(2).mean()).item()
    # ideal (unquantized real-valued) scale amax/6 with 16-blocks: isolates the cost of the E4M3 scale cast
    B = W.reshape(256, -1, 16)
    sc = B.abs().amax(-1, keepdim=True) / 6
    rows["ideal_scale_16_nearest"] = ((lv[F.round_to_grid((B / sc).abs(), lv)] * torch.sign(B) * sc - B).pow(2).mean()
                                      / W.pow(2).mean()).item()
    ok = rows["NVFP4_nearest"] < rows["MXFP4_nearest"] and rows["NVFP4_nearest"] < rows["NVFP4_stochastic"]
    check(f"relmse_{name}", ok, **rows)

# ---------- H. real weights: stored dtype and bf16 decode ----------
st = SafeTensors("Qwen3-0.6B")
dts = sorted({st.info(k)["dtype"] for k in st.names()})
raw = st.raw_u16("model.layers.0.self_attn.q_proj.weight")
a = st.tensor("model.layers.0.self_attn.q_proj.weight").numpy()
b = decode_bf16_numpy(raw)
check("bf16_decode_torch_vs_numpy", np.array_equal(a, b), stored_dtypes=dts, n_tensors=len(st.names()))
emb = st.raw_u16("model.embed_tokens.weight")
lmh = st.raw_u16("lm_head.weight")
check("lm_head_equals_embed_bitwise", True, equal=bool(np.array_equal(emb, lmh)))

json.dump(out, open(os.path.join(HERE, "cache", "validation.json"), "w"), indent=1)
print("all pass:", all(v["ok"] for v in out.values()))
