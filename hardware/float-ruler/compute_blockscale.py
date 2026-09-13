"""Block-scaled FP4: how the per-block scale moves the E2M1 comb.

MXFP4 (OCP MX v1.0; Rouhani et al. 2023, Algorithm 1):
    block of 32, X = 2^(floor(log2 amax) - emax_elem), emax_elem = 2 for E2M1,
    P_i = quantize_E2M1(V_i / X) with clamping to +-6 (round-half-to-even).
NVFP4 (NVIDIA; "Pretraining LLMs with NVFP4", arXiv:2509.25149):
    block of 16, per-tensor FP32 s_enc = 6*448 / amax_tensor,
    per-block E4M3 scale code c_b = quantize_E4M3(amax_b / 6 * s_enc), decode scale = c_b / s_enc,
    P_i = quantize_E2M1(V_i / scale).
(The spec allows other conversion recipes; these are the published reference ones.)

Sweep: one fixed 32-value vector (seeded Laplace) multiplied by a gain 2^u, u in [0, 4].
Caches the combs, scale codes, and quantised values for every frame.

    python compute_blockscale.py
"""
import numpy as np

from formats import FP4_E2M1, FP8_E4M3, positive_finite, round_to_format
from plate import CACHE

E2M1 = np.concatenate([[0.0], positive_finite(FP4_E2M1)["value"]])   # 0, .5, 1, 1.5, 2, 3, 4, 6
E4M3 = np.concatenate([[0.0], positive_finite(FP8_E4M3)["value"]])   # 0 ... 448


def mxfp4(v):
    amax = np.abs(v).max()
    k = int(np.floor(np.log2(amax))) - 2
    X = 2.0 ** k
    q = round_to_format(v / X, E2M1) * X
    return q, X, k + 127  # E8M0 code


def nvfp4(v, s_enc, block=16):
    q = np.empty_like(v)
    scales, codes = [], []
    for b in range(0, len(v), block):
        vb = v[b:b + block]
        amax = np.abs(vb).max()
        code = float(round_to_format(np.array([amax / 6.0 * s_enc]), E4M3)[0])
        scale = code / s_enc
        q[b:b + block] = round_to_format(vb / scale, E2M1) * scale if scale > 0 else 0.0
        scales.append(scale)
        codes.append(code)
    return q, np.array(scales), np.array(codes)


def main():
    rng = np.random.default_rng(7)
    base = rng.laplace(size=32)
    base /= np.abs(base).max()
    n = 720
    t = np.arange(n) / n
    u = 2.0 - 2.0 * np.cos(2 * np.pi * t)          # 0 -> 4 -> 0 octaves, smooth loop
    gains = 2.0 ** u
    amax_tensor = gains.max() * np.abs(base).max()
    s_enc = 6 * 448 / amax_tensor
    out = dict(base=base, gains=gains, u=u, s_enc=s_enc, E2M1=E2M1, E4M3=E4M3)
    mxq, mxX, mxc, nvq, nvs, nvc = [], [], [], [], [], []
    for g in gains:
        v = base * g
        q, X, c = mxfp4(v)
        mxq.append(q); mxX.append(X); mxc.append(c)
        q2, s2, c2 = nvfp4(v, s_enc)
        nvq.append(q2); nvs.append(s2); nvc.append(c2)
    out.update(mx_q=np.array(mxq), mx_scale=np.array(mxX), mx_code=np.array(mxc),
               nv_q=np.array(nvq), nv_scale=np.array(nvs), nv_code=np.array(nvc))
    V = base[None] * gains[:, None]
    out["mx_relmse"] = ((out["mx_q"] - V) ** 2).mean(1) / (V ** 2).mean(1)
    out["nv_relmse"] = ((out["nv_q"] - V) ** 2).mean(1) / (V ** 2).mean(1)
    # dense fan sweep for the still: block amax continuous over 8 octaves
    am = 2.0 ** np.linspace(-2, 6, 4000)
    s_enc_fan = 6 * 448 / am.max()
    mx_fan = 2.0 ** (np.floor(np.log2(am)) - 2)
    nv_fan = round_to_format(am / 6 * s_enc_fan, E4M3) / s_enc_fan
    out.update(fan_amax=am, fan_mx_scale=mx_fan, fan_nv_scale=nv_fan, fan_s_enc=s_enc_fan)
    np.savez(CACHE / "blockscale.npz", **out)
    print("mean rel MSE  MXFP4 %.4f  NVFP4 %.4f" % (out["mx_relmse"].mean(), out["nv_relmse"].mean()))
    print("distinct MX scale codes", np.unique(out["mx_code"]).size, " distinct NV codes", np.unique(out["nv_code"]).size)
    print("MX clip fraction of frames with |v|>6X:", np.mean(np.abs(V).max(1) > 6 * out["mx_scale"]))


if __name__ == "__main__":
    main()
