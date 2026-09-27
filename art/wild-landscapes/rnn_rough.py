"""How rough is the unrolled RNN loss, really?  Structure function of the loss along random short
segments, float64 (so the texture is the function, not rounding):

    S(delta) = RMS over base points p of [ log10 L(p + delta e) - log10 L(p) ],   e a random unit vector

for delta from 1e-9 to 1 (plane units), base points drawn uniformly in the chaotic zone (Lyapunov
exponent lambda > +0.05) and in the ordered zone (lambda < -0.05) of the span-2.5 plane, for several
horizons T.  A smooth (C^1) function has S ~ delta^1; a rough one S ~ delta^H with H < 1 (Hurst
exponent), graph dimension 3 - H for a surface.  A float32 vs float64 comparison measures how much
of the float32 hero maps is rounding.

    python rnn_rough.py  -> cache/rnn/rough.json, pilot/rnn_rough.png
"""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rnn_fractal as R


def main():
    torch.set_num_threads(16)
    rng = np.random.default_rng(0)
    deltas = np.logspace(-9, 0, 37)
    out = {"deltas": deltas.tolist()}
    for T in (10, 30, 100, 200):
        # classify candidate base points by their Lyapunov exponent (float64)
        a = rng.uniform(-2.5, 2.5, 20000); b = rng.uniform(-2.5, 2.5, 20000)
        L0, ly = R.plane(a, b, T, torch.float64)
        for zone, m in (("chaotic", ly > 0.05), ("ordered", ly < -0.05)):
            idx = np.nonzero(m)[0][:800]
            if len(idx) < 50:
                continue
            th = rng.uniform(0, 2 * np.pi, len(idx)); e = np.stack([np.cos(th), np.sin(th)], 1)
            A = (a[idx, None] + deltas[None] * e[:, :1]).ravel(); B = (b[idx, None] + deltas[None] * e[:, 1:]).ravel()
            L1, _ = R.plane(A, B, T, torch.float64, lyap=False)
            d = np.log10(L1.reshape(len(idx), -1)) - np.log10(L0[idx])[:, None]
            S = np.sqrt(np.mean(d ** 2, 0))
            # local slopes (Hurst exponent) in three bands of delta
            lg = np.log10(S); ld = np.log10(deltas)
            bands = {}
            for lo, hi in ((-9, -6), (-6, -3), (-3, -1), (-1, 0)):
                k = (ld >= lo) & (ld <= hi)
                bands[f"{lo}..{hi}"] = float(np.polyfit(ld[k], lg[k], 1)[0])
            out[f"T{T}_{zone}"] = dict(n=int(len(idx)), S=S.tolist(), H=bands, frac=float(m.mean()))
            print(T, zone, len(idx), {k: round(v, 3) for k, v in bands.items()}, flush=True)
    # float32 vs float64 on the hero grid geometry (T=30, 257^2)
    ax = np.linspace(-2.5, 2.5, 257); B_, A_ = np.meshgrid(ax, ax, indexing="ij")
    for T in (30, 100):
        L32, _ = R.plane(A_.ravel(), B_.ravel(), T, torch.float32, lyap=False)
        L64, ly = R.plane(A_.ravel(), B_.ravel(), T, torch.float64)
        dd = np.abs(np.log10(L32) - np.log10(L64))
        out[f"fp32_vs_fp64_T{T}"] = dict(median=float(np.median(dd)), p99=float(np.percentile(dd, 99)), max=float(dd.max()),
                                        median_chaotic=float(np.median(dd[ly > 0.05])))
        print("fp32 vs fp64", T, out[f"fp32_vs_fp64_T{T}"], flush=True)
    json.dump(out, open(os.path.join(HERE, "cache", "rnn", "rough.json"), "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, v in out.items():
        if k.startswith("T") and isinstance(v, dict):
            ax.loglog(deltas, v["S"], "-" if "chaotic" in k else "--", label=k)
    ax.loglog(deltas, deltas * 3, "k:", lw=0.8, label="slope 1 (smooth)")
    ax.set_xlabel("segment length delta (plane units)"); ax.set_ylabel("RMS change of log10 loss")
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(HERE, "pilot", "rnn_rough.png"), dpi=110)


if __name__ == "__main__":
    main()
