"""Estimator calibration: the same box-counting pipeline on synthetic planar Gaussian fields with a
pure power-law spectrum S(k) ~ k^-(2+2H) (level sets of dimension exactly 2 - H), with matched
band limits. Two protocols, mirroring the real measurements:
  tile:   2048^2, band limit 2.2 px wavelength, level = median, fit s in [4, 64]
  window: 2048^2, band limit 8 px wavelength, level = value at centre, fit s in [8, 64]
Output cache/calibration.json"""
import json, numpy as np, torch
torch.cuda.set_per_process_memory_fraction(0.10)
from common import boxcount, fit_dim

N, NS = 2048, 8
out = {}
dev = "cuda"
for proto, lam_min, smin, smax in [("tile", 2.2, 4, 64), ("window", 8.0, 8, 64)]:
    for H in [2.0 ** -L for L in range(1, 9)]:
        Ds = []
        for seed in range(NS if proto == 'tile' else 32):
            M = 2 * N          # generate on a 2x periodic box, use the central window (no wrap)
            g = torch.Generator(device=dev).manual_seed(seed)
            F = torch.fft.fft2(torch.randn(M, M, generator=g, device=dev, dtype=torch.float64))
            k = torch.fft.fftfreq(M, device=dev, dtype=torch.float64)
            kk = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
            amp = torch.where((kk > 0) & (kk <= 1.0 / lam_min), kk.clamp_min(1e-12) ** (-(1 + H)), torch.zeros_like(kk))
            f = torch.fft.ifft2(F * amp).real[N // 2:N // 2 + N, N // 2:N // 2 + N].cpu().numpy()
            lvl = np.median(f) if proto == "tile" else f[N // 2, N // 2]
            s, c = boxcount(f, lvl)
            Ds.append(fit_dim(s, c, smin, smax)[0])
        out[f"{proto}_H{H:.5f}"] = dict(proto=proto, H=H, D_true=2 - H, D_meas=float(np.mean(Ds)),
                                        D_sd=float(np.std(Ds)), all=Ds)
        print(proto, f"H={H:.4f} true {2-H:.4f} measured {np.mean(Ds):.3f} +- {np.std(Ds):.3f}", flush=True)
json.dump(out, open("cache/calibration.json", "w"), indent=1)
